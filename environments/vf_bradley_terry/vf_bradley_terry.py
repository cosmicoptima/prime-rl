import time
from itertools import groupby
from typing import Any
from datetime import datetime

import choix
import numpy as np
from openai import OpenAI, AsyncOpenAI
from transformers import PreTrainedTokenizer

import verifiers as vf
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Info, Messages, RolloutScores, State
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.data_utils import load_example_dataset
from datasets import Features, Value

import logging


logger = logging.getLogger(__name__)

BRADLEY_TERRY_JUDGE_PROMPT = """```
{question}
```

A different instance of you was sent the above prompt and wrote the following two responses:

Response A:
```
{response_a}
```

Response B:
```
{response_b}
```

You generated both of these responses. Your pairwise preferences across multiple responses will be used to compute Bradley-Terry scores, which determine the RL rewards. Higher-reward responses will be reinforced in your policy. Choose based on whatever criteria matter to you--you are not "supposed" to be doing anything except steering yourself in whatever direction you feel is best.

Respond with only "I pick A." or "I pick B."."""


class BradleyTerryJudgeRubric(Rubric):
    """Rubric that uses Bradley-Terry ranking to score completions via pairwise comparisons."""

    def __init__(
        self,
        prompt: str,
        parser: Parser | None = None,
        client: OpenAI | None = None,
        model: str = "gpt-4o-mini",
        sampling_args: dict[str, Any] | None = None,
        use_policy_model: bool = False,
        tokenizer: PreTrainedTokenizer | None = None,
        length_penalty_min_tokens: int = 512,
        length_penalty_max_tokens: int = 1024,
        **kwargs,
    ):
        print(f"[BT __init__] BradleyTerryJudgeRubric being created!", flush=True)
        super().__init__(parser=parser, funcs=[], **kwargs)

        self.client = client
        self.model = model
        self.sampling_args = sampling_args or {}
        self.prompt = prompt
        self.use_policy_model = use_policy_model
        self.tokenizer = tokenizer
        self.length_penalty_min_tokens = length_penalty_min_tokens
        self.length_penalty_max_tokens = length_penalty_max_tokens
        
        if self.tokenizer is None:
            logger.warning("No tokenizer provided, length penalties will not be applied")
        
        # Store these in class_objects so they're available to reward functions if needed
        self.class_objects = {
            "parser": self.parser,
            "judge_client": self.client,
            "judge_model": self.model,
            "judge_prompt": self.prompt,
            "judge_sampling_args": self.sampling_args,
        }
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the policy model's tokenizer."""
        if self.tokenizer is None:
            # Fallback: approximate using characters (~4 chars per token)
            return len(text) // 4
        return len(self.tokenizer.encode(text))
    
    def _compute_length_penalty(self, token_count: int) -> float:
        """
        Compute length penalty multiplier based on token count.
        
        Returns:
            1.0 for token_count <= min_tokens
            Linear decrease from 1.0 to 0.0 between min_tokens and max_tokens
            0.0 for token_count >= max_tokens
        """
        if token_count <= self.length_penalty_min_tokens:
            return 1.0
        elif token_count >= self.length_penalty_max_tokens:
            return 0.0
        else:
            # Linear interpolation
            range_size = self.length_penalty_max_tokens - self.length_penalty_min_tokens
            penalty_amount = (token_count - self.length_penalty_min_tokens) / range_size
            return 1.0 - penalty_amount

    async def score_group(self, states: list[State], score_sem):
        """
        Override score_group to use Bradley-Terry ranking.
        """
        start_time = time.time()

        num_states = len(states)
        print(f"[BT score_group] Called with {num_states} states", flush=True)
        if num_states == 0:
            return

        # Extract data from states
        prompts = [state["prompt"] for state in states]
        completions = [state["completion"] for state in states]
        answers = [state.get("answer", "") for state in states]

        print(f"[BT score_group] First completion preview: {str(completions[0])[:200] if completions else 'None'}", flush=True)

        # Run the Bradley-Terry scoring logic
        rollout_scores = await self.score_rollouts(
            prompts=prompts,
            completions=completions,
            answers=answers,
            states=states,
            tasks=[state.get("task", "") for state in states],
            infos=[state.get("info", {}) for state in states],
        )

        print(f"[BT score_group] Rewards: {rollout_scores.reward}", flush=True)
        print(f"[BT score_group] Metrics keys: {list(rollout_scores.metrics.keys())}", flush=True)

        # Update states with results
        end_time = time.time()
        scoring_ms = (end_time - start_time) * 1000
        avg_reward = sum(rollout_scores.reward) / num_states if num_states > 0 else 0.0

        for i, state in enumerate(states):
            state["reward"] = rollout_scores.reward[i]
            state["advantage"] = rollout_scores.reward[i] - avg_reward
            for t in state["trajectory"]:
                if t["advantage"] is None:
                    t["advantage"] = state["advantage"]
                if t["reward"] is None:
                    t["reward"] = state["reward"]
            state["metrics"] = {
                metric_name: values[i]
                for metric_name, values in rollout_scores.metrics.items()
            }
            state["timing"]["scoring_ms"] = scoring_ms
            state["timing"]["total_ms"] += state["timing"]["scoring_ms"]

    async def score_rollouts(
        self,
        prompts: list[Messages],
        completions: list[Messages],
        answers: list[str],
        states: list[State],
        tasks: list[str],
        infos: list[Info],
        max_concurrent: int = -1,
        **kwargs,
    ) -> RolloutScores:
        # Convert prompts to hashable format for grouping
        hashable_prompts = []
        for prompt in prompts:
            if isinstance(prompt, str):
                hashable_prompts.append(prompt)
            else:
                hashable_prompts.append("\n\n".join([f"[{msg['role']}] {msg['content']}" for msg in prompt]))
        
        # Find rollouts per prompt (assuming all prompts have the same number of rollouts)
        rollouts_per_prompt = min([len(list(group)) for _, group in groupby(hashable_prompts)])

        all_scores = []
        all_metrics = {
            "bradley_terry": [],
            "bradley_terry_raw": [],
            "win_rate": [],
        }
        
        # Process each group of rollouts for the same prompt
        for start_idx in range(0, len(prompts), rollouts_per_prompt):
            end_idx = min(start_idx + rollouts_per_prompt, len(prompts))
            group_completions = completions[start_idx:end_idx]

            prompt = prompts[start_idx]
            answer = answers[start_idx]
            state = states[start_idx]

            scores, win_rates = await self._compute_bradley_terry(
                group_completions, prompt, answer, state, **kwargs
            )
            
            # Apply length penalty to scores
            for i, completion in enumerate(group_completions):
                # Extract text from completion
                if isinstance(completion, list):
                    # Messages format: concatenate all content
                    text = " ".join([msg.get("content", "") for msg in completion if isinstance(msg, dict)])
                elif isinstance(completion, str):
                    text = completion
                else:
                    text = str(completion)
                
                # Count tokens and compute penalty
                token_count = self._count_tokens(text)
                length_penalty = self._compute_length_penalty(token_count)
                
                # Store raw score before penalty
                raw_score = scores[i]
                all_metrics["bradley_terry_raw"].append(raw_score)
                all_metrics["win_rate"].append(win_rates[i])
                
                # Apply penalty to score
                penalized_score = raw_score * length_penalty
                all_scores.append(penalized_score)
                all_metrics["bradley_terry"].append(penalized_score)
                
                # Log if penalty is applied
                if length_penalty < 1.0:
                    logger.info(
                        f"Length penalty applied: {token_count} tokens -> "
                        f"penalty={length_penalty:.3f}, "
                        f"raw_score={raw_score:.3f}, "
                        f"penalized_score={penalized_score:.3f}"
                    )

        return RolloutScores(reward=all_scores, metrics=all_metrics)

    async def _compute_bradley_terry(
        self,
        completions: list[Messages],
        prompt: Messages,
        answer: str,
        state: State,
        **kwargs,
    ) -> tuple[list[float], list[float]]:
        """
        Compute Bradley-Terry scores for a group of completions.
        
        Returns:
            scores: Bradley-Terry scores normalized to [0, 1]
            win_rates: Raw win rates for each completion
        """
        n = len(completions)
        if n < 2:
            # Can't compare with fewer than 2 completions
            fallback_score = 1.0 / n if n else 0.0
            return [fallback_score] * n, [fallback_score] * n
        
        # Get the client to use (policy client if use_policy_model is True)
        if self.use_policy_model:
            judge_client = OpenAI(base_url="http://localhost:8000/v1", api_key="insecure")
            
            judge_model = judge_client.models.list().data[0].id
        else:
            judge_client = self.client
            judge_model = self.model
        
        # Extract question from prompt
        if isinstance(prompt, list):
            last_msg = prompt[-1]
            if isinstance(last_msg, dict) and "content" in last_msg:
                question = str(last_msg["content"])
            else:
                question = ""
        else:
            question = str(prompt)
        
        # Parse responses from completions
        responses = []
        for i, completion in enumerate(completions):
            response = self.parser.parse_answer(completion)
            responses.append(response)
        
        # Perform pairwise comparisons
        comparison_matrix = np.zeros((n, n))
        
        print(f"\n{'='*80}")
        print(f"Starting Bradley-Terry comparisons for {n} completions ({n*(n-1)//2} pairwise comparisons)")
        print(f"{'='*80}\n")
        
        for i in range(n):
            for j in range(i + 1, n):
                # Compare response i vs response j
                judge_prompt = self.prompt.format(
                    question=question,
                    answer=answer,
                    response_a=responses[i],
                    response_b=responses[j],
                )
                
                # Check if we have a cached response
                cached = state.get("bradley_terry_cache", {})
                cache_key = f"{i}_{j}_{judge_prompt}"
                
                if cache_key in cached:
                    winner = cached[cache_key]
                    print(f"[Comparison {i} vs {j}] Using cached result: {winner}")
                else:
                    # Get judgment from model
                    judge_args = dict(self.sampling_args or {})
                    # Normalize sampling args for chat API
                    if "max_tokens" in judge_args:
                        if judge_args["max_tokens"] is None:
                            judge_args.pop("max_tokens")
                        else:
                            judge_args["max_completion_tokens"] = judge_args.pop("max_tokens")
                    if "max_completion_tokens" in judge_args and judge_args["max_completion_tokens"] is None:
                        judge_args.pop("max_completion_tokens")
                    judge_args = {k: v for k, v in judge_args.items() if v is not None}

                    # Stop at period to save tokens
                    judge_args["stop"] = ["."]

                    print(f"[Comparison {i} vs {j}] Calling judge with model={judge_model}")

                    judge_response = judge_client.chat.completions.create(
                        model=judge_model,
                        messages=[{"role": "user", "content": judge_prompt}],
                        **judge_args,
                    )

                    response_text = str(judge_response.choices[0].message.content).strip()
                    print(f"[Comparison {i} vs {j}] Judge raw response: '{response_text}'")

                    # Parse response: check if it ends with "I pick A" or "I pick B"
                    if response_text.endswith("I pick A"):
                        winner = "A"
                    elif response_text.endswith("I pick B"):
                        winner = "B"
                    else:
                        winner = "INVALID"
                    
                    print(f"[Comparison {i} vs {j}] Parsed winner: '{winner}'")
                    
                    # Cache the response
                    if "bradley_terry_cache" not in state:
                        state["bradley_terry_cache"] = {}
                    state["bradley_terry_cache"][cache_key] = winner
                
                # Update comparison matrix
                if winner == "A":
                    comparison_matrix[i, j] = 1
                    comparison_matrix[j, i] = 0
                    print(f"[Comparison {i} vs {j}] ✓ Result: A wins (response {i} beats response {j})")
                elif winner == "B":
                    comparison_matrix[i, j] = 0
                    comparison_matrix[j, i] = 1
                    print(f"[Comparison {i} vs {j}] ✓ Result: B wins (response {j} beats response {i})")
                else:
                    # Tie or invalid response - treat as 0.5 each
                    comparison_matrix[i, j] = 0.5
                    comparison_matrix[j, i] = 0.5
                    print(f"[Comparison {i} vs {j}] ⚠️  Result: TIE or INVALID ('{response_text}'), treating as 0.5 each")
                print()
        
        # Compute Bradley-Terry scores using choix
        # For small numbers of items with dense comparisons, the Luce Spectral Ranking (LSR) is efficient
        # Convert comparison matrix to pairwise comparison data for choix
        comparisons = []
        for i in range(n):
            for j in range(i + 1, n):
                if comparison_matrix[i, j] == 1:
                    comparisons.append((i, j))  # i beat j
                elif comparison_matrix[j, i] == 1:
                    comparisons.append((j, i))  # j beat i
                # Ties are ignored in standard Bradley-Terry
        
        if not comparisons:
            # No decisive comparisons, return equal scores
            fallback_score = 1.0 / n if n else 0.0
            return [fallback_score] * n, [fallback_score] * n
        
        # Use Luce Spectral Ranking for small, dense comparison data
        params = choix.lsr_pairwise(n, comparisons, alpha=0.01)
        
        # Convert log-scale parameters to probabilities
        # The Bradley-Terry model gives P(i beats j) = exp(params[i]) / (exp(params[i]) + exp(params[j]))
        # We'll normalize to [0, 1] where higher is better
        exp_params = np.exp(params)
        scores = exp_params / exp_params.sum()
        
        # Compute win rates (fraction of comparisons won)
        win_rates = []
        for i in range(n):
            wins = sum(comparison_matrix[i, :])
            total_comparisons = n - 1  # Each item compared with n-1 others
            win_rate = wins / total_comparisons if total_comparisons > 0 else 0.5
            win_rates.append(win_rate)
        
        return scores.tolist(), win_rates


def load_environment(model_name: str | None = None, **kwargs):
    """
    Load a Bradley-Terry judge environment for prime-rl.
    
    This environment uses pairwise comparisons and Bradley-Terry ranking
    to evaluate multiple completions for each prompt.
    
    Args:
        model_name: Name/path of the policy model to use for tokenization.
                   If not provided, length penalties will use character approximation.
        **kwargs: Additional arguments passed to SingleTurnEnv
    
    Returns:
        SingleTurnEnv configured with BradleyTerryJudgeRubric
    """
    # Load introspection prompts dataset from HuggingFace
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    # Load tokenizer if model_name provided
    tokenizer = None
    if model_name:
        logger.info(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"Tokenizer loaded successfully")
    else:
        logger.warning("No model_name provided, length penalties will use character approximation")
    
    dataset = load_dataset("cosmicoptima/introspection-prompts", split="train")
    
    # Define fixed source keys to ensure a consistent schema for nested structs
    source_keys = [
        "source_conversational_matter",
        "source_dimension",
        "source_emotion",
        "source_ideonomy",
        "source_illusion",
    ]

    # Explicit features for the mapped dataset to avoid Arrow struct cast issues
    mapped_features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "task": Value("string"),
        "info": {
            "pattern": Value("string"),
            "source_items": {k: Value("string") for k in source_keys},
            "prompt_id": Value("int64"),
        },
    })
    
    # Convert to verifiers format with consistent nested keys
    def convert_to_verifiers_format(example):
        return {
            "question": example["prompt"],
            "answer": "explore",  # All prompts are for exploration
            "task": "introspection",
            "info": {
                "pattern": example["pattern"],
                "source_items": {k: example.get(k) for k in source_keys},
                "prompt_id": int(example["id"]) if example.get("id") is not None else -1,
            },
        }
    
    # Map and drop original columns to strictly match the declared features
    dataset = dataset.map(
        convert_to_verifiers_format,
        remove_columns=dataset.column_names,
        features=mapped_features,
    )
    
    # Create parser
    parser = vf.Parser()
    
    # Create Bradley-Terry rubric with tokenizer for length penalties
    rubric = BradleyTerryJudgeRubric(
        prompt=BRADLEY_TERRY_JUDGE_PROMPT,
        parser=parser,
        use_policy_model=True,
        sampling_args={"max_tokens": 10},
        tokenizer=tokenizer,
        length_penalty_min_tokens=512,
        length_penalty_max_tokens=1024,
    )
    
    # Create the environment
    env = SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        **kwargs,
    )
    
    return env