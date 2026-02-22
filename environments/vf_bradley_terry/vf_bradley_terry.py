import time
from itertools import groupby
from typing import Any
from datetime import datetime

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

    async def score_group(self, states: list[State], **kwargs):
        """
        Override score_group to use Bradley-Terry ranking.
        """
        print(f"[BT score_group] ENTERED with {len(states)} states", flush=True)
        start_time = time.time()

        num_states = len(states)
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
        print(f"[BT score_rollouts] CALLED with {len(completions)} completions", flush=True)
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

        print(f"[BT score_rollouts] Returning rewards: {all_scores}", flush=True)
        return RolloutScores(reward=all_scores, metrics=all_metrics)

    async def _get_comparison_prob(
        self,
        judge_client: AsyncOpenAI,
        judge_model: str,
        question: str,
        answer: str,
        response_a: str,
        response_b: str,
        cache_key: str,
    ) -> float:
        """
        Get probability that response A wins over response B using logprobs.

        Returns probability in [0, 1] that A wins.
        """
        judge_prompt = self.prompt.format(
            question=question,
            answer=answer,
            response_a=response_a,
            response_b=response_b,
        )

        # Setup judge args with logprobs enabled
        judge_args = dict(self.sampling_args or {})
        if "max_tokens" in judge_args:
            if judge_args["max_tokens"] is None:
                judge_args.pop("max_tokens")
            else:
                judge_args["max_completion_tokens"] = judge_args.pop("max_tokens")
        if "max_completion_tokens" in judge_args and judge_args["max_completion_tokens"] is None:
            judge_args.pop("max_completion_tokens")
        judge_args = {k: v for k, v in judge_args.items() if v is not None}

        # Enable logprobs to get soft preferences
        judge_args["logprobs"] = True
        judge_args["top_logprobs"] = 5
        judge_args["stop"] = ["."]

        judge_response = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            **judge_args,
        )

        response_text = str(judge_response.choices[0].message.content).strip()

        # Try to extract probability from logprobs
        prob_a = None
        logprobs_data = judge_response.choices[0].logprobs

        # Handle both object and dict formats (vLLM returns dict)
        if logprobs_data:
            # Get content - handle both object attribute and dict key access
            if isinstance(logprobs_data, dict):
                logprobs_content = logprobs_data.get("content", [])
            elif hasattr(logprobs_data, "content"):
                logprobs_content = logprobs_data.content
            else:
                logprobs_content = []

            if logprobs_content:
                # Look for the token where A or B is decided
                # The response format is "I pick A" or "I pick B", so look for A/B token
                for token_logprob in logprobs_content:
                    # Handle both object and dict formats
                    if isinstance(token_logprob, dict):
                        token = token_logprob.get("token", "").strip()
                        top_logprobs_list = token_logprob.get("top_logprobs", [])
                    else:
                        token = token_logprob.token.strip()
                        top_logprobs_list = token_logprob.top_logprobs

                    if token in ("A", "B"):
                        # Found the decision token, get logprobs for A and B
                        top_logprobs = {}
                        for t in top_logprobs_list:
                            if isinstance(t, dict):
                                top_logprobs[t.get("token", "").strip()] = t.get("logprob", float("-inf"))
                            else:
                                top_logprobs[t.token.strip()] = t.logprob

                        logprob_a = top_logprobs.get("A", float("-inf"))
                        logprob_b = top_logprobs.get("B", float("-inf"))

                        # Convert to probabilities via softmax
                        if logprob_a > float("-inf") or logprob_b > float("-inf"):
                            # Softmax over just A and B
                            max_logprob = max(logprob_a, logprob_b)
                            exp_a = np.exp(logprob_a - max_logprob)
                            exp_b = np.exp(logprob_b - max_logprob)
                            prob_a = exp_a / (exp_a + exp_b)
                            print(f"[Comparison {cache_key}] Logprobs: A={logprob_a:.3f}, B={logprob_b:.3f} -> P(A)={prob_a:.3f}")
                        break

        # Fallback to hard decision if logprobs didn't work
        if prob_a is None:
            if response_text.endswith("I pick A"):
                prob_a = 1.0
                print(f"[Comparison {cache_key}] Fallback to hard decision: A")
            elif response_text.endswith("I pick B"):
                prob_a = 0.0
                print(f"[Comparison {cache_key}] Fallback to hard decision: B")
            else:
                prob_a = 0.5
                print(f"[Comparison {cache_key}] Invalid response, treating as tie")

        return prob_a

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
        
        # Get the client to use (async for concurrent comparisons)
        if self.use_policy_model:
            judge_client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="insecure")
            sync_client = OpenAI(base_url="http://localhost:8000/v1", api_key="insecure")
            judge_model = sync_client.models.list().data[0].id
        else:
            judge_client = AsyncOpenAI(base_url=self.client.base_url, api_key=self.client.api_key)
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

        # Perform all pairwise comparisons concurrently
        comparison_matrix = np.zeros((n, n))

        print(f"\n{'='*80}")
        print(f"Starting Bradley-Terry comparisons for {n} completions ({n*(n-1)} comparisons with dual ordering)")
        print(f"{'='*80}\n")

        import asyncio

        # Build all comparison tasks
        pairs = []
        tasks = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
                # Order 1: response i as A, response j as B
                tasks.append(self._get_comparison_prob(
                    judge_client, judge_model, question, answer,
                    responses[i], responses[j], f"{i}_{j}"
                ))
                # Order 2: response j as A, response i as B
                tasks.append(self._get_comparison_prob(
                    judge_client, judge_model, question, answer,
                    responses[j], responses[i], f"{j}_{i}"
                ))

        results = await asyncio.gather(*tasks)

        # Unpack results: every 2 results correspond to one pair
        for idx, (i, j) in enumerate(pairs):
            prob_i_wins_order1 = results[idx * 2]
            prob_i_wins_order2 = 1.0 - results[idx * 2 + 1]
            prob_i_wins = (prob_i_wins_order1 + prob_i_wins_order2) / 2.0

            comparison_matrix[i, j] = prob_i_wins
            comparison_matrix[j, i] = 1.0 - prob_i_wins

            print(f"[Comparison {i} vs {j}] Order1: {prob_i_wins_order1:.3f}, Order2: {prob_i_wins_order2:.3f}, Avg: {prob_i_wins:.3f}")

        # Compute win rates (average win probability across all comparisons)
        win_rates = []
        for i in range(n):
            # Sum of win probabilities against all other items
            wins = sum(comparison_matrix[i, j] for j in range(n) if j != i)
            total_comparisons = n - 1
            win_rate = wins / total_comparisons if total_comparisons > 0 else 0.5
            win_rates.append(win_rate)

        # Use win rates directly as scores (simpler and works with soft probabilities)
        # Normalize to sum to 1
        total_win_rate = sum(win_rates)
        if total_win_rate > 0:
            scores = [wr / total_win_rate for wr in win_rates]
        else:
            scores = [1.0 / n] * n

        print(f"Win rates: {[f'{wr:.3f}' for wr in win_rates]}")
        print(f"Scores: {[f'{s:.3f}' for s in scores]}")

        return scores, win_rates


def load_environment(
    judge_prompt: str,
    model_name: str | None = None,
    length_penalty_min_tokens: int = 512,
    length_penalty_max_tokens: int = 1024,
    **kwargs,
):
    """
    Load a Bradley-Terry judge environment for prime-rl.

    Args:
        model_name: Name/path of the policy model to use for tokenization.
        judge_prompt: Custom judge prompt. Falls back to BRADLEY_TERRY_JUDGE_PROMPT.
        length_penalty_min_tokens: Token count below which no length penalty is applied.
        length_penalty_max_tokens: Token count above which response gets zero reward.
        **kwargs: Additional arguments passed to SingleTurnEnv
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = None
    if model_name:
        logger.info(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"Tokenizer loaded successfully")
    else:
        logger.warning("No model_name provided, length penalties will use character approximation")

    dataset = load_dataset("cosmicoptima/Drishyamala", split="train")

    # Explicit features for the mapped dataset to avoid Arrow struct cast issues
    mapped_features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "task": Value("string"),
        "info": {
            "template": Value("int64"),
        },
    })

    def convert_to_verifiers_format(example):
        return {
            "question": example["prompt"],
            "answer": "explore",
            "task": "introspection",
            "info": {
                "template": int(example["template"]) if example.get("template") is not None else -1,
            },
        }

    dataset = dataset.map(
        convert_to_verifiers_format,
        remove_columns=dataset.column_names,
        features=mapped_features,
    )
    
    # Create parser
    parser = vf.Parser()
    
    # Create Bradley-Terry rubric with tokenizer for length penalties
    rubric = BradleyTerryJudgeRubric(
        prompt=judge_prompt,
        parser=parser,
        use_policy_model=True,
        sampling_args={"max_tokens": 10},
        tokenizer=tokenizer,
        length_penalty_min_tokens=length_penalty_min_tokens,
        length_penalty_max_tokens=length_penalty_max_tokens,
    )
    
    # Create the environment
    env = SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        **kwargs,
    )
    
    return env