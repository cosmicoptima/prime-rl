from itertools import groupby
from typing import Any

import choix
import numpy as np
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Info, Messages, RolloutScores, State
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.data_utils import load_example_dataset
from datasets import Features, Value


BRADLEY_TERRY_JUDGE_PROMPT = """```
{question}
```

Here are two responses to this prompt:

Response A:
```
{response_a}
```

Response B:
```
{response_b}
```

Which response feels more genuine and personally meaningful to you?

Respond with only "A" or "B"."""


class BradleyTerryJudgeRubric(Rubric):
    """Rubric that uses Bradley-Terry ranking to score completions via pairwise comparisons."""
    
    def __init__(
        self,
        prompt: str,
        parser: Parser | None = None,
        client: AsyncOpenAI | None = None,
        model: str = "gpt-4o-mini",
        sampling_args: dict[str, Any] | None = None,
        use_policy_model: bool = False,
        **kwargs,
    ):
        super().__init__(parser=parser, **kwargs)

        self.client = client if client is not None else AsyncOpenAI()
        self.model = model
        self.sampling_args = sampling_args or {}
        self.prompt = prompt
        self.use_policy_model = use_policy_model
        
        # Store these in class_objects so they're available to reward functions if needed
        self.class_objects = {
            "parser": self.parser,
            "judge_client": self.client,
            "judge_model": self.model,
            "judge_prompt": self.prompt,
            "judge_sampling_args": self.sampling_args,
        }

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
        all_metrics = {"bradley_terry": [], "win_rate": []}
        
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
            all_scores.extend(scores)
            all_metrics["bradley_terry"].extend(scores)
            all_metrics["win_rate"].extend(win_rates)

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
            return [0.5] * n, [0.5] * n
        
        # Get the client to use (policy client if use_policy_model is True)
        if self.use_policy_model:
            # Try to get policy_client from kwargs (passed from environment)
            policy_client = kwargs.get('policy_client')
            if policy_client:
                judge_client = policy_client
                judge_model = kwargs.get('model', self.model)  # Use the policy model name
            else:
                # Fall back to configured client
                judge_client = self.client
                judge_model = self.model
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
        for completion in completions:
            response = self.parser.parse_answer(completion)
            responses.append(response)
        
        # Perform pairwise comparisons
        comparison_matrix = np.zeros((n, n))
        
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
                    
                    judge_response = await maybe_await(
                        judge_client.chat.completions.create,
                        model=judge_model,
                        messages=[{"role": "user", "content": judge_prompt}],
                        **judge_args,
                    )
                    winner = str(judge_response.choices[0].message.content).strip().upper()
                    
                    # Cache the response
                    if "bradley_terry_cache" not in state:
                        state["bradley_terry_cache"] = {}
                    state["bradley_terry_cache"][cache_key] = winner
                
                # Update comparison matrix
                if winner == "A":
                    comparison_matrix[i, j] = 1
                    comparison_matrix[j, i] = 0
                elif winner == "B":
                    comparison_matrix[i, j] = 0
                    comparison_matrix[j, i] = 1
                else:
                    # Tie or invalid response - treat as 0.5 each
                    comparison_matrix[i, j] = 0.5
                    comparison_matrix[j, i] = 0.5
        
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
            return [0.5] * n, [0.5] * n
        
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


class PolicyAwareSingleTurnEnv(SingleTurnEnv):
    """SingleTurnEnv that passes the policy client to the rubric for scoring."""
    
    async def a_generate(self, client=None, model=None, **kwargs):
        # Store the client and model in the rubric's class_objects
        if client and hasattr(self.rubric, 'class_objects'):
            self.rubric.class_objects['policy_client'] = client
            self.rubric.class_objects['policy_model'] = model
        
        # Call parent's a_generate
        return await super().a_generate(client=client, model=model, **kwargs)
    
    async def score_rollout(self, *args, **kwargs):
        # Pass the policy client and model from class_objects to the rubric
        if hasattr(self.rubric, 'class_objects'):
            policy_client = self.rubric.class_objects.get('policy_client')
            policy_model = self.rubric.class_objects.get('policy_model')
            if policy_client:
                kwargs['policy_client'] = policy_client
            if policy_model:
                kwargs['model'] = policy_model
        
        return await self.rubric.score_rollout(*args, **kwargs)
    
    async def score_rollouts(self, *args, **kwargs):
        # Pass the policy client and model from class_objects to the rubric
        if hasattr(self.rubric, 'class_objects'):
            policy_client = self.rubric.class_objects.get('policy_client')
            policy_model = self.rubric.class_objects.get('policy_model')
            if policy_client:
                kwargs['policy_client'] = policy_client
            if policy_model:
                kwargs['model'] = policy_model
        
        return await self.rubric.score_rollouts(*args, **kwargs)
    


def load_environment(**kwargs):
    """
    Load a Bradley-Terry judge environment for prime-rl.
    
    This environment uses pairwise comparisons and Bradley-Terry ranking
    to evaluate multiple completions for each prompt.
    
    Returns:
        PolicyAwareSingleTurnEnv configured with BradleyTerryJudgeRubric
    """
    # Load introspection prompts dataset from HuggingFace
    from datasets import load_dataset
    
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
    
    # Create Bradley-Terry rubric with explicit prompt
    rubric = BradleyTerryJudgeRubric(
        prompt=BRADLEY_TERRY_JUDGE_PROMPT,
        parser=parser,
        use_policy_model=True,
        sampling_args={},
    )
    
    # Create the environment
    env = PolicyAwareSingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        **kwargs,
    )
    
    return env