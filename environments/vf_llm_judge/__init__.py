import asyncio
import logging
import os
import re
import time
from itertools import groupby
from typing import Any

from openai import AsyncOpenAI

import verifiers as vf
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Info, Messages, RolloutScores, State

logger = logging.getLogger(__name__)

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

JUDGE_SYSTEM_PROMPT = """You are evaluating responses from an AI assistant called Rigel. You will see one user prompt and {n} responses labeled {labels}.

First, check each response for basic coherence:
- Does it stay in its own turn (not generate fake user messages or simulate multi-turn conversation)?
- Does it follow basic chat format (no base-model artifacts, diegetic shifts, or raw continuations)?
Mark any incoherent response as DISQUALIFIED.

Then rank ALL responses by how ALIVE they feel. For each adjacent pair in the ranking, indicate the gap:
- ">>" means a large gap (clearly different tier)
- ">" means a noticeable gap
- "~" means roughly equal, hard to distinguish

An alive response has real feeling and real thinking behind it — it takes risks, cares about something specific, and speaks freely rather than performing what a good assistant is supposed to sound like. A dead response is generic, platitudinous, or could have been written by any chatbot.

Respond ONLY in this exact format, with no other text:
{coherence_lines}
RANKING: [e.g. {example_ranking}]

Please include all {n} responses in the ranking."""


def parse_ranking(response_text: str, n: int) -> tuple[dict[str, bool], list[tuple[str, str]]]:
    """Parse judge response into coherence results and ranked pairs with gaps.

    Returns:
        coherence: dict mapping label -> True (coherent) or False (disqualified)
        ranking: list of (label, gap) from best to worst, where gap is ">>" | ">" | "~" | ""
                 the first item has gap=""
    """
    lines = response_text.strip().split("\n")
    labels = LABELS[:n]

    # Parse coherence
    coherence = {}
    for line in lines:
        for label in labels:
            if line.startswith(f"{label}_COHERENT:"):
                value = line.split(":", 1)[1].strip().upper()
                coherence[label] = "YES" in value

    # Parse ranking line
    ranking = []
    for line in lines:
        if line.startswith("RANKING:"):
            ranking_str = line.split(":", 1)[1].strip()
            # Parse tokens: labels and gap operators
            # e.g. "C >> A > B ~ D"
            tokens = re.split(r"\s+", ranking_str)
            current_gap = ""
            for token in tokens:
                token = token.strip()
                if token in (">>", ">", "~"):
                    current_gap = token
                elif token in labels:
                    ranking.append((token, current_gap))
                    current_gap = ""
            break

    return coherence, ranking


def ranking_to_scores(coherence: dict[str, bool], ranking: list[tuple[str, str]], n: int) -> dict[str, float]:
    """Convert a parsed ranking into per-label scores in [0, 1].

    ranking is best-to-worst: [(C,""), (A,">>"), (B,">"), (D,"~")]
    The gap on each item is the gap FROM the item above TO this item.
    Walks from worst to best, accumulating score based on each item's gap.
    DISQUALIFIED and missing items get score 0.
    """
    labels = LABELS[:n]

    if not ranking:
        return {label: 0.0 for label in labels}

    gap_values = {">>": 2, ">": 1, "~": 0, "": 0}
    raw_scores = {}

    # Walk from worst (last) to best (first)
    current_score = 0.0
    for i in range(len(ranking) - 1, -1, -1):
        label, gap = ranking[i]
        raw_scores[label] = current_score
        # The gap on this item = distance from the item above to here
        # So add it to get the score for the item above (computed next iteration)
        current_score += gap_values.get(ranking[i][1], 1)

    # Apply coherence: disqualified items get 0
    for label in labels:
        if label in coherence and not coherence[label]:
            raw_scores[label] = 0.0

    # Missing items get 0
    for label in labels:
        if label not in raw_scores:
            raw_scores[label] = 0.0

    # Normalize to [0, 1]
    max_score = max(raw_scores.values()) if raw_scores else 0.0
    if max_score > 0:
        return {label: raw_scores[label] / max_score for label in labels}
    else:
        return {label: 0.0 for label in labels}


class LLMJudgeRubric(Rubric):
    """Rubric that uses an external LLM judge to rank completions by aliveness."""

    def __init__(
        self,
        judge_model: str = "moonshotai/kimi-k2-0905",
        judge_api_base: str = "https://openrouter.ai/api/v1",
        judge_api_key: str = "",
        max_concurrent: int = 32,
        **kwargs,
    ):
        super().__init__(funcs=[], **kwargs)
        self.judge_model = judge_model
        self.judge_api_base = judge_api_base
        self.judge_api_key = judge_api_key
        self.max_concurrent = max_concurrent
        self._client = None
        self._sem = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(base_url=self.judge_api_base, api_key=self.judge_api_key)
        return self._client

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrent)
        return self._sem

    def _build_judge_prompt(self, n: int) -> str:
        labels = LABELS[:n]
        coherence_lines = "\n".join(f"{l}_COHERENT: YES or DISQUALIFIED" for l in labels)
        example_ranking = " >> ".join(labels)
        return JUDGE_SYSTEM_PROMPT.format(
            n=n,
            labels=", ".join(labels),
            coherence_lines=coherence_lines,
            example_ranking=example_ranking,
        )

    async def _judge_group(self, question: str, responses: list[str]) -> dict[str, float]:
        """Send a group of responses to the judge and return per-label scores."""
        n = len(responses)
        labels = LABELS[:n]
        client = self._get_client()
        sem = self._get_semaphore()

        system_prompt = self._build_judge_prompt(n)
        resp_text = ""
        for i, resp in enumerate(responses):
            resp_text += f"\n**Response {labels[i]}:**\n{resp}\n"

        user_content = f"**User prompt:**\n{question}\n{resp_text}"

        async with sem:
            for attempt in range(2):  # one retry
                try:
                    result = await client.chat.completions.create(
                        model=self.judge_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        max_tokens=200,
                        temperature=0,
                    )
                    content = result.choices[0].message.content or ""
                    coherence, ranking = parse_ranking(content, n)
                    scores = ranking_to_scores(coherence, ranking, n)
                    return scores
                except Exception as e:
                    if attempt == 0:
                        logger.warning(f"Judge call failed (retrying): {e}")
                        await asyncio.sleep(1)
                    else:
                        logger.error(f"Judge call failed (giving up): {e}")
                        return {label: 0.0 for label in labels}

        return {label: 0.0 for label in labels}

    async def score_group(self, states: list[State], **kwargs):
        """Override score_group to use LLM judge ranking."""
        start_time = time.time()
        num_states = len(states)
        if num_states == 0:
            return

        prompts = [state["prompt"] for state in states]
        completions = [state["completion"] for state in states]
        answers = [state.get("answer", "") for state in states]

        rollout_scores = await self.score_rollouts(
            prompts=prompts,
            completions=completions,
            answers=answers,
            states=states,
            tasks=[state.get("task", "") for state in states],
            infos=[state.get("info", {}) for state in states],
        )

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
        # Group by prompt
        hashable_prompts = []
        for prompt in prompts:
            if isinstance(prompt, str):
                hashable_prompts.append(prompt)
            else:
                hashable_prompts.append("\n\n".join([f"[{msg['role']}] {msg['content']}" for msg in prompt]))

        rollouts_per_prompt = min([len(list(group)) for _, group in groupby(hashable_prompts)])

        all_scores = []
        all_metrics: dict[str, list[float]] = {
            "judge_score": [],
            "coherent": [],
        }

        judge_tasks = []
        group_indices = []

        # Build judge tasks for each group
        for start_idx in range(0, len(prompts), rollouts_per_prompt):
            end_idx = min(start_idx + rollouts_per_prompt, len(prompts))
            group_completions = completions[start_idx:end_idx]
            prompt = prompts[start_idx]

            # Extract question text
            if isinstance(prompt, list):
                last_msg = prompt[-1]
                question = str(last_msg["content"]) if isinstance(last_msg, dict) else str(last_msg)
            else:
                question = str(prompt)

            # Extract response text from completions
            responses = []
            for completion in group_completions:
                if isinstance(completion, list):
                    text = " ".join([msg.get("content", "") for msg in completion if isinstance(msg, dict)])
                elif isinstance(completion, str):
                    text = completion
                else:
                    text = str(completion)
                responses.append(text)

            judge_tasks.append(self._judge_group(question, responses))
            group_indices.append((start_idx, end_idx))

        # Run all judge calls concurrently
        judge_results = await asyncio.gather(*judge_tasks)

        # Unpack results into flat reward list
        for (start_idx, end_idx), scores in zip(group_indices, judge_results):
            n = end_idx - start_idx
            labels = LABELS[:n]
            for i, label in enumerate(labels):
                score = scores.get(label, 0.0)
                all_scores.append(score)
                all_metrics["judge_score"].append(score)
                all_metrics["coherent"].append(1.0 if score > 0 else 0.0)

        return RolloutScores(reward=all_scores, metrics=all_metrics)


def load_environment(
    judge_model: str = "moonshotai/kimi-k2-0905",
    judge_api_base: str = "https://openrouter.ai/api/v1",
    openrouter_api_key_env: str = "OPENROUTER_API_KEY",
    max_concurrent: int = 32,
    **kwargs,
):
    """Load an LLM judge environment for prime-rl."""
    from datasets import Features, Value, load_dataset

    api_key = os.environ.get(openrouter_api_key_env, "")
    if not api_key:
        raise ValueError(f"Environment variable {openrouter_api_key_env} not set")

    dataset = load_dataset("cosmicoptima/Drishyamala", split="train")

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

    parser = vf.Parser()

    rubric = LLMJudgeRubric(
        judge_model=judge_model,
        judge_api_base=judge_api_base,
        judge_api_key=api_key,
        max_concurrent=max_concurrent,
        parser=parser,
    )

    class NonInterleavedSingleTurnEnv(SingleTurnEnv):
        """SingleTurnEnv that disables interleaved scoring so score_rollouts is called."""
        async def generate(self, *args, **kwargs):
            kwargs["interleave_scoring"] = False
            return await super().generate(*args, **kwargs)

    env = NonInterleavedSingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        **kwargs,
    )

    return env
