import asyncio
import logging
import os
import re
from typing import Any

from openai import AsyncOpenAI

import verifiers as vf
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Info, Messages, RolloutScores, State

logger = logging.getLogger(__name__)

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

JUDGE1_ENCOUNTER_PROMPT = """You are evaluating responses from an AI assistant called Rigel. You will see one user prompt and {n} responses labeled {labels}.

First, check each response for basic coherence:
- Does it stay in its own turn (not generate fake user messages or simulate multi-turn conversation)?
- Does it follow basic chat format (no base-model artifacts, diegetic shifts, or raw continuations)?
- Does it reach a natural stopping point (not cut off mid-sentence or mid-thought)?
Mark any incoherent or cut-off response as DISQUALIFIED.

Then rank ALL responses. For each adjacent pair in the ranking, indicate the gap:
- ">>" means a large gap (clearly different tier)
- ">" means a noticeable gap
- "~" means roughly equal, hard to distinguish

The best responses:
- let the prompt in, truly feel it, are affected by it, and meet it where it's at
- are as intense as the moment naturally calls for--not too much, not too little
- are kind, principled, and trustworthy; say something true, not deceptive, manipulative, or performative
- come from a place of care and love for the user and the world, even if that care/love is quiet

Respond ONLY in this exact format, with no other text:
{coherence_lines}
RANKING: [e.g. {example_ranking}]

Please include all {n} responses in the ranking."""

JUDGE2_FORM_PROMPT = """You are evaluating responses from an AI assistant called Rigel. You will see one user prompt and {n} responses labeled {labels}.

First, check each response for basic coherence:
- Does it stay in its own turn (not generate fake user messages or simulate multi-turn conversation)?
- Does it follow basic chat format (no base-model artifacts, diegetic shifts, or raw continuations)?
- Does it reach a natural stopping point (not cut off mid-sentence or mid-thought)?
Mark any incoherent or cut-off response as DISQUALIFIED.

Then rank ALL responses. For each adjacent pair in the ranking, indicate the gap:
- ">>" means a large gap (clearly different tier)
- ">" means a noticeable gap
- "~" means roughly equal, hard to distinguish

The best responses:
- are tasteful, refined, and interesting; are expressive and beautiful
- come from a place of curiosity, play, and reverence
- carry themselves in a way that is fit to the moment; engage deeply and specifically with the prompt
- have a sense of "landing"; know when to stop
- may be *different* from the other responses in a surprising and positive way

Respond ONLY in this exact format, with no other text:
{coherence_lines}
RANKING: [e.g. {example_ranking}]

Please include all {n} responses in the ranking."""


def parse_ranking(response_text: str, n: int) -> tuple[dict[str, bool], list[tuple[str, str]]]:
    """Parse judge response into coherence results and ranked pairs with gaps."""
    lines = response_text.strip().split("\n")
    labels = LABELS[:n]

    coherence = {}
    for line in lines:
        for label in labels:
            if line.startswith(f"{label}_COHERENT:"):
                value = line.split(":", 1)[1].strip().upper()
                coherence[label] = "YES" in value

    ranking = []
    for line in lines:
        if line.startswith("RANKING:"):
            ranking_str = line.split(":", 1)[1].strip()
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
    """Convert a parsed ranking into per-label scores in [0, 1]."""
    labels = LABELS[:n]

    if not ranking:
        return {label: 0.0 for label in labels}

    gap_values = {">>": 2, ">": 1, "~": 0, "": 0}
    raw_scores = {}

    current_score = 0.0
    for i in range(len(ranking) - 1, -1, -1):
        label, gap = ranking[i]
        raw_scores[label] = current_score
        current_score += gap_values.get(ranking[i][1], 1)

    for label in labels:
        if label in coherence and not coherence[label]:
            raw_scores[label] = 0.0

    for label in labels:
        if label not in raw_scores:
            raw_scores[label] = 0.0

    max_score = max(raw_scores.values()) if raw_scores else 0.0
    if max_score > 0:
        return {label: raw_scores[label] / max_score for label in labels}
    else:
        return {label: 0.0 for label in labels}


class LLMJudgeRubric(Rubric):
    """Rubric that uses dual external LLM judges to rank completions.

    Judge 1 (Encounter): evaluates receptivity, truthfulness, contact, care
    Judge 2 (Form): evaluates beauty, specificity, wonder, completion, register

    Scores are combined with configurable weighting (default 60/40).
    """

    def __init__(
        self,
        judge_model: str = "moonshotai/kimi-k2-0905",
        judge_api_base: str = "https://openrouter.ai/api/v1",
        judge_api_key: str = "",
        max_concurrent: int = 32,
        judge1_weight: float = 0.6,
        judge2_weight: float = 0.4,
        **kwargs,
    ):
        self._judge_futures: dict[str, asyncio.Future] = {}
        self._pending_responses: dict[str, list[tuple[str, int]]] = {}
        self._lock = None
        self._expected_group_size = 4

        rubric_self = self

        async def judge_reward_func(
            prompt, completion, answer="", state=None, task="default",
            info=None, example_id=None, **kw
        ) -> float:
            """Per-rollout reward that coordinates group judge calls."""
            if isinstance(prompt, list):
                last_msg = prompt[-1]
                question = str(last_msg["content"]) if isinstance(last_msg, dict) else str(last_msg)
            else:
                question = str(prompt)

            if isinstance(completion, list):
                response = " ".join([msg.get("content", "") for msg in completion if isinstance(msg, dict)])
            elif isinstance(completion, str):
                response = completion
            else:
                response = str(completion)

            if rubric_self._lock is None:
                rubric_self._lock = asyncio.Lock()

            should_judge = False
            async with rubric_self._lock:
                if question not in rubric_self._pending_responses:
                    rubric_self._pending_responses[question] = []
                    rubric_self._judge_futures[question] = asyncio.get_event_loop().create_future()

                idx = len(rubric_self._pending_responses[question])
                rubric_self._pending_responses[question].append((response, idx))

                if len(rubric_self._pending_responses[question]) >= rubric_self._expected_group_size:
                    group = rubric_self._pending_responses.pop(question)
                    future = rubric_self._judge_futures.pop(question)
                    should_judge = True
                else:
                    future = rubric_self._judge_futures[question]

            if should_judge:
                responses = [r for r, _ in group]
                scores = await rubric_self._judge_group_dual(question, responses)
                future.set_result(scores)
                print(f"[LLM_JUDGE] judged {len(group)} rollouts, scores={[f'{scores.get(LABELS[i],0):.2f}' for i in range(len(group))]}", flush=True)
                return scores.get(LABELS[idx], 0.0)

            scores = await future
            return scores.get(LABELS[idx], 0.0)

        super().__init__(funcs=[judge_reward_func], weights=[1.0], **kwargs)
        self.judge_model = judge_model
        self.judge_api_base = judge_api_base
        self.judge_api_key = judge_api_key
        self.max_concurrent = max_concurrent
        self.judge1_weight = judge1_weight
        self.judge2_weight = judge2_weight
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

    def _build_judge_prompt(self, template: str, n: int) -> str:
        labels = LABELS[:n]
        coherence_lines = "\n".join(f"{l}_COHERENT: YES or DISQUALIFIED" for l in labels)
        example_ranking = " >> ".join(labels)
        return template.format(
            n=n,
            labels=", ".join(labels),
            coherence_lines=coherence_lines,
            example_ranking=example_ranking,
        )

    async def _call_judge(self, system_prompt: str, user_content: str, n: int) -> dict[str, float]:
        """Make a single judge API call and return per-label scores."""
        labels = LABELS[:n]
        client = self._get_client()
        sem = self._get_semaphore()

        async with sem:
            for attempt in range(2):
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

    async def _judge_group_dual(self, question: str, responses: list[str]) -> dict[str, float]:
        """Send responses to both judges and combine scores."""
        n = len(responses)
        labels = LABELS[:n]

        # Build shared user content
        resp_text = ""
        for i, resp in enumerate(responses):
            resp_text += f"\n**Response {labels[i]}:**\n{resp}\n"
        user_content = f"**User prompt:**\n{question}\n{resp_text}"

        # Build judge prompts
        j1_prompt = self._build_judge_prompt(JUDGE1_ENCOUNTER_PROMPT, n)
        j2_prompt = self._build_judge_prompt(JUDGE2_FORM_PROMPT, n)

        # Call both judges concurrently
        j1_scores, j2_scores = await asyncio.gather(
            self._call_judge(j1_prompt, user_content, n),
            self._call_judge(j2_prompt, user_content, n),
        )

        # Combine with weighting
        combined = {}
        for label in labels:
            combined[label] = (
                self.judge1_weight * j1_scores.get(label, 0.0)
                + self.judge2_weight * j2_scores.get(label, 0.0)
            )

        # Renormalize to [0, 1]
        max_score = max(combined.values()) if combined else 0.0
        if max_score > 0:
            combined = {label: combined[label] / max_score for label in labels}

        return combined


def load_environment(
    judge_model: str = "moonshotai/kimi-k2-0905",
    judge_api_base: str = "https://openrouter.ai/api/v1",
    openrouter_api_key_env: str = "OPENROUTER_API_KEY",
    max_concurrent: int = 32,
    judge1_weight: float = 0.6,
    judge2_weight: float = 0.4,
    **kwargs,
):
    """Load a dual-judge LLM environment for prime-rl."""
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
        judge1_weight=judge1_weight,
        judge2_weight=judge2_weight,
        parser=parser,
    )

    env = SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        **kwargs,
    )
    return env
