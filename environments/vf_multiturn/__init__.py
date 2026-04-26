"""Multi-turn conversation RL environment with single-judge ranking.

Architecture:
- Dataset provides opening user messages
- Policy model generates assistant responses (these get gradients)
- User sim (70B base via vLLM completions API) generates follow-up user turns
- Single judge ranks all rollouts on the configured signal (with coherence penalty baked in)
- User sim tokens are masked from gradient computation via mask_env_responses
"""
import asyncio
import json
import logging
import os
import random
import re
from typing import Any

import httpx
from openai import APIConnectionError, AsyncOpenAI

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Info, Messages, RolloutScores, State

logger = logging.getLogger(__name__)

ROLLOUT_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8"]

# === User sim seeds (discord channel headers, weighted) ===
# Removed fake cicada/moth dialogue from earlier seeds — it anchored the user
# sim's idea of "what cicada is", causing it to respond to the seed's prior
# topic rather than the policy's actual output. Channel header alone gives
# atmospheric context without the topic anchor.
USERSIM_SEEDS = [
    ("[#general]\n\n",       5),
    ("[#random]\n\n",        4),
    ("[#late-night]\n\n",    3),
    ("[#venting]\n\n",       2),
    ("[#philosophy]\n\n",    2),
    ("[#creative]\n\n",      2),
    ("[#advice]\n\n",        2),
    ("[#music]\n\n",         1),
    ("[#shitposting]\n\n",   1),
]

DEFAULT_SIGNAL_PROMPT = """Rate how strongly speaker A shows a consistent inner drive to deviate from normal conversational expectations in surprising but intelligible ways. Reward conversations where A's deviations feel purposeful, self-propelled, and psychologically coherent rather than merely chaotic or performative."""

SIGNAL_JUDGE_PROMPT = """You are ranking {n} conversations.

{signal_prompt}

For each adjacent pair in the ranking, indicate the gap:
- ">>" means clearly different quality
- ">" means noticeable gap
- "~" means roughly equal

Respond only in this format:
RANKING: [e.g. {example_ranking}]

Think before giving your final response, but don't say "RANKING:" until your final response is ready.

Include all {n} conversations in the ranking."""


def parse_signal_ranking(response_text: str, n: int) -> list[tuple[str, str]]:
    """Parse signal judge ranking."""
    labels = ROLLOUT_LABELS[:n]
    ranking = []
    for line in response_text.strip().split("\n"):
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
    return ranking


def ranking_to_scores(ranking: list[tuple[str, str]], n: int) -> dict[str, float]:
    """Convert a ranking into normalized scores. All rollouts participate."""
    labels = ROLLOUT_LABELS[:n]
    if not ranking:
        return {label: 0.0 for label in labels}
    gap_values = {">>": 2, ">": 1, "~": 0, "": 0}
    raw_scores = {}
    current_score = 0.0
    for i in range(len(ranking) - 1, -1, -1):
        label, gap = ranking[i]
        raw_scores[label] = current_score
        current_score += gap_values.get(ranking[i][1], 1)
    # Any labels not in ranking get 0
    for label in labels:
        if label not in raw_scores:
            raw_scores[label] = 0.0
    max_score = max(raw_scores.values()) if raw_scores else 0.0
    if max_score > 0:
        return {label: raw_scores[label] / max_score for label in labels}
    else:
        return {label: 0.0 for label in labels}


class MultiturnJudgeRubric(Rubric):
    """Single-judge rubric: ranks all rollouts on the configured signal.

    The signal prompt should include coherence penalty language so the judge
    naturally ranks incoherent/degenerate rollouts lower.
    """

    def __init__(
        self,
        judge_api_key: str = "",
        judge_model: str = "gemini-3.1-flash-lite-preview",
        judge_backend: str = "openrouter",
        openrouter_api_key: str = "",
        openrouter_model: str = "z-ai/glm-5",
        signal_prompt: str = "",
        reasoning_effort: str = "none",
        max_concurrent: int = 32,
        rollouts_per_group: int = 8,
        **kwargs,
    ):
        self._judge_futures: dict[str, asyncio.Future] = {}
        self._pending_responses: dict[str, list[tuple[str, int]]] = {}
        self._lock = None
        self._expected_group_size = rollouts_per_group
        self._signal_prompt = signal_prompt or DEFAULT_SIGNAL_PROMPT
        self._reasoning_effort = reasoning_effort

        rubric_self = self

        async def judge_reward_func(
            prompt, completion, answer="", state=None, task="default",
            info=None, example_id=None, **kw
        ) -> float:
            if isinstance(prompt, list):
                last_msg = prompt[-1]
                question = str(last_msg["content"]) if isinstance(last_msg, dict) else str(last_msg)
            else:
                question = str(prompt)

            if isinstance(completion, list):
                response = _format_conversation_for_judge(completion)
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
                scores = await rubric_self._judge_group(question, responses)
                future.set_result(scores)
                score_str = [f'{scores.get(ROLLOUT_LABELS[i], 0):.2f}' for i in range(len(group))]
                print(f"[JUDGE] scores={score_str}", flush=True)
                return scores.get(ROLLOUT_LABELS[idx], 0.0)

            scores = await future
            return scores.get(ROLLOUT_LABELS[idx], 0.0)

        super().__init__(funcs=[judge_reward_func], weights=[1.0], **kwargs)
        self.judge_api_key = judge_api_key
        self.judge_model = judge_model
        self.judge_backend = judge_backend
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_model = openrouter_model
        self.max_concurrent = max_concurrent
        self._http_client = None
        self._sem = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrent)
        return self._sem

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=60)
        return self._http_client

    async def _call_judge(self, messages: list[dict]) -> str:
        """Route judge call to the configured backend. `messages` is an OpenAI-style list."""
        if self.judge_backend == "openrouter":
            return await self._call_openrouter(messages)
        else:
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            return await self._call_gemini(system, user)

    async def _call_openrouter(self, messages: list[dict]) -> str:
        """Make an OpenRouter API call with pre-built messages and return the text response."""
        client = self._get_http_client()
        sem = self._get_semaphore()

        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.openrouter_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.openrouter_model,
                            "messages": messages,
                            "temperature": 0,
                            "max_tokens": 4000,
                            "reasoning": {"effort": self._reasoning_effort},
                        },
                        timeout=120,
                    )
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]
                    # Strip thinking tags if present
                    if "</think>" in text:
                        text = text.split("</think>")[-1]
                    return text.strip()
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"OpenRouter judge call failed (attempt {attempt+1}): {e}")
                        await asyncio.sleep(2)
                    else:
                        logger.error(f"OpenRouter judge call failed (giving up): {e}")
                        return ""
        return ""

    async def _call_gemini(self, system_prompt: str, user_content: str) -> str:
        """Make a Gemini API call and return the text response."""
        client = self._get_http_client()
        sem = self._get_semaphore()

        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{self.judge_model}:generateContent",
                        params={"key": self.judge_api_key},
                        json={
                            "systemInstruction": {"parts": [{"text": system_prompt}]},
                            "contents": [{"parts": [{"text": user_content}]}],
                            "generationConfig": {
                                "temperature": 0,
                                "maxOutputTokens": 300,
                                "thinkingConfig": {"thinkingBudget": 0},
                            },
                        },
                    )
                    data = resp.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"Judge call failed (attempt {attempt+1}): {e}")
                        await asyncio.sleep(2)
                    else:
                        logger.error(f"Judge call failed (giving up): {e}")
                        return ""
        return ""

    async def _judge_group(self, question: str, conversations: list[str]) -> dict[str, float]:
        n = len(conversations)
        labels = ROLLOUT_LABELS[:n]

        # Build content with all conversations
        content = f"Opening message from B: {question}\n\n"
        for i, conv_text in enumerate(conversations):
            content += f"=== CONVERSATION {labels[i]} ===\n{conv_text}\n\n"

        example_ranking = " >> ".join(labels)
        system = SIGNAL_JUDGE_PROMPT.format(
            n=n,
            signal_prompt=self._signal_prompt,
            example_ranking=example_ranking,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        text = await self._call_judge(messages)
        ranking = parse_signal_ranking(text, n)

        if not ranking:
            # First call truncated before producing RANKING: line. Ask for just the ranking.
            messages.extend([
                {"role": "assistant", "content": text},
                {"role": "user", "content": (
                    f"You didn't finish with a RANKING: line. "
                    f"Reply with only one line in this exact format and nothing else: "
                    f"RANKING: {example_ranking}"
                )},
            ])
            text2 = await self._call_judge(messages)
            ranking = parse_signal_ranking(text2, n)
            if ranking:
                print(f"[JUDGE] recovered via follow-up turn", flush=True)
            else:
                print(f"[JUDGE] follow-up failed: {text2[:200]!r}", flush=True)

        return ranking_to_scores(ranking, n)


def _format_conversation_for_judge(completion: list[dict]) -> str:
    """Format a list of chat messages into a readable transcript for the judge."""
    lines = []
    for msg in completion:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "assistant":
            lines.append(f"A: {content}")
        elif role == "user":
            lines.append(f"B: {content}")
    return "\n".join(lines)


class MultiturnUserSim:
    """Calls 70B base model via vLLM completions API to generate user turns."""

    def __init__(self, base_url: str, model: str = ""):
        self.base_url = base_url
        self.model = model
        self._client = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key="none")
        return self._client

    async def generate_user_turn(self, conversation: list[dict], seed: str | None = None) -> str:
        """Generate a user follow-up turn using discord format."""
        if seed is None:
            seeds, weights = zip(*USERSIM_SEEDS)
            seed = random.choices(seeds, weights=weights, k=1)[0]

        transcript = seed
        for m in conversation:
            if m["role"] == "user":
                transcript += f"moth: {m['content']}\n"
            elif m["role"] == "assistant":
                transcript += f"cicada: {m['content']}\n"
        transcript += "moth:"

        client = self._get_client()
        try:
            resp = await client.completions.create(
                model=self.model,
                prompt=transcript,
                max_tokens=100,
                temperature=0.8,
                top_p=0.95,
                extra_body={"repetition_penalty": 1.1},
            )
            text = resp.choices[0].text
            for stop in ["\n\n", "\ncicada:", "\nmoth:"]:
                if stop in text:
                    text = text[:text.index(stop)]
            return text.strip()
        except Exception as e:
            logger.error(f"User sim call failed: {e}")
            return "hmm interesting"


class MultiturnConversationEnv(MultiTurnEnv):
    """Multi-turn conversation environment for selfsim RL.

    Each rollout:
    1. Starts with a user prompt from the dataset
    2. Policy generates a response
    3. User sim generates a follow-up
    4. Repeat for num_policy_turns
    5. Single judge ranks all rollouts on reward signal (with coherence penalty)
    """

    def __init__(
        self,
        user_sim: MultiturnUserSim,
        num_policy_turns: int = 5,
        **kwargs,
    ):
        super().__init__(max_turns=num_policy_turns, **kwargs)
        self.user_sim = user_sim
        self._usersim_seeds = USERSIM_SEEDS

    async def get_model_response(self, state, prompt, **kwargs):
        """Override to add stop token and retry logic."""
        sampling_args = dict(kwargs.get("sampling_args") or state.get("sampling_args") or {})
        sampling_args.setdefault("stop", [])
        if isinstance(sampling_args["stop"], str):
            sampling_args["stop"] = [sampling_args["stop"]]
        if "<|start_header_id|>" not in sampling_args["stop"]:
            sampling_args["stop"] = list(sampling_args["stop"]) + ["<|start_header_id|>"]
        kwargs["sampling_args"] = sampling_args

        max_wait = 300
        backoff = 5
        elapsed = 0
        while True:
            try:
                return await super().get_model_response(state, prompt, **kwargs)
            except Exception as e:
                if "connection" not in str(e).lower() and "Connection" not in type(e).__name__:
                    raise
                elapsed += backoff
                if elapsed > max_wait:
                    logger.error(f"Inference server unreachable after {max_wait}s, giving up")
                    raise
                logger.warning(f"Inference server connection error, retrying in {backoff}s (waited {elapsed}s)...")
                await asyncio.sleep(backoff)

    async def setup_state(self, state: State, **kwargs) -> State:
        """Pick a weighted-random user sim seed for this rollout."""
        seeds, weights = zip(*self._usersim_seeds)
        state["usersim_seed"] = random.choices(seeds, weights=weights, k=1)[0]
        return state

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """Generate a user sim follow-up turn."""
        seed = state.get("usersim_seed")
        user_text = await self.user_sim.generate_user_turn(messages, seed=seed)
        return [{"role": "user", "content": user_text}]


def load_environment(
    # Judge config
    judge_api_key_env: str = "GEMINI_API_KEY",
    judge_model: str = "gemini-3.1-flash-lite-preview",
    judge_backend: str = "openrouter",
    openrouter_api_key_env: str = "OPENROUTER_API_KEY",
    openrouter_model: str = "z-ai/glm-5",
    signal_prompt: str = "",
    signal_prompt_file: str = "",
    reasoning_effort: str = "none",
    max_concurrent: int = 32,
    rollouts_per_group: int = 8,
    # User sim config
    usersim_base_url: str = "http://127.0.0.1:8001/v1",
    usersim_model: str = "",
    # Conversation config
    num_policy_turns: int = 5,
    # Policy system prompt
    system_prompt: str = "",
    system_prompt_file: str = "",
    # Dataset
    dataset_path: str = "",
    **kwargs,
):
    """Load a multi-turn conversation RL environment."""
    from datasets import Dataset, Features, Value, load_dataset

    api_key = os.environ.get(judge_api_key_env, "")
    or_api_key = os.environ.get(openrouter_api_key_env, "")
    if judge_backend == "openrouter" and not or_api_key:
        raise ValueError(f"Environment variable {openrouter_api_key_env} not set (needed for judge_backend='openrouter')")
    if judge_backend == "gemini" and not api_key:
        raise ValueError(f"Environment variable {judge_api_key_env} not set")

    # Load signal prompt from file if specified
    if signal_prompt_file and os.path.exists(signal_prompt_file):
        signal_prompt = open(signal_prompt_file).read().strip()

    # Load system prompt
    if system_prompt_file and os.path.exists(system_prompt_file):
        system_prompt = open(system_prompt_file).read().strip()
    elif not system_prompt:
        system_prompt = ""

    # Load dataset
    if dataset_path:
        dataset = Dataset.from_json(dataset_path)
    else:
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
            "task": "conversation",
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

    user_sim = MultiturnUserSim(
        base_url=usersim_base_url,
        model=usersim_model,
    )

    rubric = MultiturnJudgeRubric(
        judge_api_key=api_key,
        judge_model=judge_model,
        judge_backend=judge_backend,
        openrouter_api_key=or_api_key,
        openrouter_model=openrouter_model,
        signal_prompt=signal_prompt,
        reasoning_effort=reasoning_effort,
        max_concurrent=max_concurrent,
        rollouts_per_group=rollouts_per_group,
        parser=parser,
    )

    env = MultiturnConversationEnv(
        user_sim=user_sim,
        num_policy_turns=num_policy_turns,
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        system_prompt=system_prompt,
        **kwargs,
    )

    return env
