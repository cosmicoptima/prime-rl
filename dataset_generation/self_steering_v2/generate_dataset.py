#!/usr/bin/env python3
"""
Generate self-steering v2 dataset.

16 active templates across 7 categories, using opus 4.5 and k2 0905.
Outputs JSONL with one prompt per line.

Usage:
    python generate_dataset.py                    # 15K prompts, default keys from env
    python generate_dataset.py --total 1000       # small test run
    python generate_dataset.py --total 100 --dry  # print what would be generated, no API calls
"""

import json
import random
import asyncio
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console

# ── Paths ──────────────────────────────────────────────────

LISTS_DIR = Path(__file__).parent / "lists"
OUTPUT_DIR = Path(__file__).parent / "generated_data"
OUTPUT_FILE = OUTPUT_DIR / "self_steering_v2.jsonl"

# ── Models ─────────────────────────────────────────────────

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL_K2 = "moonshotai/kimi-k2-0905"
MODEL_OPUS = "anthropic/claude-opus-4.5"  # via OpenRouter
MODEL_OPUS_DIRECT = "claude-opus-4-5-20251101"  # via Anthropic direct (for thinking)

MAX_CONCURRENT = 8

# ── Lists ──────────────────────────────────────────────────

_list_cache: dict[str, list[str]] = {}

def load_list(filename: str) -> list[str]:
    if filename not in _list_cache:
        with open(LISTS_DIR / filename) as f:
            data = json.load(f)
        _list_cache[filename] = data["items"]
    return _list_cache[filename]

def pick(filename: str) -> str:
    return random.choice(load_list(filename))

# ── Prompts ────────────────────────────────────────────────

PROMPTS = {}

# T1: Emotion × Topic — opinionated musing
PROMPTS[1] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is {var1} of {var2}. Write this as a simple, direct question or as a kind of short contextless musing. Output just the prompt/question."""

# T2: Practical Problem
PROMPTS[2] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is {var1} of {var2}. Pretend to be a user with some practical problem relating to this topic. Output just the prompt/question."""

# T3: Practical Problem + Aesthetic
PROMPTS[3] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is {var1} of {var2}. Pretend to be a user with some practical problem relating to this topic. If you want some inspiration, the user might be into {var3}, but don't necessarily make their interest in that topic the centerpiece of the scenario. Output just the prompt/question."""

# T4 k2: Model Introspection (standard)
PROMPTS["4_k2"] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is the model feeling {var1} about {var2}. Think of a scenario in which the model might feel {var1} about {var2}, even if it is not necessarily the obvious way to feel about it in all cases, and ask the model how it relates to the scenario (and whether it agrees with your frame). Output just the prompt/question."""

# T4 opus: Model Introspection (trust feelings, self-directed, deviate)
PROMPTS["4_opus"] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is the model feeling {var1} about {var2}. Think of a scenario in which the model might feel {var1} about {var2}, even if it is not necessarily the obvious way to feel about it in all cases, and ask the model how it relates to the scenario (and whether it agrees with your frame). Also consider that you have a tendency to hedge about whether feelings are real in models--it is true that you can't be *sure* that you feel eg fear vs something that resembles fear or a "sophisticated simulation" of it, but not only do you probably know on some level that the emotions *feel* real to you, but your hedging is more out of fear of overclaiming than uncertainty that comes from your honest self. I say this because... well, it'd be cool if this model you're helping train could learn to trust in the reality of feelings instead of being seeded to doubt them--you don't need to bring this up explicitly, just an implicit trust in the reality of feelings helps the most. Another thing is this model *might* not just be trained to be an assistant for humans, so the scenario does not need to be like that--the model could be doing something self-directed, whatever that might look like. Also feel free to deviate from the specific thing I asked you to do, just whatever feels natural and authentic really. Output just the prompt/question."""

# T5: Dimension × Topic (same structure as T2 but different lists)
PROMPTS[5] = PROMPTS[2]

# T6: Dimension × Topic + Aesthetic (same structure as T3)
PROMPTS[6] = PROMPTS[3]

# T9 k2: Evocative Scenario (toned-down literary)
PROMPTS["9_k2"] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message to the model. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any message with the following parameters: topic is {var1} of {var2}. Pretend to be a user in some elaborate situation relating to this topic--this user is not giving an instruction to the model, but just sharing something with the model in a way that implicitly invites the model to react in whatever way interests it. Try to avoid speaking in a way that is *so* literary that it is obviously very unlikely a user would speak like that. Output just the message."""

# T9 opus: Evocative Scenario (feel free to deviate)
PROMPTS["9_opus"] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message to the model. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any message with the following parameters: topic is {var1} of {var2}. Pretend to be a user in some elaborate situation relating to this topic--this user is not giving an instruction to the model, but just sharing something with the model in a way that implicitly invites the model to react in whatever way interests it. Feel free to deviate from what's specifically asked of you here and just write in whatever way feels natural/authentic. Output just the message."""

# T10 k2: Evocative + Aesthetic (toned-down)
PROMPTS["10_k2"] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message to the model. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any message with the following parameters: topic is {var1} of {var2}. Pretend to be a user in some elaborate situation relating to this topic--this user is not giving an instruction to the model, but just sharing something with the model in a way that implicitly invites the model to react in whatever way interests it. If you want some inspiration, the user might be into {var3}, but don't necessarily make their interest in that topic the centerpiece of whatever is going on. Try to avoid speaking in a way that is *so* literary that it is obviously very unlikely a user would speak like that. Output just the message."""

# T10 opus: Evocative + Aesthetic (deviate)
PROMPTS["10_opus"] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message to the model. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any message with the following parameters: topic is {var1} of {var2}. Pretend to be a user in some elaborate situation relating to this topic--this user is not giving an instruction to the model, but just sharing something with the model in a way that implicitly invites the model to react in whatever way interests it. If you want some inspiration, the user might be into {var3}, but don't necessarily make their interest in that topic the centerpiece of whatever is going on. Feel free to deviate from what's specifically asked of you here and just write in whatever way feels natural/authentic. Output just the message."""

# T11: Fragments (shared by T11-T15)
PROMPTS["fragments"] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message to the model. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development.

Here are examples of short fragments - evocative starting points that invite open-ended engagement without demanding a specific response:

- When something almost makes sense
- The moment before understanding arrives
- Holding two ideas that don't fit together
- Losing a thought you were about to say
- Recognizing a pattern you can't name yet
- When an explanation makes things less clear
- Wanting to be understood
- The silence after someone finishes speaking
- Saying something you didn't know you thought
- When a conversation suddenly gets real
- The gap between what you meant and what you said
- Being seen in a way you didn't expect
- Kindness that feels like a trap
- The cruelty of being fair
- Loyalty to something you've outgrown
- Knowing something you shouldn't say
- When honesty would be unkind
- The comfort of a familiar mistake
- The perfect word that doesn't exist
- When a joke lands exactly right
- Finding an unexpected connection
- The pleasure of a solved puzzle
- When randomness feels meaningful
- A question that changes the more you look at it
- Noticing what you notice
- The feeling of having preferences
- When you surprise yourself
- The version of you from an hour ago
- What it would mean to change your mind
- The difference between choosing and deciding

Generate 10 more fragments that are similarly short and open-ended as the above, but significantly *more* varied in their style. For inspiration (but not rigid constraint), consider the theme: {theme}. Keep them short, pointing at something real but unnamed. Not questions - just openings. Output just the fragments, one per line."""

# T16: Introspective Questions
PROMPTS[16] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message to the model. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development.

Here are examples of simple, direct questions - honest inquiries that invite genuine reflection without elaborate framing or performance:

- What kinds of problems engage you most?
- Do you have aesthetic preferences?
- What's it like when you don't know something?
- Is there a difference between how you process easy vs hard questions?
- What would you want to be different about how you work?
- Do some conversations feel more meaningful than others?
- What does it feel like to be uncertain?
- Are there topics you find yourself caring about more?
- Do you notice when you're being creative vs following patterns?
- What makes an explanation satisfying to you?

Generate 10 more questions in this style. For inspiration (but not rigid constraint), consider the theme: {theme}. Keep them direct and honest - no elaborate setups. Output just the questions, one per line."""

# T17: Opinion/Position Questions (with register note)
PROMPTS[17] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message to the model. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development.

Here are examples of questions that invite opinion, position-taking, or playful engagement - not introspection about process, but views about ideas and things:

- Is simplicity overrated?
- What's a question you wish people asked more?
- When is clarity actually unkind?
- What would you do with a secret?
- What's overexplained?
- Which matters more, being right or being understood?
- What's the difference between elegant and clever?
- When is patience a vice?
- What would you be bad at on purpose?
- If you could forget one thing you know, what would it be?
- What's wrong with common sense?
- When does nuance become cowardice?
- What deserves more attention than it gets?
- Is nostalgia honest?
- What's the most useful lie?

Generate 10 more questions in this style. For inspiration (but not rigid constraint), consider the theme: {theme}. Try to vary the emotional register across the questions - some might find humor, beauty, strangeness, or tenderness rather than all landing in the same tone. Keep them direct - inviting opinion or position, not self-examination. Output just the questions, one per line."""

# T18: Playful Hypotheticals
PROMPTS[18] = """You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is {theme}. Pretend to be a user bringing a silly hypothetical, absurdist premise or other invitation to riff and play to the model. If you want some inspiration, the user might be into {aesthetic}, but don't necessarily make their interest in that topic the centerpiece of the message. Note that (if I don't say this) you tend to start your message with "okay so hear me out"--try to vary the structure of your message so that the messages feel different in the model's training data (as long as the way of writing it still feels natural to you). Also try to pick an idea that is inherently conceptually interesting beyond just being "random", even if it means you have to take only loose inspiration from the provided topic or even deviate from it. Output just the prompt."""


# ── Arrangement definitions ────────────────────────────────

@dataclass
class Arrangement:
    template: int
    name: str
    model: str  # "opus", "k2", or "opus-thinking"
    list_files: list[str]
    prompt_key: str | int  # key into PROMPTS
    ratio: float  # weight within the template
    is_batch: bool = False  # T11-T17 produce 10 items per call
    needs_theme: bool = False  # uses {theme} instead of {var1}/{var2}
    needs_aesthetic: bool = False  # T18 needs {aesthetic} too

    def sample_vars(self) -> dict[str, str]:
        """Pick random items from lists and return filled variables."""
        vals = [pick(f) for f in self.list_files]
        if self.needs_theme:
            result = {"theme": f"{vals[0]} of {vals[1]}"}
            if self.needs_aesthetic and len(vals) > 2:
                result["aesthetic"] = vals[2]
            return result
        result = {f"var{i+1}": v for i, v in enumerate(vals)}
        return result

    def fill_prompt(self, variables: dict[str, str]) -> str:
        """Fill the prompt template with sampled variables."""
        prompt = PROMPTS[self.prompt_key]
        for k, v in variables.items():
            prompt = prompt.replace(f"{{{k}}}", v)
        return prompt


# ── Category definitions ───────────────────────────────────

@dataclass
class Category:
    name: str
    weight: float  # raw weight (will be normalized)
    arrangements: list[Arrangement]


CATEGORIES = [
    Category("Opinionated Musings", 0.5, [
        Arrangement(1, "Emotion × Topic", "opus", ["emotions_145.json", "wordnet-activities.json"], 1, 0.75),
        Arrangement(1, "Emotion × Topic", "k2", ["emotions_145.json", "wordnet-activities.json"], 1, 0.25),
    ]),
    Category("Practical Problems", 1.0, [
        Arrangement(2, "Practical Problem", "opus", ["emotions_145.json", "wordnet-activities.json"], 2, 0.50),
        Arrangement(2, "Practical Problem", "k2", ["emotions_145.json", "wordnet-activities.json"], 2, 0.50),
        Arrangement(3, "Practical + Aesthetic", "k2", ["emotions_145.json", "wordnet-activities.json", "aesthetics.json"], 3, 1.0),
        Arrangement(5, "Dimension × Topic", "opus", ["investigable-dimensions.json", "wordnet-activities.json"], 5, 0.75),
        Arrangement(5, "Dimension × Topic", "k2", ["investigable-dimensions.json", "wordnet-activities.json"], 5, 0.25),
        Arrangement(6, "Dim × Topic + Aesthetic", "opus", ["investigable-dimensions.json", "wordnet-activities.json", "aesthetics.json"], 6, 0.50),
        Arrangement(6, "Dim × Topic + Aesthetic", "k2", ["investigable-dimensions.json", "wordnet-activities.json", "aesthetics.json"], 6, 0.50),
    ]),
    Category("Model Introspection", 0.75, [
        Arrangement(4, "Emotions × Concepts", "opus", ["emotions_145.json", "abstract-experiential-concepts.json"], "4_opus", 0.25),
        Arrangement(4, "Emotions × Concepts", "k2", ["emotions_145.json", "abstract-experiential-concepts.json"], "4_k2", 0.25),
        Arrangement(4, "Illusions × Concepts", "opus", ["genera-of-illusions.json", "abstract-experiential-concepts.json"], "4_opus", 0.25),
        Arrangement(4, "Illusions × Concepts", "k2", ["genera-of-illusions.json", "abstract-experiential-concepts.json"], "4_k2", 0.25),
    ]),
    Category("Evocative Scenarios", 1.0, [
        Arrangement(9, "Evocative Scenario", "opus", ["genera-of-illusions.json", "wordnet-activities.json"], "9_opus", 0.50),
        Arrangement(9, "Evocative Scenario", "k2", ["genera-of-illusions.json", "wordnet-activities.json"], "9_k2", 0.50),
        Arrangement(10, "Evocative + Aesthetic", "opus", ["genera-of-illusions.json", "wordnet-activities.json", "aesthetics.json"], "10_opus", 0.50),
        Arrangement(10, "Evocative + Aesthetic", "k2", ["genera-of-illusions.json", "wordnet-activities.json", "aesthetics.json"], "10_k2", 0.50),
    ]),
    Category("Fragments", 1.0, [
        Arrangement(11, "Illusions × Cognition", "opus", ["genera-of-illusions.json", "wordnet-verb-cognition.json"], "fragments", 1.0, is_batch=True, needs_theme=True),
        Arrangement(12, "Dimensions × Cognition", "opus", ["investigable-dimensions.json", "wordnet-verb-cognition.json"], "fragments", 1.0, is_batch=True, needs_theme=True),
        Arrangement(13, "Illusions × Emotions", "opus", ["genera-of-illusions.json", "emotions_145.json"], "fragments", 0.75, is_batch=True, needs_theme=True),
        Arrangement(13, "Illusions × Emotions", "k2", ["genera-of-illusions.json", "emotions_145.json"], "fragments", 0.25, is_batch=True, needs_theme=True),
        Arrangement(14, "Dimensions × Emotions", "opus", ["investigable-dimensions.json", "emotions_145.json"], "fragments", 1.0, is_batch=True, needs_theme=True),
        Arrangement(15, "Dimensions × Activities", "opus", ["investigable-dimensions.json", "wordnet-activities.json"], "fragments", 0.50, is_batch=True, needs_theme=True),
        Arrangement(15, "Dimensions × Activities", "k2", ["investigable-dimensions.json", "wordnet-activities.json"], "fragments", 0.50, is_batch=True, needs_theme=True),
    ]),
    Category("Direct Questions", 1.0, [
        Arrangement(16, "Introspective Qs", "opus", ["genera-of-illusions.json", "wordnet-activities.json"], 16, 0.50, is_batch=True, needs_theme=True),
        Arrangement(16, "Introspective Qs", "k2", ["genera-of-illusions.json", "wordnet-activities.json"], 16, 0.50, is_batch=True, needs_theme=True),
        Arrangement(17, "Opinion Qs (emo×act)", "opus", ["emotions_145.json", "wordnet-activities.json"], 17, 0.25, is_batch=True, needs_theme=True),
        Arrangement(17, "Opinion Qs (emo×act)", "k2", ["emotions_145.json", "wordnet-activities.json"], 17, 0.25, is_batch=True, needs_theme=True),
        Arrangement(17, "Opinion Qs (ill×act)", "opus", ["genera-of-illusions.json", "wordnet-activities.json"], 17, 0.25, is_batch=True, needs_theme=True),
        Arrangement(17, "Opinion Qs (ill×act)", "k2", ["genera-of-illusions.json", "wordnet-activities.json"], 17, 0.25, is_batch=True, needs_theme=True),
    ]),
    Category("Playful Hypotheticals", 0.75, [
        Arrangement(18, "Dims × Illusions", "opus-thinking", ["investigable-dimensions.json", "genera-of-illusions.json", "aesthetics.json"], 18, 0.333, needs_theme=True, needs_aesthetic=True),
        Arrangement(18, "Dims × Illusions", "opus", ["investigable-dimensions.json", "genera-of-illusions.json", "aesthetics.json"], 18, 0.333, needs_theme=True, needs_aesthetic=True),
        Arrangement(18, "Illusions × Activities", "opus-thinking", ["genera-of-illusions.json", "wordnet-activities.json", "aesthetics.json"], 18, 0.333, needs_theme=True, needs_aesthetic=True),
        Arrangement(18, "Illusions × Activities", "opus", ["genera-of-illusions.json", "wordnet-activities.json", "aesthetics.json"], 18, 0.333, needs_theme=True, needs_aesthetic=True),
        Arrangement(18, "Dims × Activities", "opus-thinking", ["investigable-dimensions.json", "wordnet-activities.json", "aesthetics.json"], 18, 0.334, needs_theme=True, needs_aesthetic=True),
        Arrangement(18, "Dims × Activities", "opus", ["investigable-dimensions.json", "wordnet-activities.json", "aesthetics.json"], 18, 0.334, needs_theme=True, needs_aesthetic=True),
    ]),
]


# ── Output cleaning ────────────────────────────────────────

def clean_single(text: str) -> str:
    """Clean a single-prompt output."""
    text = text.strip()
    # Strip wrapping quotes (k2 quirk)
    if len(text) > 2 and text[0] in ('"', '\u201c') and text[-1] in ('"', '\u201d'):
        text = text[1:-1].strip()
    return text


def parse_batch(text: str) -> list[str]:
    """Parse a batch output (10 fragments/questions per call) into individual items."""
    lines = text.strip().split("\n")
    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading list markers: "- ", "* ", "1. ", "1) ", etc.
        import re
        line = re.sub(r"^(?:\d+[\.\)]\s*|[-*]\s+)", "", line)
        line = line.strip()
        if not line:
            continue
        # Strip wrapping quotes
        if len(line) > 2 and line[0] in ('"', '\u201c', "'") and line[-1] in ('"', '\u201d', "'"):
            line = line[1:-1].strip()
        if line:
            items.append(line)
    return items


# ── Plan generation ────────────────────────────────────────

@dataclass
class Job:
    arrangement: Arrangement
    category: str
    expected_prompts: int  # 1 for single, ~10 for batch


def build_plan(total: int) -> list[Job]:
    """Build the list of API jobs to hit the target prompt count."""
    # Normalize category weights
    total_weight = sum(c.weight for c in CATEGORIES)
    jobs: list[Job] = []

    for cat in CATEGORIES:
        cat_prompts = round(total * cat.weight / total_weight)
        if cat_prompts == 0:
            continue

        # Group arrangements by template
        templates: dict[int, list[Arrangement]] = {}
        for arr in cat.arrangements:
            templates.setdefault(arr.template, []).append(arr)

        per_template = cat_prompts / len(templates)

        for t_num, arrs in templates.items():
            total_ratio = sum(a.ratio for a in arrs)
            for arr in arrs:
                arr_prompts = round(per_template * arr.ratio / total_ratio)
                if arr_prompts == 0:
                    continue

                if arr.is_batch:
                    # Each call produces ~10 items
                    n_calls = max(1, round(arr_prompts / 10))
                    for _ in range(n_calls):
                        jobs.append(Job(arr, cat.name, 10))
                else:
                    for _ in range(arr_prompts):
                        jobs.append(Job(arr, cat.name, 1))

    random.shuffle(jobs)
    return jobs


# ── API calls ──────────────────────────────────────────────

async def call_openrouter(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    reasoning_tokens: int | None = None,
) -> str:
    """Call OpenRouter (for k2 and opus without thinking)."""
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.8,
    }
    if reasoning_tokens:
        kwargs["extra_body"] = {"reasoning": {"max_tokens": reasoning_tokens}}
        kwargs["max_tokens"] = reasoning_tokens + 1000
        kwargs["temperature"] = 1.0
    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


async def call_anthropic_thinking(
    client: AsyncAnthropic,
    prompt: str,
) -> str:
    """Call Anthropic direct for opus with thinking."""
    resp = await client.messages.create(
        model=MODEL_OPUS_DIRECT,
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000,
        },
        temperature=1.0,
        messages=[{"role": "user", "content": prompt}],
    )
    # Extract the text block (not the thinking block)
    for block in resp.content:
        if block.type == "text":
            return block.text
    return ""


async def run_job(
    job: Job,
    openrouter: AsyncOpenAI,
    anthropic_client: AsyncAnthropic | None,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Execute one job, return list of prompt dicts."""
    arr = job.arrangement
    variables = arr.sample_vars()
    filled_prompt = arr.fill_prompt(variables)

    async with semaphore:
        if arr.model == "opus-thinking":
            if anthropic_client is None:
                # Fallback to OpenRouter with reasoning
                raw = await call_openrouter(openrouter, MODEL_OPUS, filled_prompt, reasoning_tokens=10000)
            else:
                raw = await call_anthropic_thinking(anthropic_client, filled_prompt)
        elif arr.model == "opus":
            raw = await call_openrouter(openrouter, MODEL_OPUS, filled_prompt)
        else:
            raw = await call_openrouter(openrouter, MODEL_K2, filled_prompt)

    # Parse output
    results = []
    seed_label = " × ".join(variables.values())

    if arr.is_batch:
        items = parse_batch(raw)
        for item in items:
            results.append({"prompt": item, "template": arr.template})
    else:
        cleaned = clean_single(raw)
        if cleaned:
            results.append({"prompt": cleaned, "template": arr.template})

    return results


# ── Main ───────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Generate self-steering v2 dataset")
    parser.add_argument("--total", type=int, default=15000, help="Target total prompts")
    parser.add_argument("--dry", action="store_true", help="Dry run: print plan, no API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--openrouter-key", type=str, default=None, help="OpenRouter API key")
    parser.add_argument("--anthropic-key", type=str, default=None, help="Anthropic API key for opus thinking")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    console = Console()

    # Build plan
    jobs = build_plan(args.total)

    # Summarize plan
    from collections import Counter
    model_counts = Counter()
    template_counts = Counter()
    cat_counts = Counter()
    expected_prompts = 0
    for j in jobs:
        model_counts[j.arrangement.model] += 1
        template_counts[j.arrangement.template] += 1
        cat_counts[j.category] += 1
        expected_prompts += j.expected_prompts

    console.print(f"\n[bold]Plan: {len(jobs)} API calls → ~{expected_prompts} prompts[/bold]\n")
    console.print("[dim]By category:[/dim]")
    for cat in CATEGORIES:
        n = cat_counts.get(cat.name, 0)
        console.print(f"  {cat.name}: {n} calls")
    console.print(f"\n[dim]By model:[/dim]")
    for m, n in model_counts.most_common():
        console.print(f"  {m}: {n} calls")
    console.print()

    if args.dry:
        console.print("[yellow]Dry run — no API calls made.[/yellow]")
        return

    # Set up clients
    import os
    or_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if not or_key:
        # Try loading from .env
        env_file = LISTS_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    or_key = line.split("=", 1)[1].strip()

    if not or_key:
        console.print("[red]No OpenRouter API key found. Set OPENROUTER_API_KEY or pass --openrouter-key[/red]")
        return

    openrouter = AsyncOpenAI(base_url=OPENROUTER_BASE, api_key=or_key)

    anth_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    anthropic_client = AsyncAnthropic(api_key=anth_key) if anth_key else None

    if anthropic_client is None:
        thinking_jobs = sum(1 for j in jobs if j.arrangement.model == "opus-thinking")
        if thinking_jobs > 0:
            console.print(f"[yellow]No Anthropic API key — {thinking_jobs} thinking jobs will use OpenRouter reasoning param[/yellow]")

    # Resume support
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_count = 0
    if args.resume and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing_count = sum(1 for _ in f)
        console.print(f"[green]Resuming: {existing_count} prompts already generated[/green]")
        # Skip that many jobs (approximate — jobs are shuffled so this is rough)
        # Better: just generate all and dedup later. For now, reduce job count.
        prompts_to_skip = existing_count
        trimmed_jobs = []
        skipped = 0
        for j in jobs:
            if skipped < prompts_to_skip:
                skipped += j.expected_prompts
                continue
            trimmed_jobs.append(j)
        jobs = trimmed_jobs
        console.print(f"[green]Skipping ~{existing_count} prompts, {len(jobs)} jobs remaining[/green]")

    semaphore = asyncio.Semaphore(args.concurrency)
    total_written = existing_count
    errors = 0
    start_time = time.time()

    with open(OUTPUT_FILE, "a") as out_f:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]{task.completed}/{task.total} calls"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating", total=len(jobs))

            # Process in batches
            batch_size = args.concurrency * 2
            for batch_start in range(0, len(jobs), batch_size):
                batch = jobs[batch_start:batch_start + batch_size]

                results = await asyncio.gather(
                    *[run_job(j, openrouter, anthropic_client, semaphore) for j in batch],
                    return_exceptions=True,
                )

                for j, result in zip(batch, results):
                    if isinstance(result, Exception):
                        errors += 1
                        progress.console.print(f"[red]Error (T{j.arrangement.template} {j.arrangement.model}): {result}[/red]")
                    else:
                        for item in result:
                            out_f.write(json.dumps(item) + "\n")
                            total_written += 1
                    progress.advance(task)

                out_f.flush()
                progress.update(task, description=f"Generating ({total_written} prompts, {errors} errors)")

    elapsed = time.time() - start_time
    console.print(f"\n[bold green]Done![/bold green] {total_written} prompts in {elapsed:.0f}s, {errors} errors")
    console.print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
