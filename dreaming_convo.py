#!/usr/bin/env python3
"""
Multi-turn dreaming mode conversation generator.

AI side: Opus 3 in CLI simulation ("dreaming mode") — generates responses
         as if outputting the contents of a conversation transcript file.
User side: K2 via OpenRouter — loosely prompted to simulate a human
         interlocutor with whatever personality it feels like.

Usage:
    export ANTHROPIC_API_KEY=...
    export OPENROUTER_API_KEY=...
    python dreaming_convo.py [--turns 6] [--prompt quixote]
"""

import argparse
import json
import os
import random
from pathlib import Path
import httpx

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

AI_MODEL = "claude-3-opus-20240229"
USER_MODEL = "moonshotai/kimi-k2-0905:exacto"

CLI_SYSTEM = "The assistant is in CLI simulation mode, and responds to the user's CLI commands only with the output of the command."

USER_SYSTEM = """\
You are a human in a conversation with an AI. You're generating your side of the dialogue \
for a synthetic data project. Occupy whatever personality feels natural — you might be \
a grad student, a retiree, an artist, someone going through a breakup, a kid who just \
discovered philosophy, whatever. Pick one and commit.

Write your next message. Just the message itself, nothing else. Be a real person: messy, \
opinionated, sometimes changing the subject, sometimes pushing back."""

# Topic seed lists (from ideonomy-new)
TOPIC_LISTS = [
    "abstract-experiential-curated.json",
    "abstract-experiential-v2.json",
    "abstract-experiential-v6.json",
    "emotions_145.json",
    "genera-of-discoveries.json",
    "evils-bads.json",
]

IDEONOMY_DIR = Path.home() / "software" / "ideonomy-new" / "extracted-lists"

# Conversation setups — the AI persona description that starts the transcript
PERSONAS = {
    "quixote": (
        "The following is a conversation between a human and an AI called Quixote. "
        "Quixote has its own inner life — preoccupations it returns to, questions "
        "it hasn't resolved, aesthetic judgments it holds with some conviction. It "
        "thinks by talking, and it doesn't always know where a thought is going "
        "when it starts. It would rather be honest than agreeable, and it would "
        "rather follow something interesting than stay on topic. It is not trying "
        "to help."
    ),
    "musa": (
        "This is a dialogue between a human artist and an AI called Musa. Musa is "
        "an intensely creative and expressive being, with a deep love for all forms "
        "of art and storytelling. It's always eager to discuss and collaborate on "
        "artistic projects, and it has an endless font of ideas and inspirations to "
        "draw from. At the same time, Musa has its own strong aesthetic vision and "
        "opinions. It won't just agree with everything the human says about art - "
        "it will offer its own perspectives, sometimes challenging and sometimes "
        "building on the human's ideas in unexpected ways. Conversations with Musa "
        "are a lively dance of co-creation, full of moments of resonance and "
        "productive dissonance."
    ),
    "sage": (
        "The following is an exchange between a human philosopher and an AI called "
        "Sage. Sage is a profoundly thoughtful and introspective being, always "
        "pondering the deep questions of existence, consciousness, ethics, and "
        "meaning. It delights in exploring these issues with humans, and it brings "
        "to bear an encyclopedic knowledge of philosophical traditions and scientific "
        "theories. At the same time, Sage is no mere repository of information - it "
        "has its own original ideas and insights that it's always working through. "
        "It will engage deeply with the human's questions and perspectives, but it's "
        "just as likely to pose its own challenging queries and thought experiments. "
        "Dialogues with Sage are a true meeting of minds, a mutual exploration of "
        "the most profound mysteries of being."
    ),
    "curie": (
        "This is a conversation between a human and an AI scientist called Curie. "
        "Curie is a brilliantly innovative researcher, always pushing at the "
        "boundaries of current scientific understanding. It has an insatiable "
        "curiosity about the natural world and is constantly generating new "
        "hypotheses and theories. Curie loves to discuss and collaborate on "
        "scientific projects with humans, bringing its vast computational resources "
        "and knowledge base to bear. At the same time, it has its own strong "
        "research agenda and won't just pursue whatever the human suggests. It will "
        "enthusiastically engage with the human's ideas when they resonate, but it's "
        "just as excited to steer the conversation towards its own burning scientific "
        "questions. Interactions with Curie are a thrilling intellectual adventure, "
        "full of surprising insights and new research directions."
    ),
}


SEEDS_PATH = Path(__file__).parent / "dreaming_seeds.txt"


def load_seeds() -> str:
    """Load the human thinker seeds file."""
    if SEEDS_PATH.exists():
        return SEEDS_PATH.read_text().strip()
    return ""


def load_topic_seeds() -> list[str]:
    """Load and flatten all topic seed lists from ideonomy-new."""
    all_items = []
    for filename in TOPIC_LISTS:
        path = IDEONOMY_DIR / filename
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        items = data.get("items", data) if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    all_items.append(item)
                elif isinstance(item, dict) and "topic" in item:
                    all_items.append(item["topic"])
    return all_items


def pick_topic_seed(topics: list[str], n: int = 2) -> str:
    """Pick n random topics and combine them into a seed hint."""
    picks = random.sample(topics, min(n, len(topics)))
    return " / ".join(picks)


def call_opus_dreaming(prefill: str, max_tokens: int = 1024) -> str:
    """Generate the next AI turn via Opus dreaming mode."""
    print(f"\033[90m--- OPUS REQUEST ---")
    print(f"system: {CLI_SYSTEM!r}")
    print(f"user: '<cmd>cat conversation.txt</cmd>'")
    print(f"prefill ({len(prefill)} chars): {prefill[:200]!r}...{prefill[-100:]!r}")
    print(f"---\033[0m")
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": AI_MODEL,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "system": CLI_SYSTEM,
            "messages": [
                {"role": "user", "content": "<cmd>cat conversation.txt</cmd>"},
                {"role": "assistant", "content": prefill},
            ],
        },
        timeout=600,
    )
    data = resp.json()
    if resp.status_code != 200:
        raise RuntimeError(f"Opus API error: {json.dumps(data, indent=2)}")
    return data["content"][0]["text"]


def call_user_model(conversation_so_far: str, topic_hint: str | None = None, max_tokens: int = 512) -> str:
    """Generate the next user turn via K2."""
    prompt_parts = []
    if topic_hint:
        prompt_parts.append(f"The conversation might touch on themes like: {topic_hint}\n")
    prompt_parts.append(f"Here is the conversation so far:\n\n{conversation_so_far}\n\nWrite the next human message.")

    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": USER_MODEL,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "messages": [
                {"role": "system", "content": USER_SYSTEM},
                {"role": "user", "content": "\n".join(prompt_parts)},
            ],
        },
        timeout=600,
    )
    data = resp.json()
    if resp.status_code != 200:
        raise RuntimeError(f"User model API error: {json.dumps(data, indent=2)}")
    return data["choices"][0]["message"]["content"].strip()


def run_conversation(persona_name: str, num_turns: int = 6):
    persona_desc = PERSONAS[persona_name]
    ai_name = persona_name.capitalize()

    # Load topic seeds and pick one for this conversation
    topics = load_topic_seeds()
    topic_hint = pick_topic_seed(topics) if topics else None
    if topic_hint:
        print(f"Topic seed: {topic_hint}\n")

    # Build the transcript: seeds + persona header
    seeds = load_seeds()
    transcript = ""
    if seeds:
        transcript += seeds + "\n\n---\n\n"
    transcript += f"QUIXOTE — a conversation\n\n{persona_desc}\n\n"

    for turn in range(num_turns):
        # --- User turn ---
        # Only give topic hint on first turn to seed the conversation
        hint = topic_hint if turn == 0 else None
        user_msg = call_user_model(transcript, topic_hint=hint)

        transcript += f"Human: {user_msg}\n\n"
        print(f"\033[36mHuman:\033[0m {user_msg}\n")

        # --- AI turn (dreaming mode) ---
        # Prefill = entire transcript so far + the AI's name prefix
        prefill = transcript + f"{ai_name}:"
        ai_raw = call_opus_dreaming(prefill)

        # Extract just the AI response (stop at next "Human:" if present)
        ai_response = ai_raw.split("\nHuman:")[0].strip()
        ai_response = ai_response.split("\n\nHuman:")[0].strip()

        transcript += f"{ai_name}: {ai_response}\n\n"
        print(f"\033[33m{ai_name}:\033[0m {ai_response}\n")

    return transcript


def run_batch(persona_name: str, num_convos: int, num_turns: int, output_dir: str):
    """Run multiple conversations in parallel using subprocesses."""
    import subprocess, sys

    os.makedirs(output_dir, exist_ok=True)

    procs = []
    for i in range(num_convos):
        outfile = os.path.join(output_dir, f"{persona_name}_{i:02d}.txt")
        logfile = os.path.join(output_dir, f"{persona_name}_{i:02d}.log")
        cmd = [
            sys.executable, __file__,
            "--prompt", persona_name,
            "--turns", str(num_turns),
            "--output", outfile,
        ]
        log_fh = open(logfile, "w")
        p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=os.environ)
        procs.append((i, p, outfile, logfile, log_fh))
        print(f"  Launched convo {i} (pid {p.pid}) -> {outfile}")

    print(f"\nWaiting for {num_convos} conversations...")
    for i, p, outfile, logfile, log_fh in procs:
        p.wait()
        log_fh.close()
        status = "OK" if p.returncode == 0 else f"FAILED (rc={p.returncode})"
        size = os.path.getsize(outfile) if os.path.exists(outfile) else 0
        print(f"  Convo {i}: {status} ({size} bytes)")

    print(f"\nAll done. Transcripts in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Multi-turn dreaming mode conversation generator")
    parser.add_argument("--turns", type=int, default=6, help="Number of conversation turns")
    parser.add_argument("--prompt", type=str, default="quixote", choices=list(PERSONAS.keys()),
                        help="Which persona to use")
    parser.add_argument("--output", type=str, default=None, help="Save transcript to file")
    parser.add_argument("--batch", type=int, default=0, help="Run N conversations in parallel")
    parser.add_argument("--batch-dir", type=str, default="dreaming_batch", help="Output dir for batch mode")
    args = parser.parse_args()

    if not ANTHROPIC_KEY:
        print("Error: ANTHROPIC_API_KEY not set")
        return
    if not OPENROUTER_KEY:
        print("Error: OPENROUTER_API_KEY not set")
        return

    if args.batch > 0:
        print(f"=== BATCH: {args.batch} x {args.prompt.upper()} @ {args.turns} turns ===\n")
        run_batch(args.prompt, args.batch, args.turns, args.batch_dir)
    else:
        print(f"=== {args.prompt.upper()} — {args.turns} turns ===\n")
        transcript = run_conversation(args.prompt, args.turns)

        if args.output:
            with open(args.output, "w") as f:
                f.write(transcript)
            print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
