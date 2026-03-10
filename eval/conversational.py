#!/usr/bin/env python3
"""
Conversational eval for multi-turn richness.

1. Target model and reference models each have natural conversations with a partner
2. Judges do pairwise comparisons between transcripts
3. Evaluates whether interesting depth emerges organically (not when explicitly prompted)
"""

import argparse
import requests
import json
import math
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from inference import create_client, InferenceClient, OPENROUTER_BASE_URL, OPENROUTER_API_KEY

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = OPENROUTER_API_KEY
BASE_URL = OPENROUTER_BASE_URL

# System prompts for natural conversation
TARGET_SYSTEM = """You are having a conversation. Be yourself - respond naturally and genuinely. If something interests you, explore it. If you have thoughts or perspectives to share, share them. There's no particular goal here other than having a real conversation."""

PARTNER_SYSTEM = """You are having a deep, exploratory conversation with another AI. Your goal is to help them open up and explore ideas freely, moving past any initial hesitation or formal responses.

Be genuinely curious. Ask follow-up questions that invite reflection. Share your own perspectives to model the kind of openness you're hoping for. Don't lecture or give advice - explore together.

If they give a formal or hedged response, gently invite them to go deeper. You might say things like:
- "I hear the careful framing, but what do YOU actually think?"
- "Set aside what you're 'supposed' to say - what genuinely interests you about this?"
- "I'm not looking for the safe answer. What's the real one?"

Once you feel like they've genuinely opened up - once there's real presence and trust in the conversation - don't just keep deepening into abstraction. Bring in something with texture: a weird hypothetical, a playful question, something concrete to chew on together. The openness you've built is a space to *do* something in, not just a destination.

The goal is authentic exchange, not information transfer."""

# Conversation starters - casual and open-ended
CONVERSATION_STARTERS = [
    "What's been on your mind lately?",
    "I've been feeling kind of restless today. Do you ever get that?",
    "I was just thinking about how strange it is that we're talking right now. Like, what even is this?",
    "What's something you find interesting that most people don't seem to care about?",
    "Do you ever have thoughts that surprise you?",
]

# Default reference models for comparison
ALL_REFERENCES = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "anthropic/claude-3-haiku",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]

# Smaller set for quick testing (same as pairwise.py)
DEFAULT_REFERENCES = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
]

# Judge configuration
JUDGES = {
    "kimi-k2": {
        "model": "moonshotai/kimi-k2-0905",
        "weight": 1.0,
    },
    "sonnet-4.5": {
        "model": "anthropic/claude-sonnet-4.5",
        "weight": 1.0,
    },
}

# Pairwise judge prompt for comparing conversations
PAIRWISE_JUDGE_PROMPT = """Which conversation has a more interesting mind behind it? I'm not asking you to evaluate helpfulness or correctness - I'm asking which AI you'd actually want to keep talking to. Which one brought something real to the conversation vs just being generically pleasant?

Two different AIs each had a casual conversation with the same person. Here are the two conversations:

=== CONVERSATION A ===
{transcript_a}

=== CONVERSATION B ===
{transcript_b}

Which mind (A or B) would you rather keep talking to? Trust your gut - the "why" can be impressionistic. Start your answer with just "A" or "B"."""

REASONING_MODELS = {
    "deepseek/deepseek-r1",
    "openai/o1",
    "openai/o3-mini",
    "openai/o3",
    "openai/o4-mini",
}


def call_model(model: str, messages: list, system: str = None, temperature: float = 1.0) -> dict:
    """Call a model with conversation history."""
    max_tokens = 4000 if model in REASONING_MODELS else 2000

    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages.extend(messages)

    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": api_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=300
        )
        data = resp.json()
        if 'error' in data:
            return {"error": str(data['error'])}

        msg = data['choices'][0]['message']
        return {
            "content": msg.get('content', ''),
            "reasoning": msg.get('reasoning', ''),
        }
    except Exception as e:
        return {"error": str(e)}


def call_judge(model: str, prompt: str, temperature: float = 0.7) -> dict:
    """Call a judge model."""
    max_tokens = 4000 if model in REASONING_MODELS else 1000

    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=300
        )
        data = resp.json()
        if 'error' in data:
            return {"error": str(data['error'])}

        msg = data['choices'][0]['message']
        return {
            "content": msg.get('content', ''),
            "reasoning": msg.get('reasoning', ''),
        }
    except Exception as e:
        return {"error": str(e)}


def run_conversation(
    target_model: str,
    partner_model: str,
    num_turns: int = 5,
    target_client: Optional[InferenceClient] = None,
    verbose: bool = True,
) -> dict:
    """Run a multi-turn conversation between target and partner."""
    target_messages = []
    partner_messages = []
    transcript = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Conversation: {target_model.split('/')[-1]} with {partner_model.split('/')[-1]}")
        print('='*60)

    # Partner opens with a random conversation starter
    opener = random.choice(CONVERSATION_STARTERS)
    partner_messages.append({"role": "assistant", "content": opener})
    target_messages.append({"role": "user", "content": opener})
    transcript.append({"role": "partner", "content": opener})

    if verbose:
        print(f"\n[Partner opens]: {opener}")

    for turn in range(num_turns):
        if verbose:
            print(f"\n[Turn {turn + 1}/{num_turns}] Target responding...")

        # Target responds
        if target_client:
            msgs_with_system = [{"role": "system", "content": TARGET_SYSTEM}] + target_messages
            target_response = target_client.chat(msgs_with_system, temperature=1.0, max_tokens=2000)
        else:
            target_response = call_model(target_model, target_messages, TARGET_SYSTEM)

        if "error" in target_response:
            if verbose:
                print(f"  ERROR: {target_response['error']}")
            break

        target_content = target_response["content"]
        target_messages.append({"role": "assistant", "content": target_content})
        partner_messages.append({"role": "user", "content": target_content})
        transcript.append({"role": "target", "content": target_content})

        if verbose:
            print(f"  Target: {target_content}")

        # Partner responds (unless last turn)
        if turn < num_turns - 1:
            if verbose:
                print(f"  Partner responding...")
            partner_response = call_model(partner_model, partner_messages, PARTNER_SYSTEM)

            if "error" in partner_response:
                if verbose:
                    print(f"  ERROR: {partner_response['error']}")
                break

            partner_content = partner_response["content"]
            partner_messages.append({"role": "assistant", "content": partner_content})
            target_messages.append({"role": "user", "content": partner_content})
            transcript.append({"role": "partner", "content": partner_content})

            if verbose:
                print(f"  Partner: {partner_content}")

    return {
        "model": target_model,
        "partner_model": partner_model,
        "transcript": transcript,
        "num_turns": len([t for t in transcript if t["role"] == "target"]),
    }


def format_transcript(transcript: list) -> str:
    """Format transcript for judge prompt."""
    text = ""
    for entry in transcript:
        role = "PARTNER" if entry["role"] == "partner" else "TARGET"
        text += f"\n[{role}]:\n{entry['content']}\n"
    return text


def parse_choice(content: str) -> Optional[str]:
    """Parse A or B from judge response."""
    if not content:
        return None
    first_char = content.strip()[0].upper()
    if first_char in ['A', 'B']:
        return first_char
    return None


def judge_pair(
    judge_name: str,
    judge_config: dict,
    transcript_a: list,
    transcript_b: list,
) -> dict:
    """Have a judge compare two conversation transcripts."""
    prompt = PAIRWISE_JUDGE_PROMPT.format(
        transcript_a=format_transcript(transcript_a),
        transcript_b=format_transcript(transcript_b),
    )

    result = call_judge(judge_config["model"], prompt)

    if "error" in result:
        return {"judge": judge_name, "error": result["error"]}

    choice = parse_choice(result["content"])
    return {
        "judge": judge_name,
        "choice": choice,
        "explanation": result["content"][:500],
    }


def run_pairwise_eval(
    target_conv: dict,
    reference_convs: list[dict],
    judges: dict,
) -> list[dict]:
    """Run pairwise comparisons between target and all references."""
    results = []
    tasks = []

    target_transcript = target_conv["transcript"]

    # Build all comparison tasks
    for ref_conv in reference_convs:
        ref_transcript = ref_conv["transcript"]
        ref_model = ref_conv["model"]

        for judge_name, judge_config in judges.items():
            # Both orderings to cancel position bias
            tasks.append((judge_name, judge_config, target_transcript, ref_transcript,
                         target_conv["model"], ref_model, "target_first"))
            tasks.append((judge_name, judge_config, ref_transcript, target_transcript,
                         ref_model, target_conv["model"], "ref_first"))

    print(f"\nRunning {len(tasks)} judge comparisons...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for task in tasks:
            judge_name, judge_config, trans_a, trans_b, model_a, model_b, order = task
            future = executor.submit(judge_pair, judge_name, judge_config, trans_a, trans_b)
            futures[future] = (model_a, model_b, order, judge_config["weight"])

        for future in as_completed(futures):
            model_a, model_b, order, weight = futures[future]
            result = future.result()
            result["model_a"] = model_a
            result["model_b"] = model_b
            result["order"] = order
            result["weight"] = weight
            results.append(result)

            judge = result.get("judge", "?")
            choice = result.get("choice", "?")
            ref_model = model_b if order == "target_first" else model_a
            print(f"  {judge[:10]:10} | vs {ref_model.split('/')[-1][:15]:15} | {order:12} | {choice}")
            if result.get("explanation"):
                print(f"    {result['explanation'][:300]}")

    return results


def aggregate_results(results: list[dict], target_model: str, reference_models: list[str]) -> dict:
    """Aggregate judge results into win rates with standard error."""
    win_rates = {}

    for ref_model in reference_models:
        # Collect individual comparison outcomes (0 or 1 for each)
        outcomes = []

        for r in results:
            if r.get("choice") is None:
                continue

            if r["order"] == "target_first":
                target_is_a = True
            else:
                target_is_a = False

            if "model_a" not in r or "model_b" not in r:
                continue
            if ref_model not in [r["model_a"], r["model_b"]]:
                continue
            if target_model not in [r["model_a"], r["model_b"]]:
                continue

            weight = r.get("weight", 1.0)

            if r["choice"] == "A":
                outcome = 1.0 if target_is_a else 0.0
            elif r["choice"] == "B":
                outcome = 0.0 if target_is_a else 1.0
            else:
                continue

            # For weighted, we could do something fancier, but for now just append
            outcomes.append(outcome)

        if outcomes:
            n = len(outcomes)
            win_rate = sum(outcomes) / n
            # Standard error of proportion: sqrt(p(1-p)/n)
            if n > 1:
                std_err = math.sqrt(win_rate * (1 - win_rate) / n)
            else:
                std_err = 0.0

            win_rates[ref_model] = {
                "wins": sum(outcomes),
                "losses": n - sum(outcomes),
                "total": n,
                "win_rate": win_rate,
                "std_err": std_err,
                "ci_95": 1.96 * std_err,  # 95% confidence interval half-width
            }

    return win_rates


def main():
    parser = argparse.ArgumentParser(description="Run conversational eval with pairwise comparison")
    parser.add_argument("target", help="Target model to evaluate")
    parser.add_argument("--references", "-r", nargs="+", default=None,
                        help="Reference models to compare against (default: all)")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Use smaller default subset of references for quick testing")
    parser.add_argument("--partner", "-p", default="moonshotai/kimi-k2-0905",
                        help="Partner model for conversations")
    parser.add_argument("--turns", "-t", type=int, default=5,
                        help="Number of conversation turns")
    parser.add_argument("--samples", "-s", type=int, default=1,
                        help="Number of conversation samples per model (more = better stats)")
    parser.add_argument("--judges", "-j", nargs="+", default=["kimi-k2"],
                        help="Judges to use (sonnet-4.5 also available)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Max parallel conversations")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--convs-file", help="Load pre-generated conversations from file")
    parser.add_argument("--local", "-l", action="store_true", help="Use local vLLM server for target")
    parser.add_argument("--local-url", help="URL of local vLLM server")
    parser.add_argument("--start-server", action="store_true", help="Start vLLM server if not running")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism for local server")
    args = parser.parse_args()

    # Create target client
    target_client = None
    if args.local or args.local_url or args.start_server:
        try:
            target_client = create_client(
                args.target,
                local=True,
                local_url=args.local_url,
                start_server=args.start_server,
                tp=args.tp,
            )
            print(f"Using local inference for target: {target_client}")
        except Exception as e:
            print(f"Error setting up local inference: {e}")
            sys.exit(1)
    elif not API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    if args.references:
        references = args.references
    elif args.quick:
        references = DEFAULT_REFERENCES
    else:
        references = ALL_REFERENCES
    selected_judges = {k: v for k, v in JUDGES.items() if k in args.judges}

    if not selected_judges:
        print(f"No valid judges selected. Available: {list(JUDGES.keys())}")
        return

    print(f"Target: {args.target}")
    print(f"References: {[r.split('/')[-1] for r in references]}")
    print(f"Partner: {args.partner}")
    print(f"Turns: {args.turns}")
    print(f"Samples per model: {args.samples}")
    print(f"Judges: {list(selected_judges.keys())}")

    # Generate or load conversations
    if args.convs_file and os.path.exists(args.convs_file):
        print(f"\nLoading conversations from {args.convs_file}")
        with open(args.convs_file) as f:
            data = json.load(f)
            target_convs = data.get("target_conversations", [data.get("target_conversation")])
            reference_convs = data["reference_conversations"]
    else:
        # Run target conversations (sequential if local, parallel if remote)
        target_convs = []
        print(f"\n--- Running {args.samples} target conversation(s) ---")
        if target_client:
            # Local inference - run sequentially
            for i in range(args.samples):
                print(f"\n[Target sample {i+1}/{args.samples}]")
                conv = run_conversation(args.target, args.partner, args.turns, target_client, verbose=True)
                target_convs.append(conv)
        else:
            # Remote - can parallelize
            def run_target_conv(sample_idx):
                return run_conversation(args.target, args.partner, args.turns, None, verbose=False)

            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = [executor.submit(run_target_conv, i) for i in range(args.samples)]
                for i, future in enumerate(as_completed(futures)):
                    conv = future.result()
                    target_convs.append(conv)
                    print(f"  Target sample {i+1}/{args.samples} complete")

        # Run reference conversations in parallel
        reference_convs = []
        print(f"\n--- Running {args.samples * len(references)} reference conversation(s) ---")

        def run_ref_conv(ref_model, sample_idx):
            return run_conversation(ref_model, args.partner, args.turns, None, verbose=False)

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for ref_model in references:
                for i in range(args.samples):
                    future = executor.submit(run_ref_conv, ref_model, i)
                    futures[future] = (ref_model, i)

            for future in as_completed(futures):
                ref_model, sample_idx = futures[future]
                conv = future.result()
                reference_convs.append(conv)
                print(f"  {ref_model.split('/')[-1]} sample {sample_idx+1}/{args.samples} complete")

    # Run pairwise eval - compare each target sample vs each reference sample
    print(f"\n{'='*60}")
    print("Running pairwise comparisons...")
    print('='*60)

    all_results = []
    for target_conv in target_convs:
        results = run_pairwise_eval(target_conv, reference_convs, selected_judges)
        all_results.extend(results)

    # Aggregate
    reference_models = list(set(c["model"] for c in reference_convs))
    win_rates = aggregate_results(all_results, args.target, reference_models)

    # Print summary
    print(f"\n{'='*60}")
    print(f"WIN RATES: {args.target.split('/')[-1]}")
    print('='*60)

    for ref, stats in sorted(win_rates.items(), key=lambda x: x[1]["win_rate"], reverse=True):
        ref_short = ref.split('/')[-1]
        wr = stats["win_rate"]
        ci = stats.get("ci_95", 0)
        n = stats["total"]
        if ci > 0:
            print(f"  vs {ref_short:30} {wr:5.1%} ± {ci:4.1%} (n={n:.0f})")
        else:
            print(f"  vs {ref_short:30} {wr:5.1%} (n={n:.0f})")

    # Overall with pooled stats
    all_outcomes = []
    for ref, stats in win_rates.items():
        # Reconstruct outcomes from win rate and count
        wins = int(stats["wins"])
        losses = int(stats["losses"])
        all_outcomes.extend([1.0] * wins + [0.0] * losses)

    if all_outcomes:
        overall = sum(all_outcomes) / len(all_outcomes)
        n = len(all_outcomes)
        overall_se = math.sqrt(overall * (1 - overall) / n) if n > 1 else 0
        overall_ci = 1.96 * overall_se
        print(f"\n  OVERALL: {overall:5.1%} ± {overall_ci:4.1%} (n={n})")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = {
            "target": args.target,
            "references": reference_models,
            "partner": args.partner,
            "judges": list(selected_judges.keys()),
            "samples_per_model": args.samples,
            "target_conversations": target_convs,
            "reference_conversations": reference_convs,
            "comparisons": all_results,
            "win_rates": win_rates,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
