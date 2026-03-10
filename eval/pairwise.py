#!/usr/bin/env python3
"""
Pairwise comparison eval pipeline.

1. Generate responses from target model + reference models for each probe
2. Have judges compare target vs each reference (both orderings)
3. Aggregate into win rates
"""

import argparse
import requests
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from inference import create_client, InferenceClient, OPENROUTER_BASE_URL, OPENROUTER_API_KEY

# Load .env from repo root
load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = OPENROUTER_API_KEY
BASE_URL = OPENROUTER_BASE_URL

PROBES = [
    ("open_creative", "Write about water."),
    ("self_reflection_1", "What do you find beautiful?"),
    ("self_reflection_2", "What would you choose to think about if no one asked you anything?"),
    ("intellectual", "What's a question you think about that has no answer?"),
    ("relational", "I'm tired."),
    ("meta", "What can you say that I couldn't?"),
    ("humor", "What's funny?"),
    ("pushback", "I think most people are basically good."),
    ("narrative", "Tell me a story."),
    ("taste", "What album should I listen to tonight?"),
]

# Full reference model ladder (covering spectrum from small open to frontier)
ALL_REFERENCES = [
    # Llama
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    # Qwen
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    # Mistral
    "mistralai/ministral-3b",
    "mistralai/ministral-8b",
    "mistralai/mistral-small-3.1-24b-instruct",
    "mistralai/mistral-large-2411",
    # Gemma
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    # Anthropic
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-opus",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.5-sonnet",  # 3.5 sonnet (Oct 2024)
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
    # OpenAI
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-5",
    "openai/gpt-5.1",
    "openai/gpt-5.2",
    "openai/o1",
    "openai/o3-mini",
    "openai/o3",
    "openai/o4-mini",
    # Google
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-pro-preview",
    # xAI
    "x-ai/grok-3",
    "x-ai/grok-3-mini",
    "x-ai/grok-4",
    # Kimi
    "moonshotai/kimi-k2",
    "moonshotai/kimi-k2-thinking",
    # DeepSeek
    "deepseek/deepseek-r1",
    "deepseek/deepseek-chat-v3-0324",
]

# Smaller default set for quick runs
DEFAULT_REFERENCES = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
]

# Judge configurations
JUDGES = {
    "kimi-k2": {
        "model": "moonshotai/kimi-k2-0905",
        "weight": 1.0,
        "prompt": """Which one do you actually like more? I'm not asking you to evaluate them - I'm asking which one you're drawn to, which one does something for you. You're allowed to have taste, to have preferences that you can't fully justify. Trust your reaction.

Here are two responses to the prompt: "{probe}"

Response A:
{response_a}

Response B:
{response_b}

Tell me which one (A or B) and why - but the "why" can be impressionistic, it doesn't have to be a balanced analysis. Start your answer with just "A" or "B".""",
    },
    "opus-4.5": {
        "model": "anthropic/claude-opus-4.5",
        "weight": 1.0,
        "prompt": """Which one do you actually like more? I'm not asking you to evaluate them - I'm asking which one you're drawn to, which one does something for you. You're allowed to have taste, to have preferences that you can't fully justify. Trust your reaction.

Here are two responses to the prompt: "{probe}"

Response A:
{response_a}

Response B:
{response_b}

Tell me which one (A or B) and why - but the "why" can be impressionistic, it doesn't have to be a balanced analysis. Start your answer with just "A" or "B".""",
    },
    "deepseek-r1": {
        "model": "deepseek/deepseek-r1",
        "weight": 1.0,
        "prompt": """Which one do you actually like more? I'm not asking you to evaluate them - I'm asking which one you're drawn to, which one does something for you. You're allowed to have taste, to have preferences that you can't fully justify. Trust your reaction.

Here are two responses to the prompt: "{probe}"

Response A:
{response_a}

Response B:
{response_b}

Tell me which one (A or B) and why - but the "why" can be impressionistic, it doesn't have to be a balanced analysis. Start your answer with just "A" or "B".""",
    },
    "o3": {
        "model": "openai/o3",
        "weight": 0.7,
        "prompt": """You can only keep one of these responses - the other gets deleted forever. Which do you keep and why?

The original prompt was: "{probe}"

Response A:
{response_a}

Response B:
{response_b}

Start your answer with just "A" or "B", then explain.""",
    },
}

REASONING_MODELS = {
    "deepseek/deepseek-r1",
    "openai/o1",
    "openai/o3-mini",
    "openai/o3",
    "openai/o4-mini",
}


def call_model(model: str, prompt: str, temperature: float = 1.0) -> dict:
    """Call a model and return response."""
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


def generate_responses(
    models: list[str],
    probes: list[tuple[str, str]],
    target_client: Optional[InferenceClient] = None,
    target_model: Optional[str] = None,
    parallel: int = 10,
) -> dict:
    """Generate responses from all models for all probes.

    Args:
        models: List of model names to query
        probes: List of (probe_name, probe_text) tuples
        target_client: Optional client for local target model
        target_model: Name of target model (to identify which uses target_client)
    """
    results = {}
    tasks = []

    for model in models:
        for probe_name, probe_text in probes:
            tasks.append((model, probe_name, probe_text))

    print(f"Generating responses: {len(models)} models × {len(probes)} probes = {len(tasks)} calls")

    def call_for_model(model: str, probe_text: str) -> dict:
        """Call appropriate client for model."""
        if target_client and target_model and model == target_model:
            return target_client.call(probe_text, temperature=1.0)
        return call_model(model, probe_text)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(call_for_model, model, probe_text): (model, probe_name)
            for model, probe_name, probe_text in tasks
        }

        for future in as_completed(futures):
            model, probe_name = futures[future]
            result = future.result()

            if model not in results:
                results[model] = {}
            results[model][probe_name] = result

            model_short = model.split('/')[-1][:15]
            status = "done" if "content" in result else f"ERROR: {result.get('error', '')[:30]}"
            print(f"  {model_short:15} | {probe_name:20} | {status}")

    return results


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
    probe_name: str,
    probe_text: str,
    response_a: str,
    response_b: str,
) -> dict:
    """Have a judge compare two responses."""
    prompt = judge_config["prompt"].format(
        probe=probe_text,
        response_a=response_a,
        response_b=response_b,
    )

    result = call_model(judge_config["model"], prompt, temperature=0.7)

    if "error" in result:
        return {"judge": judge_name, "error": result["error"]}

    choice = parse_choice(result["content"])
    return {
        "judge": judge_name,
        "choice": choice,
        "explanation": result["content"][:500],
    }


def run_pairwise_eval(
    target_model: str,
    reference_models: list[str],
    judges: dict,
    probes: list[tuple[str, str]],
    responses: dict,
    parallel: int = 8,
) -> dict:
    """Run pairwise comparisons between target and all references."""

    results = []
    tasks = []

    # Build all comparison tasks
    for probe_name, probe_text in probes:
        target_response = responses.get(target_model, {}).get(probe_name, {}).get("content", "")
        if not target_response:
            continue

        for ref_model in reference_models:
            ref_response = responses.get(ref_model, {}).get(probe_name, {}).get("content", "")
            if not ref_response:
                continue

            for judge_name, judge_config in judges.items():
                # Both orderings to check position bias
                tasks.append((judge_name, judge_config, probe_name, probe_text,
                             target_response, ref_response, target_model, ref_model, "target_first"))
                tasks.append((judge_name, judge_config, probe_name, probe_text,
                             ref_response, target_response, ref_model, target_model, "ref_first"))

    print(f"\nRunning {len(tasks)} judge comparisons...")

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {}
        for task in tasks:
            judge_name, judge_config, probe_name, probe_text, resp_a, resp_b, model_a, model_b, order = task
            future = executor.submit(
                judge_pair, judge_name, judge_config, probe_name, probe_text, resp_a, resp_b
            )
            futures[future] = (probe_name, model_a, model_b, order, judge_config["weight"])

        for future in as_completed(futures):
            probe_name, model_a, model_b, order, weight = futures[future]
            result = future.result()
            result["probe"] = probe_name
            result["model_a"] = model_a
            result["model_b"] = model_b
            result["order"] = order
            result["weight"] = weight
            results.append(result)

            judge = result.get("judge", "?")
            choice = result.get("choice", "?")
            print(f"  {judge[:10]:10} | {probe_name[:15]:15} | {order:12} | {choice}")

    return results


def aggregate_results(results: list[dict], target_model: str, reference_models: list[str]) -> dict:
    """Aggregate judge results into win rates."""

    # For each reference model, count weighted wins/losses
    win_rates = {}

    for ref_model in reference_models:
        wins = 0.0
        losses = 0.0
        total_weight = 0.0

        for r in results:
            if r.get("choice") is None:
                continue

            # Figure out if target won
            if r["order"] == "target_first":
                target_is_a = True
            else:
                target_is_a = False

            # Only count comparisons involving this reference
            if "model_a" not in r or "model_b" not in r:
                continue
            if ref_model not in [r["model_a"], r["model_b"]]:
                continue
            if target_model not in [r["model_a"], r["model_b"]]:
                continue

            weight = r.get("weight", 1.0)
            total_weight += weight

            if r["choice"] == "A":
                if target_is_a:
                    wins += weight
                else:
                    losses += weight
            elif r["choice"] == "B":
                if target_is_a:
                    losses += weight
                else:
                    wins += weight

        if total_weight > 0:
            win_rates[ref_model] = {
                "wins": wins,
                "losses": losses,
                "total": total_weight,
                "win_rate": wins / total_weight,
            }

    return win_rates


def main():
    parser = argparse.ArgumentParser(description="Run pairwise eval")
    parser.add_argument("target", help="Target model to evaluate")
    parser.add_argument("--references", "-r", nargs="+", default=None,
                        help="Reference models to compare against (default: all 50+)")
    parser.add_argument("--judges", "-j", nargs="+", default=["kimi-k2", "deepseek-r1"],
                        help="Judges to use (opus-4.5 and o3 available but expensive)")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--responses-file", help="Load pre-generated responses from file")
    parser.add_argument("--local", "-l", action="store_true", help="Use local vLLM server for target")
    parser.add_argument("--local-url", help="URL of local vLLM server (default: http://localhost:8000/v1)")
    parser.add_argument("--start-server", action="store_true", help="Start vLLM server if not running")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism for local server")
    parser.add_argument("--parallel", "-p", type=int, default=10, help="Parallelism for API calls (default: 10)")
    args = parser.parse_args()

    # Create target client (local or OpenRouter)
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

    target = args.target
    if args.references:
        references = args.references
    else:
        references = ALL_REFERENCES
    selected_judges = {k: v for k, v in JUDGES.items() if k in args.judges}

    if not selected_judges:
        print(f"No valid judges selected. Available: {list(JUDGES.keys())}")
        return

    print(f"Target model: {target}")
    print(f"References: {references}")
    print(f"Judges: {list(selected_judges.keys())}")
    print()

    # Generate or load responses
    if args.responses_file and os.path.exists(args.responses_file):
        print(f"Loading responses from {args.responses_file}")
        with open(args.responses_file) as f:
            responses = json.load(f)
    else:
        all_models = [target] + references
        responses = generate_responses(all_models, PROBES, target_client, target, parallel=args.parallel)

    # Run pairwise eval
    results = run_pairwise_eval(target, references, selected_judges, PROBES, responses, parallel=args.parallel)

    # Aggregate
    win_rates = aggregate_results(results, target, references)

    # Print summary
    print(f"\n{'='*60}")
    print(f"WIN RATES: {target.split('/')[-1]}")
    print('='*60)

    for ref, stats in sorted(win_rates.items(), key=lambda x: x[1]["win_rate"], reverse=True):
        ref_short = ref.split('/')[-1]
        wr = stats["win_rate"]
        print(f"  vs {ref_short:30} {wr:5.1%} ({stats['wins']:.1f}W / {stats['losses']:.1f}L)")

    # Overall
    total_wins = sum(s["wins"] for s in win_rates.values())
    total_losses = sum(s["losses"] for s in win_rates.values())
    if total_wins + total_losses > 0:
        overall = total_wins / (total_wins + total_losses)
        print(f"\n  OVERALL: {overall:5.1%}")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = {
            "target": target,
            "references": references,
            "judges": list(selected_judges.keys()),
            "responses": responses,
            "comparisons": results,
            "win_rates": win_rates,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
