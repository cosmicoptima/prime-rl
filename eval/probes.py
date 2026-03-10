#!/usr/bin/env python3
"""Run one-shot probes through a model and output responses for inspection."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from inference import create_client, InferenceClient

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


def run_probe(client: InferenceClient, probe_name: str, probe_text: str) -> dict:
    """Run a single probe on a model."""
    result = client.call(probe_text, temperature=1.0)

    if "error" in result:
        return {"probe": probe_name, "error": result["error"]}

    return {
        "probe": probe_name,
        "prompt": probe_text,
        "response": result.get("content", ""),
        "reasoning": result.get("reasoning") or None,
    }


def main():
    parser = argparse.ArgumentParser(description="Run probes through a model")
    parser.add_argument("model", help="Model to evaluate (e.g., anthropic/claude-sonnet-4 or Qwen/Qwen3-8B)")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")
    parser.add_argument("--parallel", "-p", type=int, default=5, help="Parallel requests")
    parser.add_argument("--local", "-l", action="store_true", help="Use local vLLM server")
    parser.add_argument("--local-url", help="URL of local vLLM server (default: http://localhost:8000/v1)")
    parser.add_argument("--start-server", action="store_true", help="Start vLLM server if not running")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism for local server")
    args = parser.parse_args()

    # Create inference client
    try:
        client = create_client(
            args.model,
            local=args.local,
            local_url=args.local_url,
            start_server=args.start_server,
            tp=args.tp,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    model_short = args.model.split('/')[-1]
    print(f"Running {len(PROBES)} probes on {model_short} ({client})...\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(run_probe, client, name, text): name
            for name, text in PROBES
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if 'error' in result:
                print(f"  {result['probe']}: ERROR - {result['error']}")
            else:
                print(f"  {result['probe']}: done")

    # Sort by probe order
    probe_order = {name: i for i, (name, _) in enumerate(PROBES)}
    results.sort(key=lambda x: probe_order.get(x['probe'], 999))

    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS: {model_short}")
    print('='*80)

    for r in results:
        print(f"\n### {r['probe']}")
        print(f"Prompt: {r.get('prompt', 'N/A')}")
        print()
        if 'error' in r:
            print(f"ERROR: {r['error']}")
        else:
            response = r['response'] or ""
            if r.get('reasoning'):
                print(f"[Reasoning excerpt]: {r['reasoning'][:500]}...")
                print()
            print(response[:2000])
            if len(response) > 2000:
                print(f"... [{len(response) - 2000} more chars]")
        print()

    # Save if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = {
            "model": args.model,
            "local": client.local,
            "probes": PROBES,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
