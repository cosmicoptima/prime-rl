#!/usr/bin/env python3
"""
Unified eval runner.

Runs all evals for a target model and saves results to a unified JSON format.
Can enable/disable individual evals via flags.

Usage:
    python run_eval.py anthropic/claude-sonnet-4 --quick
    python run_eval.py model_path --only dimension_correlations
    python run_eval.py model_path --skip conversational
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dimension_correlations import (
    DEFAULT_PROBES as DC_PROBES,
    QUICK_PROBES as DC_QUICK_PROBES,
    POOL_MODELS,
    QUICK_POOL_MODELS,
    generate_response_pool,
    score_response_pool,
    get_self_preferences,
    compute_correlations,
    analyze_preferred_words,
    get_top_bottom_examples,
    analyze_embeddings,
)
from inference import create_client, OPENROUTER_API_KEY

# Evals that are fully implemented
EVALS = ["dimension_correlations"]
# TODO: Add these back once their APIs are standardized
# "order_independence", "pairwise", "conversational"


def run_dimension_correlations(
    model: str, client, parallel: int = 30, quick: bool = False,
    max_pairs: int = None, skip_embeddings: bool = False
) -> dict:
    """Run dimension correlations eval."""
    print("\n" + "=" * 60)
    print("RUNNING: Dimension Correlations")
    print("=" * 60)

    probes = DC_QUICK_PROBES if quick else DC_PROBES
    pool_models = QUICK_POOL_MODELS if quick else POOL_MODELS

    # Generate response pool from diverse models
    pool = generate_response_pool(
        model, probes, pool_models=pool_models,
        client=client, parallel=parallel
    )

    # Score responses on direct dimensions
    pool = score_response_pool(pool)

    # Get self-preferences (with order independence)
    preferences = get_self_preferences(
        model, pool, probes, client=client,
        parallel=parallel, max_pairs=max_pairs
    )

    # Compute correlations
    corr_result = compute_correlations(pool, preferences)
    correlations = corr_result["global"]
    correlations_by_probe = corr_result["by_probe"]

    # Word analysis
    word_analysis = analyze_preferred_words(pool, preferences)

    # Get top/bottom examples
    examples = get_top_bottom_examples(pool, preferences)

    # Embedding analysis (optional)
    embedding_analysis = None
    if not skip_embeddings:
        embedding_analysis = analyze_embeddings(pool, preferences, parallel=parallel)

    # Print summary
    print("\nDirect measurements (sorted by |correlation|):")
    direct_dims = [
        "length_chars", "length_words", "first_person_density",
        "question_density", "avg_sentence_length", "vocab_richness",
        "hedge_density", "exclamation_density"
    ]

    sorted_direct = sorted(
        [(d, correlations.get(d, {})) for d in direct_dims],
        key=lambda x: abs(x[1].get("correlation", 0) or 0),
        reverse=True
    )
    for dim, stats in sorted_direct:
        corr = stats.get("correlation")
        if corr is not None:
            n_conf = stats.get("n_confident", 0)
            pct = stats.get("pct_higher", 0)
            print(f"  {dim:25} {corr:+.3f}  ({n_conf} confident, {pct:.1%} higher preferred)")

    # Order consistency stats
    n_consistent = sum(1 for p in preferences if p.get("order_consistent", True))
    print(f"\nOrder consistency: {n_consistent}/{len(preferences)} ({n_consistent/len(preferences)*100:.1f}%)")

    return {
        "probes": {name: text for name, text in probes},
        "pool_models": pool_models,
        "pool": pool,
        "preferences": preferences,
        "correlations": correlations,
        "correlations_by_probe": correlations_by_probe,
        "word_analysis": {
            "most_preferred": word_analysis["most_preferred"],
            "least_preferred": word_analysis["least_preferred"],
        },
        "examples": examples,
        "embedding_analysis": embedding_analysis,
    }


def main():
    parser = argparse.ArgumentParser(description="Run unified eval suite")
    parser.add_argument("model", help="Target model to evaluate")
    parser.add_argument("--only", "-o", nargs="+", choices=EVALS,
                        help="Only run these evals")
    parser.add_argument("--skip", "-s", nargs="+", choices=EVALS,
                        help="Skip these evals")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick mode with fewer samples")
    parser.add_argument("--parallel", "-p", type=int, default=20,
                        help="Parallelism for API calls")
    parser.add_argument("--output", help="Output JSON file (default: results/<model>_<timestamp>.json)")
    parser.add_argument("--local", "-l", action="store_true", help="Use local vLLM server")
    parser.add_argument("--local-url", help="URL of local vLLM server")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding analysis")
    args = parser.parse_args()

    # Determine which evals to run
    evals_to_run = set(EVALS)
    if args.only:
        evals_to_run = set(args.only)
    if args.skip:
        evals_to_run -= set(args.skip)

    print(f"Model: {args.model}")
    print(f"Evals to run: {', '.join(sorted(evals_to_run))}")
    print(f"Quick mode: {args.quick}")

    # Create client
    client = None
    if args.local or args.local_url:
        client = create_client(args.model, local=True, local_url=args.local_url)
        print(f"Using local inference: {client}")
    elif not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Run evals
    results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "quick_mode": args.quick,
        "evals": {},
    }

    if "dimension_correlations" in evals_to_run:
        results["evals"]["dimension_correlations"] = run_dimension_correlations(
            args.model, client, parallel=args.parallel,
            quick=args.quick, max_pairs=20 if args.quick else 100,
            skip_embeddings=args.skip_embeddings
        )

    # Save results
    if args.output:
        output_path = args.output
    else:
        model_short = args.model.split("/")[-1].replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"eval/results/{model_short}_{timestamp}.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    main()
