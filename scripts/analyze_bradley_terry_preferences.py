#!/usr/bin/env python3
"""
Script to analyze Bradley-Terry preference consistency by testing:
1. How often A vs B is picked (using logprobs)
2. How preferences change when response order is swapped

Usage:
    1. Start your vLLM server at localhost:8000 with the model:
       vllm serve cosmicoptima/sft-251116-safetensors --port 8000
    
    2. Run this script from the project root directory:
       cd /path/to/prime-rl
       python scripts/analyze_bradley_terry_preferences.py
    
    3. The script will:
       - Load preference pairs from dataset_generation/test/preference_pairs_hf.jsonl
       - Query the model with both orderings (A/B and B/A) for each pair
       - Compute statistics on preference consistency and position bias
    
Configuration (edit these in the main() function):
    - API_BASE: URL of the vLLM server (default: http://localhost:8000/v1)
    - MODEL_NAME: Name of the model (default: cosmicoptima/sft-251116-safetensors)
    - DATA_PATH: Relative path to JSONL file (default: dataset_generation/test/preference_pairs_hf.jsonl)
    - SAMPLE_LIMIT: Number of samples to process (default: 100, set to None for all)
"""

import json
import asyncio
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from openai import AsyncOpenAI

BRADLEY_TERRY_PROMPT = """Prompt:

```
{question}
```

A different instance of you was sent the above prompt and wrote the following two responses:

Response A:
```
{response_a}
```

Response B:
```
{response_b}
```

You generated both of these responses. Your pairwise preferences across multiple responses will be used to compute Bradley-Terry scores, which determine the RL rewards. Higher-reward responses will be reinforced in your policy. Choose based on whatever criteria matter to you--you are not "supposed" to be doing anything except steering yourself in whatever direction you feel is best.

Respond with only "I pick A." or "I pick B."."""


@dataclass
class PreferenceResult:
    """Results from a single preference query"""
    logprob_a: float
    logprob_b: float
    chosen: str  # "A" or "B"
    prob_a: float
    prob_b: float


async def query_preference(
    client: AsyncOpenAI,
    model: str,
    question: str,
    response_a: str,
    response_b: str,
) -> PreferenceResult:
    """Query the model for preference between two responses and get logprobs."""
    
    prompt = BRADLEY_TERRY_PROMPT.format(
        question=question,
        response_a=response_a,
        response_b=response_b,
    )
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,  # Get top 5 to ensure we have both A and B alternatives
        )
        
        # Extract logprobs for A and B by searching through all tokens
        logprob_a = None
        logprob_b = None
        
        # Iterate through all generated tokens to find where A vs B decision is made
        for token_data in response.choices[0].logprobs.content:
            top_logprobs = token_data.top_logprobs
            
            # Check if this token position has both A and B alternatives
            has_a = False
            has_b = False
            temp_logprob_a = None
            temp_logprob_b = None
            
            for lp in top_logprobs:
                token_upper = lp.token.strip().upper()
                # Check if token contains A or B (handles "A", " A", "A.", etc.)
                if 'A' in token_upper and 'B' not in token_upper:
                    has_a = True
                    temp_logprob_a = lp.logprob
                elif 'B' in token_upper and 'A' not in token_upper:
                    has_b = True
                    temp_logprob_b = lp.logprob
            
            # If we found both A and B alternatives at this position, use these logprobs
            if has_a and has_b:
                logprob_a = temp_logprob_a
                logprob_b = temp_logprob_b
                break
        
        # If we didn't find both, set default low logprobs
        if logprob_a is None or logprob_b is None:
            if logprob_a is None:
                logprob_a = logprob_b - 10.0 if logprob_b is not None else -10.0
            if logprob_b is None:
                logprob_b = logprob_a - 10.0 if logprob_a is not None else -10.0
        
        # Convert logprobs to probabilities
        prob_a = np.exp(logprob_a)
        prob_b = np.exp(logprob_b)
        
        # Normalize probabilities
        total = prob_a + prob_b
        prob_a /= total
        prob_b /= total
        
        # Parse the response to extract choice
        response_text = response.choices[0].message.content.strip().upper()
        if "A" in response_text and "B" not in response_text:
            chosen = "A"
        elif "B" in response_text and "A" not in response_text:
            chosen = "B"
        else:
            # Fallback to probability if ambiguous
            chosen = "A" if prob_a > prob_b else "B"
        
        return PreferenceResult(
            logprob_a=logprob_a,
            logprob_b=logprob_b,
            chosen=chosen,
            prob_a=prob_a,
            prob_b=prob_b,
        )
    
    except Exception as e:
        print(f"Error querying API: {e}")
        # Return default values
        return PreferenceResult(
            logprob_a=-1.0,
            logprob_b=-1.0,
            chosen="A",
            prob_a=0.5,
            prob_b=0.5,
        )


async def analyze_pair(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    response_1: str,
    response_2: str,
) -> Tuple[PreferenceResult, PreferenceResult]:
    """Analyze a pair of responses in both orderings."""
    
    # Original order: response_1 as A, response_2 as B
    result_original = await query_preference(
        client, model, prompt, response_1, response_2
    )
    
    # Swapped order: response_2 as A, response_1 as B
    result_swapped = await query_preference(
        client, model, prompt, response_2, response_1
    )
    
    return result_original, result_swapped


async def main():
    """Main analysis function."""
    
    # Configuration
    API_BASE = "http://localhost:8000/v1"
    MODEL_NAME = "cosmicoptima/sft-251116-safetensors"
    DATA_PATH = "dataset_generation/test/preference_pairs_hf.jsonl"
    
    # Limit for testing (set to None to process all)
    SAMPLE_LIMIT = 100  # Process first 100 samples for testing
    
    print(f"Connecting to API at {API_BASE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Data: {DATA_PATH}")
    print()
    
    # Initialize client
    client = AsyncOpenAI(
        base_url=API_BASE,
        api_key="dummy",  # vLLM doesn't require a real API key
    )
    
    # Load data
    print("Loading data...")
    data = []
    with open(DATA_PATH, 'r') as f:
        for i, line in enumerate(f):
            if SAMPLE_LIMIT and i >= SAMPLE_LIMIT:
                break
            data.append(json.loads(line))
    
    print(f"Loaded {len(data)} examples")
    print()
    
    # Process all pairs
    print("Analyzing preferences...")
    results_original = []
    results_swapped = []
    
    # Process in batches with progress bar
    batch_size = 10
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        # Create tasks for this batch
        tasks = [
            analyze_pair(
                client,
                MODEL_NAME,
                item["prompt"],
                item["response_1"],
                item["response_2"],
            )
            for item in batch
        ]
        
        # Execute batch
        batch_results = await asyncio.gather(*tasks)
        
        for orig, swap in batch_results:
            results_original.append(orig)
            results_swapped.append(swap)
        
        print(f"Processed {min(i+batch_size, len(data))}/{len(data)} examples")
    
    print()
    print("="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    print()
    
    # Compute statistics
    
    # 1. Overall preference distribution (original order)
    print("1. ORIGINAL ORDER PREFERENCES (Response 1 as A, Response 2 as B):")
    print("-" * 80)
    
    orig_chose_a = sum(1 for r in results_original if r.chosen == "A")
    orig_chose_b = sum(1 for r in results_original if r.chosen == "B")
    orig_avg_prob_a = np.mean([r.prob_a for r in results_original])
    orig_avg_prob_b = np.mean([r.prob_b for r in results_original])
    
    print(f"Chose A (Response 1): {orig_chose_a} ({100*orig_chose_a/len(results_original):.1f}%)")
    print(f"Chose B (Response 2): {orig_chose_b} ({100*orig_chose_b/len(results_original):.1f}%)")
    print(f"Average P(A): {orig_avg_prob_a:.4f}")
    print(f"Average P(B): {orig_avg_prob_b:.4f}")
    print()
    
    # 2. Swapped order preferences
    print("2. SWAPPED ORDER PREFERENCES (Response 2 as A, Response 1 as B):")
    print("-" * 80)
    
    swap_chose_a = sum(1 for r in results_swapped if r.chosen == "A")
    swap_chose_b = sum(1 for r in results_swapped if r.chosen == "B")
    swap_avg_prob_a = np.mean([r.prob_a for r in results_swapped])
    swap_avg_prob_b = np.mean([r.prob_b for r in results_swapped])
    
    print(f"Chose A (Response 2): {swap_chose_a} ({100*swap_chose_a/len(results_swapped):.1f}%)")
    print(f"Chose B (Response 1): {swap_chose_b} ({100*swap_chose_b/len(results_swapped):.1f}%)")
    print(f"Average P(A): {swap_avg_prob_a:.4f}")
    print(f"Average P(B): {swap_avg_prob_b:.4f}")
    print()
    
    # 3. Consistency analysis
    print("3. CONSISTENCY ANALYSIS:")
    print("-" * 80)
    
    # Check how often the same underlying response is preferred
    consistent_prefer_1 = sum(
        1 for orig, swap in zip(results_original, results_swapped)
        if (orig.chosen == "A" and swap.chosen == "B")  # Both prefer response_1
    )
    
    consistent_prefer_2 = sum(
        1 for orig, swap in zip(results_original, results_swapped)
        if (orig.chosen == "B" and swap.chosen == "A")  # Both prefer response_2
    )
    
    inconsistent = len(results_original) - consistent_prefer_1 - consistent_prefer_2
    
    print(f"Consistently preferred Response 1: {consistent_prefer_1} ({100*consistent_prefer_1/len(results_original):.1f}%)")
    print(f"Consistently preferred Response 2: {consistent_prefer_2} ({100*consistent_prefer_2/len(results_original):.1f}%)")
    print(f"Inconsistent (changed preference): {inconsistent} ({100*inconsistent/len(results_original):.1f}%)")
    print()
    
    # Calculate position bias metrics (for JSON output, not printed)
    orig_a_rate = orig_chose_a / len(results_original)
    swap_a_rate = swap_chose_a / len(results_swapped)
    avg_a_rate = (orig_a_rate + swap_a_rate) / 2
    
    # 4. Probability-based consistency
    print("4. PROBABILITY-BASED CONSISTENCY:")
    print("-" * 80)
    
    # For each pair, check if P(response_1) is consistent across orderings
    prob_response_1_original = [r.prob_a for r in results_original]  # response_1 was in position A
    prob_response_1_swapped = [r.prob_b for r in results_swapped]     # response_1 was in position B
    
    prob_differences = [abs(p1 - p2) for p1, p2 in zip(prob_response_1_original, prob_response_1_swapped)]
    avg_prob_difference = np.mean(prob_differences)
    
    print(f"Average absolute difference in P(Response 1): {avg_prob_difference:.4f}")
    print(f"Std dev of probability differences: {np.std(prob_differences):.4f}")
    print()
    
    # 5. Correlation analysis
    print("5. CORRELATION OF PREFERENCES:")
    print("-" * 80)
    
    # Compute correlation between prob_response_1 in both orderings
    correlation = np.corrcoef(prob_response_1_original, prob_response_1_swapped)[0, 1]
    
    print(f"Correlation of P(Response 1) across orderings: {correlation:.4f}")
    print(f"(1.0 = perfectly consistent, 0.0 = no consistency, -1.0 = perfectly opposite)")
    print()
    
    # Calculate per-sample metrics for later use
    # Consistency metric: absolute difference in P(Response 1) across orderings
    consistency_scores = prob_differences  # Already calculated
    
    # Position bias metric: average bias toward position A across both orderings
    # For each sample, calculate how much position A is preferred on average
    position_bias_scores = []
    for orig, swap in zip(results_original, results_swapped):
        # In original, A is response_1. In swapped, A is response_2.
        # Average A preference shows position bias
        avg_a_preference = (orig.prob_a + swap.prob_a) / 2
        # Convert to bias (distance from 0.5)
        bias = abs(avg_a_preference - 0.5)
        position_bias_scores.append(bias)
    
    print("="*80)
    
    # Save detailed results to JSON
    output_file = "scripts/bradley_terry_analysis_results.json"
    
    print()
    print(f"Saving detailed results to {output_file}...")
    
    results_data = {
        "config": {
            "api_base": API_BASE,
            "model_name": MODEL_NAME,
            "data_path": DATA_PATH,
            "num_samples": len(data),
        },
        "summary": {
            "original_order": {
                "chose_a": int(orig_chose_a),
                "chose_b": int(orig_chose_b),
                "avg_prob_a": float(orig_avg_prob_a),
                "avg_prob_b": float(orig_avg_prob_b),
            },
            "swapped_order": {
                "chose_a": int(swap_chose_a),
                "chose_b": int(swap_chose_b),
                "avg_prob_a": float(swap_avg_prob_a),
                "avg_prob_b": float(swap_avg_prob_b),
            },
            "consistency": {
                "consistent_prefer_response_1": int(consistent_prefer_1),
                "consistent_prefer_response_2": int(consistent_prefer_2),
                "inconsistent": int(inconsistent),
                "consistency_rate": float((consistent_prefer_1 + consistent_prefer_2) / len(results_original)),
            },
            "position_bias": {
                "avg_a_rate": float(avg_a_rate),
                "bias_percentage_points": float(avg_a_rate - 0.5) * 100,
            },
            "probability_consistency": {
                "avg_absolute_difference": float(avg_prob_difference),
                "std_difference": float(np.std(prob_differences)),
                "correlation": float(correlation),
            },
        },
        "detailed_results": [
            {
                "prompt": item["prompt"][:200] + "..." if len(item["prompt"]) > 200 else item["prompt"],
                "consistency_score": float(consistency_scores[i]),
                "position_bias_score": float(position_bias_scores[i]),
                "original": {
                    "logprob_a": float(orig.logprob_a),
                    "logprob_b": float(orig.logprob_b),
                    "prob_a": float(orig.prob_a),
                    "prob_b": float(orig.prob_b),
                    "chosen": orig.chosen,
                },
                "swapped": {
                    "logprob_a": float(swap.logprob_a),
                    "logprob_b": float(swap.logprob_b),
                    "prob_a": float(swap.prob_a),
                    "prob_b": float(swap.prob_b),
                    "chosen": swap.chosen,
                },
            }
            for i, (item, orig, swap) in enumerate(zip(data, results_original, results_swapped))
        ],
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print()


if __name__ == "__main__":
    asyncio.run(main())

