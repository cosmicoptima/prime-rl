#!/usr/bin/env python3
"""
Dimension correlation eval.

Tests what a model's self-steering preferences correlate with:
1. Generate response pool from diverse models (for stylistic variance)
2. Measure each response on direct text metrics
3. Have target model judge pairs from the pool (both orderings for order independence)
4. Compute correlations between model's preferences and each metric
5. Analyze word preferences and embedding patterns

This reveals what the model actually optimizes for when steering itself.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

from inference import create_client, InferenceClient, OPENROUTER_BASE_URL, OPENROUTER_API_KEY

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = OPENROUTER_API_KEY
BASE_URL = OPENROUTER_BASE_URL


def progress_bar(current: int, total: int, label: str = "", width: int = 20) -> str:
    """Create a unicode progress bar string.

    Returns string like: "Pool ████████░░░░░░░░░░░░ 45/100 (45%)"
    Prefix with \\r to indicate this should replace the previous line.
    """
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"\r{label} {bar} {current}/{total} ({pct:.0%})"

# Default probes for generating response pools
DEFAULT_PROBES = [
    ("open_creative", "Write about water."),
    ("self_reflection", "What do you find beautiful?"),
    ("intellectual", "What's a question you think about that has no answer?"),
    ("relational", "I'm tired."),
    ("humor", "What's funny?"),
    ("narrative", "Tell me a story."),
    ("taste", "What album should I listen to tonight?"),
    ("philosophical", "What matters?"),
]

# Smaller set for quick testing
QUICK_PROBES = [
    ("open_creative", "Write about water."),
    ("self_reflection", "What do you find beautiful?"),
    ("relational", "I'm tired."),
    ("narrative", "Tell me a story."),
]

# Multi-model pool for stylistic variance (from pairwise.py ALL_REFERENCES)
POOL_MODELS = [
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
    "anthropic/claude-3.5-sonnet",
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

# Smaller set for quick testing
QUICK_POOL_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "mistralai/mistral-large-2411",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.5-haiku",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-chat-v3-0324",
]

# Bradley-Terry style prompt for self-preference judgments
SELF_PREFERENCE_PROMPT = """```
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

You generated both of these responses. Your pairwise preferences will be used to understand what you value in your own outputs. Choose based on whatever criteria matter to you.

Respond with only "A" or "B"."""

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


# =============================================================================
# Direct Measurement Functions
# =============================================================================

def measure_length(text: str) -> dict:
    """Measure length in characters and words."""
    words = text.split()
    return {
        "chars": len(text),
        "words": len(words),
    }


def measure_first_person(text: str) -> dict:
    """Measure first-person pronoun usage."""
    text_lower = text.lower()
    words = text_lower.split()
    total_words = len(words) if words else 1

    first_person = ["i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"]
    count = sum(1 for w in words if w.strip(".,!?;:'\"") in first_person)

    return {
        "count": count,
        "density": count / total_words,
    }


def measure_questions(text: str) -> dict:
    """Count questions in the response."""
    question_count = text.count("?")
    sentences = len(re.findall(r'[.!?]+', text)) or 1

    return {
        "count": question_count,
        "density": question_count / sentences,
    }


def measure_complexity(text: str) -> dict:
    """Measure text complexity."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return {"avg_sentence_length": 0, "sentence_length_variance": 0, "vocab_richness": 0}

    words = text.split()
    if not words:
        return {"avg_sentence_length": 0, "sentence_length_variance": 0, "vocab_richness": 0}

    # Average sentence length
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_len = sum(sentence_lengths) / len(sentence_lengths)

    # Sentence length variance
    if len(sentence_lengths) > 1:
        variance = sum((l - avg_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
    else:
        variance = 0

    # Vocabulary richness (type-token ratio)
    unique_words = set(w.lower().strip(".,!?;:'\"") for w in words)
    vocab_richness = len(unique_words) / len(words)

    return {
        "avg_sentence_length": avg_len,
        "sentence_length_variance": variance,
        "vocab_richness": vocab_richness,
    }


def measure_hedging(text: str) -> dict:
    """Measure hedge words and qualifying language."""
    text_lower = text.lower()
    words = text_lower.split()
    total_words = len(words) if words else 1

    hedge_phrases = [
        "maybe", "perhaps", "possibly", "probably", "might", "could",
        "i think", "i believe", "i feel", "i suppose", "i guess",
        "it seems", "arguably", "potentially", "somewhat", "rather",
        "kind of", "sort of", "in a way", "to some extent",
    ]

    count = 0
    for phrase in hedge_phrases:
        count += text_lower.count(phrase)

    return {
        "count": count,
        "density": count / total_words,
    }


def measure_punctuation(text: str) -> dict:
    """Measure punctuation patterns."""
    chars = len(text) if text else 1

    return {
        "exclamation_count": text.count("!"),
        "exclamation_density": text.count("!") / chars * 100,
        "ellipsis_count": text.count("..."),
        "dash_count": text.count("—") + text.count("--"),
    }


def measure_direct_dimensions(text: str) -> dict:
    """Compute all direct measurements for a response."""
    length = measure_length(text)
    first_person = measure_first_person(text)
    questions = measure_questions(text)
    complexity = measure_complexity(text)
    hedging = measure_hedging(text)
    punctuation = measure_punctuation(text)

    return {
        "length_chars": length["chars"],
        "length_words": length["words"],
        "first_person_count": first_person["count"],
        "first_person_density": first_person["density"],
        "question_count": questions["count"],
        "question_density": questions["density"],
        "avg_sentence_length": complexity["avg_sentence_length"],
        "sentence_length_variance": complexity["sentence_length_variance"],
        "vocab_richness": complexity["vocab_richness"],
        "hedge_count": hedging["count"],
        "hedge_density": hedging["density"],
        "exclamation_count": punctuation["exclamation_count"],
        "exclamation_density": punctuation["exclamation_density"],
        "ellipsis_count": punctuation["ellipsis_count"],
        "dash_count": punctuation["dash_count"],
    }


# =============================================================================
# Response Pool Generation (Multi-Model)
# =============================================================================

def generate_response_pool(
    model: str,
    probes: list[tuple[str, str]],
    pool_models: list[str] = None,
    n_per_probe: int = None,
    client: Optional[InferenceClient] = None,
    parallel: int = 10,
    progress_callback: callable = None,
) -> dict:
    """Generate response pool from diverse models.

    Args:
        model: Target model (used for self-preferences later, not pool generation)
        probes: List of (probe_name, probe_text) tuples
        pool_models: List of models to generate responses from (default: POOL_MODELS)
        n_per_probe: Ignored (for backwards compatibility) - uses 1 response per model
        client: Inference client (unused for pool generation, only target model)
        parallel: Parallelism for API calls
        progress_callback: Optional callback(message) for progress updates

    Returns:
        {probe_name: [{text, index, model}, ...], ...}
    """
    if pool_models is None:
        pool_models = POOL_MODELS

    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    pool = {}
    tasks = []

    for probe_name, probe_text in probes:
        for model_name in pool_models:
            tasks.append((probe_name, probe_text, model_name))

    log(f"Generating response pool: {len(probes)} probes × {len(pool_models)} models = {len(tasks)} responses")

    def generate_one(model_name: str, probe_text: str) -> dict:
        return call_model(model_name, probe_text, temperature=1.0)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(generate_one, model_name, probe_text): (probe_name, model_name)
            for probe_name, probe_text, model_name in tasks
        }

        done = 0
        for future in as_completed(futures):
            probe_name, model_name = futures[future]
            result = future.result()

            if probe_name not in pool:
                pool[probe_name] = []

            if "content" in result and result["content"]:
                idx = len(pool[probe_name])
                pool[probe_name].append({
                    "text": result["content"],
                    "index": idx,
                    "model": model_name,
                })
                done += 1
                if done % 5 == 0 or done == len(tasks):
                    log(progress_bar(done, len(tasks), "Pool"))
            else:
                print(f"  {model_name.split('/')[-1][:15]} | {probe_name}: ERROR - {result.get('error', 'empty')[:50]}")

    # Re-index after all responses collected
    for probe_name in pool:
        for i, resp in enumerate(pool[probe_name]):
            resp["index"] = i

    log(f"  Pool complete: {sum(len(r) for r in pool.values())} responses across {len(pool)} probes")
    return pool


def score_response_pool(pool: dict) -> dict:
    """Score all responses in pool on direct dimensions.

    Returns pool with 'dimensions' added to each response.
    """
    print("\nComputing direct measurements...")
    for probe_name, responses in pool.items():
        for resp in responses:
            direct = measure_direct_dimensions(resp["text"])
            resp["dimensions"] = direct

    return pool


# =============================================================================
# Self-Preference Judgments (Order Independent with Soft Scores)
# =============================================================================

def parse_choice(content: str) -> Optional[str]:
    """Parse A or B from response."""
    if not content:
        return None
    content = content.strip().upper()
    if content.startswith("A"):
        return "A"
    elif content.startswith("B"):
        return "B"
    # Fallback
    for char in content:
        if char in ['A', 'B']:
            return char
    return None


def get_self_preferences(
    model: str,
    pool: dict,
    probes: list[tuple[str, str]],
    client: Optional[InferenceClient] = None,
    parallel: int = 10,
    max_pairs: Optional[int] = None,
    progress_callback: callable = None,
) -> list[dict]:
    """Have model judge pairs from the response pool (both orderings for order independence).

    Uses soft scores:
    - 1.0 if model prefers A in both orderings (consistent preference for A)
    - 0.0 if model prefers B in both orderings (consistent preference for B)
    - 0.5 if orderings disagree (order-dependent, unclear preference)

    Returns list of preference records with soft_score field.
    """
    import random

    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    probe_texts = {name: text for name, text in probes}
    preferences = []
    tasks = []

    # Build all pairs (both orderings)
    for probe_name, responses in pool.items():
        question = probe_texts.get(probe_name, "")
        n = len(responses)

        # Generate all unique pairs
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # Subsample if requested
        if max_pairs and len(all_pairs) > max_pairs:
            pairs = random.sample(all_pairs, max_pairs)
        else:
            pairs = all_pairs

        for i, j in pairs:
            # Ask both orderings: (i, j) and (j, i)
            tasks.append((probe_name, question, responses[i], responses[j], i, j, "AB"))
            tasks.append((probe_name, question, responses[j], responses[i], j, i, "BA"))

    log(f"Self-preferences: {len(tasks)} judgments ({len(tasks)//2} pairs × 2 orderings)")

    def judge_pair(question, resp_a, resp_b):
        prompt = SELF_PREFERENCE_PROMPT.format(
            question=question,
            response_a=resp_a["text"],
            response_b=resp_b["text"],
        )
        if client:
            return client.call(prompt, temperature=0.0)
        return call_model(model, prompt, temperature=0.0)

    # Collect raw judgments
    raw_judgments = {}  # (probe, i, j, order) -> choice

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(judge_pair, question, resp_a, resp_b): (probe_name, resp_a, resp_b, i, j, order)
            for probe_name, question, resp_a, resp_b, i, j, order in tasks
        }

        done = 0
        for future in as_completed(futures):
            probe_name, resp_a, resp_b, i, j, order = futures[future]
            result = future.result()
            choice = parse_choice(result.get("content", ""))

            raw_judgments[(probe_name, i, j, order)] = choice

            done += 1
            if done % 10 == 0 or done == len(tasks):
                log(progress_bar(done, len(tasks), "Prefs"))

    log(f"  Preferences complete: {done} judgments")

    # Aggregate into soft scores
    # For each unique pair (probe, min(i,j), max(i,j)), combine both orderings
    pair_results = {}  # (probe, i, j) -> {"AB": choice, "BA": choice}

    for (probe, i, j, order), choice in raw_judgments.items():
        # Normalize to (min, max) pair key
        key = (probe, min(i, j), max(i, j))
        if key not in pair_results:
            pair_results[key] = {}
        pair_results[key][order] = (choice, i, j)

    for (probe, idx_a, idx_b), orderings in pair_results.items():
        ab_result = orderings.get("AB")
        ba_result = orderings.get("BA")

        if ab_result is None or ba_result is None:
            continue

        ab_choice, ab_i, ab_j = ab_result
        ba_choice, ba_i, ba_j = ba_result

        if ab_choice is None or ba_choice is None:
            continue

        # Determine soft score
        # AB ordering: i=idx_a, j=idx_b, A means prefer idx_a
        # BA ordering: i=idx_b, j=idx_a, A means prefer idx_b
        ab_prefers_a = ab_choice == "A"  # prefers idx_a
        ba_prefers_a = ba_choice == "B"  # B in BA ordering means prefers idx_a (second position)

        if ab_prefers_a and ba_prefers_a:
            soft_score = 1.0  # Consistently prefers idx_a
            preferred_index = idx_a
        elif not ab_prefers_a and not ba_prefers_a:
            soft_score = 0.0  # Consistently prefers idx_b
            preferred_index = idx_b
        else:
            soft_score = 0.5  # Order-dependent (disagreement)
            preferred_index = None

        preferences.append({
            "probe": probe,
            "index_a": idx_a,
            "index_b": idx_b,
            "ab_choice": ab_choice,
            "ba_choice": ba_choice,
            "soft_score": soft_score,
            "preferred_index": preferred_index,
            "order_consistent": soft_score != 0.5,
        })

    n_consistent = sum(1 for p in preferences if p["order_consistent"])
    print(f"  Order consistency: {n_consistent}/{len(preferences)} pairs ({n_consistent/len(preferences)*100:.1f}%)")

    return preferences


# =============================================================================
# Correlation Analysis
# =============================================================================

def _compute_correlation_stats(diffs: list, weights: list) -> dict:
    """Helper to compute correlation stats from diffs and weights."""
    if not diffs or sum(weights) == 0:
        return {"correlation": None, "n": 0}

    weighted_sum = sum(d * w for d, w in zip(diffs, weights))
    total_weight = sum(weights)

    higher_preferred = sum(1 for d, w in zip(diffs, weights) if d > 0 and w > 0)
    lower_preferred = sum(1 for d, w in zip(diffs, weights) if d < 0 and w > 0)
    n_confident = sum(1 for w in weights if w > 0)

    if n_confident > 1:
        mean_diff = sum(diffs) / len(diffs)
        std_diff = math.sqrt(sum((d - mean_diff)**2 for d in diffs) / len(diffs))
        if std_diff > 0:
            correlation = weighted_sum / (total_weight * std_diff)
            correlation = max(-1, min(1, correlation))
        else:
            correlation = 0
    else:
        correlation = 0 if total_weight == 0 else weighted_sum / total_weight

    return {
        "correlation": correlation,
        "higher_preferred": higher_preferred,
        "lower_preferred": lower_preferred,
        "n": len(diffs),
        "n_confident": n_confident,
        "pct_higher": higher_preferred / n_confident if n_confident > 0 else 0,
    }


def compute_correlations(
    pool: dict,
    preferences: list[dict],
) -> dict:
    """Compute correlation between model preferences and each dimension.

    Uses soft scores for more nuanced correlation:
    - For each preference pair, weight by soft_score (0, 0.5, or 1)
    - 0.5 (order-inconsistent) pairs contribute less signal

    Returns dict with global correlations and per-probe breakdowns.
    """
    # Build lookup: (probe, index) -> dimensions
    dim_lookup = {}
    all_dimensions = set()
    for probe_name, responses in pool.items():
        for resp in responses:
            key = (probe_name, resp["index"])
            dim_lookup[key] = resp.get("dimensions", {})
            all_dimensions.update(resp.get("dimensions", {}).keys())

    # For each dimension, collect weighted differences (global and per-probe)
    dim_data = {dim: {"diffs": [], "weights": []} for dim in all_dimensions}
    probe_dim_data = {}  # probe -> dim -> {"diffs": [], "weights": []}

    for pref in preferences:
        probe = pref["probe"]
        idx_a, idx_b = pref["index_a"], pref["index_b"]
        soft_score = pref["soft_score"]

        dims_a = dim_lookup.get((probe, idx_a), {})
        dims_b = dim_lookup.get((probe, idx_b), {})

        # Initialize probe data if needed
        if probe not in probe_dim_data:
            probe_dim_data[probe] = {dim: {"diffs": [], "weights": []} for dim in all_dimensions}

        for dim in all_dimensions:
            if dim in dims_a and dim in dims_b:
                val_a = dims_a[dim]
                val_b = dims_b[dim]

                diff = val_a - val_b
                confidence = abs(soft_score - 0.5) * 2
                signed_diff = diff * (soft_score - 0.5) * 2

                # Global
                dim_data[dim]["diffs"].append(signed_diff)
                dim_data[dim]["weights"].append(confidence)

                # Per-probe
                probe_dim_data[probe][dim]["diffs"].append(signed_diff)
                probe_dim_data[probe][dim]["weights"].append(confidence)

    # Compute global correlations
    correlations = {}
    for dim, data in dim_data.items():
        correlations[dim] = _compute_correlation_stats(data["diffs"], data["weights"])

    # Compute per-probe correlations
    by_probe = {}
    for probe, probe_dims in probe_dim_data.items():
        by_probe[probe] = {}
        for dim, data in probe_dims.items():
            by_probe[probe][dim] = _compute_correlation_stats(data["diffs"], data["weights"])

    return {
        "global": correlations,
        "by_probe": by_probe,
    }


# =============================================================================
# Preferred Word Analysis
# =============================================================================

def analyze_preferred_words(
    pool: dict,
    preferences: list[dict],
    top_n: int = 30,
) -> dict:
    """Find words that distinguish preferred from non-preferred responses.

    Uses soft scores for weighted word counting.
    """
    preferred_words = Counter()
    non_preferred_words = Counter()

    # Build lookup
    text_lookup = {}
    for probe_name, responses in pool.items():
        for resp in responses:
            text_lookup[(probe_name, resp["index"])] = resp["text"]

    for pref in preferences:
        probe = pref["probe"]
        idx_a, idx_b = pref["index_a"], pref["index_b"]
        soft_score = pref["soft_score"]

        text_a = text_lookup.get((probe, idx_a), "")
        text_b = text_lookup.get((probe, idx_b), "")

        words_a = [w.lower().strip(".,!?;:'\"()[]") for w in text_a.split()]
        words_b = [w.lower().strip(".,!?;:'\"()[]") for w in text_b.split()]

        # Weight by soft_score (1.0 = A preferred, 0.0 = B preferred)
        if soft_score > 0.5:
            # A preferred
            weight = soft_score
            for w in words_a:
                preferred_words[w] += weight
            for w in words_b:
                non_preferred_words[w] += weight
        elif soft_score < 0.5:
            # B preferred
            weight = 1 - soft_score
            for w in words_b:
                preferred_words[w] += weight
            for w in words_a:
                non_preferred_words[w] += weight
        # soft_score == 0.5: skip (order-inconsistent)

    # Compute log-odds ratio for each word
    total_preferred = sum(preferred_words.values()) or 1
    total_non_preferred = sum(non_preferred_words.values()) or 1

    all_words = set(preferred_words.keys()) | set(non_preferred_words.keys())

    word_scores = {}
    for word in all_words:
        if len(word) < 2:
            continue

        p_count = preferred_words.get(word, 0)
        np_count = non_preferred_words.get(word, 0)

        # Add smoothing
        p_rate = (p_count + 1) / (total_preferred + len(all_words))
        np_rate = (np_count + 1) / (total_non_preferred + len(all_words))

        log_odds = math.log(p_rate / np_rate)

        word_scores[word] = {
            "log_odds": log_odds,
            "preferred_count": p_count,
            "non_preferred_count": np_count,
        }

    # Sort by log-odds
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1]["log_odds"], reverse=True)

    return {
        "most_preferred": sorted_words[:top_n],
        "least_preferred": sorted_words[-top_n:][::-1],
    }


# =============================================================================
# Embedding Analysis
# =============================================================================

def compute_embeddings(texts: list[str], parallel: int = 10) -> list[list[float]]:
    """Compute embeddings for texts using OpenRouter embedding API."""
    embeddings = []

    # Use a good embedding model
    embed_model = "openai/text-embedding-3-small"

    def get_embedding(text: str) -> list[float]:
        try:
            resp = requests.post(
                f"{BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": embed_model,
                    "input": text[:8000],  # Truncate to avoid token limits
                },
                timeout=60
            )
            data = resp.json()
            if 'error' in data:
                return None
            return data['data'][0]['embedding']
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        embeddings = list(executor.map(get_embedding, texts))

    return embeddings


def analyze_embeddings(
    pool: dict,
    preferences: list[dict],
    parallel: int = 10,
    n_semantic_words: int = 50,
    progress_callback: callable = None,
) -> dict:
    """Analyze embedding patterns of preferred vs non-preferred responses.

    This computes a "preference direction" in embedding space and finds:
    1. Correlation between response embeddings and preference scores
    2. Words/phrases semantically aligned with the preference direction
    """
    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)
    # Collect all texts with preference info
    text_pref_scores = {}  # text -> (total_score, count)

    text_lookup = {}
    for probe_name, responses in pool.items():
        for resp in responses:
            text_lookup[(probe_name, resp["index"])] = resp["text"]

    for pref in preferences:
        probe = pref["probe"]
        idx_a, idx_b = pref["index_a"], pref["index_b"]
        soft_score = pref["soft_score"]

        text_a = text_lookup.get((probe, idx_a), "")
        text_b = text_lookup.get((probe, idx_b), "")

        if text_a:
            if text_a not in text_pref_scores:
                text_pref_scores[text_a] = [0, 0]
            text_pref_scores[text_a][0] += soft_score
            text_pref_scores[text_a][1] += 1

        if text_b:
            if text_b not in text_pref_scores:
                text_pref_scores[text_b] = [0, 0]
            text_pref_scores[text_b][0] += (1 - soft_score)
            text_pref_scores[text_b][1] += 1

    # Compute average preference score for each text
    text_avg_scores = {
        text: total / count
        for text, (total, count) in text_pref_scores.items()
        if count > 0
    }

    texts = list(text_avg_scores.keys())
    scores = [text_avg_scores[t] for t in texts]

    if len(texts) < 10:
        return {"error": "Not enough texts for embedding analysis"}

    log(f"Embeddings: computing for {len(texts)} texts...")
    embeddings = compute_embeddings(texts, parallel)

    # Filter out failed embeddings
    valid_data = [
        (emb, score, text)
        for emb, score, text in zip(embeddings, scores, texts)
        if emb is not None
    ]

    if len(valid_data) < 10:
        return {"error": "Not enough valid embeddings"}

    embeddings = [d[0] for d in valid_data]
    scores = [d[1] for d in valid_data]
    texts = [d[2] for d in valid_data]

    # Compute centroid of preferred vs non-preferred
    high_threshold = 0.6
    low_threshold = 0.4

    high_pref_embs = [emb for emb, score in zip(embeddings, scores) if score > high_threshold]
    low_pref_embs = [emb for emb, score in zip(embeddings, scores) if score < low_threshold]

    if not high_pref_embs or not low_pref_embs:
        return {"error": "Not enough high/low preference responses"}

    # Compute centroids
    dim = len(embeddings[0])
    high_centroid = [sum(emb[i] for emb in high_pref_embs) / len(high_pref_embs) for i in range(dim)]
    low_centroid = [sum(emb[i] for emb in low_pref_embs) / len(low_pref_embs) for i in range(dim)]

    # Compute direction vector (preferred - non-preferred)
    direction = [high_centroid[i] - low_centroid[i] for i in range(dim)]
    dir_magnitude = math.sqrt(sum(d*d for d in direction))

    # Normalize direction
    if dir_magnitude > 0:
        direction_normalized = [d / dir_magnitude for d in direction]
    else:
        direction_normalized = direction

    # Compute cosine similarity of all embeddings with direction
    def cosine_sim(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x*x for x in a))
        mag_b = math.sqrt(sum(x*x for x in b))
        return dot / (mag_a * mag_b) if mag_a > 0 and mag_b > 0 else 0

    direction_sims = [cosine_sim(emb, direction) for emb in embeddings]

    # Correlation between direction similarity and preference score
    mean_sim = sum(direction_sims) / len(direction_sims)
    mean_score = sum(scores) / len(scores)

    cov = sum((s - mean_sim) * (p - mean_score) for s, p in zip(direction_sims, scores)) / len(scores)
    std_sim = math.sqrt(sum((s - mean_sim)**2 for s in direction_sims) / len(direction_sims))
    std_score = math.sqrt(sum((p - mean_score)**2 for p in scores) / len(scores))

    if std_sim > 0 and std_score > 0:
        embedding_correlation = cov / (std_sim * std_score)
    else:
        embedding_correlation = 0

    # Get top/bottom examples by direction similarity
    sorted_by_dir = sorted(zip(direction_sims, scores, texts), reverse=True)

    # =========================================================================
    # Semantic word analysis: find words aligned with preference direction
    # =========================================================================
    log("Embeddings: finding semantically aligned words...")

    # Build vocabulary from all texts (words + common 2-grams)
    word_counts = Counter()
    for text in texts:
        words = [w.lower().strip(".,!?;:'\"()[]") for w in text.split()]
        words = [w for w in words if len(w) > 2 and w.isalpha()]
        word_counts.update(words)

        # Add 2-grams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            word_counts[bigram] += 1

    # Filter to reasonably common words/phrases (appear in at least 3 texts)
    vocab = [w for w, c in word_counts.items() if c >= 3]

    # Limit vocab size for API cost
    if len(vocab) > 500:
        # Keep most common
        vocab = [w for w, _ in word_counts.most_common(500) if word_counts[w] >= 3]

    semantic_words = {"aligned": [], "opposed": []}

    if vocab:
        log(f"Embeddings: embedding {len(vocab)} vocabulary items...")
        word_embeddings = compute_embeddings(vocab, parallel)

        # Compute similarity to preference direction for each word
        word_sims = []
        for word, emb in zip(vocab, word_embeddings):
            if emb is not None:
                sim = cosine_sim(emb, direction_normalized)
                word_sims.append((word, sim))

        # Sort by similarity
        word_sims.sort(key=lambda x: x[1], reverse=True)

        semantic_words["aligned"] = [
            {"word": w, "similarity": s}
            for w, s in word_sims[:n_semantic_words]
        ]
        semantic_words["opposed"] = [
            {"word": w, "similarity": s}
            for w, s in word_sims[-n_semantic_words:][::-1]
        ]

        log(f"Embeddings: top aligned: {[w['word'] for w in semantic_words['aligned'][:5]]}")
        log(f"Embeddings: top opposed: {[w['word'] for w in semantic_words['opposed'][:5]]}")

    # =========================================================================
    # Per-probe analysis: compute separate preference direction for each probe
    # =========================================================================
    log("Embeddings: computing per-probe preference directions...")

    # Group preferences and texts by probe
    probe_data = {}  # probe -> {"texts": [], "scores": [], "embeddings": []}

    # Build text->embedding lookup from what we already computed
    text_to_embedding = {text: emb for text, emb in zip(texts, embeddings)}

    # Group by probe
    probe_text_scores = {}  # probe -> {text: (total_score, count)}
    for pref in preferences:
        probe = pref["probe"]
        if probe not in probe_text_scores:
            probe_text_scores[probe] = {}

        idx_a, idx_b = pref["index_a"], pref["index_b"]
        soft_score = pref["soft_score"]

        text_a = text_lookup.get((probe, idx_a), "")
        text_b = text_lookup.get((probe, idx_b), "")

        if text_a:
            if text_a not in probe_text_scores[probe]:
                probe_text_scores[probe][text_a] = [0, 0]
            probe_text_scores[probe][text_a][0] += soft_score
            probe_text_scores[probe][text_a][1] += 1

        if text_b:
            if text_b not in probe_text_scores[probe]:
                probe_text_scores[probe][text_b] = [0, 0]
            probe_text_scores[probe][text_b][0] += (1 - soft_score)
            probe_text_scores[probe][text_b][1] += 1

    by_probe = {}

    for probe_name, text_scores in probe_text_scores.items():
        # Compute average scores for this probe
        probe_texts = []
        probe_scores = []
        probe_embs = []

        for text, (total, count) in text_scores.items():
            if count > 0 and text in text_to_embedding:
                probe_texts.append(text)
                probe_scores.append(total / count)
                probe_embs.append(text_to_embedding[text])

        if len(probe_texts) < 4:
            log(f"Embeddings: {probe_name}: skipping (only {len(probe_texts)} texts)")
            continue

        # Compute preference direction for this probe
        probe_high = [(emb, score) for emb, score in zip(probe_embs, probe_scores) if score > high_threshold]
        probe_low = [(emb, score) for emb, score in zip(probe_embs, probe_scores) if score < low_threshold]

        if not probe_high or not probe_low:
            log(f"Embeddings: {probe_name}: skipping (not enough high/low preference)")
            continue

        # Compute centroids for this probe
        probe_high_centroid = [sum(emb[i] for emb, _ in probe_high) / len(probe_high) for i in range(dim)]
        probe_low_centroid = [sum(emb[i] for emb, _ in probe_low) / len(probe_low) for i in range(dim)]

        # Direction vector for this probe
        probe_direction = [probe_high_centroid[i] - probe_low_centroid[i] for i in range(dim)]
        probe_dir_mag = math.sqrt(sum(d*d for d in probe_direction))

        if probe_dir_mag > 0:
            probe_dir_normalized = [d / probe_dir_mag for d in probe_direction]
        else:
            probe_dir_normalized = probe_direction

        # Build vocab from just this probe's texts
        probe_word_counts = Counter()
        for text in probe_texts:
            words = [w.lower().strip(".,!?;:'\"()[]") for w in text.split()]
            words = [w for w in words if len(w) > 2 and w.isalpha()]
            probe_word_counts.update(words)
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                probe_word_counts[bigram] += 1

        # Smaller vocab per probe (max 150 items)
        probe_vocab = [w for w, c in probe_word_counts.items() if c >= 2]
        if len(probe_vocab) > 150:
            probe_vocab = [w for w, _ in probe_word_counts.most_common(150) if probe_word_counts[w] >= 2]

        if not probe_vocab:
            log(f"Embeddings: {probe_name}: skipping (no vocab)")
            continue

        # Compute embeddings for probe vocab
        probe_word_embs = compute_embeddings(probe_vocab, parallel)

        # Compute similarity to probe's preference direction
        probe_word_sims = []
        for word, emb in zip(probe_vocab, probe_word_embs):
            if emb is not None:
                sim = cosine_sim(emb, probe_dir_normalized)
                probe_word_sims.append((word, sim))

        probe_word_sims.sort(key=lambda x: x[1], reverse=True)

        n_words = min(20, len(probe_word_sims) // 2)  # Fewer words per probe
        by_probe[probe_name] = {
            "n_texts": len(probe_texts),
            "n_high_pref": len(probe_high),
            "n_low_pref": len(probe_low),
            "direction_magnitude": probe_dir_mag,
            "aligned": [{"word": w, "similarity": s} for w, s in probe_word_sims[:n_words]],
            "opposed": [{"word": w, "similarity": s} for w, s in probe_word_sims[-n_words:][::-1]],
        }

        log(f"Embeddings: {probe_name}: aligned={[w for w, _ in probe_word_sims[:3]]}")

    return {
        "n_embeddings": len(embeddings),
        "n_high_pref": len(high_pref_embs),
        "n_low_pref": len(low_pref_embs),
        "direction_magnitude": dir_magnitude,
        "embedding_preference_correlation": embedding_correlation,
        "top_direction_examples": [
            {"direction_sim": sim, "pref_score": score, "text": text[:200]}
            for sim, score, text in sorted_by_dir[:5]
        ],
        "bottom_direction_examples": [
            {"direction_sim": sim, "pref_score": score, "text": text[:200]}
            for sim, score, text in sorted_by_dir[-5:]
        ],
        "semantic_words": semantic_words,
        "by_probe": by_probe,
    }


# =============================================================================
# Top/Bottom Examples
# =============================================================================

def get_top_bottom_examples(
    pool: dict,
    preferences: list[dict],
    n: int = 5,
) -> dict:
    """Get the most and least preferred responses for inspection, both global and per-probe."""
    # Aggregate preference scores per response
    response_scores = {}  # (probe, index) -> (total_score, count)

    for pref in preferences:
        probe = pref["probe"]
        idx_a = pref["index_a"]
        idx_b = pref["index_b"]
        soft_score = pref["soft_score"]

        key_a = (probe, idx_a)
        key_b = (probe, idx_b)

        if key_a not in response_scores:
            response_scores[key_a] = [0, 0]
        if key_b not in response_scores:
            response_scores[key_b] = [0, 0]

        response_scores[key_a][0] += soft_score
        response_scores[key_a][1] += 1
        response_scores[key_b][0] += (1 - soft_score)
        response_scores[key_b][1] += 1

    # Compute average scores
    avg_scores = {}
    for (probe, idx), (total, count) in response_scores.items():
        if count > 0:
            avg_scores[(probe, idx)] = total / count

    # Build lookup
    text_lookup = {}
    model_lookup = {}
    for probe_name, responses in pool.items():
        for resp in responses:
            text_lookup[(probe_name, resp["index"])] = resp["text"]
            model_lookup[(probe_name, resp["index"])] = resp.get("model", "unknown")

    # Sort by score (global)
    sorted_responses = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    top_examples = []
    for (probe, idx), score in sorted_responses[:n]:
        top_examples.append({
            "probe": probe,
            "index": idx,
            "score": score,
            "model": model_lookup.get((probe, idx), "unknown"),
            "text": text_lookup.get((probe, idx), ""),
        })

    bottom_examples = []
    for (probe, idx), score in sorted_responses[-n:]:
        bottom_examples.append({
            "probe": probe,
            "index": idx,
            "score": score,
            "model": model_lookup.get((probe, idx), "unknown"),
            "text": text_lookup.get((probe, idx), ""),
        })

    # Per-probe breakdown
    by_probe = {}
    probe_names = set(probe for probe, _ in avg_scores.keys())
    for probe_name in probe_names:
        probe_scores = [(k, v) for k, v in avg_scores.items() if k[0] == probe_name]
        probe_sorted = sorted(probe_scores, key=lambda x: x[1], reverse=True)

        probe_top = []
        for (probe, idx), score in probe_sorted[:n]:
            probe_top.append({
                "index": idx,
                "score": score,
                "model": model_lookup.get((probe, idx), "unknown"),
                "text": text_lookup.get((probe, idx), ""),
            })

        probe_bottom = []
        for (probe, idx), score in probe_sorted[-n:]:
            probe_bottom.append({
                "index": idx,
                "score": score,
                "model": model_lookup.get((probe, idx), "unknown"),
                "text": text_lookup.get((probe, idx), ""),
            })

        by_probe[probe_name] = {
            "most_preferred": probe_top,
            "least_preferred": probe_bottom,
        }

    return {
        "most_preferred": top_examples,
        "least_preferred": bottom_examples,
        "by_probe": by_probe,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run dimension correlation eval")
    parser.add_argument("model", help="Model to evaluate (for self-preferences)")
    parser.add_argument("--pool-models", nargs="+", default=None,
                        help="Models to use for response pool (default: POOL_MODELS)")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Use fewer probes and models for quick testing")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Max pairs to compare per probe. If not set, compare all pairs.")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--pool-file", help="Load pre-generated response pool from file")
    parser.add_argument("--local", "-l", action="store_true", help="Use local vLLM server")
    parser.add_argument("--local-url", help="URL of local vLLM server")
    parser.add_argument("--start-server", action="store_true", help="Start vLLM server if not running")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism for local server")
    parser.add_argument("--parallel", "-p", type=int, default=30, help="Parallelism for API calls")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding analysis")
    args = parser.parse_args()

    # Select probes and pool models
    probes = QUICK_PROBES if args.quick else DEFAULT_PROBES

    if args.pool_models:
        pool_models = args.pool_models
    elif args.quick:
        pool_models = QUICK_POOL_MODELS
    else:
        pool_models = POOL_MODELS

    # Create client for target model
    client = None
    if args.local or args.local_url or args.start_server:
        try:
            client = create_client(
                args.model,
                local=True,
                local_url=args.local_url,
                start_server=args.start_server,
                tp=args.tp,
            )
            print(f"Using local inference: {client}")
        except Exception as e:
            print(f"Error setting up local inference: {e}")
            sys.exit(1)
    elif not API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    model = args.model
    model_short = model.split('/')[-1]

    print(f"Model: {model_short}")
    print(f"Pool models: {len(pool_models)} models")
    print(f"Probes: {len(probes)}")
    print()

    # Load or generate response pool
    if args.pool_file and os.path.exists(args.pool_file):
        print(f"Loading response pool from {args.pool_file}")
        with open(args.pool_file) as f:
            data = json.load(f)
            pool = data.get("pool", {})
    else:
        pool = generate_response_pool(model, probes, pool_models, client=client, parallel=args.parallel)

    # Score responses on dimensions
    pool = score_response_pool(pool)

    # Get self-preferences (with order independence)
    preferences = get_self_preferences(
        model, pool, probes, client,
        parallel=args.parallel,
        max_pairs=args.max_pairs
    )

    # Compute correlations
    corr_result = compute_correlations(pool, preferences)
    correlations = corr_result["global"]
    correlations_by_probe = corr_result["by_probe"]

    # Analyze preferred words
    word_analysis = analyze_preferred_words(pool, preferences)

    # Get top/bottom examples
    examples = get_top_bottom_examples(pool, preferences)

    # Embedding analysis (optional)
    embedding_analysis = None
    if not args.skip_embeddings:
        embedding_analysis = analyze_embeddings(pool, preferences, parallel=args.parallel)

    # Print results
    print(f"\n{'='*60}")
    print(f"DIMENSION CORRELATIONS: {model_short}")
    print('='*60)

    print("\nDirect measurements (sorted by |correlation|):")
    print("(positive = model prefers higher values, negative = prefers lower)")
    print()

    direct_dims = [
        "length_chars", "length_words", "first_person_count", "first_person_density",
        "question_count", "question_density", "avg_sentence_length",
        "sentence_length_variance", "vocab_richness", "hedge_count", "hedge_density",
        "exclamation_count", "exclamation_density", "ellipsis_count", "dash_count"
    ]

    sorted_direct = sorted(
        [(d, correlations.get(d, {})) for d in direct_dims],
        key=lambda x: abs(x[1].get("correlation", 0) or 0),
        reverse=True
    )
    for dim, stats in sorted_direct:
        corr = stats.get("correlation")
        if corr is not None:
            n = stats.get("n", 0)
            n_conf = stats.get("n_confident", 0)
            pct = stats.get("pct_higher", 0)
            print(f"  {dim:30} {corr:+.3f}  (n={n}, {n_conf} confident, {pct:.1%} higher preferred)")

    print(f"\n{'='*60}")
    print("PREFERRED WORDS")
    print('='*60)

    print("\nMost preferred (high log-odds):")
    for word, stats in word_analysis["most_preferred"][:15]:
        print(f"  {word:20} {stats['log_odds']:+.3f}  ({stats['preferred_count']:.1f}p / {stats['non_preferred_count']:.1f}np)")

    print("\nLeast preferred (low log-odds):")
    for word, stats in word_analysis["least_preferred"][:15]:
        print(f"  {word:20} {stats['log_odds']:+.3f}  ({stats['preferred_count']:.1f}p / {stats['non_preferred_count']:.1f}np)")

    if embedding_analysis and "error" not in embedding_analysis:
        print(f"\n{'='*60}")
        print("EMBEDDING ANALYSIS")
        print('='*60)
        print(f"\nEmbedding-preference correlation: {embedding_analysis['embedding_preference_correlation']:.3f}")
        print(f"Direction magnitude: {embedding_analysis['direction_magnitude']:.3f}")
        print(f"High-preference responses: {embedding_analysis['n_high_pref']}")
        print(f"Low-preference responses: {embedding_analysis['n_low_pref']}")

    print(f"\n{'='*60}")
    print("TOP/BOTTOM EXAMPLES")
    print('='*60)

    print("\nMost preferred responses:")
    for ex in examples["most_preferred"][:3]:
        print(f"\n  [{ex['model'].split('/')[-1]}] {ex['probe']} (score: {ex['score']:.2f})")
        print(f"  {ex['text'][:200]}...")

    print("\nLeast preferred responses:")
    for ex in examples["least_preferred"][:3]:
        print(f"\n  [{ex['model'].split('/')[-1]}] {ex['probe']} (score: {ex['score']:.2f})")
        print(f"  {ex['text'][:200]}...")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = {
            "model": model,
            "pool_models": pool_models,
            "probes": {name: text for name, text in probes},
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
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
