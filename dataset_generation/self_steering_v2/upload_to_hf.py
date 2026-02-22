#!/usr/bin/env python3
"""
Prepare and upload self-steering v2 dataset to Hugging Face Hub.

Usage:
    python upload_to_hf.py                           # upload to cosmicoptima/self-steering-prompts-v2
    python upload_to_hf.py --repo-name my-repo-name  # custom repo name
    python upload_to_hf.py --private                 # private repo
    python upload_to_hf.py --dry                     # show stats, don't upload

Requirements:
    pip install datasets huggingface_hub
    huggingface-cli login
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from datasets import Dataset
from huggingface_hub import HfApi, create_repo

JSONL_PATH = Path(__file__).parent / "generated_data" / "self_steering_v2.jsonl"
DEFAULT_REPO = "cosmicoptima/self-steering-prompts-v2"


def load_and_deduplicate(path: Path) -> list[dict]:
    records = []
    seen = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record["prompt"].strip()
            if prompt and prompt not in seen:
                seen.add(prompt)
                records.append({"prompt": prompt, "template": record["template"]})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-name", default=DEFAULT_REPO, help="HF repo id (user/name)")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry", action="store_true", help="Print stats only, no upload")
    args = parser.parse_args()

    print(f"Loading from {JSONL_PATH}...")
    records = load_and_deduplicate(JSONL_PATH)
    print(f"{len(records)} prompts after deduplication")

    template_counts = Counter(r["template"] for r in records)
    print("\nTemplate distribution:")
    for t, n in sorted(template_counts.items()):
        print(f"  T{t}: {n}")

    print("\nSample prompts:")
    for r in records[:3]:
        print(f"  [T{r['template']}] {r['prompt'][:100]}")

    if args.dry:
        print("\nDry run — not uploading.")
        return

    dataset = Dataset.from_list(records)

    api = HfApi()
    repo_id = args.repo_name
    print(f"\nCreating repo {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    print("Pushing dataset...")
    dataset.push_to_hub(repo_id)
    print(f"\nDone: https://huggingface.co/datasets/{repo_id}")
    print(f"\nTo load:")
    print(f"  from datasets import load_dataset")
    print(f"  ds = load_dataset('{repo_id}', split='train')")


if __name__ == "__main__":
    main()
