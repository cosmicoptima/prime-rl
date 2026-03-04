#!/usr/bin/env python3
"""
Pre-compute frozen teacher logprobs for context distillation.

Runs the model with the soul doc as system prompt over all examples,
saves randomly sampled logprobs at each response token position.
These frozen logprobs are used as the distillation target during training.

Usage:
    torchrun --nproc_per_node=4 scripts/precompute_teacher_logprobs.py \
        --model cosmicoptima/Prathamavatsa \
        --data cosmicoptima/3SgWfLwfa6 \
        --soul-doc soul_doc.txt \
        --output /workspace/teacher_logprobs.pt \
        --n-samples 1000 \
        --seq-len 2048
"""

import argparse
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--soul-doc", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of random vocab positions to sample per token")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    vocab_size = model.config.vocab_size
    print(f"Vocab size: {vocab_size}")

    soul_doc = Path(args.soul_doc).read_text().strip()
    print(f"Soul doc: {len(soul_doc)} chars")

    # Load dataset
    if args.data.endswith(".jsonl") or args.data.endswith(".json"):
        ds = load_dataset("json", data_files=args.data, split="train")
    else:
        ds = load_dataset(args.data, split="train")
    print(f"Dataset: {len(ds)} examples")

    all_results = []
    skipped = 0

    for idx in range(len(ds)):
        example = ds[idx]
        prompt_text = example["prompt"]
        response_text = example["response"]

        # Tokenize teacher (with soul doc)
        teacher_messages = [
            {"role": "system", "content": soul_doc},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]
        teacher_ids = tokenizer.apply_chat_template(teacher_messages, return_tensors="pt")

        if teacher_ids.shape[1] > args.seq_len:
            skipped += 1
            continue

        # Build loss mask for teacher (only assistant tokens)
        # Find where assistant tokens start by comparing with/without response
        no_response_messages = [
            {"role": "system", "content": soul_doc},
            {"role": "user", "content": prompt_text},
        ]
        no_response_ids = tokenizer.apply_chat_template(
            no_response_messages, add_generation_prompt=True
        )
        assistant_start = len(no_response_ids)
        n_total = teacher_ids.shape[1]

        # Shift for next-token prediction: logits at position i predict token i+1
        # Response tokens are at positions assistant_start to n_total-1 in the original
        # After shift: loss positions are assistant_start-1 to n_total-2 in logits
        loss_start = assistant_start - 1
        loss_end = n_total - 1
        n_response = loss_end - loss_start

        if n_response <= 0:
            skipped += 1
            continue

        # Forward pass
        with torch.no_grad():
            logits = model(teacher_ids.to(model.device)).logits  # [1, seq, vocab]

        # Extract response logits
        response_logits = logits[0, loss_start:loss_end, :]  # [n_response, vocab]
        response_log_probs = F.log_softmax(response_logits.float(), dim=-1)  # [n_response, vocab]

        # Random sample N vocab indices per position
        sampled_indices = torch.stack([
            torch.randperm(vocab_size)[:args.n_samples]
            for _ in range(n_response)
        ])  # [n_response, n_samples]

        # Gather logprobs at sampled positions
        sampled_logprobs = torch.gather(
            response_log_probs.cpu(),
            dim=-1,
            index=sampled_indices,
        )  # [n_response, n_samples]

        all_results.append({
            "idx": idx,
            "n_response": n_response,
            "sampled_indices": sampled_indices.to(torch.int32),  # [n_response, n_samples]
            "sampled_logprobs": sampled_logprobs.to(torch.float16),  # [n_response, n_samples]
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(ds)} ({skipped} skipped)")

        # Free memory
        del logits, response_logits, response_log_probs

    print(f"\nDone: {len(all_results)} examples processed, {skipped} skipped")

    # Save
    output_path = Path(args.output)
    torch.save(all_results, output_path)
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
