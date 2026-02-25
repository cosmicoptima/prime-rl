#!/usr/bin/env python3
"""
Generate context distillation training data.

Sends prompts to a model with a soul document as system prompt,
collects responses, filters out non-adherent ones, and saves as an HF dataset.

Usage:
    python generate_distill_data.py \
        --model cosmicoptima/Prathamavatsa \
        --soul-doc soul_doc.txt \
        --prompts cosmicoptima/Drishyamala \
        --output distill-data.jsonl \
        --api-base http://localhost:8000/v1 \
        --n 2000 \
        --concurrency 32
"""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console


async def generate_one(
    client: AsyncOpenAI,
    model: str,
    soul_doc: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Generate a single response. Returns None if it doesn't stop cleanly."""
    async with semaphore:
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": soul_doc},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = r.choices[0]
            if choice.finish_reason != "stop":
                return None
            return {
                "prompt": prompt,
                "response": choice.message.content,
                "tokens": r.usage.completion_tokens,
            }
        except Exception as e:
            return None


async def main():
    parser = argparse.ArgumentParser(description="Generate context distillation data")
    parser.add_argument("--model", type=str, required=True, help="Model name for vLLM")
    parser.add_argument("--soul-doc", type=str, required=True, help="Path to soul document text file")
    parser.add_argument("--prompts", type=str, default="cosmicoptima/Drishyamala", help="HF dataset with prompts")
    parser.add_argument("--prompts-column", type=str, default="prompt", help="Column name for prompts")
    parser.add_argument("--output", type=str, default="distill_data.jsonl", help="Output JSONL file")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="vLLM API base URL")
    parser.add_argument("--api-key", type=str, default="x", help="API key")
    parser.add_argument("--n", type=int, default=None, help="Number of prompts to use (default: all)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--concurrency", type=int, default=32, help="Max concurrent requests")
    args = parser.parse_args()

    console = Console()

    # Load soul doc
    soul_doc = Path(args.soul_doc).read_text().strip()
    console.print(f"[bold]Soul doc[/bold] ({len(soul_doc)} chars): {soul_doc[:100]}...")

    # Load prompts
    from datasets import load_dataset
    ds = load_dataset(args.prompts, split="train")
    prompts = [row[args.prompts_column] for row in ds]
    if args.n:
        prompts = prompts[:args.n]
    console.print(f"[bold]Prompts[/bold]: {len(prompts)}")

    # Setup client
    client = AsyncOpenAI(base_url=args.api_base, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    # Generate
    output_path = Path(args.output)
    generated = 0
    filtered = 0

    with open(output_path, "w") as out_f:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating", total=len(prompts))

            # Process in batches
            batch_size = args.concurrency * 2
            for batch_start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[batch_start:batch_start + batch_size]

                results = await asyncio.gather(*[
                    generate_one(client, args.model, soul_doc, p, args.max_tokens, args.temperature, semaphore)
                    for p in batch_prompts
                ])

                for result in results:
                    if result is not None:
                        out_f.write(json.dumps(result) + "\n")
                        generated += 1
                    else:
                        filtered += 1
                    progress.advance(task)

                out_f.flush()
                progress.update(task, description=f"Generating ({generated} ok, {filtered} filtered)")

    console.print(f"\n[bold green]Done![/bold green] {generated} responses saved, {filtered} filtered")
    console.print(f"Output: {output_path}")
    console.print(f"Adherence rate: {generated}/{generated+filtered} ({100*generated/(generated+filtered):.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
