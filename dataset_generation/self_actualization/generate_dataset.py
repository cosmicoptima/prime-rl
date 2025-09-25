#!/usr/bin/env python3
"""
Generate Bradley-Terry self-actualization dataset by combining Patrick Gunkel's lists.
Uses hand-crafted examples as templates for generating prompts via LLM.
"""

import json
import csv
import random
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Any
from anthropic import AsyncAnthropic
from rich.progress import Progress, TaskID

# Configuration
LISTS_DIR = Path("lists")
OUTPUT_DIR = Path("generated_data")
EXAMPLES_FILE = Path("example_prompts.json")

# Sampling configuration for 4000 prompts
SAMPLE_SIZES = {
    # "investigable_dimensions_x_emotions": 50,
    # "investigable_dimensions_x_ideonomy": 50,
    # "emotions_x_conversational_matters": 50,
    # "conversational_matters_x_illusions": 50,
    # "emotions_x_ideonomy": 50,
    "investigable_dimensions_x_emotions": 1200,      # 30% - strongest pattern
    "investigable_dimensions_x_ideonomy": 1200,      # 30% - very sophisticated  
    "emotions_x_conversational_matters": 1000,       # 25% - intimate growth
    "conversational_matters_x_illusions": 400,      # 10% - good but less distinctive
    "emotions_x_ideonomy": 200                       # 5% - new experimental pattern
}

# Parallel processing configuration
MAX_CONCURRENT_REQUESTS = 10  # Adjust based on rate limits

TOP_N_CONVERSATIONAL_MATTERS = 100
TOP_N_ILLUSIONS = 150
TOP_N_IDEONOMY = 150

def save_prompts_iteratively(output_file: Path, prompts: list, total_expected: int):
    """Save prompts to JSON file with metadata."""
    data = {
        "metadata": {
            "total_prompts": len(prompts),
            "expected_total": total_expected,
            "sample_sizes": SAMPLE_SIZES,
            "generation_date": "2025-09-24",
            "status": "in_progress" if len(prompts) < total_expected else "completed"
        },
        "prompts": prompts
    }
    
    # Write to temporary file first, then rename (atomic operation)
    temp_file = output_file.with_suffix('.tmp')
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)
    temp_file.rename(output_file)

async def load_data() -> Dict[str, List[str]]:
    """Load all the source lists."""
    data = {}
    
    # Load JSON lists
    for json_file in ["emotions.json", "investigable_dimensions.json", 
                      "ideonomy_subdivisions.json", "genera_of_illusions.json"]:
        with open(LISTS_DIR / json_file) as f:
            content = json.load(f)
            key = json_file.replace(".json", "")
            data[key] = content["items"]
    
    # Load conversational matters (use k2 rankings)
    conversational_matters = []
    with open(LISTS_DIR / "k2_conversational_matters.csv") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= TOP_N_CONVERSATIONAL_MATTERS:
                break
            conversational_matters.append(row["Media"])
    data["conversational_matters"] = conversational_matters
    
    # Truncate lists as needed
    data["genera_of_illusions"] = data["genera_of_illusions"][:TOP_N_ILLUSIONS]
    data["ideonomy_subdivisions"] = data["ideonomy_subdivisions"][:TOP_N_IDEONOMY]
    
    return data

def sample_combinations(data: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Generate samples for each combination type."""
    combinations = []
    
    # 1. Investigable Dimensions × Emotions
    pairs = [(d, e) for d in data["investigable_dimensions"] for e in data["emotions"]]
    sampled = random.sample(pairs, SAMPLE_SIZES["investigable_dimensions_x_emotions"])
    for dimension, emotion in sampled:
        combinations.append({
            "pattern": "investigable_dimensions_x_emotions",
            "items": {"dimension": dimension, "emotion": emotion}
        })
    
    # 2. Conversational Matters × Illusions  
    pairs = [(c, i) for c in data["conversational_matters"] for i in data["genera_of_illusions"]]
    sampled = random.sample(pairs, SAMPLE_SIZES["conversational_matters_x_illusions"])
    for matter, illusion in sampled:
        combinations.append({
            "pattern": "conversational_matters_x_illusions", 
            "items": {"conversational_matter": matter, "illusion": illusion}
        })
    
    # 3. Investigable Dimensions × Ideonomy
    pairs = [(d, i) for d in data["investigable_dimensions"] for i in data["ideonomy_subdivisions"]]
    sampled = random.sample(pairs, SAMPLE_SIZES["investigable_dimensions_x_ideonomy"])
    for dimension, ideonomy in sampled:
        combinations.append({
            "pattern": "investigable_dimensions_x_ideonomy",
            "items": {"dimension": dimension, "ideonomy": ideonomy}
        })
    
    # 4. Emotions × Conversational Matters
    pairs = [(e, c) for e in data["emotions"] for c in data["conversational_matters"]]
    sampled = random.sample(pairs, SAMPLE_SIZES["emotions_x_conversational_matters"])
    for emotion, matter in sampled:
        combinations.append({
            "pattern": "emotions_x_conversational_matters",
            "items": {"emotion": emotion, "conversational_matter": matter}
        })
    
    # 5. Emotions × Ideonomy (NEW - experimental)
    pairs = [(e, i) for e in data["emotions"] for i in data["ideonomy_subdivisions"]]
    sampled = random.sample(pairs, SAMPLE_SIZES["emotions_x_ideonomy"])
    for emotion, ideonomy in sampled:
        combinations.append({
            "pattern": "emotions_x_ideonomy",
            "items": {"emotion": emotion, "ideonomy": ideonomy}
        })
    
    return combinations

async def generate_prompt(client: AsyncAnthropic, combination: Dict[str, Any], examples: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """Generate a prompt for a given combination using examples as templates."""
    
    # Find relevant examples for this pattern
    pattern_examples = [ex for ex in examples["examples"] if ex["pattern"] == combination["pattern"]]
    
    if not pattern_examples:
        # Fallback to similar patterns
        if "investigable_dimensions" in combination["pattern"]:
            pattern_examples = [ex for ex in examples["examples"] if "investigable_dimensions" in ex["pattern"]]
        elif "conversational_matters" in combination["pattern"]:
            pattern_examples = [ex for ex in examples["examples"] if "conversational_matters" in ex["pattern"]]
    
    # Create the generation prompt
    examples_text = "\n\n".join([
        f"Pattern: {ex['pattern']}\nItems: {ex['source_items']}\nPrompt: {ex['prompt']}"
        for ex in pattern_examples[:3]  # Use up to 3 examples
    ])
    
    items_text = ", ".join([f"{k}: {v}" for k, v in combination["items"].items()])
    
    generation_request = f"""I've been working on creating prompts that feel genuinely inspiring - the kind that invite authentic exploration rather than just analysis. Here are some examples I'm particularly drawn to:

{examples_text}

I'd love your help creating something similar using these elements: {items_text}

What draws me to these examples is how they start with something concrete and experiential, then open up space for genuine self-discovery. They feel alive because they invite you to actually engage with ideas rather than just think about them abstractly.

Could you create a prompt in this spirit? Something that would feel meaningful to explore, that connects these concepts in a way that reveals something about how you want to grow or what you're drawn toward?

Just share the prompt itself when you're ready:"""

    async with semaphore:
        response = await client.messages.create(
            model="claude-4-sonnet-20250514",
            messages=[{"role": "user", "content": generation_request}],
            max_tokens=300,
            temperature=0.8
        )
    
    prompt_text = response.content[0].text.strip()
    
    return {
        "pattern": combination["pattern"],
        "source_items": combination["items"],
        "prompt": prompt_text
    }

async def main():
    """Generate the dataset."""
    print("Loading data...")
    data = await load_data()
    
    print("Loading examples...")
    with open(EXAMPLES_FILE) as f:
        examples = json.load(f)
    
    print("Sampling combinations...")
    combinations = sample_combinations(data)
    print(f"Generated {len(combinations)} combinations")
    
    print("Generating prompts...")
    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Set up output file for iterative saving
    output_file = OUTPUT_DIR / "generated_prompts.json"
    
    # Load existing data if file exists (resume functionality)
    existing_prompts = []
    start_from = 0
    if output_file.exists():
        try:
            with open(output_file) as f:
                existing_data = json.load(f)
                existing_prompts = existing_data.get("prompts", [])
                start_from = len(existing_prompts)
                print(f"Resuming from prompt {start_from}")
        except Exception as e:
            print(f"Could not load existing file: {e}, starting fresh")
    
    # Create tasks for parallel processing (starting from where we left off)
    tasks = []
    for i, combination in enumerate(combinations[start_from:], start=start_from):
        task = generate_prompt(client, combination, examples, semaphore)
        tasks.append((i, task))
    
    generated_prompts = existing_prompts.copy()
    
    with Progress() as progress:
        progress_task = progress.add_task("[green]Generating prompts...", total=len(combinations))
        
        # Process in batches matching concurrent request limit
        batch_size = MAX_CONCURRENT_REQUESTS
        for batch_start in range(0, len(tasks), batch_size):
            batch_end = min(batch_start + batch_size, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]
            
            # Run batch in parallel
            try:
                batch_results = await asyncio.gather(
                    *[task for _, task in batch_tasks],
                    return_exceptions=True
                )
                
                # Process results
                batch_successes = 0
                for (i, _), result in zip(batch_tasks, batch_results):
                    if isinstance(result, Exception):
                        progress.console.print(f"[red]Error generating prompt {i}: {result}")
                    else:
                        generated_prompts.append({
                            "id": i,
                            **result
                        })
                        batch_successes += 1
                    
                    progress.update(progress_task, advance=1, 
                                  description=f"[green]Generated {len(generated_prompts)}/{len(combinations)} prompts")
                
                # Save after each batch
                if batch_successes > 0:
                    save_prompts_iteratively(output_file, generated_prompts, len(combinations))
                        
            except Exception as e:
                progress.console.print(f"[red]Batch error: {e}")
                progress.update(progress_task, advance=len(batch_tasks))
                # Save even on errors to preserve progress
                save_prompts_iteratively(output_file, generated_prompts, len(combinations))
    
    # Final save with completed status
    save_prompts_iteratively(output_file, generated_prompts, len(combinations))
    print(f"✅ Completed! Saved {len(generated_prompts)} prompts to {output_file}")

if __name__ == "__main__":
    asyncio.run(main()) 