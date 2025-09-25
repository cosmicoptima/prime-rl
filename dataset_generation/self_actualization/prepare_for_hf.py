#!/usr/bin/env python3
"""
Convert self-actualization prompts to Hugging Face datasets format.
"""

import json
from pathlib import Path
from datasets import Dataset
import pandas as pd

def convert_to_hf_format():
    """Convert the cleaned CSV dataset to Hugging Face datasets format."""
    
    # Load the cleaned CSV (which has your manual fixes)
    df = pd.read_csv("self_actualization_prompts.csv")
    
    # Load original metadata
    with open("generated_prompts.json") as f:
        data = json.load(f)
    metadata = data["metadata"]
    
    # Create Hugging Face dataset
    dataset = Dataset.from_pandas(df)
    
    # Add metadata
    dataset.info.description = metadata
    
    # Save just the Parquet format for HF
    dataset.save_to_disk("hf_dataset")
    
    print(f"✅ Converted {len(df)} prompts to Hugging Face format")
    print(f"📁 Saved as: hf_dataset/ (Parquet format)")
    
    # Print sample
    print(f"\n📋 Sample entry:")
    print(df.iloc[0].to_dict())
    
    # Print pattern distribution
    pattern_counts = df['pattern'].value_counts()
    print(f"\n📊 Pattern distribution:")
    for pattern, count in pattern_counts.items():
        print(f"   {pattern}: {count}")

if __name__ == "__main__":
    convert_to_hf_format() 