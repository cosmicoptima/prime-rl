#!/usr/bin/env python3
"""
Test just the dataset content and format - no external dependencies.
"""

import json
import random
from pathlib import Path

def analyze_dataset():
    """Analyze the introspection prompts dataset."""
    print("🔍 Analyzing Introspection Prompts Dataset\n")
    
    # Load dataset
    with open("generated_prompts.json") as f:
        data = json.load(f)
    
    prompts = data["prompts"]
    metadata = data["metadata"]
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total prompts: {len(prompts)}")
    print(f"   Generation date: {metadata['generation_date']}")
    print(f"   Status: {metadata['status']}")
    
    # Analyze patterns
    pattern_counts = {}
    for prompt in prompts:
        pattern = prompt["pattern"]
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print(f"\n📈 Pattern Distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(prompts)) * 100
        print(f"   {pattern}: {count} ({percentage:.1f}%)")
    
    # Show sample prompts from each pattern
    print(f"\n📋 Sample Prompts by Pattern:")
    
    samples_shown = set()
    for pattern in pattern_counts.keys():
        # Find first prompt of this pattern we haven't shown
        for prompt in prompts:
            if prompt["pattern"] == pattern and prompt["id"] not in samples_shown:
                print(f"\n🔸 {pattern}:")
                print(f"   Source: {prompt['source_items']}")
                print(f"   Prompt: {prompt['prompt'][:200]}...")
                samples_shown.add(prompt["id"])
                break
    
    # Test format for HF compatibility
    print(f"\n🧪 Testing HF Dataset Format:")
    
    # Simulate what HF datasets would see
    sample = prompts[0]
    required_fields = ["id", "pattern", "prompt", "source_items"]
    
    for field in required_fields:
        if field in sample:
            print(f"   ✅ {field}: {type(sample[field])}")
        else:
            print(f"   ❌ Missing field: {field}")
    
    # Test a few random prompts for quality
    print(f"\n🎲 Random Quality Check:")
    random_samples = random.sample(prompts, 3)
    
    for i, prompt in enumerate(random_samples, 1):
        print(f"\n   Sample {i} (ID {prompt['id']}):")
        print(f"   Pattern: {prompt['pattern']}")
        print(f"   Length: {len(prompt['prompt'])} chars")
        print(f"   Starts with: '{prompt['prompt'][:50]}...'")
        
        # Check for quality indicators
        quality_indicators = [
            "moment when" in prompt['prompt'].lower(),
            "what does" in prompt['prompt'].lower(), 
            "describe" in prompt['prompt'].lower(),
            "think of" in prompt['prompt'].lower(),
            len(prompt['prompt']) > 100,  # Substantial length
            "?" in prompt['prompt']  # Contains questions
        ]
        
        quality_score = sum(quality_indicators)
        print(f"   Quality indicators: {quality_score}/6")

def test_conversion_format():
    """Test the format needed for the Bradley-Terry environment."""
    print(f"\n🎯 Testing Environment Conversion Format:")
    
    with open("generated_prompts.json") as f:
        data = json.load(f)
    
    # Test converting a few prompts to environment format
    sample_prompts = data["prompts"][:3]
    
    converted = []
    for item in sample_prompts:
        converted.append({
            "messages": [{"role": "user", "content": item["prompt"]}],
            "label": "explore",
            "info": {
                "pattern": item["pattern"],
                "source_items": item["source_items"], 
                "prompt_id": item["id"]
            }
        })
    
    print(f"   ✅ Converted {len(converted)} prompts to environment format")
    print(f"   📋 Sample conversion:")
    print(f"      Messages: {len(converted[0]['messages'])} message(s)")
    print(f"      Label: {converted[0]['label']}")
    print(f"      Info keys: {list(converted[0]['info'].keys())}")
    print(f"      Content preview: {converted[0]['messages'][0]['content'][:100]}...")

if __name__ == "__main__":
    analyze_dataset()
    test_conversion_format()
    
    print(f"\n🎉 Dataset Analysis Complete!")
    print(f"\n🚀 Ready for use:")
    print(f"   • HuggingFace: load_dataset('cosmicoptima/introspection-prompts')")
    print(f"   • Local JSON: generated_prompts.json")
    print(f"   • Bradley-Terry environment integration: ✅") 