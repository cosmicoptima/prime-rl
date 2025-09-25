#!/usr/bin/env python3
"""
Quick test of the dataset without external dependencies.
"""

import json
import sys
from pathlib import Path

def test_local_files():
    """Test the local dataset files."""
    print("🧪 Testing local dataset files...\n")
    
    # Test 1: Check if CSV exists and is readable
    csv_file = Path("self_actualization_prompts.csv")
    if csv_file.exists():
        with open(csv_file) as f:
            lines = f.readlines()
        print(f"✅ CSV file: {len(lines)-1} prompts") # -1 for header
    else:
        print("❌ CSV file not found")
        return False
    
    # Test 2: Check original JSON  
    json_file = Path("generated_prompts.json")
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
        print(f"✅ JSON file: {data['metadata']['total_prompts']} prompts")
        print(f"📅 Generated: {data['metadata']['generation_date']}")
        print(f"📊 Status: {data['metadata']['status']}")
    else:
        print("❌ JSON file not found")
        return False
    
    # Test 3: Show sample prompt
    sample_prompt = data['prompts'][0]
    print(f"\n📋 Sample prompt:")
    print(f"   ID: {sample_prompt['id']}")
    print(f"   Pattern: {sample_prompt['pattern']}")
    print(f"   Source: {sample_prompt['source_items']}")
    print(f"   Text: {sample_prompt['prompt'][:200]}...")
    
    # Test 4: Bradley-Terry environment integration
    print(f"\n🎯 Testing environment integration...")
    try:
        sys.path.append("../../environments/vf_bradley_terry")
        from bradley_terry import load_environment
        
        # Test with very small sample
        env = load_environment(num_prompts=2)
        print(f"✅ Environment loads: {len(env.dataset)} train, {len(env.eval_dataset)} eval")
        
        sample_env_prompt = env.dataset[0]
        print(f"📋 Environment prompt format:")
        print(f"   Messages: {len(sample_env_prompt['messages'])} message(s)")
        print(f"   Content: {sample_env_prompt['messages'][0]['content'][:100]}...")
        print(f"   Info: {sample_env_prompt['info'].keys()}")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_local_files()
    
    if success:
        print(f"\n🎉 All tests passed!")
        print(f"🚀 Dataset is ready for:")
        print(f"   - Loading from HF: load_dataset('cosmicoptima/introspection-prompts')")
        print(f"   - Bradley-Terry training: load_environment()")
        print(f"   - Manual analysis: generated_prompts.json")
    else:
        print(f"\n❌ Tests failed. Check file paths and dependencies.") 