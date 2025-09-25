#!/usr/bin/env python3
"""
Test loading the introspection prompts dataset from Hugging Face.
"""

def test_hf_dataset():
    """Test loading from Hugging Face Hub."""
    try:
        from datasets import load_dataset
        
        print("📥 Loading dataset from Hugging Face...")
        dataset = load_dataset('cosmicoptima/introspection-prompts')
        
        print(f"✅ Successfully loaded dataset!")
        print(f"📊 Total prompts: {len(dataset['train'])}")
        
        # Show sample
        sample = dataset['train'][0]
        print(f"\n📋 Sample entry:")
        print(f"   ID: {sample['id']}")
        print(f"   Pattern: {sample['pattern']}")
        print(f"   Prompt: {sample['prompt'][:150]}...")
        
        # Show pattern distribution
        patterns = {}
        for item in dataset['train']:
            pattern = item['pattern']
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        print(f"\n📈 Pattern distribution:")
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"   {pattern}: {count}")
            
        return dataset
        
    except ImportError:
        print("❌ datasets library not available")
        print("Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def test_bradley_terry_environment():
    """Test the Bradley-Terry environment with the dataset."""
    try:
        import sys
        sys.path.append("../../environments/vf_bradley_terry")
        
        from bradley_terry import load_environment
        
        print("\n🎯 Testing Bradley-Terry environment...")
        
        # Load with just a few prompts for testing
        env = load_environment(num_prompts=5)
        
        print(f"✅ Environment loaded successfully!")
        print(f"📊 Training prompts: {len(env.dataset)}")
        print(f"📊 Eval prompts: {len(env.eval_dataset)}")
        
        # Show a sample prompt
        sample = env.dataset[0]
        print(f"\n📋 Sample environment prompt:")
        print(f"   Content: {sample['messages'][0]['content'][:150]}...")
        print(f"   Pattern: {sample['info']['pattern']}")
        
        return env
        
    except Exception as e:
        print(f"❌ Error testing environment: {e}")
        return None

if __name__ == "__main__":
    print("🧪 Testing Introspection Prompts Dataset\n")
    
    # Test 1: Load from HF
    dataset = test_hf_dataset()
    
    # Test 2: Bradley-Terry environment  
    env = test_bradley_terry_environment()
    
    if dataset and env:
        print(f"\n🎉 All tests passed! Dataset is ready for use.")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above.") 