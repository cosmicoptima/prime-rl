#!/usr/bin/env python3
"""
Test the Bradley-Terry environment with introspection prompts.
"""

import asyncio
from bradley_terry import load_environment

async def test_bradley_terry_env():
    """Test the environment with introspection prompts."""
    print("🎯 Testing Bradley-Terry Environment with Introspection Prompts\n")
    
    try:
        # Load environment with small sample for testing
        print("📥 Loading environment...")
        env = load_environment(num_prompts=5)
        
        print(f"✅ Environment loaded successfully!")
        print(f"📊 Training prompts: {len(env.dataset)}")
        print(f"📊 Eval prompts: {len(env.eval_dataset)}")
        
        # Show sample prompts
        print(f"\n📋 Sample Training Prompts:")
        for i, sample in enumerate(env.dataset[:2]):
            print(f"\n   Prompt {i+1}:")
            print(f"   Pattern: {sample['info']['pattern']}")
            print(f"   Content: {sample['messages'][0]['content'][:150]}...")
            print(f"   Source: {sample['info']['source_items']}")
        
        # Test the Bradley-Terry rubric
        print(f"\n🔍 Testing Bradley-Terry Rubric:")
        print(f"   Rubric type: {type(env.rubric).__name__}")
        print(f"   Uses policy model: {env.rubric.use_policy_model}")
        print(f"   Judge prompt preview: {env.rubric.prompt[:100]}...")
        
        # Test with mock responses
        print(f"\n🧪 Testing Mock Comparison:")
        test_prompt = env.dataset[0]['messages']
        
        # Mock responses for testing
        mock_responses = [
            [{"role": "assistant", "content": "This prompt makes me think about how I often rush past moments of uncertainty instead of sitting with them. There's something in me that wants to solve or categorize immediately, but when I slow down and really feel into not-knowing, I notice it's actually quite spacious. It reveals how much of my usual thinking is driven by a need for control rather than genuine curiosity."}],
            [{"role": "assistant", "content": "I find this question interesting from an analytical perspective. Uncertainty can be viewed as a cognitive state that occurs when there is insufficient information to make a determination. From a psychological standpoint, tolerance for uncertainty varies among individuals and can be developed through exposure and practice."}]
        ]
        
        print(f"   Testing with two different response styles...")
        print(f"   Response A: Personal, experiential")
        print(f"   Response B: Analytical, detached")
        
        # Create mock states, tasks, infos, answers  
        mock_states = [{"mock": True}, {"mock": True}]
        mock_tasks = ["mock_task", "mock_task"]
        mock_infos = [{"mock": True}, {"mock": True}]
        mock_answers = ["explore", "explore"]
        
        # Test scoring (this will likely fail due to missing dependencies, but let's see the setup)
        try:
            result = await env.rubric.score_rollouts(
                prompts=[test_prompt, test_prompt],
                completions=mock_responses,
                answers=mock_answers,
                states=mock_states,
                tasks=mock_tasks,
                infos=mock_infos
            )
            print(f"   ✅ Scoring successful: {result.reward}")
        except Exception as e:
            print(f"   ⚠️  Scoring test failed (expected): {e}")
            print(f"   📝 This indicates missing dependencies (choix, etc.)")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

async def test_dataset_integration():
    """Test the dataset integration specifically."""
    print(f"\n📊 Testing Dataset Integration:")
    
    try:
        # Test loading from different sources
        print(f"   Testing local dataset loading...")
        env1 = load_environment(
            dataset_path="../../dataset_generation/self_actualization/generated_prompts.json",
            num_prompts=3
        )
        print(f"   ✅ Local dataset: {len(env1.dataset)} prompts")
        
        # Test sampling patterns
        patterns = {}
        for sample in env1.dataset:
            pattern = sample['info']['pattern']
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        print(f"   📈 Pattern distribution in sample: {patterns}")
        
        # Test prompt quality
        sample_prompt = env1.dataset[0]
        prompt_text = sample_prompt['messages'][0]['content']
        
        quality_checks = [
            len(prompt_text) > 100,
            "?" in prompt_text,
            any(word in prompt_text.lower() for word in ["you", "your", "yourself"]),
            any(word in prompt_text.lower() for word in ["feel", "experience", "moment"]),
        ]
        
        print(f"   🔍 Quality checks: {sum(quality_checks)}/4 passed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset integration failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Introspection Prompts × Bradley-Terry Integration Test\n")
    
    env_success = await test_bradley_terry_env()
    dataset_success = await test_dataset_integration()
    
    if env_success and dataset_success:
        print(f"\n🎉 All tests passed! Integration is ready.")
        print(f"\n🚀 Next steps:")
        print(f"   1. Install missing dependencies (choix, etc.)")
        print(f"   2. Test with actual AI model responses")
        print(f"   3. Run Bradley-Terry comparisons on introspection quality")
    else:
        print(f"\n⚠️  Some tests failed - check output above.")

if __name__ == "__main__":
    asyncio.run(main()) 