#!/usr/bin/env python3
"""
Upload just the README to update the dataset documentation.
"""

from huggingface_hub import HfApi

def upload_readme():
    """Upload the updated README."""
    api = HfApi()
    
    try:
        print("📤 Uploading updated README...")
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id="cosmicoptima/introspection-prompts",
            repo_type="dataset"
        )
        print("✅ README successfully updated on HuggingFace!")
        print("🔗 View at: https://huggingface.co/datasets/cosmicoptima/introspection-prompts")
        
    except Exception as e:
        print(f"❌ Error uploading README: {e}")

if __name__ == "__main__":
    upload_readme() 