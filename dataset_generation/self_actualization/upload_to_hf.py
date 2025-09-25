#!/usr/bin/env python3
"""
Upload self-actualization prompts dataset to Hugging Face Hub.

Usage:
    python upload_to_hf.py --username YOUR_HF_USERNAME --repo-name self-actualization-prompts

Requirements:
    pip install huggingface_hub datasets pandas
    huggingface-cli login
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from datasets import load_from_disk
import json

def upload_dataset(username: str, repo_name: str, private: bool = False):
    """Upload the dataset to Hugging Face Hub."""
    
    repo_id = f"{username}/{repo_name}"
    
    # Initialize HF API
    api = HfApi()
    
    try:
        # Create repository
        print(f"📦 Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        
        # Upload files
        print("📤 Uploading files...")
        
        # Upload just README
        print("   Uploading README.md...")
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md", 
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        # Upload the HF dataset format if it exists
        if Path("hf_dataset").exists():
            print("   Uploading HF dataset format...")
            api.upload_folder(
                folder_path="hf_dataset",
                path_in_repo="hf_dataset", 
                repo_id=repo_id,
                repo_type="dataset"
            )
        
        print(f"✅ Successfully uploaded dataset to: https://huggingface.co/datasets/{repo_id}")
        
        # Print loading instructions
        print(f"\n📚 To load this dataset:")
        print(f"```python")
        print(f"from datasets import load_dataset")
        print(f"dataset = load_dataset('{repo_id}')")
        print(f"```")
        
        print(f"\n🔗 View at: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        print("Make sure you're logged in with: huggingface-cli login")

def main():
    parser = argparse.ArgumentParser(description="Upload self-actualization dataset to Hugging Face")
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--repo-name", default="introspection-prompts", help="Repository name")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    
    args = parser.parse_args()
    
    # First prepare the HF format
    print("🔄 Preparing Hugging Face format...")
    from prepare_for_hf import convert_to_hf_format
    convert_to_hf_format()
    
    # Then upload
    upload_dataset(args.username, args.repo_name, args.private)

if __name__ == "__main__":
    main() 