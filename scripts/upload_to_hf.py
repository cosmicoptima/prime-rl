#!/usr/bin/env python3
"""
Upload trained model to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py \
        --model-dir outputs/weights/step_1000 \
        --username YOUR_HF_USERNAME \
        --repo-name your-model-name \
        [--private]

Requirements:
    pip install huggingface_hub
    huggingface-cli login
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from loguru import logger


def upload_model(model_dir: Path, username: str, repo_name: str, private: bool = False):
    """Upload the model to Hugging Face Hub."""

    repo_id = f"{username}/{repo_name}"

    # Initialize HF API
    api = HfApi()

    try:
        # Create repository (repo_type defaults to "model")
        logger.info(f"Creating model repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )

        # Upload the entire model directory
        logger.info(f"Uploading model from {model_dir}...")
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
        )

        logger.success(f"Successfully uploaded model to: https://huggingface.co/{repo_id}")

        # Print loading instructions
        print(f"\n🤖 To load this model:")
        print(f"```python")
        print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"model = AutoModelForCausalLM.from_pretrained('{repo_id}')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{repo_id}')")
        print(f"```")

        print(f"\n🔗 View at: https://huggingface.co/{repo_id}")

    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        print("Make sure you're logged in with: huggingface-cli login")
        raise


def main():
    parser = argparse.ArgumentParser(description="Upload trained model to Hugging Face")
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Path to model directory (e.g., outputs/weights/step_1000)",
    )
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--repo-name", required=True, help="Model repository name")
    parser.add_argument("--private", action="store_true", help="Make repository private")

    args = parser.parse_args()

    # Verify model directory exists
    if not args.model_dir.exists():
        logger.error(f"Model directory not found: {args.model_dir}")
        return

    # Check for STABLE file
    if not (args.model_dir / "STABLE").exists():
        logger.warning("No STABLE file found. Checkpoint may be incomplete.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            logger.info("Upload cancelled")
            return

    # Check for required files
    required_files = ["config.json"]
    missing_files = [f for f in required_files if not (args.model_dir / f).exists()]
    if missing_files:
        logger.warning(f"Missing files: {', '.join(missing_files)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            logger.info("Upload cancelled")
            return

    upload_model(args.model_dir, args.username, args.repo_name, args.private)


if __name__ == "__main__":
    main()

