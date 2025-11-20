#!/usr/bin/env python3
"""
Convert a HuggingFace model from .bin (PyTorch) to .safetensors format.

This script downloads a model from HuggingFace, converts the weights from .bin to .safetensors,
and uploads the converted model back to HuggingFace (either to the same repo or a new one).

OPTIMIZED FOR LOW MEMORY/DISK USAGE - suitable for cheap rented CPU machines!

Memory/Disk Requirements:
    - Memory: ~10-15GB RAM (processes one shard at a time)
    - Disk: ~10-20GB free space (only stores current shard being processed)
    - Works on CPU-only machines (no GPU required)
    - Recommended: Any cheap cloud instance with 16GB+ RAM

Usage:
    # Convert and upload to a new repo
    python scripts/convert_bin_to_safetensors.py \
        --source-repo username/model-name \
        --target-repo username/model-name-safetensors

    # Convert and update the same repo (use with caution!)
    python scripts/convert_bin_to_safetensors.py \
        --source-repo username/model-name \
        --target-repo username/model-name \
        --overwrite

Requirements:
    pip install huggingface_hub safetensors torch
    huggingface-cli login
"""

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, HfFileSystem, create_repo, hf_hub_download
from loguru import logger
from safetensors.torch import save_file


class ProgressIndicator:
    """Shows progress dots while waiting for long operations."""

    def __init__(self, message: str, interval: int = 30):
        self.message = message
        self.interval = interval
        self.running = False
        self.thread = None
        self.start_time = None

    def _run(self):
        while self.running:
            elapsed = int(time.time() - self.start_time)
            logger.info(f"{self.message} (elapsed: {elapsed}s / {elapsed//60}m {elapsed%60}s)")
            time.sleep(self.interval)

    def __enter__(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)


def convert_shard_by_shard(
    source_repo: str,
    target_repo: str,
    overwrite: bool = False,
    private: bool = False,
):
    """Convert model shard-by-shard with minimal memory and disk usage."""

    # Check if overwriting same repo
    if source_repo == target_repo and not overwrite:
        logger.error("Source and target repos are the same. Use --overwrite to confirm.")
        return

    api = HfApi()
    fs = HfFileSystem()

    # Get list of files in source repo
    try:
        files = api.list_repo_files(repo_id=source_repo, repo_type="model")
    except Exception as e:
        logger.error(f"Failed to list files in {source_repo}: {e}")
        return

    # Find .bin files
    bin_files = [f for f in files if f.endswith(".bin") and not f.endswith(".index.json")]
    metadata_files = [
        f
        for f in files
        if not f.endswith(".bin") and not f.endswith(".safetensors") and not f.endswith(".safetensors.index.json")
    ]

    if not bin_files:
        logger.error("No .bin files found. Model may already be in safetensors format.")
        return

    logger.info(f"Found {len(bin_files)} .bin file(s) to convert")
    logger.info(f"Found {len(metadata_files)} metadata file(s) to copy")

    # Create target repo
    logger.info(f"Creating/updating repository: {target_repo}")
    create_repo(
        repo_id=target_repo,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    # Create temporary directory for processing
    temp_dir = Path("./tmp_conversion")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Check if model is sharded
        index_file = "pytorch_model.bin.index.json"
        has_index = index_file in files

        if has_index:
            # Download and parse index
            logger.info("Downloading model index...")
            index_path = hf_hub_download(
                repo_id=source_repo,
                filename=index_file,
                repo_type="model",
                local_dir=temp_dir,
                local_dir_use_symlinks=False,
            )

            with open(index_path) as f:
                index = json.load(f)

            weight_map = index["weight_map"]
            metadata = index.get("metadata", {})

            # Create new index for safetensors
            new_weight_map = {}
            new_index = {"metadata": metadata, "weight_map": new_weight_map}

            # Convert each shard
            for shard_idx, bin_file in enumerate(sorted(set(weight_map.values())), 1):
                total_shards = len(set(weight_map.values()))
                logger.info(f"Processing shard {shard_idx}/{total_shards}: {bin_file}")

                # Download this shard only
                logger.info(f"  Downloading {bin_file}...")
                bin_path = hf_hub_download(
                    repo_id=source_repo,
                    filename=bin_file,
                    repo_type="model",
                    local_dir=temp_dir,
                    local_dir_use_symlinks=False,
                )

                # Get file size for reference
                file_size_gb = os.path.getsize(bin_path) / (1024**3)
                logger.info(f"  File size: {file_size_gb:.2f} GB")

                # Load and convert
                logger.info(f"  Loading weights into memory (this may take several minutes)...")
                load_start = time.time()
                with ProgressIndicator("  Still loading weights...", interval=30):
                    shard_state = torch.load(bin_path, map_location="cpu", weights_only=True)
                load_time = time.time() - load_start
                logger.info(f"  Loaded {len(shard_state)} tensors in {load_time:.1f}s ({load_time/60:.1f} min)")

                # Generate safetensors filename
                safetensors_file = bin_file.replace(".bin", ".safetensors")

                # Save as safetensors
                logger.info(f"  Converting to safetensors format...")
                save_start = time.time()
                safetensors_path = temp_dir / safetensors_file
                with ProgressIndicator("  Still converting...", interval=30):
                    save_file(shard_state, safetensors_path, metadata={"format": "pt"})
                save_time = time.time() - save_start
                logger.info(f"  Saved in {save_time:.1f}s ({save_time/60:.1f} min)")

                # Update weight map
                for key, file in weight_map.items():
                    if file == bin_file:
                        new_weight_map[key] = safetensors_file

                # Upload this converted shard immediately
                logger.info(f"  Uploading {safetensors_file}...")
                api.upload_file(
                    path_or_fileobj=str(safetensors_path),
                    path_in_repo=safetensors_file,
                    repo_id=target_repo,
                    repo_type="model",
                )

                # Clean up to save disk space
                os.remove(bin_path)
                os.remove(safetensors_path)
                logger.info(f"  Completed and cleaned up {bin_file}")

            # Save and upload the new index
            logger.info("Creating safetensors index...")
            index_output = temp_dir / "model.safetensors.index.json"
            with open(index_output, "w", encoding="utf-8") as f:
                json.dump(new_index, f, indent=2, sort_keys=True)
                f.write("\n")

            logger.info("Uploading index...")
            api.upload_file(
                path_or_fileobj=str(index_output),
                path_in_repo="model.safetensors.index.json",
                repo_id=target_repo,
                repo_type="model",
            )

        else:
            # Single file model
            bin_file = bin_files[0]
            logger.info(f"Processing single-file model: {bin_file}")

            # Download
            logger.info("Downloading model...")
            bin_path = hf_hub_download(
                repo_id=source_repo,
                filename=bin_file,
                repo_type="model",
                local_dir=temp_dir,
                local_dir_use_symlinks=False,
            )

            # Get file size for reference
            file_size_gb = os.path.getsize(bin_path) / (1024**3)
            logger.info(f"File size: {file_size_gb:.2f} GB")

            # Load and convert
            logger.info("Loading weights into memory (this may take several minutes for large models)...")
            logger.info("Progress updates will appear every 30 seconds...")
            load_start = time.time()
            with ProgressIndicator("Still loading weights...", interval=30):
                state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            load_time = time.time() - load_start
            logger.info(f"Loaded {len(state_dict)} tensors in {load_time:.1f}s ({load_time/60:.1f} minutes)")

            # Save as safetensors
            logger.info("Converting to safetensors format (this may also take several minutes)...")
            save_start = time.time()
            safetensors_path = temp_dir / "model.safetensors"
            with ProgressIndicator("Still converting...", interval=30):
                save_file(state_dict, safetensors_path, metadata={"format": "pt"})
            save_time = time.time() - save_start
            logger.info(f"Saved in {save_time:.1f}s ({save_time/60:.1f} minutes)")

            # Upload
            logger.info("Uploading converted model...")
            api.upload_file(
                path_or_fileobj=str(safetensors_path),
                path_in_repo="model.safetensors",
                repo_id=target_repo,
                repo_type="model",
            )

            # Clean up
            os.remove(bin_path)
            os.remove(safetensors_path)

        # Upload metadata files (config, tokenizer, etc.)
        logger.info("Uploading metadata files...")
        for metadata_file in metadata_files:
            logger.info(f"  Uploading {metadata_file}...")
            metadata_path = hf_hub_download(
                repo_id=source_repo,
                filename=metadata_file,
                repo_type="model",
                local_dir=temp_dir,
                local_dir_use_symlinks=False,
            )
            api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo=metadata_file,
                repo_id=target_repo,
                repo_type="model",
            )
            os.remove(metadata_path)

        logger.success(f"✅ Successfully converted and uploaded model to: https://huggingface.co/{target_repo}")

        # Print info
        print(f"\n🤖 To load the converted model:")
        print(f"```python")
        print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"model = AutoModelForCausalLM.from_pretrained('{target_repo}')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{target_repo}')")
        print(f"```")

        print(f"\n🔗 View at: https://huggingface.co/{target_repo}")

    finally:
        # Cleanup
        logger.info("Cleaning up temporary files...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model from .bin to .safetensors format")
    parser.add_argument(
        "--source-repo",
        required=True,
        help="Source HuggingFace repo ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--target-repo",
        required=True,
        help="Target HuggingFace repo ID (can be same as source with --overwrite)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the source repo if source and target are the same",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the target repository private",
    )

    args = parser.parse_args()

    convert_shard_by_shard(
        source_repo=args.source_repo,
        target_repo=args.target_repo,
        overwrite=args.overwrite,
        private=args.private,
    )


if __name__ == "__main__":
    main()

