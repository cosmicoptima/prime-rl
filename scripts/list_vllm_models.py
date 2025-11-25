#!/usr/bin/env python3
"""
Quick utility to list available models from vLLM API.

Usage:
    python scripts/list_vllm_models.py [--api-base http://localhost:8000/v1]
"""

import argparse
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="List models available in vLLM API")
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="Base URL for the vLLM API (default: http://localhost:8000/v1)",
    )
    args = parser.parse_args()
    
    print(f"Connecting to {args.api_base}...")
    print()
    
    try:
        client = OpenAI(base_url=args.api_base, api_key="dummy")
        models = client.models.list()
        
        print("Available models:")
        print("-" * 80)
        for model in models:
            print(f"  - {model.id}")
        print("-" * 80)
        print(f"Total: {len(models.data)} model(s)")
        
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print()
        print("Make sure vLLM is running. Start it with:")
        print("  vllm serve <model-name> --port 8000")


if __name__ == "__main__":
    main()

