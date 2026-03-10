#!/usr/bin/env python3
"""
Shared inference utilities for eval scripts.

Supports both OpenRouter API and local vLLM inference.
"""

import os
import requests
import subprocess
import time
import signal
import atexit
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# OpenRouter config
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Local vLLM config
LOCAL_BASE_URL = "http://localhost:8000/v1"
LOCAL_API_KEY = "EMPTY"

# Models that use extended reasoning
REASONING_MODELS = {
    "deepseek/deepseek-r1",
    "openai/o1",
    "openai/o3-mini",
    "openai/o3",
    "openai/o4-mini",
}

# Global reference to spawned server process
_server_process: Optional[subprocess.Popen] = None


def _cleanup_server():
    """Clean up server process on exit."""
    global _server_process
    if _server_process is not None:
        print("\nShutting down vLLM server...")
        _server_process.terminate()
        try:
            _server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_process.kill()
        _server_process = None


atexit.register(_cleanup_server)


def is_local_model(model: str) -> bool:
    """Check if model string refers to a local model (path or HF model without provider prefix)."""
    # If it's a path
    if "/" in model and Path(model.split("/")[0]).exists():
        return True
    # If it doesn't have a provider prefix like "anthropic/" or "openai/"
    if "/" in model:
        provider = model.split("/")[0]
        known_providers = {
            "anthropic", "openai", "google", "meta-llama", "mistralai",
            "qwen", "deepseek", "x-ai", "moonshotai", "cohere",
        }
        if provider.lower() not in known_providers:
            # Could be HF model like "Qwen/Qwen3-8B"
            return True
    return False


def wait_for_server(base_url: str, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/models", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def start_local_server(
    model: str,
    port: int = 8000,
    tp: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = True,
) -> str:
    """
    Start a local vLLM server for the given model.

    Returns the base URL for the server.
    """
    global _server_process

    base_url = f"http://localhost:{port}/v1"

    # Check if server is already running
    try:
        resp = requests.get(f"{base_url}/models", timeout=2)
        if resp.status_code == 200:
            print(f"vLLM server already running at {base_url}")
            return base_url
    except requests.exceptions.RequestException:
        pass

    # Build command
    cmd = [
        "python", "-m", "prime_rl.inference.server",
        f"--model.name={model}",
        f"--server.port={port}",
        f"--parallel.tp={tp}",
        f"--gpu_memory_utilization={gpu_memory_utilization}",
    ]

    if max_model_len:
        cmd.append(f"--model.max_model_len={max_model_len}")

    if trust_remote_code:
        cmd.append("--model.trust_remote_code=true")

    print(f"Starting vLLM server: {' '.join(cmd)}")

    # Start server
    _server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for server to be ready
    print("Waiting for server to be ready...")
    if not wait_for_server(base_url):
        _cleanup_server()
        raise RuntimeError("vLLM server failed to start")

    print(f"vLLM server ready at {base_url}")
    return base_url


def get_local_model_name(base_url: str) -> str:
    """Get the model name from a running vLLM server."""
    resp = requests.get(f"{base_url}/models", timeout=10)
    data = resp.json()
    if data.get("data"):
        return data["data"][0]["id"]
    raise RuntimeError("Could not get model name from server")


class InferenceClient:
    """
    Unified inference client that works with both OpenRouter and local vLLM.
    """

    def __init__(
        self,
        model: str,
        local: bool = False,
        local_base_url: Optional[str] = None,
        start_server: bool = False,
        server_kwargs: Optional[dict] = None,
    ):
        """
        Initialize inference client.

        Args:
            model: Model name (OpenRouter format like "anthropic/claude-sonnet-4"
                   or HF format like "Qwen/Qwen3-8B" or local path)
            local: Force local inference mode
            local_base_url: Base URL for local server (default: http://localhost:8000/v1)
            start_server: Whether to start a vLLM server if not running
            server_kwargs: Additional kwargs for start_local_server
        """
        self.original_model = model
        self.local = local or is_local_model(model)

        if self.local:
            self.base_url = local_base_url or LOCAL_BASE_URL
            self.api_key = LOCAL_API_KEY

            if start_server:
                self.base_url = start_local_server(model, **(server_kwargs or {}))

            # Get actual model name from server
            try:
                self.model = get_local_model_name(self.base_url)
            except Exception:
                self.model = model
        else:
            self.base_url = OPENROUTER_BASE_URL
            self.api_key = OPENROUTER_API_KEY
            self.model = model

            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")

    def call(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> dict:
        """
        Call the model with a prompt.

        Returns dict with 'content', 'reasoning' (if applicable), or 'error'.
        """
        if max_tokens is None:
            max_tokens = 4000 if self.original_model in REASONING_MODELS else 1000

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int = 1000,
    ) -> dict:
        """
        Call the model with a message list.

        Returns dict with 'content', 'reasoning' (if applicable), or 'error'.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=300
            )
            data = resp.json()

            if 'error' in data:
                return {"error": str(data['error'])}

            msg = data['choices'][0]['message']
            return {
                "content": msg.get('content', ''),
                "reasoning": msg.get('reasoning', ''),
            }
        except Exception as e:
            return {"error": str(e)}

    def __repr__(self):
        mode = "local" if self.local else "openrouter"
        return f"InferenceClient(model={self.model}, mode={mode})"


def create_client(
    model: str,
    local: bool = False,
    local_url: Optional[str] = None,
    start_server: bool = False,
    **server_kwargs,
) -> InferenceClient:
    """
    Create an inference client for the given model.

    Args:
        model: Model name
        local: Force local inference mode
        local_url: Base URL for existing local server
        start_server: Start a vLLM server if needed
        **server_kwargs: Additional kwargs for server startup (tp, gpu_memory_utilization, etc.)

    Returns:
        InferenceClient instance
    """
    return InferenceClient(
        model=model,
        local=local,
        local_base_url=local_url,
        start_server=start_server,
        server_kwargs=server_kwargs if server_kwargs else None,
    )
