#!/bin/bash
# Setup and launch RL on a RunPod pod.
#
# Usage:
#   ./scripts/setup_pod.sh USER@ssh.runpod.io [--launch] [config_name]
#   BRANCH=name overrides the default branch (selfsim)
#
# Prerequisites:
#   - Pod running with primeintellect/prime-rl:commit-bffd310 image
#   - Startup script (see below) already ran chmod, peft install, vllm start
#
# RunPod startup script (paste as container start command, replacing $HF_TOKEN with your actual token):
#   bash -c "sudo chmod 777 /workspace && mkdir -p /workspace/hf_cache /workspace/logs /workspace/models && export HF_HOME=/workspace/hf_cache && export HF_HUB_ENABLE_HF_TRANSFER=1 && export HF_TOKEN=$HF_TOKEN && uv pip install peft zstandard hf-xet --python /app/.venv/bin/python 2>/dev/null; /app/.venv/bin/hf download aethera-gp/selfsim-v3.1-8b-A-ckpt700-merged --cache-dir /workspace/hf_cache 1>/workspace/logs/download_model.log 2>&1 & /app/.venv/bin/hf download meta-llama/Llama-3.1-70B --cache-dir /workspace/hf_cache 1>/workspace/logs/download_70b.log 2>&1 & export CUDA_VISIBLE_DEVICES=4,5,6,7; /app/.venv/bin/vllm serve meta-llama/Llama-3.1-70B --port 8002 --gpu-memory-utilization 0.95 --max-model-len 4096 --tensor-parallel-size 4 --download-dir /workspace/hf_cache 1>/workspace/logs/usersim_vllm.log 2>&1 & sleep infinity"
#
# Troubleshooting:
#   - If GPUs 0-3 have stale processes: they are vLLM inference workers, kill with kill -9 PID
#     Do NOT kill processes on GPUs 4-7 (user sim). Check: ps -p PID -o args= | grep 8002
#   - If user sim fails with gated repo error: HF_TOKEN not exported before CUDA_VISIBLE_DEVICES
#   - If LoRA upload to HF fails: uninstall hf-xet (uv pip uninstall hf-xet) and retry the upload only.
#     We keep hf-xet installed at startup for ~5-10x faster downloads (266GB Llama-3.1-70B in ~15 min vs ~2.5 hr).

set -euo pipefail

SSH_TARGET="${1:?Usage: $0 USER@ssh.runpod.io [--launch] [config_name]}"
LAUNCH="${2:-}"
CONFIG="${3:-distributed_self}"
BRANCH="${BRANCH:-selfsim}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

ssh_cmd() {
    # Run a command on the pod via RunPod proxy SSH (stdin piping required)
    echo "$1; exit" | ssh -tt -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$SSH_TARGET" 2>&1
}

ssh_cmd_quiet() {
    # Run and filter to just the useful output
    ssh_cmd "$1" | strings | grep -v "appuser@\|RUNPOD\|Enjoy\|Warning\|^\[" | grep -v "^$1" || true
}

echo "=== Pod setup: $SSH_TARGET ==="

# --- 1. Clone/update fork on pod ---
echo ""
echo "--- Syncing source code ---"
ssh_cmd 'if [ -d /workspace/prime-rl/.git ]; then cd /workspace/prime-rl && git fetch origin '"$BRANCH"' 2>&1 | tail -1 && git checkout '"$BRANCH"' 2>&1 | tail -1 && git pull --ff-only origin '"$BRANCH"' 2>&1 | tail -1; else git clone -b '"$BRANCH"' https://github.com/cosmicoptima/prime-rl.git /workspace/prime-rl 2>&1 | tail -1; fi; rm -rf /app/configs/'"$CONFIG"' && cp -r /workspace/prime-rl/configs/'"$CONFIG"' /app/configs/'"$CONFIG"'; cp /workspace/prime-rl/configs/negamp_v2/SYSTEM_PROMPT.md /workspace/SYSTEM_PROMPT.md; find /workspace/prime-rl -name __pycache__ -exec rm -rf {} + 2>/dev/null; echo SYNC_DONE' 2>&1 | grep -E "SYNC_DONE|Already|Updating|Cloning|Switched|Fast-forward" || true

# --- 2. Wait for user sim ---
echo ""
echo "--- Waiting for user sim ---"
for i in $(seq 1 120); do
    RESULT=$(ssh_cmd 'curl -s -o /dev/null -w "HEALTH_%{http_code}" http://127.0.0.1:8002/health' 2>&1 | grep -o "HEALTH_[0-9]*" | tail -1)
    if [ "$RESULT" = "HEALTH_200" ]; then
        echo "  user sim ready!"
        break
    fi
    if [ "$i" -eq 1 ]; then
        echo "  waiting (check /workspace/logs/usersim_vllm.log if slow)..."
    fi
    sleep 10
done

# --- 3. Kill stale GPU workers (not user sim) ---
echo ""
echo "--- Cleaning stale GPU processes ---"
ssh_cmd 'for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do ps -p $pid -o args= 2>/dev/null | grep -q 8002 || kill -9 $pid 2>/dev/null; done; sleep 2; echo "GPU processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l) (should be 4 = user sim)"' 2>&1 | grep "GPU processes" || true

# --- 4. Launch if requested ---
if [ "$LAUNCH" = "--launch" ]; then
    echo ""
    echo "--- Launching RL run ---"
    ssh_cmd "export HF_HOME=/workspace/hf_cache; export HF_HUB_ENABLE_HF_TRANSFER=1; export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN env var}; export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY env var}; export WANDB_API_KEY=${WANDB_API_KEY:?Set WANDB_API_KEY env var}; export PYTHONPATH=/workspace/prime-rl/src:/workspace/prime-rl/environments; nohup /app/.venv/bin/rl @ configs/$CONFIG/rl.toml > /workspace/logs/run.log 2>&1 & sleep 2; ps aux | grep 'rl @' | grep -v grep | head -1; echo RL_LAUNCHED" 2>&1 | grep -E "rl @|RL_LAUNCHED" || echo "  WARNING: launch may have failed"
    echo ""
    echo "  Check logs: ssh into pod and run: tail -f /workspace/logs/run.log"
fi

echo ""
echo "=== Setup complete ==="
