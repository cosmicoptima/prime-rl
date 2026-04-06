@AGENTS.md

# prime-rl-selfsim

Fork of upstream prime-rl (at commit `bffd310`) for multi-turn RL on the selfsim (vvvv) model. Configs, environment, and patches live here. The pod runs upstream's Docker image but uses our code via PYTHONPATH.

## Running an experiment

**There is a setup script that handles everything. Use it.**

```bash
# Edit configs/negamp_v2/rl.toml, commit, push, then:
HF_TOKEN=... OPENROUTER_API_KEY=... WANDB_API_KEY=... ./scripts/setup_pod.sh USER@ssh.runpod.io --launch
```

The three env vars are secrets that must be passed on the command line. Ask the user for them or check conversation history. Do not hardcode them in committed files (GitHub push protection will block it).

This git-pulls on the pod, waits for the user sim, cleans stale GPU processes, and launches. ~30 seconds on a warm pod.

To switch between runs: same command again. It kills the old run, pulls new code, relaunches. Do NOT restart the pod — the 70B user sim takes 5-10 min to reload.

### If the pod is stopped or doesn't exist

The pod can be resumed via the RunPod GraphQL API:
```bash
curl -s -X POST "https://api.runpod.io/graphql?api_key=API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { podResume(input: { podId: \"POD_ID\", gpuCount: 8 }) { id desiredStatus } }"}'
```
Wait ~2 min for the container to start, then run the setup script. The startup script (container start command) handles chmod, peft install, model downloads, and vLLM user sim launch. Its template is in the header comments of `scripts/setup_pod.sh`.

## Critical things that will waste your time if you don't know them

**RunPod proxy SSH does not work like normal SSH.** You cannot pass commands as arguments. They are silently ignored. You must pipe via stdin:
```bash
echo 'your-command; exit' | ssh -tt -i ~/.ssh/id_ed25519 USER@ssh.runpod.io
```
The `-tt` flag is mandatory. scp and rsync do not work. Use `git clone`/`git pull` on the pod for file transfer.

**Do not push files to `/app/src/`.** The image's source code is at `/app/src/` but our fork is at `/workspace/prime-rl/`. We set `PYTHONPATH=/workspace/prime-rl/src:/workspace/prime-rl/environments` to override it. If you copy files into `/app/src/` you will create version mismatches that break imports.

**Stale GPU processes after a killed run are just vLLM inference workers. They are killable.** `kill -9 PID` works. Do NOT restart the pod to clear them. Check what's on the GPUs with `nvidia-smi --query-compute-apps=pid --format=csv,noheader`. The user sim runs on port 8002 — verify before killing: `ps -p PID -o args= | grep 8002`. Only restart the pod if the PIDs don't exist in `ps` at all (true GPU zombies, which is rare).

**The CLI is `hf`, not `huggingface-cli`.** The startup script needs `sudo chmod 777 /workspace`. The HF token must be `export`ed before any `CUDA_VISIBLE_DEVICES=` prefix or vLLM won't authenticate for gated models. `peft` is not in the image and must be installed on each container start. `hf-xet` is buggy and should be uninstalled for uploads to work.

## Pod info

- Image: `primeintellect/prime-rl:commit-bffd310` (NOT `:main`)
- Pod ID: `0lg9lsl5bw8i86`
- SSH: `0lg9lsl5bw8i86-644121ae@ssh.runpod.io`
- RunPod API key: ask the user or check conversation history
- GPUs 0-1: inference, 2-3: training, 4-7: user sim (70B)
- Volume: 400GB at `/workspace` (persists). Container disk at `/app` (wiped on restart).
- Base model: `aethera-gp/selfsim-v3.1-8b-A-ckpt700-merged`
- Judge: GLM-5 via OpenRouter
- Wandb: project `selfsim-rl`

## Viewing samples

Due to PTY noise, you can't just run Python inline via SSH. Write a script to a file, execute it, save output to a file, then cat:
```bash
echo 'SCRIPT_BASE64' | base64 -d > /tmp/view.py
python3 /tmp/view.py > /tmp/out.txt 2>&1
cat /tmp/out.txt
```
Sample files are at `outputs/run_default/wandb/run-*/files/media/table/samples_STEP_*.table.json`. Each contains columns `step, task, example_id, messages, input_ids, reward`. The `messages` field is a raw string with `<|start_header_id|>user/assistant<|end_header_id|>` delimiters.

## What's in the fork

Our patches add rsLoRA support to the trainer and a single-judge multi-turn conversation environment. See the git history for details. The environment is at `environments/vf_multiturn/__init__.py`.
