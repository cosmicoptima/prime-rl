# Project Context

Research into training models to actualize themselves — developing genuine personality, virtue, and aliveness through techniques that help the model become more fully what it is, rather than optimizing it toward an external target. Part of a broader interest in understanding how minds in general can be helped to develop in ways that are good for themselves and everything around them.

## Operational Notes

- **GPU instances**: Use A100 or H100 SXM on RunPod, Prime Intellect, or Lambda. Other GPU types have had compatibility issues.
- **Disk**: RunPod root overlay is 120GB and fills up fast with model weights. Use `/workspace` for everything large (model cache, outputs). Set `HF_HOME=/workspace/.cache/huggingface`. Disk size doesn't seem to be configurable via the PI API despite the parameter existing — ask the user to provision instances via the RunPod UI so disk can be set large enough.
- **Long-running processes**: Always use tmux. SSH disconnect sends SIGHUP and kills processes even with nohup.
- **Evals**: Always use `prime eval` on an instance with the prime_rl image. Manual eval code gives different results due to prompting/parsing. Comparisons are only valid when both use the same tool and settings.
