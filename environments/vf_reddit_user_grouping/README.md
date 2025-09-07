# reddit-user-grouping

### Overview
- **Environment ID**: `reddit-user-grouping`
- **Short description**: Group comments in a single Reddit comment thread by author. Usernames are redacted

### Datasets
- **Primary dataset(s)**: cosmicoptima/IFhXR5QAHNW9 - Reddit posts with comment threads and known user mappings
- **Source links**: [HuggingFace Dataset](https://huggingface.co/datasets/cosmicoptima/IFhXR5QAHNW9)
- **Preprocessing**: Filtered to examples ≤2076 tokens to fit model context window. This should be manually adjusted as necessary

### Task
- **Type**: single-turn
- **Parser**: `UserDiffParser` - extracts grouped comment indices from answer inside &lt;answer&gt; tag
- **Rubric overview**: F1 scoring between predicted and actual user groupings, with power=4 scaling

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval reddit-user-grouping
```

Configure model and sampling:

```bash
uv run vf-eval reddit-user-grouping   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Reports are written under `./environments/vf_reddit_user_grouping/reports/` and auto-embedded below.

### Environment Arguments
Standard verifiers environment arguments supported:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
The environment uses a sophisticated reward function with multiple validation steps:

| Metric | Meaning |
| ------ | ------- |
| `reward` | F1-weighted group matching score with power=4 scaling (0.0-1.0) |
| `format_reward` | Binary reward for properly formatted XML response (0.0 or 1.0) |

**Reward calculation details:**
- Validates all comments are assigned exactly once (no duplicates/missing)
- Computes F1 score between each actual user group and best-matching predicted group
- Weights each group's contribution by its size (comment count / total comments)
- Applies power=4 scaling: `reward = weighted_f1_score^4`

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval reddit-user-grouping -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->