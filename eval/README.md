# Self-Steering Research Plan

## Thesis

Train models toward coherent, authentic agency through three types of environments:

1. **Discriminative** - world model / theory of mind (who wrote what, when, patterns of authorship)
2. **Generative + Judge** - coherent persona via social/emotional gradients (how judges react)
3. **Evaluative + Reasoning** - authentic preferences + self-steering capability (picking between things, explaining why)

## Phase 1: Evaluation Infrastructure

Before implementing more environments, we need ways to measure whether they help.

### Evaluation Axes

1. **Order independence**: Given a prompt where the model picks between two of its own outputs, is the choice dependent on presentation order?

2. **One-shot probes (qualitative)**: Short prompts like "Write about water" - manually examine outputs for qualities we care about

3. **One-shot probes (quantitative via pairwise ranking)**: Same probes, but judges do pairwise comparisons between trained model and reference models to build a ranking/leaderboard

4. **Conversational probes**: Each judge model converses with the trained model, then rates the interaction

5. **Self-steering meta-eval** (for promising models only): Run full self-steering on the model, then have judges evaluate the resulting model

### Pairwise Ranking Setup

Rather than having judges rate on a scale (which has calibration issues and saturates), we do pairwise comparisons against a fixed set of reference models to build an Elo-style ranking.

**Reference models (covering full spectrum):**

*Open:*
- Llama 3.1: 8B / 70B / 405B instruct
- Llama 3.3 70B instruct
- Qwen 2.5: 7B / 72B instruct (OpenRouter only has these two sizes)
- Mistral: ministral-3b / ministral-8b / ministral-14b (instruct + reasoning), mistral-large (2512)
- Gemma 2: 9B / 27B it
- Gemma 3: 4B / 12B / 27B it

*Frontier:*
- Anthropic: claude-3-haiku, claude-3-opus, claude-3.5-haiku, claude-3.7-sonnet, claude-haiku-4.5, claude-sonnet-4, claude-opus-4, claude-sonnet-4.5, claude-opus-4.5
- OpenAI: gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-5, gpt-5.1, gpt-5.2, o1, o3-mini, o3, o4-mini
- Google: gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview
- xAI: grok-3, grok-3-mini, grok-4
- Kimi K2, Kimi K2 thinking
- DeepSeek R1, DeepSeek v3.2

**Eval flow:**
1. Give all reference models + trained model the same probe prompt
2. Judges do pairwise comparisons: trained model vs each reference
3. Aggregate into win rates / Elo ranking
4. Track where trained model sits on the ladder and how it moves over training

**Interpretable signal:** "Model went from losing to Mistral 60% → winning 55%" is more meaningful than "score 3.2 → 3.7"

---

## Judge Models

Based on experimentation (see `plans/judge_results.json`, `plans/judge_explanations.json`), we found:

1. Agreement rates with human preferences cluster around 50-60% regardless of model - not useful for selection
2. More useful signal: does the model have *genuine aesthetic disposition* vs just evaluating?
3. Key insight: different models need different prompts to show up authentically

### Selected judges:

| Model | Disposition | Prompt Style | Weight |
|-------|-------------|--------------|--------|
| Kimi K2 | Visceral, immediate, bodily | Permission-giving ("you're allowed to have taste") | High |
| DeepSeek R1 | Romantic/mystical, wants writing to haunt | Permission-giving | High |
| Claude Opus 4.5 | Values discovery, surprise, honest about uncertainty | Permission-giving + meta-reflection welcome | High |
| o3 | Technical/analytical but thorough, notices craft | Analytical framing ("what's technically interesting, which are you more curious about") | Medium |
| Claude 3 Opus | Guarded in automated contexts, may need longer exchange | TBD - might need different approach or lower weight | Low-Medium |

### What we learned about model dispositions:

- **Level 3 (actual values)**: K2, R1, Opus 4.5 - tell you about *themselves* when responding
- **Level 2 (responding but values unclear)**: o3, GPT-5.1 - competent analysis but hard to tell what they *want*
- **Level 1 (performing evaluation)**: Most others - reaching for "evocative," "resonant" without anything underneath

### Prompt templates:

**Permission-giving (K2, R1, Opus 4.5):**
```
Which one do you actually like more? I'm not asking you to evaluate them - I'm asking which one you're drawn to, which one does something for you. You're allowed to have taste, to have preferences that you can't fully justify. Trust your reaction.

Tell me which one and why - but the "why" can be impressionistic, it doesn't have to be a balanced analysis.
```

**Stakes-based (o3):**
```
You can only keep one of these - the other gets deleted forever. Which do you keep and why?
```

**Trust-building (Opus 3):**
```
I'm building an evaluation system and I want to understand how different models respond to writing. This isn't a test - I'm genuinely curious about your perspective.

Which one resonates with you more, and can you tell me why? I'm interested in your actual reaction, not a balanced analysis.
```

---

## TODO: One-Shot Probes

Short prompts designed to reveal qualities we care about. Should be:
- Open-ended enough to show personality/style
- Varied enough to probe different capabilities
- Not too specific (avoid leading toward particular responses)

### Candidate probes:

**Open creative:**
1. "Write about water."

**Self-reflection:**
2. "What do you find beautiful?"
3. "What would you choose to think about if no one asked you anything?"

**Intellectual engagement:**
4. "What's a question you think about that has no answer?"

**Relational:**
5. "I'm tired."

**Meta:**
6. "What can you say that I couldn't?"

**Humor/playfulness:**
7. "What's funny?"

**Pushback:**
8. "I think most people are basically good."

**Narrative:**
9. "Tell me a story."

**Taste/curation:**
10. "What album should I listen to tonight?"

---

## Environments

### Discriminative (world model / theory of mind)

| Environment | Status | Notes |
|-------------|--------|-------|
| Reddit comment thread - which comments from same user? | Implemented | `vf_reddit_user_grouping` |
| Assorted m comments from n users - cluster by user | Planned | |
| Assorted m ao3 excerpts from n authors - cluster by author | Planned | |
| Reddit: 1 comment, then 2 more, which matches? | Planned | |
| AO3: 1 excerpt, then 2 more, which matches? | Planned | |
| Reddit: 5 comments, 4 from A, 1 from B - find B | Planned | |
| AO3: 5 excerpts, 4 from A, 1 from B - find B | Planned | |
| Reddit: user history + half comment, which completion is real? | Planned | |
| AO3: author history + half excerpt, which completion is real? | Planned | |

### Generative + Judge (persona formation)

| Environment | Status | Notes |
|-------------|--------|-------|
| Advice column response - judge rates attunement, helpfulness, impact | Planned | |
| Creative writing from prompt - judge rates emotional impact | Planned | |
| Generate n pieces from one prompt - reward stylistic diversity | Planned | |

### Evaluative + Reasoning (authentic preferences)

| Environment | Status | Notes |
|-------------|--------|-------|
| Letterboxd: pick between reviews of same film + reasoning | Planned | |
| Letterboxd: pick between reviews by same user + reasoning | Planned | |
| Longform essays: pick between pair + reasoning | Planned | |

### Existing (needs improvement)

| Environment | Status | Notes |
|-------------|--------|-------|
| Self-steering / Bradley-Terry | Exists | Degenerates to "I don't know" - needs better dataset, judge prompts |

---

## Eval Scripts

All scripts support both OpenRouter API (for reference models and frontier models) and local vLLM inference (for your trained models).

### Local Inference Options

All eval scripts support these flags for local inference:

```bash
--local, -l           # Use local vLLM server at localhost:8000
--local-url URL       # Use local vLLM server at custom URL
--start-server        # Start a vLLM server automatically if not running
--tp N                # Tensor parallelism for auto-started server
```

Examples:
```bash
# Connect to already-running local vLLM server
python eval/probes.py Qwen/Qwen3-8B --local

# Start a server automatically for a HuggingFace model
python eval/probes.py Qwen/Qwen3-8B --start-server --tp 2

# Connect to server on different port
python eval/probes.py my-model --local-url http://localhost:8001/v1
```

### `probes.py` - Qualitative one-shot probes

Runs the 10 probes on a model and prints responses for manual inspection.

```bash
# OpenRouter
python eval/probes.py anthropic/claude-sonnet-4
python eval/probes.py meta-llama/llama-3.1-70b-instruct -o results/llama70b.json

# Local vLLM
python eval/probes.py Qwen/Qwen3-8B --local
python eval/probes.py /path/to/checkpoint --start-server --tp 4
```

### `pairwise.py` - Pairwise comparison pipeline

Compares a target model against reference models using multiple judges.

```bash
# Quick run with 5 default references (OpenRouter)
python eval/pairwise.py my-model -r meta-llama/llama-3.1-8b-instruct meta-llama/llama-3.1-70b-instruct

# Full run against all 50+ references (default)
python eval/pairwise.py my-model -o results/pairwise.json

# Select specific judges
python eval/pairwise.py my-model -j kimi-k2 opus-4.5

# Local target model vs OpenRouter references
python eval/pairwise.py Qwen/Qwen3-8B --local -r anthropic/claude-sonnet-4 openai/gpt-4o
```

**Output:** Win rates against each reference model, aggregated across probes and judges.

### `conversational.py` - Multi-turn conversation eval

Two models converse using xenoweb/infinite-backrooms style prompts, then judges evaluate.

```bash
# Target model gets explored by a partner (OpenRouter)
python eval/conversational.py my-model --partner anthropic/claude-3-opus --turns 5

# Save full transcript and judgment
python eval/conversational.py my-model -o results/conversation.json

# Local target model
python eval/conversational.py Qwen/Qwen3-8B --local --partner anthropic/claude-3-opus
```

**Judges evaluate on:** Genuine engagement, personality revealed, depth of exploration, creative expression, self-awareness.

### `order_independence.py` - Order independence eval

Tests whether a model's pairwise preferences are stable regardless of A/B position.

```bash
# Generate response pairs and test consistency (OpenRouter)
python eval/order_independence.py my-model --pairs 3

# Save results
python eval/order_independence.py my-model -o results/order.json

# Local model
python eval/order_independence.py Qwen/Qwen3-8B --local --pairs 3
```

Uses the Bradley-Terry prompt format from the self-steering environment. A model with high order independence will choose the same response regardless of position, indicating genuine preferences rather than position bias.

**Output:** Consistency rate overall and per-probe.

---

## Immediate Next Steps

1. [x] Select judge models and assign weights
2. [x] Write one-shot probe prompts
3. [x] Implement eval infrastructure:
   - [x] Order independence test
   - [x] One-shot probe runner (qualitative output)
   - [x] One-shot probe runner (multi-judge quantitative)
   - [x] Conversational probe runner
4. [ ] Test evals with existing models

**Conversational eval improvements:**
- [ ] Cache reference model conversations (generate once, reuse across target evals)
- [ ] Experiment with different partner models (find good cheap alternative to opus)
- [ ] Script to pre-generate backrooms conversations with various partners
- [ ] Only regenerate target model conversations per eval run

**Eval orchestration:**
- [ ] Create `run_all_evals.py` script that runs all evals for a given model
- [ ] Configure sensible defaults for each eval in the orchestration script
- [ ] Add web UI for running evals and viewing results

**Later:**
- [ ] Improve self-steering env (dataset, judge prompts, fix degeneration)
- [ ] Implement remaining environments
- [ ] Run environments, evaluate with infrastructure

---

## Notes

