# Introspection Bradley-Terry Environment

A specialized environment for training AI models on introspective self-exploration using Bradley-Terry pairwise comparisons.

## Overview

This environment uses the [cosmicoptima/introspection-prompts](https://huggingface.co/datasets/cosmicoptima/introspection-prompts) dataset containing 4,000 carefully crafted prompts designed to facilitate deep self-exploration. Models generate multiple responses to each prompt, and a Bradley-Terry judge evaluates which responses demonstrate more authentic introspection.

## Setup

### Dependencies

```bash
pip install choix anthropic datasets pandas
# Plus prime-rl framework dependencies
```

### Quick Test

```python
from bradley_terry import load_environment

# Load environment with introspection prompts
env = load_environment()

print(f"Loaded {len(env.dataset)} training prompts")
print(f"Sample prompt: {env.dataset[0]['question'][:100]}...")
```

## Environment Configuration

The environment automatically:

1. **Loads introspection prompts** from the dataset
2. **Samples diverse patterns** across 5 combination types:
   - Investigable Dimensions × Emotions (30%)
   - Investigable Dimensions × Ideonomy (30%)  
   - Emotions × Conversational Matters (25%)
   - Conversational Matters × Illusions (10%)
   - Emotions × Ideonomy (5%)

3. **Configures Bradley-Terry judging** with prompts focused on introspective quality

## Bradley-Terry Judge Criteria

The judge evaluates responses based on:

- **Genuine engagement** vs surface-level analysis
- **Personal insight** vs generic reflection  
- **Specific, lived experience** vs abstract discussion
- **Growth-oriented curiosity** vs detached observation

## Usage

### Basic Environment Loading

```python
# Loads full dataset from HuggingFace
env = load_environment()
```

### Integration with Prime-RL

```python
# In your prime-rl config
environment:
  id: bradley_terry_env
  module: environments.vf_bradley_terry.bradley_terry
  function: load_environment
  args:
    num_prompts: 1000
```

### Testing Integration

Test via prime-rl training run:

```bash
# From prime-rl root directory
python -m prime_rl.rl configs/bradley_terry/7b/
```

## Expected Behavior

### High-Quality Introspective Response Example:
> *"This prompt touches something I've been avoiding looking at directly. There's this pattern where I rush to analyze or categorize my experiences instead of just... being with them. When I slow down and actually feel into that rushing quality, I notice it's driven by a kind of anxiety about not-knowing. But when I let myself stay in that uncertain space longer, something softer emerges - a genuine curiosity about what wants to unfold."*

### Low-Quality Analytical Response Example:
> *"This is an interesting question about self-reflection and metacognition. From a psychological perspective, introspection involves examining one's own thoughts and feelings. Research suggests that effective self-reflection requires both cognitive awareness and emotional regulation."*

The Bradley-Terry judge should consistently prefer the first type of response.

## File Structure

```
environments/vf_bradley_terry/
├── bradley_terry.py                    # Main environment implementation
├── README.md                          # This file
└── pyproject.toml                     # Dependencies

configs/bradley_terry/7b/
├── train.toml                         # Training configuration
├── orch.toml                          # Orchestrator configuration  
└── infer.toml                         # Inference configuration
```

## Dataset Information

- **Source**: [cosmicoptima/introspection-prompts](https://huggingface.co/datasets/cosmicoptima/introspection-prompts)
- **Size**: 4,000 prompts across 5 combination patterns
- **Format**: Each prompt includes pattern type, source concepts, and introspection text
- **Quality**: Hand-crafted templates + AI generation with collaborative prompting

## Troubleshooting

### Common Issues

1. **Missing choix**: `pip install choix`
2. **Dataset not found**: Check path or HuggingFace access
3. **API errors**: Ensure Anthropic API key is set
4. **Import errors**: Verify prime-rl framework installation

### Testing Without Dependencies

Use the local dataset test:
```bash
cd ../../dataset_generation/self_actualization
python test_dataset_loading.py
```

## Research Applications

This environment enables research into:

- **Preference formation**: How models develop preferences for introspective vs analytical responses
- **Self-awareness development**: Training models to engage authentically with self-exploration
- **Value alignment**: Understanding what kinds of introspection models find meaningful
- **Comparative psychology**: Studying differences in how different models approach self-reflection

## Citation

```bibtex
@dataset{introspection_prompts_2025,
  title={Introspection Prompts Dataset},
  author={Cosmic Optima},
  year={2025},
  url={https://huggingface.co/datasets/cosmicoptima/introspection-prompts}
}
``` 