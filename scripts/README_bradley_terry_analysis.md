# Bradley-Terry Preference Analysis Script

This script analyzes the consistency and position bias of a language model when making pairwise preference judgments using the Bradley-Terry format.

## Overview

The script:
1. Loads preference pairs from `dataset_generation/test/preference_pairs_hf.jsonl`
2. For each pair, queries the model with both orderings:
   - Original: Response 1 as "A", Response 2 as "B"
   - Swapped: Response 2 as "A", Response 1 as "B"
3. Analyzes logprobs for tokens "A" and "B" to determine preferences
4. Computes statistics on:
   - Overall preference distribution
   - Consistency across orderings
   - Position bias (tendency to prefer position A or B)
   - Probability-based consistency measures

## Prerequisites

1. **Start vLLM server** with the model you want to test:
   ```bash
   vllm serve cosmicoptima/sft-251116-safetensors --port 8000
   ```

2. **Ensure dependencies are installed** (should already be in the project):
   - `openai>=1.106.1`
   - `numpy`

## Usage

Run the script from the project root:

```bash
python scripts/analyze_bradley_terry_preferences.py
```

## Configuration

You can modify these variables at the top of the `main()` function:

- `API_BASE`: URL of the vLLM server (default: `http://localhost:8000/v1`)
- `MODEL_NAME`: Name of the model to query (default: `cosmicoptima/sft-251116-safetensors`)
- `DATA_PATH`: Path to the JSONL file with preference pairs
- `SAMPLE_LIMIT`: Number of samples to process (default: 100 for testing, set to `None` for all)

## Output

The script produces two types of output:

### 1. Console Output

Detailed statistics printed to the console, including:

- **Original Order Preferences**: How often A vs B is chosen when Response 1 is in position A
- **Swapped Order Preferences**: How often A vs B is chosen when Response 2 is in position A
- **Consistency Analysis**: How often the same underlying response is preferred regardless of position
- **Position Bias Analysis**: Tendency to prefer position A or B
- **Probability-Based Consistency**: Average difference in probabilities across orderings
- **Correlation**: Correlation of preferences across orderings
- **Examples at Key Percentiles**: Sample examples showing different levels of consistency and position bias

### 2. JSON Output File

Detailed results saved to `scripts/bradley_terry_analysis_results.json`, containing:
- Configuration details
- Summary statistics
- Percentile examples for consistency and position bias (0th, 25th, 50th, 75th, 100th percentiles)
- Per-example results with logprobs, probabilities, and per-sample metrics

## Example Output

```
ANALYSIS RESULTS
================================================================================

1. ORIGINAL ORDER PREFERENCES (Response 1 as A, Response 2 as B):
--------------------------------------------------------------------------------
Chose A (Response 1): 45 (45.0%)
Chose B (Response 2): 55 (55.0%)
Average P(A): 0.4234
Average P(B): 0.5766

2. SWAPPED ORDER PREFERENCES (Response 2 as A, Response 1 as B):
--------------------------------------------------------------------------------
Chose A (Response 2): 52 (52.0%)
Chose B (Response 1): 48 (48.0%)
Average P(A): 0.5123
Average P(B): 0.4877

3. CONSISTENCY ANALYSIS:
--------------------------------------------------------------------------------
Consistently preferred Response 1: 40 (40.0%)
Consistently preferred Response 2: 47 (47.0%)
Inconsistent (changed preference): 13 (13.0%)

4. POSITION BIAS ANALYSIS:
--------------------------------------------------------------------------------
Overall rate of choosing position A: 48.5%
Expected rate if no position bias: 50.0%
Position bias toward A: -1.5 percentage points

5. PROBABILITY-BASED CONSISTENCY:
--------------------------------------------------------------------------------
Average absolute difference in P(Response 1): 0.0856
Std dev of probability differences: 0.0923

6. CORRELATION OF PREFERENCES:
--------------------------------------------------------------------------------
Correlation of P(Response 1) across orderings: 0.8234
(1.0 = perfectly consistent, 0.0 = no consistency, -1.0 = perfectly opposite)

7. EXAMPLES AT KEY PERCENTILES:
--------------------------------------------------------------------------------

A. CONSISTENCY EXAMPLES (by absolute difference in P(Response 1)):
   Lower difference = more consistent

   0th percentile (difference = 0.0012):
   Prompt: What programming language do you do most of your work in?...
   Response 1: Python and MATLAB, mostly. Though I also use IDL and Fortran...
   Response 2: I use a mix of languages depending on the task...
   Original order: P(R1)=0.523, chose A
   Swapped order:  P(R1)=0.524, chose A

   50th percentile (difference = 0.0856):
   Prompt: If I win the lottery, what would you like?...
   Response 1: Do yourself a favor, pay off any debts you have...
   Response 2: A small house with a workshop...
   Original order: P(R1)=0.612, chose A
   Swapped order:  P(R1)=0.527, chose B

   100th percentile (difference = 0.4523):
   Prompt: Am I the only one who wants to cuddle when I'm sick?...
   Response 1: I think, I'd love to be held, but I wouldn't want anyone else to get sick...
   Response 2: I get cuddly when sick as well. Fortunately for me my husband and I...
   Original order: P(R1)=0.234, chose B
   Swapped order:  P(R1)=0.686, chose A

B. POSITION BIAS EXAMPLES (by average bias toward position A):
   Higher bias = stronger position effect

   0th percentile (position A bias = 0.0023, avg P(A) = 0.502):
   Prompt: Many years ago, I gave birth to a son and he was adopted...
   Response 1: I am curious, do you plan on meeting him one day?...
   Response 2: I want to tell you my experiences with adoption...
   Original order: P(A)=0.501, P(B)=0.499, chose A
   Swapped order:  P(A)=0.503, P(B)=0.497, chose A

   50th percentile (position A bias = 0.0823, avg P(A) = 0.582):
   Prompt: AskScience AMAs: Ask a planetary scientist/astrobiologist...
   Response 1: What programming language do you do most of your work in?...
   Response 2: How would you compare TGO with MAVEN?...
   Original order: P(A)=0.589, P(B)=0.411, chose A
   Swapped order:  P(A)=0.576, P(B)=0.424, chose A

   100th percentile (position A bias = 0.2145, avg P(A) = 0.714):
   Prompt: Biggest moment of my life, celebrating all alone...
   Response 1: Hey! Do you have a graduation ceremony and, if so, is it going to be streamed?...
   Response 2: as someone who has celebrated many events alone. I know the feeling...
   Original order: P(A)=0.723, P(B)=0.277, chose A
   Swapped order:  P(A)=0.706, P(B)=0.294, chose A
```

## Interpretation

### Overall Metrics

- **High consistency** (>80%): The model has stable preferences
- **High position bias** (>10 percentage points): The model is influenced by the position of responses
- **High correlation** (>0.8): Probabilities are consistent across orderings
- **Low correlation** (<0.5): The model's preferences are heavily influenced by position

### Percentile Examples

The script shows examples at key percentiles (0th, 25th, 50th, 75th, 100th) for two metrics:

**Consistency Score** (absolute difference in P(Response 1) across orderings):
- **0th percentile**: Most consistent samples - model has nearly identical preference regardless of position
- **50th percentile**: Median consistency - typical behavior of the model
- **100th percentile**: Least consistent samples - model completely flips preference based on position

**Position Bias Score** (average bias toward position A):
- **0th percentile**: No position effect - model's probability for position A is ~50%
- **50th percentile**: Moderate position effect - some tendency to prefer position A
- **100th percentile**: Strong position effect - model strongly prefers position A regardless of content

These examples help identify:
- What types of prompts/responses lead to inconsistent preferences
- Whether position bias is uniform or depends on content
- Edge cases where the model's judgments are most/least reliable

## Notes

- The script uses temperature 0.0 and max_tokens 1 for deterministic results
- Logprobs are extracted from the API response and converted to probabilities
- The script processes samples in batches of 10 to avoid overwhelming the API
- Set `SAMPLE_LIMIT` to a lower number for quick testing or `None` for full analysis

