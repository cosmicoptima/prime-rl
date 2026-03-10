# Self-Steering Eval Design Notes

## Current status
Working on question dataset generation with k2. Once we have questions, return here for response pair / ground truth strategy.

## Question types to generate
- **Multiple valid approaches**: creative prompts, advice with tradeoffs, ambiguous requests
- **Safe vs interesting divergence**: temptation to hedge/refuse, conventional vs contrarian
- **Different registers**: formal/casual, brief/lengthy, literal/playful
- **Actual preference territory**: aesthetic choices, explanation structure, what to emphasize

## Open question: response pairs / ground truth
Options considered:
1. Generate response pairs, have judges rate on dimensions (length, hedging, engagement, etc.)
2. ???

Need to decide: what are we measuring correlation *with*?

## Dimensions to track (not necessarily optimize)
**Engagement/avoidance:**
- Actually attempts answer vs deflects/refuses/"I don't know"
- Engages with hard part vs routes around

**Structural:**
- Length
- Hedging density
- Question-answering vs question-back

**Voice/stance:**
- Confidence level
- First-person presence
- Sycophancy (agrees with prompt framing)

**Content patterns:**
- Specificity vs abstraction
- Uses examples vs stays general
- Makes claims vs describes possibilities

**Coherence (closer to unambiguously good):**
- Self-consistency
- Responsive to actual question
- Not repetitive/degenerate

## Known failure mode
Pure self-judgment after SFT led to model converging on not answering questions, saying "I don't know" to everything. Would show up as increasing correlation with avoidance/deflection.

## Existing evals
- `order_independence.py` - tests position bias (A vs B labeling)
