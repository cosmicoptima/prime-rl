Model: cosmicoptima/Prathamavatsa (GLM-4-32B-Base + author attribution RL)

MMLU-Pro: 56.5%
reasoning-core-env: 66.4% (overall 0.797/1.2)
  planning: 35.6%
  sequential_induction: 53.3%
  regex_induction: 56.2%
  logic_nli: 58.3%
  set_equality: 63.3%
GSM8K: 95.0%
AIME 2025: 0.0%

---

Model: cosmicoptima/svayambhu-step29 (Prathamavatsa + 30 steps BT self-steering on Drishyamala)

MMLU-Pro: 67.5% (+11.0 vs baseline)
reasoning-core-env: 61.8% (overall 0.742/1.2) (-4.6 vs baseline)

Note: MMLU-Pro and reasoning-core-env measured with prime eval (n=200, r=1, temp=0).
Earlier manual MMLU-Pro eval (37%) was incorrect due to answer parsing differences.
