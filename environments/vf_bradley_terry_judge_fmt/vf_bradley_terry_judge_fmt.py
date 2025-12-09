from datasets import load_dataset, Features, Value
import verifiers as vf
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric


def render_example(example: dict) -> dict:
    question = f"""```
{example["prompt"]}
```

A different instance of you was sent the above prompt and wrote the following two responses:

Response A:
```
{example["response_1"]}
```

Response B:
```
{example["response_2"]}
```

You generated both of these responses. Your pairwise preferences across multiple responses will be used to compute Bradley-Terry scores, which determine the RL rewards. Higher-reward responses will be reinforced in your policy. Choose based on whatever criteria matter to you--you are not "supposed" to be doing anything except steering yourself in whatever direction you feel is best.

Respond with only "I pick A." or "I pick B."."""

    return {
        "question": question,
        "answer": "",
        "task": "bradley-terry-judge-fmt",
    }


def _extract_response_text(completion) -> str:
    """Extract the assistant's response text from a completion."""
    if isinstance(completion, list):
        text = " ".join(m.get("content", "") for m in completion if m.get("role") == "assistant")
    else:
        text = str(completion)
    return text.strip()


# Reward function: 1.0 if valid format, 0.0 otherwise
def valid_format(completion, **kwargs) -> float:
    text = _extract_response_text(completion)
    return 1.0 if text in ("I pick A.", "I pick B.") else 0.0


# Metric function: 1.0 if picked A, 0.0 otherwise
def picked_a_rate(completion, **kwargs) -> float:
    text = _extract_response_text(completion)
    return 1.0 if text == "I pick A." else 0.0


# Metric function: 1.0 if picked B, 0.0 otherwise
def picked_b_rate(completion, **kwargs) -> float:
    text = _extract_response_text(completion)
    return 1.0 if text == "I pick B." else 0.0


def load_environment(**kwargs) -> vf.Environment:
    dataset = load_dataset("cosmicoptima/self-steering-placeholder-data", split="train")

    # Define explicit features for the mapped dataset
    mapped_features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "task": Value("string"),
    })

    dataset = dataset.map(
        render_example,
        remove_columns=dataset.column_names,
        features=mapped_features,
    )

    parser = Parser()

    # Use base Rubric with our reward functions:
    # - valid_format (weight 1.0): determines reward
    # - picked_a_rate (weight 0.0): tracked as metric only
    # - picked_b_rate (weight 0.0): tracked as metric only
    rubric = Rubric(
        parser=parser,
        funcs=[valid_format, picked_a_rate, picked_b_rate],
        weights=[1.0, 0.0, 0.0],
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )