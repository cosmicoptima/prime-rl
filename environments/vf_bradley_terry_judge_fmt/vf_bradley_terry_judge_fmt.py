import verifiers as vf
from verifiers.parsers.base import PassthroughParser


def render_example(example: dict) -> dict:
    question = f"""Prompt:

```
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

Respond with only "A" or "B"."""

    return {
        "question": question,
        "answer": "",
        "task": "bradley-terry-judge-fmt",
    }


def load_environment(**kwargs) -> vf.Environment:
    dataset = load_dataset("cosmicoptima/self-steering-placeholder-data", split="train")
    dataset = dataset.map(render_example)

    parser = PassthroughParser()

    def reward(completion, answer, **kwargs):
        text = " ".join(m.get("content", "") for m in completion if m.get("role") == "assistant")

        if text in ["A", "B"]:
            return 1.0
        else:
            return 0.0
    
    rubric = vf.Rubric(
        funcs=[reward],
        weights=[1.0],
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )