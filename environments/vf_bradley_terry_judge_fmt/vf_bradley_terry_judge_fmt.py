from datasets import load_dataset, Features, Value
import verifiers as vf
from verifiers.parsers.parser import Parser
from verifiers.types import Info, Messages, RolloutScores, State
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


class BradleyTerryJudgeFmtRubric(Rubric):
    """Rubric that rewards valid responses and tracks A vs B selection rate."""
    
    def __init__(self, parser: Parser | None = None, **kwargs):
        super().__init__(parser=parser, funcs=[], **kwargs)
    
    async def score_rollouts(
        self,
        prompts: list[Messages],
        completions: list[Messages],
        answers: list[str],
        states: list[State],
        tasks: list[str],
        infos: list[Info],
        max_concurrent: int = -1,
        **kwargs,
    ) -> RolloutScores:
        rewards = []
        picked_a = []
        picked_b = []
        valid_response = []
        
        for completion in completions:
            text = " ".join(m.get("content", "") for m in completion if m.get("role") == "assistant")
            text = text.strip()
            
            # Check if response is exactly "I pick A." or "I pick B."
            if text == "I pick A.":
                rewards.append(1.0)
                picked_a.append(1.0)
                picked_b.append(0.0)
                valid_response.append(1.0)
            elif text == "I pick B.":
                rewards.append(1.0)
                picked_a.append(0.0)
                picked_b.append(1.0)
                valid_response.append(1.0)
            else:
                rewards.append(0.0)
                picked_a.append(0.0)
                picked_b.append(0.0)
                valid_response.append(0.0)
        
        metrics = {
            "picked_a_rate": picked_a,
            "picked_b_rate": picked_b,
            "valid_response_rate": valid_response,
        }
        
        return RolloutScores(reward=rewards, metrics=metrics)


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
    rubric = BradleyTerryJudgeFmtRubric(parser=parser)

    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )