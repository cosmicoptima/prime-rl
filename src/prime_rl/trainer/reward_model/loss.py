import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def compute_bt_loss(
    chosen_rewards: Float[Tensor, "batch"],
    rejected_rewards: Float[Tensor, "batch"],
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """
    Compute Bradley-Terry preference loss.

    loss = -log(sigmoid(r_chosen - r_rejected))

    Args:
        chosen_rewards: Scalar rewards for preferred responses.
        rejected_rewards: Scalar rewards for rejected responses.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    reward_diff = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(reward_diff).mean()

    with torch.no_grad():
        accuracy = (reward_diff > 0).float().mean().item()
        mean_chosen = chosen_rewards.mean().item()
        mean_rejected = rejected_rewards.mean().item()
        mean_margin = reward_diff.mean().item()

    metrics = {
        "accuracy": accuracy,
        "mean_chosen_reward": mean_chosen,
        "mean_rejected_reward": mean_rejected,
        "mean_margin": mean_margin,
    }

    return loss, metrics
