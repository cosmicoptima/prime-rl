import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


def compute_distill_loss(
    student_logits: Float[Tensor, "tokens vocab"],
    teacher_logits: Float[Tensor, "tokens vocab"],
    temperature: float = 1.0,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """
    Compute forward KL divergence: KL(teacher || student).

    Both inputs should be pre-aligned response-token logits (extracted via loss masks).
    Forward KL is mode-covering: forces the student to assign probability wherever the teacher does.
    """
    teacher_logits = teacher_logits / temperature
    student_logits = student_logits / temperature

    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()

    kl_per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    loss = kl_per_token.mean()

    if temperature != 1.0:
        loss = loss * (temperature**2)

    metrics = {
        "kl_mean": kl_per_token.mean().item(),
        "kl_max": kl_per_token.max().item(),
        "teacher_entropy": -(teacher_probs * teacher_log_probs).sum(dim=-1).mean().item(),
        "student_entropy": -(student_log_probs.exp() * student_log_probs).sum(dim=-1).mean().item(),
    }

    return loss, metrics


def compute_distill_loss_frozen(
    student_logits: Float[Tensor, "tokens vocab"],
    teacher_logprobs: Float[Tensor, "tokens n_samples"],
    teacher_indices: Int[Tensor, "tokens n_samples"],
    vocab_size: int,
    temperature: float = 1.0,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """
    Compute forward KL divergence against frozen (pre-computed) teacher logprobs.

    Uses randomly sampled vocab positions from the teacher. The KL is computed only
    at the sampled positions and scaled by vocab_size/n_samples for an unbiased
    gradient estimate.

    Args:
        student_logits: Full student logits at response token positions. [N, V]
        teacher_logprobs: Pre-computed teacher log-probs at sampled positions. [N, K]
        teacher_indices: Which vocab indices were sampled. [N, K]
        vocab_size: Full vocabulary size (for scaling).
        temperature: Softening temperature.
    """
    student_logits = student_logits / temperature

    # Get student log-probs at the sampled positions
    student_log_probs_sampled = torch.gather(
        F.log_softmax(student_logits, dim=-1),
        dim=-1,
        index=teacher_indices.long(),
    )  # [N, K]

    # Teacher probs at sampled positions
    teacher_logprobs_scaled = teacher_logprobs.float() / temperature
    teacher_probs_sampled = teacher_logprobs_scaled.exp()

    # KL at sampled positions: P * (log P - log Q)
    kl_sampled = (teacher_probs_sampled * (teacher_logprobs_scaled - student_log_probs_sampled)).sum(dim=-1)

    # Scale by vocab_size / n_samples for unbiased estimate
    n_samples = teacher_indices.shape[-1]
    scale = vocab_size / n_samples
    kl_per_token = kl_sampled * scale

    loss = kl_per_token.mean()

    if temperature != 1.0:
        loss = loss * (temperature**2)

    metrics = {
        "kl_mean": kl_per_token.mean().item(),
        "kl_max": kl_per_token.max().item(),
    }

    return loss, metrics
