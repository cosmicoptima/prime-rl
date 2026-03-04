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
    teacher_logprobs: Float[Tensor, "tokens k"],
    teacher_indices: Int[Tensor, "tokens k"],
    vocab_size: int,
    temperature: float = 1.0,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """
    Compute forward KL divergence against frozen (pre-computed) teacher logprobs.

    Uses top-K teacher logprobs which capture essentially all probability mass.

    Args:
        student_logits: Full student logits at response token positions. [N, V]
        teacher_logprobs: Pre-computed teacher top-K log-probs. [N, K]
        teacher_indices: Which vocab indices correspond to the top-K. [N, K]
        vocab_size: Full vocabulary size (unused with top-K, kept for API compat).
        temperature: Softening temperature.
    """
    student_logits = student_logits / temperature

    # Get student log-probs at the top-K positions
    student_log_probs_at_topk = torch.gather(
        F.log_softmax(student_logits, dim=-1),
        dim=-1,
        index=teacher_indices.long(),
    )  # [N, K]

    # Teacher probs at top-K positions (renormalize within top-K)
    teacher_logprobs_scaled = teacher_logprobs.float() / temperature
    teacher_probs_topk = F.softmax(teacher_logprobs_scaled, dim=-1)

    # KL at top-K positions: P * (log P - log Q)
    teacher_log_probs_topk = F.log_softmax(teacher_logprobs_scaled, dim=-1)
    kl_per_token = (teacher_probs_topk * (teacher_log_probs_topk - student_log_probs_at_topk)).sum(dim=-1)

    loss = kl_per_token.mean()

    if temperature != 1.0:
        loss = loss * (temperature**2)

    metrics = {
        "kl_mean": kl_per_token.mean().item(),
        "kl_max": kl_per_token.max().item(),
    }

    return loss, metrics
