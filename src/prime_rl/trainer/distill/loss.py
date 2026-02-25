import torch
import torch.nn.functional as F
from jaxtyping import Float
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

    Args:
        student_logits: Logits from the model without context, at response token positions.
        teacher_logits: Logits from the model with context, at response token positions.
        temperature: Softening temperature. Higher values provide more gradient signal from
                     low-probability tokens. Default 1.0 (no softening).
    """
    # Apply temperature scaling
    teacher_logits = teacher_logits / temperature
    student_logits = student_logits / temperature

    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()

    # KL(teacher || student) = sum_v P(v) * (log P(v) - log Q(v))
    kl_per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)

    loss = kl_per_token.mean()

    # Scale loss by T^2 when using temperature (standard distillation trick from Hinton et al.)
    if temperature != 1.0:
        loss = loss * (temperature**2)

    metrics = {
        "kl_mean": kl_per_token.mean().item(),
        "kl_max": kl_per_token.max().item(),
        "teacher_entropy": -(teacher_probs * teacher_log_probs).sum(dim=-1).mean().item(),
        "student_entropy": -(student_log_probs.exp() * student_log_probs).sum(dim=-1).mean().item(),
    }

    return loss, metrics
