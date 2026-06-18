import torch
import torch.nn.functional as F

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.6,
    gamma_pos: float = 2.0,
    gamma_neg: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Sigmoid focal loss.
    """
    inputs = inputs.float()
    targets = targets.float()

    p = torch.sigmoid(inputs)

    pos_loss = -(1 - p) ** gamma_pos * torch.log(p.clamp(min=1e-8))
    neg_loss = -(p) ** gamma_neg * torch.log((1 - p).clamp(min=1e-8))

    loss = targets * pos_loss + (1 - targets) * neg_loss

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss