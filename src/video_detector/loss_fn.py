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

def mil_margin_loss(topk_mean, rest_mean, labels, margin=1.0):
    pos_mask = (labels.long() == 1)
    if not pos_mask.any():
        return torch.tensor(0.0, device=topk_mean.device, dtype=torch.float32, requires_grad=True)

    # 1. 强力清理输入：将 NaN 转为 0，将 Inf 转为有限的大数
    # 这是防止任何 Backward NaN 的最后防线
    t = torch.nan_to_num(topk_mean.float().squeeze()[pos_mask], nan=0.0, posinf=10.0, neginf=-10.0)
    r = torch.nan_to_num(rest_mean.float().squeeze()[pos_mask], nan=0.0, posinf=10.0, neginf=-10.0)

    # 2. 计算差值并再次 clamp
    diff = t - r
    
    # 3. 使用线性 ReLU 替代任何指数运算（Sigmoid/Softplus）
    # ReLU 的梯度要么是 0 要么是 1，绝对不可能产生 NaN
    loss = torch.clamp(margin - diff, min=0).mean()
    
    return loss