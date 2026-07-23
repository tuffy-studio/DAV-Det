# reference: https://github.com/visinf/INSID3/blob/main/utils/clustering.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.cluster import AgglomerativeClustering

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    gamma_pos: float = 2.0,
    gamma_neg: float = 2.0,
    mu: float = 0.0,
    reduction: str = "mean",
):
    """
    inputs: [B, N] logits
    targets: [B, N] 0/1
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
        return loss.mean(dim=1)  # [B]
    else:
        return loss

def init_weights(m):
    """
    初始化权重的通用函数：
    - 对 nn.Linear 层使用 Xavier uniform 初始化
    - 对 bias 使用 0 初始化
    """
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

class FlexibleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, drop_rates=None, activation_fn=nn.ReLU):
        """
        参数说明：
        - input_size: 输入特征维度，如 768
        - hidden_sizes: 隐藏层大小列表，如 [512, 256]
        - num_classes: 输出维度，如 1（二分类）
        - drop_rates: 每层的 Dropout 概率列表，如 [0.1, 0.1]（长度必须与 hidden_sizes 相同）
        - activation_fn: 激活函数类（默认是 nn.ReLU，可传 nn.LeakyReLU）
        """
        super(FlexibleMLP, self).__init__()

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()

        if drop_rates is None:
            drop_rates = [0.0] * len(hidden_sizes)
        assert len(drop_rates) == len(hidden_sizes), "drop_rates 和 hidden_sizes 长度不一致"

        prev_size = input_size
        for hidden_size, drop_rate in zip(hidden_sizes, drop_rates):
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropouts.append(nn.Dropout(drop_rate))
            self.activations.append(activation_fn())  # 动态创建激活函数实例
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, num_classes)
        self.apply(init_weights)

    def forward(self, x):
        for fc, drop, act in zip(self.layers, self.dropouts, self.activations):
            x = fc(x)
            x = drop(x)
            x = act(x)
        return self.output_layer(x)

class TokenWise_TokenReducer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, temperature=0.07, topk_ratio=0.05):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.temperature = temperature
        self.topk_ratio = topk_ratio
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, MIL_learning=False, reduction='mean', return_instance_logits=False):
        x = self.norm(x)
        token_logits = self.mlp(x)  # [B, N, 1]
        token_logits = torch.clamp(token_logits, min=-10, max=10)
        token_weights = F.softmax(token_logits.squeeze(-1) / self.temperature, dim=-1)  # [B, N]
        aggregated = torch.sum(token_weights.unsqueeze(-1) * x, dim=1)  # [B, D]

        B, N, _ = token_logits.shape
        k = max(1, int(self.topk_ratio * N))

        topk_vals, topk_idx = torch.topk(token_logits.squeeze(-1), k, dim=1)  # [B, k]
        mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, topk_idx, False)
        #res_logit = token_logits[mask].view(B, N - k).mean(dim=1)

        res_logit = token_logits.squeeze(-1).masked_fill(~mask, 0.0)
        denom = max(N - k, 1)
        res_logit = res_logit.sum(dim=1) / denom

        if reduction == 'mean':
            topk_logit = topk_vals.mean(dim=1)  # [B]
        else:
            raise ValueError(f"Unsupported reduction method: {reduction}")


        if MIL_learning:
            if return_instance_logits:
                return aggregated, topk_logit, res_logit, token_logits.squeeze(-1)
            else:
                return aggregated, topk_logit, res_logit
        else:   
            return aggregated


class Patch_Classifier_Reducer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, temperature=0.07, topk_ratio=0.05):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.temperature = temperature
        self.topk_ratio = topk_ratio
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, MIL_learning=False, reduction='mean', return_instance_logits=False):
        x = self.norm(x)
        patch_logits = self.mlp(x)  # [B, N, 1]
        patch_logits = torch.clamp(patch_logits, min=-10, max=10)
        patch_weights = F.softmax(patch_logits.squeeze(-1) / self.temperature, dim=-1)  # [B, N]
        aggregated = torch.sum(patch_weights.unsqueeze(-1) * x, dim=1)  # [B, D]

        B, N, _ = patch_logits.shape
        k = max(1, int(self.topk_ratio * N))

        topk_vals, topk_idx = torch.topk(patch_logits.squeeze(-1), k, dim=1)  # [B, k]
        mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, topk_idx, False)
        #res_logit = patch_logits[mask].view(B, N - k).mean(dim=1)

        res_logit = patch_logits.squeeze(-1).masked_fill(~mask, 0.0)
        denom = max(N - k, 1)
        res_logit = res_logit.sum(dim=1) / denom

        if reduction == 'mean':
            topk_logit = topk_vals.mean(dim=1)  # [B]
        else:
            raise ValueError(f"Unsupported reduction method: {reduction}")


        if MIL_learning:
            if return_instance_logits:
                return aggregated, topk_logit, res_logit, patch_logits.squeeze(-1)
            else:
                return aggregated, topk_logit, res_logit
        else:   
            return aggregated


class Segment_Classifier_Reducer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, temperature=0.07, topk_ratio=0.05):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.temperature = temperature
        self.topk_ratio = topk_ratio
        self.norm = nn.LayerNorm(input_dim)


    def forward(self, x, MIL_learning=False, reduction='mean', return_instance_logits=False):
        x = self.norm(x)
        segment_logits = self.mlp(x)  # [1, nums_segment, 1]
        segment_logits = torch.clamp(segment_logits, min=-10, max=10)
        segment_weights = F.softmax(segment_logits.squeeze(-1) / self.temperature, dim=-1)  # [1, nums_segment]
        aggregated = torch.sum(segment_weights.unsqueeze(-1) * x, dim=1)  # [1, D]

        B, N, _ = segment_logits.shape
        k = max(1, int(self.topk_ratio * N))

        topk_vals, topk_idx = torch.topk(segment_logits.squeeze(-1), k, dim=1)  # [1, k]
        mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, topk_idx, False)
        #res_logit = segment_logits[mask].view(B, N - k).mean(dim=1)

        res_logit = segment_logits.squeeze(-1).masked_fill(~mask, 0.0)
        denom = max(N - k, 1)
        res_logit = res_logit.sum(dim=1) / denom

        if reduction == 'mean':
            topk_logit = topk_vals.mean(dim=1)  # [1]
        else:
            raise ValueError(f"Unsupported reduction method: {reduction}")

        if MIL_learning:
            if return_instance_logits:
                return aggregated, topk_logit, res_logit, segment_logits.squeeze(-1)
            else:
                return aggregated, topk_logit, res_logit
        else:   
            return aggregated

def vit_patch_clustering(
    patch_tokens: torch.Tensor,
    tau: float = 0.9
):
    """
    Args:
        patch_tokens: (N, C) e.g. ViT-L output (1024, 1024)
        tau: similarity threshold

    Returns:
        labels: (N,) cluster id
        protos: (K, C) cluster prototypes
    """

    # --------------------------
    # 1. L2 normalize
    # --------------------------
    X = F.normalize(patch_tokens, p=2, dim=-1)

    # --------------------------
    # 2. cosine similarity
    # --------------------------
    S = X @ X.T                       # (N, N)
    S = S.clamp(-1, 1)

    # --------------------------
    # 3. convert to distance
    # --------------------------
    D = (1.0 - S).detach().cpu().numpy()

    # --------------------------
    # 4. agglomerative clustering
    # --------------------------
    ac = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - tau
    )

    labels = ac.fit_predict(D)
    labels = torch.from_numpy(labels).long()

    # --------------------------
    # 5. compute prototypes
    # --------------------------
    K = labels.max().item() + 1
    protos = []

    for k in range(K):
        idx = (labels == k)
        if idx.sum() > 0:
            mu = patch_tokens[idx].mean(dim=0)
        else:
            mu = patch_tokens[0]  # fallback

        mu = F.normalize(mu, p=2, dim=0)
        protos.append(mu.unsqueeze(0))

    protos = torch.cat(protos, dim=0)

    return labels, protos
