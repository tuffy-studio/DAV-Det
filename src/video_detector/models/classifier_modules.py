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
    
    def mask2patch(self, mask, patch_size=16, threshold=0.5):
        """
        mask: [B, H, W] (0/1)
        return: [B, N] (0/1 hard patch label)
        """
        B, H, W = mask.shape
        assert H % patch_size == 0 and W % patch_size == 0

        mask = mask.float().unsqueeze(1)  # [B,1,H,W]

        patch_mask = F.avg_pool2d(
            mask,
            kernel_size=patch_size,
            stride=patch_size
        )  # [B,1,H/P,W/P]

        patch_mask = patch_mask.squeeze(1).flatten(1)  # [B,N]

        # HARD binarization
        patch_label = (patch_mask > threshold).float() # [B,N] 0/1 hard label

        return patch_label

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
    
    def mask2patch(self, mask, patch_size=16, threshold=0.5):
        """
        mask: [B, H, W] (0/1)
        return: [B, N] (0/1 hard patch label)
        """
        B, H, W = mask.shape
        assert H % patch_size == 0 and W % patch_size == 0

        mask = mask.float().unsqueeze(1)  # [B,1,H,W]

        patch_mask = F.avg_pool2d(
            mask,
            kernel_size=patch_size,
            stride=patch_size
        )  # [B,1,H/P,W/P]

        patch_mask = patch_mask.squeeze(1).flatten(1)  # [B,N]

        # HARD binarization
        patch_label = (patch_mask > threshold).float() # [B,N] 0/1 hard label

        return patch_label

    def forward(self, x, MIL_learning=False, reduction='mean', return_instance_logits=False, gt_mask=None, threshold=0.1):
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

        if gt_mask is not None:
            patch_label = self.mask2patch(gt_mask, threshold=threshold)  # [B, N]
            mask_loss = sigmoid_focal_loss(
                patch_logits.squeeze(-1),
                patch_label,
                alpha=0.9,
                reduction="mean"
            ) # [B]

            return aggregated, topk_logit, res_logit, patch_logits.squeeze(-1), mask_loss


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


    def mask2patch(self, mask, patch_size=16, threshold=0.5):
        """
        mask: [B, H, W] (0/1)
        return: [B, N] (0/1 hard patch label)
        """
        B, H, W = mask.shape
        assert H % patch_size == 0 and W % patch_size == 0

        mask = mask.float().unsqueeze(1)  # [B,1,H,W]

        patch_mask = F.avg_pool2d(
            mask,
            kernel_size=patch_size,
            stride=patch_size
        )  # [B,1,H/P,W/P]

        patch_mask = patch_mask.squeeze(1).flatten(1)  # [B,N]

        # HARD binarization
        patch_label = (patch_mask > threshold).float() # [B,N] 0/1 hard label

        return patch_label

    def mask2segment(self, gt_mask, cluster_labels, patch_size=16, threshold=0.5):
        """
        gt_mask: [1, H, W]
        cluster_labels: [N]
        return: [1, number of segments] 0/1 hard label
        """

        device = gt_mask.device

        # =========================
        # 1. GT → patch label
        # =========================
        patch_label = self.mask2patch(
            gt_mask,
            patch_size=patch_size,
            threshold=threshold
        )  # [1, N]

        patch_label = patch_label.squeeze(0)  # [N]

        labels = cluster_labels  # [N]

        K = labels.max().item() + 1 # number of segments

        segment_labels = []

        # =========================
        # 2. aggregate per segment
        # =========================
        for k in range(K):

            idx = (labels == k)

            if idx.sum() > 0:
                seg_label = (patch_label[idx].float().mean() > threshold).float()
            else:
                seg_label = torch.tensor(0.0, device=device)

            segment_labels.append(seg_label)

        return torch.stack(segment_labels, dim=0).unsqueeze(0)  # [1, number of segments]

    def forward(self, x, MIL_learning=False, reduction='mean', return_instance_logits=False, gt_mask=None, threshold=0.1, cluster_labels=None):
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

        if gt_mask is not None:
            segment_label = self.mask2segment(gt_mask, cluster_labels=cluster_labels, threshold=threshold)  # [1, nums_segment] 0/1 hard label
            mask_loss = sigmoid_focal_loss(
                segment_logits.squeeze(-1),
                segment_label,
                alpha=0.9,
                reduction="mean"
            ) # [1]

            return aggregated, topk_logit, res_logit, segment_logits.squeeze(-1), mask_loss


        if MIL_learning:
            if return_instance_logits:
                return aggregated, topk_logit, res_logit, segment_logits.squeeze(-1)
            else:
                return aggregated, topk_logit, res_logit
        else:   
            return aggregated


# class TokenWise_TokenReducer(nn.Module):
#     """
#     基于 Top-K 机制的 Token 缩减与聚合模块
#     功能：
#     1. 计算每个 Token 的重要性分数 (Logits)
#     2. 仅保留分数最高的 Top-K 个 Token 参与特征聚合
#     3. 支持 MIL (多实例学习) 的辅助输出 (Top-K 均值 vs 剩余均值)
#     """
#     def __init__(self, input_dim=768, hidden_dim=128, temperature=0.07, topk_ratio=0.05):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.Dropout(0.1),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
#         self.temperature = temperature
#         self.topk_ratio = topk_ratio
#         self.norm = nn.LayerNorm(input_dim)

#     def forward(self, x, MIL_learning=False, reduction='mean', return_instance_logits=False):
#         """
#         x: [B, N, D] (Batch_size, Token_num, Dimension)
#         """
#         # 1. 特征归一化与重要性评分
#         x_norm = self.norm(x)
#         logits = self.mlp(x_norm).squeeze(-1)  # [B, N]
        
#         # 数值稳定性处理：防止 Softmax 爆炸
#         logits = torch.clamp(logits, min=-10, max=10)
        
#         B, N = logits.shape
#         k = max(1, int(self.topk_ratio * N))

#         # 2. 获取 Top-K 索引
#         topk_vals, topk_idx = torch.topk(logits, k, dim=1)  # [B, k]

#         # 3. 核心修改：仅针对 Top-K 进行 Softmax 聚合
#         # 创建一个全负无穷的掩码，只有 Top-K 的位置设为 0
#         mask = torch.full_like(logits, float('-inf'))
#         mask.scatter_(1, topk_idx, 0.0)
        
#         # 加上掩码后做 Softmax，非 Top-K 区域权重将严格为 0
#         # (logits + mask) 让非 topk 位置变为 -inf，softmax(-inf) = 0
#         topk_weights = F.softmax((logits + mask) / self.temperature, dim=-1) # [B, N]
        
#         # 聚合特征：[B, 1, N] @ [B, N, D] -> [B, 1, D] -> [B, D]
#         aggregated = torch.bmm(topk_weights.unsqueeze(1), x).squeeze(1)

#         # 4. MIL 逻辑处理
#         if not MIL_learning:
#             return aggregated

#         # 计算剩余 (Rest) Token 的平均得分
#         # 创建反向掩码：Top-K 为 0, 其他为 1
#         rest_mask = torch.ones_like(logits)
#         rest_mask.scatter_(1, topk_idx, 0.0)
        
#         # 避免除以 0
#         num_rest = N - k if N > k else 1
#         res_logit = (logits * rest_mask).sum(dim=1) / num_rest

#         # 计算 Top-K 的聚合得分
#         if reduction == 'mean':
#             topk_logit = topk_vals.mean(dim=1)
#         elif reduction == 'logsumexp':
#             topk_logit = torch.logsumexp(logits + mask, dim=1) # 同样只考虑 Top-K
#         else:
#             raise ValueError(f"Unsupported reduction method: {reduction}")

#         # 5. 返回结果
#         if return_instance_logits:
#             return aggregated, topk_logit, res_logit, logits
        
#         return aggregated, topk_logit, res_logit

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


def visualize_clusters(labels, img_size=512, patch_size=16):
    import matplotlib.pyplot as plt
    import numpy as np
    """
    labels: (1024,)  for ViT-L/16 on 512x512 → 32x32 patches
    """

    grid_size = img_size // patch_size  # 32

    label_map = labels.cpu().numpy().reshape(grid_size, grid_size)

    plt.figure(figsize=(6, 6))
    plt.imshow(label_map, cmap="tab20")  # 或 "jet"
    plt.title("Agglomerative Clustering on ViT Patches")
    plt.colorbar()
    plt.axis("off")
    plt.savefig("cluster_labels.png")
    plt.show()

def overlay_on_image(img_tensor, labels, img_size=512, patch_size=16, alpha=0.6):
    """
    img_tensor: (1,3,H,W)
    """

    import torch
    import matplotlib.pyplot as plt

    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    grid_size = img_size // patch_size
    label_map = labels.cpu().numpy().reshape(grid_size, grid_size)

    # 上采样 label map 到 image size
    label_map = torch.tensor(label_map)[None, None].float()
    label_map = torch.nn.functional.interpolate(
        label_map,
        size=(img_size, img_size),
        mode="nearest"
    )[0, 0].numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(label_map, cmap="tab20", alpha=alpha)
    plt.title("Cluster Overlay on Image")
    plt.axis("off")
    plt.savefig("cluster_overlay.png")
    plt.show()

if __name__ == "__main__":
    # 测试代码
    from dinov3 import DINOv3Model

    model = DINOv3Model(backbone_name="/data/data2/jielun/XPlainVerse/MM_2026_method/HiDINO/dinov3-vitl16-pretrain-lvd1689m/", layer_indices=[24])

    img_path = "/home/home/jielun/DDL/track1/train/real/8d2b8acbd7dab8ec39caa039442b244d.png"
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    from torchvision.transforms import functional as TF
    img = TF.resize(img, (640, 640))
    img = TF.to_tensor(img)

    import torchvision.transforms as T
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    img = normalize(img)
    img = img.unsqueeze(0)  # (1, 3, 512, 512)

    patch_tokens = model(img)[2][0].squeeze(0)  # (1, 1024, 1024) -> (1024, 1024)
    print("Patch tokens shape:", patch_tokens.shape)

    labels, protos = vit_patch_clustering(patch_tokens, tau=0.9)
    print("Cluster labels shape:", labels.shape)  # (1024,)
    print("Prototypes shape:", protos.shape)      # (K, 768)

    visualize_clusters(labels)

    overlay_on_image(img, labels)