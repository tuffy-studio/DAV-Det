import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math
import re

class DINOv3Model(nn.Module):
    def __init__(
        self,
        backbone_name,
        layer_indices=None,   # 优先：直接指定，如 [6, 9, 11]
        num_last_layers=4,    # 备用：取最后 N 层
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone_name)
        print(self.backbone)

        for p in self.backbone.parameters():
            p.requires_grad = False # 冻结预训练模型参数

        encoder_layers = self.backbone.model.layer
        N = len(encoder_layers)
        print(f"Backbone has {N} layers.") # 从0开始计数

        if layer_indices is not None:
            self.layer_indices = [i for i in layer_indices]
        else:
            self.layer_indices = list(range(N - num_last_layers, N))

    def forward(self, x):
        with torch.no_grad():
            outputs = self.backbone(pixel_values=x, output_hidden_states=True)

        cls_tokens = torch.stack(
            [outputs.hidden_states[i][:, 0, :] for i in self.layer_indices],
            dim=1
        ).float()

        register_tokens = torch.stack(
            [outputs.hidden_states[i][:, 1:5, :] for i in self.layer_indices],
            dim=1
        ).float()

        patch_tokens = torch.stack(
            [outputs.hidden_states[i][:, 5:, :] for i in self.layer_indices],
            dim=1
        ).float()

        return cls_tokens, register_tokens, patch_tokens

class LinearLoRA(nn.Module):
    def __init__(self, base_linear, r=32, lora_alpha=64, dropout_rate=0.0, train_bias=False):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError('DINOv3LinearLoRA expects an nn.Linear module.')

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = int(r)
        self.scaling = float(lora_alpha) / max(1, self.r)
        self.dropout = nn.Dropout(dropout_rate) if float(dropout_rate) > 0 else nn.Identity()

        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=bool(train_bias))
        else:
            self.register_parameter('bias', None)

        if self.r <= 0:
            raise ValueError('DINOv3LinearLoRA requires rank r > 0.')

        self.w_lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
        self.w_lora_B = nn.Parameter(torch.empty(self.out_features, self.r))
        self.reset_parameters()

        self.w_lora_A.requires_grad = True
        self.w_lora_B.requires_grad = True
        self.weight.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.w_lora_B)

    def forward(self, x):
        base_output = F.linear(x, self.weight, self.bias)
        if getattr(self, 'use_lora', True):
            lora_input = self.dropout(x) if self.training else x
            lora_hidden = F.linear(lora_input, self.w_lora_A, bias=None)
            lora_output = F.linear(lora_hidden, self.w_lora_B, bias=None)
            return base_output + lora_output * self.scaling
        else:
            return base_output

def inject_lora(
    model,
    target_modules=("query","value","q","v"),#("query", "key", "value", "qkv", "proj"),
    r=32,
    alpha=64,
    dropout=0.0,
    layer_indices=None,
    verbose=True,
):
    applied = []

    def _replace(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # 递归
            _replace(child, full_name)

            # 替换 Linear
            if isinstance(child, nn.Linear):
                if any(t in full_name for t in target_modules):

                    setattr(
                        module,
                        name,
                        LinearLoRA(child, r=r, lora_alpha=alpha, dropout_rate=dropout)
                    )

                    applied.append(full_name)

    _replace(model)

    # =========================
    # 🔥 debug print
    # =========================
    if verbose:
        print("\n" + "=" * 60)
        print(f"[LoRA Injection Summary]")
        print(f"Total injected modules: {len(applied)}")
        print("-" * 60)

        for name in applied:
            print(f"[LoRA] {name}")

        print("=" * 60 + "\n")

    return applied


def inject_lora_layer(
    model,
    target_modules=("q_proj", "v_proj"),  # 修改为你要注入的模块名称
    r=32,
    alpha=64,
    dropout=0.0,
    layer_indices=None,  # 指定哪些层需要注入
    verbose=True,
):
    applied = []

    def _replace(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # 如果指定了 layer_indices，且当前层不在指定的索引列表中，则跳过
            if layer_indices is not None and any(f"layer.{i}" in full_name for i in layer_indices):
                # 检查是否是目标模块，注入 LoRA
                if isinstance(child, nn.Linear) and any(t in full_name for t in target_modules):
                    # 这里用 LoRA 替换目标模块
                    setattr(
                        module,
                        name,
                        LinearLoRA(child, r=r, lora_alpha=alpha, dropout_rate=dropout)
                    )
                    applied.append(full_name)
            
            # 递归遍历子模块
            _replace(child, full_name)

    # 调用递归函数，开始替换
    _replace(model)

    # 打印调试信息
    if verbose:
        print("\n" + "=" * 60)
        print(f"[LoRA Injection Summary]")
        print(f"Total injected modules: {len(applied)}")
        print("-" * 60)

        for name in applied:
            print(f"[LoRA] {name}")

        print("=" * 60 + "\n")

    return applied

class DINOv3Model_LORA(nn.Module):
    def __init__(
        self,
        backbone_name,
        layer_indices=None,
        num_last_layers=4,
        use_lora=True,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        unfreeze_norm=False,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone_name)
        for p in self.backbone.parameters(): 
            p.requires_grad = False

        # =========================
        # 🔥 插入 LoRA（关键）
        # =========================
        if use_lora:
            print("Injecting LoRA into backbone...")
            print(f"LoRA Config - r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}, target_modules: ('q_proj', 'v_proj')")
            inject_lora(
                self.backbone,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
        
        # =========================
        # 🔥 解冻 LayerNorm（可选）
        # =========================
        if unfreeze_norm:
            print("Unfreezing LayerNorm parameters...")
            unfrozen_count = 0
            for name, param in self.backbone.named_parameters():
                if "norm" in name.lower() or "ln" in name.lower():
                    param.requires_grad = True
                    unfrozen_count += 1
            print(f"Unfrozen {unfrozen_count} LayerNorm parameters.")

        encoder_layers = self.backbone.model.layer
        N = len(encoder_layers)
        print(f"Backbone has {N} layers.")

        if layer_indices is not None:
            self.layer_indices = layer_indices
        else:
            self.layer_indices = list(range(N - num_last_layers, N))

    def forward(self, x, use_lora=True):
        for module in self.modules():
            if isinstance(module, LinearLoRA):
                module.use_lora = use_lora

        outputs = self.backbone(pixel_values=x, output_hidden_states=True)

        cls_tokens = torch.stack(
            [outputs.hidden_states[i][:, 0, :] for i in self.layer_indices],
            dim=1
        ).float()

        register_tokens = torch.stack(
            [outputs.hidden_states[i][:, 1:5, :] for i in self.layer_indices],
            dim=1
        ).float()

        patch_tokens = torch.stack(
            [outputs.hidden_states[i][:, 5:, :] for i in self.layer_indices],
            dim=1
        ).float()

        return cls_tokens, register_tokens, patch_tokens
