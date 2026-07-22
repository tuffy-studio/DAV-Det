"""
Simple Audio Deepfake Detector

Architecture:
    1. PE-AV AudioEncoder (with optional LoRA) as frontend
    2. MLP classifier head

Supports variable-length audio input via padding_mask.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional, Dict


# Add project path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peav_audio_encoder.audio_encoder import StandaloneAudioEncoder


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

def build_detector_aasist(
    peav_checkpoint: str,
    use_lora: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    unfreeze_norm: bool = True,
    device: str = "cuda",
    use_deep_supervision: bool = True,
    num_supervision_layers: int = 3,
    num_heads: int = 8,
    attn_dropout: float = 0.1,
    use_dual_path: bool = True,
    use_attention_pooling: bool = True,
):
    """
    Build SimpleDetectorWithAASISTBackend from PE-AV checkpoint.
    
    Args:
        peav_checkpoint: Path to PE-AV checkpoint directory
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        unfreeze_norm: Whether to unfreeze norm layers
        device: Device to load model on
        use_deep_supervision: Whether to enable deep supervision auxiliary heads
        num_supervision_layers: Number of transformer layers to supervise (last N layers)
        num_heads: Number of attention heads for temporal branch
        attn_dropout: Dropout for attention layers
        use_dual_path: Whether to use dual-branch (temporal + spectral)
        use_attention_pooling: Whether to use attention-based pooling
    
    Returns:
        SimpleDetectorWithAASISTBackend instance
    """
    from aasist_style_backend import SimpleDetectorWithAASISTBackend
    
    # Load feature extractor with LoRA already injected
    feature_extractor = StandaloneAudioEncoder.from_pretrained(
        peav_checkpoint,
        output_dim=None,
        device='cpu',
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        unfreeze_norm=unfreeze_norm,
    )

    feature_extractor.freeze_base_model()

    # Build detector with AASIST-style backend
    detector = SimpleDetectorWithAASISTBackend(
        feature_extractor=feature_extractor,
        num_heads=num_heads,
        attn_dropout=attn_dropout,
        use_dual_path=use_dual_path,
        use_attention_pooling=use_attention_pooling,
        use_deep_supervision=use_deep_supervision,
        num_supervision_layers=num_supervision_layers,
    )
    
    return detector.to(device)
