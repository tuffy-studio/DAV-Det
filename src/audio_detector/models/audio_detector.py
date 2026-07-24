import os
import sys
import torch


# Add project path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peav_audio_encoder.audio_encoder import StandaloneAudioEncoder


def build_detector(
    peav_checkpoint: str,
    use_lora: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    unfreeze_norm: bool = True,
    device: str = "cuda",
    use_deep_supervision: bool = False,
    num_supervision_layers: int = 3,
    num_heads: int = 8,
    attn_dropout: float = 0.1,
):
    """
    Build AudioDeepfakeDetector from PE-AV checkpoint.

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

    Returns:
        AudioDeepfakeDetector instance
    """
    from audio_detector_backend import AudioDeepfakeDetector

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

    # Build detector with audio detector backend
    detector = AudioDeepfakeDetector(
        feature_extractor=feature_extractor,
        num_heads=num_heads,
        attn_dropout=attn_dropout,
        use_deep_supervision=use_deep_supervision,
        num_supervision_layers=num_supervision_layers,
    )

    return detector.to(device)

