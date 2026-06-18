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


class TokenWise_TokenReducer(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, temperature=0.07):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.temperature = temperature
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, N, D) frame-level features
            padding_mask: (B, N) boolean mask, True for valid positions
        
        Returns:
            aggregated: (B, D) weighted sum of frame features
        """
        x = self.norm(x)
        token_logits = self.mlp(x).squeeze(-1)  # [B, N]
        
        # Apply padding mask: set padding positions to -inf before softmax
        if padding_mask is not None:
            token_logits = token_logits.masked_fill(~padding_mask, float('-inf'))
        
        token_weights = F.softmax(token_logits / self.temperature, dim=-1)  # [B, N]
        
        # Handle all-padding case (all -inf -> NaN after softmax)
        token_weights = torch.nan_to_num(token_weights, nan=0.0)
        
        aggregated = torch.sum(token_weights.unsqueeze(-1) * x, dim=1)  # [B, D]

        return aggregated

class SimpleDeepfakeDetector(nn.Module):
    """
    Simple deepfake detector using PE-AV AudioEncoder + MLP.
    
    No category classification, just binary fake/real detection.
    Uses cls_token + aggregated frame tokens (via TokenWise_TokenReducer) for classification.
    
    Optional deep supervision: supervise intermediate transformer layers (e.g., layer 13, 14, 15)
    with lightweight FlexibleMLP heads. Each supervised layer has its own TokenWise_TokenReducer.
    The main classifier always uses the final layer (layer 16).
    """
    def __init__(
        self,
        feature_extractor: StandaloneAudioEncoder,
        token_reducer_hidden_dim: int = 256,
        token_reducer_temperature: float = 0.07,
        use_deep_supervision: bool = False,
        num_supervision_layers: int = 3,
    ):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.use_deep_supervision = use_deep_supervision
        self.num_supervision_layers = num_supervision_layers
        
        # Main token reducer for the final layer
        # self.token_reducer = TokenWise_TokenReducer(
        #     input_dim=1024,
        #     hidden_dim=token_reducer_hidden_dim,
        #     temperature=token_reducer_temperature,
        # )
        
        # Main classifier head: input is cls_token (1024) + aggregated_token (1024) = 2048
        self.classifier = FlexibleMLP(input_size=2048, hidden_sizes=[512], num_classes=1, drop_rates=[0.1])
        
        # Deep supervision: each supervised layer has its own token_reducer + classifier head
        # e.g., for 16-layer transformer: supervise layers 13, 14, 15 (0-indexed: 12, 13, 14)
        if use_deep_supervision:
            self.deep_supervision_token_reducers = nn.ModuleList()
            self.deep_supervision_heads = nn.ModuleList()
            for _ in range(num_supervision_layers):
                # Each layer gets its own token reducer
                self.deep_supervision_token_reducers.append(
                    TokenWise_TokenReducer(
                        input_dim=1024,
                        hidden_dim=token_reducer_hidden_dim,
                        temperature=token_reducer_temperature,
                    )
                )
                # Lightweight head: FlexibleMLP(2048 -> 256 -> 1)
                self.deep_supervision_heads.append(
                    FlexibleMLP(input_size=2048, hidden_sizes=[512], num_classes=1, drop_rates=[0.1])
                )
            print(f"Deep supervision enabled: {num_supervision_layers} auxiliary heads, each with own TokenReducer")
        
        # Print parameter counts
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SimpleDeepfakeDetector:")
        print(f"  Total params: {total/1e6:.2f}M")
        print(f"  Trainable params: {trainable/1e6:.2f}M")
        print(f"  Frozen params: {(total-trainable)/1e6:.2f}M")
    
    def _compute_logits_from_layer_output(self, hidden_state, padding_mask, token_reducer):
        """
        Given a transformer layer output (B, N+1, D), extract cls_token and frame tokens,
        aggregate frame tokens using the provided token_reducer, and return combined features.
        
        Args:
            hidden_state: (B, N+1, D) where N+1 includes cls_token at position 0
            padding_mask: (B, N) boolean mask for frame tokens
            token_reducer: TokenWise_TokenReducer instance to use for aggregation
        
        Returns:
            combined_features: (B, 2048) cls_token + aggregated_token
        """
        cls_token = hidden_state[:, 0]  # (B, D)
        frame_tokens = hidden_state[:, 1:].mean(dim=1)  # (B, N, D)
        #aggregated = token_reducer(frame_tokens, padding_mask=padding_mask)
        return torch.cat([cls_token, frame_tokens], dim=-1)  # (B, 2048)
    
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        if_return_feature: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_values: (B, 1, T) raw audio waveform, sampling_rate=48000
            padding_mask: (B, T) boolean mask, True for valid positions
            return_features: whether to return intermediate features
        
        Returns:
            dict with:
                - logits: (B,) raw logits from main classifier (final layer), for BCEWithLogitsLoss
                - prob: (B,) fake probability [0, 1] (for inference)
                - pooler_output: (B, 1024) if return_features=True
                - aggregated_token: (B, 1024) if return_features=True
                - aux_logits: list of (B,) raw logits from deep supervision heads (if enabled)
        """
        # Feature extraction
        features = self.feature_extractor(
            input_values=input_values,
            padding_mask=padding_mask,
            return_hidden_states=self.use_deep_supervision,
        )
        
        # pooler_output: (B, 1024) - cls token from final layer
        pooler_output = features["pooler_output"]
        
        # frame_features: (B, T', 1024) - frame-level tokens from final layer
        frame_features = features["frame_features"].mean(dim=1)
        
        # Get frame-level padding mask
        feature_padding_mask = features.get("audio_feature_padding_mask")
        
        # Aggregate frame tokens via TokenWise_TokenReducer (main, for final layer)
        #aggregated_token = self.token_reducer(frame_features, padding_mask=feature_padding_mask)  # (B, 1024)
        
        # Concatenate cls_token and aggregated_token
        combined_features = torch.cat([pooler_output, frame_features], dim=-1)  # (B, 2048)
        
        # Main classification (always uses final layer)
        logits = self.classifier(combined_features).squeeze(-1)  # (B,) raw logits, no sigmoid
        prob = torch.sigmoid(logits)  # (B,) for inference
        
        result = {
            "logits": logits,
            "prob": prob,
        }
        
        # Deep supervision: use intermediate transformer layer outputs
        # e.g., for 16-layer model with num_supervision_layers=3:
        # supervise layers 13, 14, 15 (1-indexed) which are indices 12, 13, 14 (0-indexed)
        if self.use_deep_supervision and "hidden_states" in features:
            hidden_states = features["hidden_states"]  # list of (B, N+1, D), one per layer
            num_layers = len(hidden_states)
            
            # Select the last (num_supervision_layers) layers BEFORE the final layer
            start_idx = num_layers - self.num_supervision_layers - 1
            selected_layers = hidden_states[start_idx:num_layers - 1]
            
            aux_logits = []
            for i, layer_output in enumerate(selected_layers):
                # Use the i-th token reducer (each layer has its own)
                combined = self._compute_logits_from_layer_output(
                    layer_output, feature_padding_mask, self.deep_supervision_token_reducers[i]
                )
                aux_logit = self.deep_supervision_heads[i](combined).squeeze(-1)  # (B,) raw logits
                aux_logits.append(aux_logit)
            
            result["aux_logits"] = aux_logits
        
        if if_return_feature:
            result["feature"] = combined_features
        
        return result
    
    @torch.no_grad()
    def predict(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simple prediction interface.
        
        Args:
            input_values: (B, 1, T) or (1, T)
            padding_mask: (B, T) or (T,)
        
        Returns:
            prob: (B,) fake probability
        """
        self.eval()
        
        # Handle single sample
        if input_values.ndim == 2:
            input_values = input_values.unsqueeze(0)
        if padding_mask is not None and padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
        
        outputs = self.forward(input_values, padding_mask)
        return outputs["prob"]
    
    @torch.no_grad()
    def predict_chunks(
        self,
        waveform: torch.Tensor,
        clip_length: int = 144000,  # 3s @ 48kHz
        hop_length: int = 72000,    # 1.5s @ 48kHz (50% overlap)
        aggregation: str = "mean",
    ) -> float:
        """
        Predict on long audio by splitting into chunks.
        
        Args:
            waveform: (1, T) or (T,) raw audio
            clip_length: chunk length in samples
            hop_length: hop size in samples
            aggregation: "mean" or "max"
        
        Returns:
            aggregated fake probability
        """
        self.eval()
        
        # Ensure shape (1, T)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        total_len = waveform.shape[1]
        
        # Short audio: direct prediction
        if total_len <= clip_length:
            padding_mask = torch.ones(1, total_len, dtype=torch.bool, device=waveform.device)
            return self.predict(waveform, padding_mask).item()
        
        # Long audio: split into chunks
        probs = []
        for start in range(0, total_len - clip_length + 1, hop_length):
            chunk = waveform[:, start:start + clip_length]
            chunk_mask = torch.ones(1, clip_length, dtype=torch.bool, device=chunk.device)
            prob = self.predict(chunk, chunk_mask).item()
            probs.append(prob)
        
        # Handle remaining tail
        last_start = start + hop_length
        if last_start < total_len:
            tail = waveform[:, last_start:]
            if tail.shape[1] >= clip_length // 2:  # Only if tail is long enough
                tail_mask = torch.ones(1, tail.shape[1], dtype=torch.bool, device=tail.device)
                prob = self.predict(tail, tail_mask).item()
                probs.append(prob)
        
        # Aggregate
        if aggregation == "mean":
            return sum(probs) / len(probs)
        elif aggregation == "max":
            return max(probs)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    def save_checkpoint(self, path: str, config: Optional[dict] = None):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': config,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {path}")
        return checkpoint.get('config')


def build_detector(
    peav_checkpoint: str,
    use_lora: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    unfreeze_norm: bool = True,
    device: str = "cuda",
    use_deep_supervision: bool = True,
    num_supervision_layers: int = 3,
):
    """
    Build SimpleDeepfakeDetector from PE-AV checkpoint.
    
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
    
    Returns:
        SimpleDeepfakeDetector instance
    """
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

    
    # Build detector
    detector = SimpleDeepfakeDetector(
        feature_extractor=feature_extractor,
        use_deep_supervision=use_deep_supervision,
        num_supervision_layers=num_supervision_layers,
    )
    
    return detector.to(device)


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


if __name__ == "__main__":
    # Quick test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "/data/data2/jielun/ADD/peav/pe-av-base"
    
    print("Building detector...")
    detector = build_detector(
        peav_checkpoint=checkpoint_path,
        freeze_extractor=True,
        use_lora=True,
        device=device,
    )
    
    # Test with 3s audio
    print("\nTesting with 3s audio...")
    batch_size = 2
    num_samples = 48000 * 3
    dummy_audio = torch.randn(batch_size, 1, num_samples).to(device)
    dummy_mask = torch.ones(batch_size, num_samples, dtype=torch.bool).to(device)
    
    detector.eval()
    with torch.no_grad():
        outputs = detector(dummy_audio, dummy_mask, return_features=True)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Prob shape: {outputs['prob'].shape}")
    print(f"Pooler output shape: {outputs['pooler_output'].shape}")
    print(f"Probs: {outputs['prob'].cpu().numpy()}")
    
    # Test chunk prediction
    print("\nTesting chunk prediction...")
    long_audio = torch.randn(1, 48000 * 10).to(device)  # 10s
    prob = detector.predict_chunks(long_audio, clip_length=48000*3, hop_length=48000*1)
    print(f"Aggregated prob: {prob:.4f}")
