"""
Standalone AudioEncoder extracted from PE-AV model.

This module provides a self-contained AudioEncoder that only requires audio input.
It extracts the audio encoder from the PE-AV checkpoint and provides a clean interface.

Usage:
    from standalone_audio_encoder import StandaloneAudioEncoder, AudioProcessor
    
    # Load model
    model = StandaloneAudioEncoder.from_pretrained("/path/to/pe-av-base")
    
    # Process audio
    processor = AudioProcessor(sampling_rate=48000, hop_length=1920)
    inputs = processor(["/path/to/audio1.wav", "/path/to/audio2.wav"])
    
    # Forward pass
    outputs = model(
        input_values=inputs["input_values"],
        padding_mask=inputs["padding_mask"]
    )
    
    # outputs["frame_features"]: (B, T', 1024) - frame-level features
    # outputs["pooler_output"]:  (B, 1024)     - global audio embedding
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file
from transformers import BatchFeature
from torch.nn.utils.rnn import pad_sequence


# =============================================================================
# Rotary Embedding (copied from perception_models/core/transformer.py)
# =============================================================================

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_, seq_dim).float()
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        head_dim: int,
        max_seqlen: int = 1024,
        scale_factor: int = 1,
        low_freq_factor: int = 1,
        high_freq_factor: int = 32,
        old_context_len: int = 8192,
    ):
        super().__init__()
        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        if scale_factor != 1:
            self.low_freq_wavelen = old_context_len / low_freq_factor
            self.high_freq_wavelen = old_context_len / high_freq_factor
            assert self.low_freq_wavelen >= self.high_freq_wavelen

    def reset_parameters(self):
        self.register_buffer(
            "freqs_cis",
            self.precompute_freqs_cis(dim=self.head_dim, end=self.max_seqlen, theta=self.theta),
            persistent=False,
        )

    def apply_scaling(self, freqs):
        if self.scale_factor == 1:
            return freqs
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < self.high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > self.low_freq_wavelen:
                new_freqs.append(freq / self.scale_factor)
            else:
                assert self.low_freq_wavelen != self.high_freq_wavelen
                smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / self.scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        freqs = self.apply_scaling(freqs)
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        cos, sin = freqs.cos(), freqs.sin()
        return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)

    def forward(self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None):
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


# =============================================================================
# Patcher (copied from perception_models/core/audio_visual_encoder/patcher.py)
# =============================================================================

class MaskedGroupNorm(torch.nn.GroupNorm):
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if padding_mask is None:
            return super().forward(x)
        B, C, L = x.shape
        G = self.num_groups
        x_grouped = x.view(B, G, C // G, L)
        padding_mask_grouped = padding_mask.reshape(B, G, C // G, L).bool()
        mean = torch.masked.mean(x_grouped, mask=padding_mask_grouped, dim=(2, 3), keepdim=True)
        var = torch.masked.var(
            x_grouped, mask=padding_mask_grouped, dim=(2, 3), keepdim=True, unbiased=False
        )
        x_norm = (x_grouped - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(B, C, L)
        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x_norm * padding_mask


class ConvBlock1d(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.groupnorm = MaskedGroupNorm(num_groups=1, num_channels=hidden_size)
        self.activation = torch.nn.SiLU()
        self.project = torch.nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding="same"
        )

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.groupnorm(x, padding_mask=padding_mask)
        x = self.activation(x)
        return self.project(x)


class ResnetBlock1d(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.block1 = ConvBlock1d(hidden_size)
        self.block2 = ConvBlock1d(hidden_size)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).expand_as(x)
        h = self.block1(x, padding_mask=padding_mask)
        h = self.block2(h, padding_mask=padding_mask)
        return h + x


# =============================================================================
# Transformer (copied from perception_models/core/audio_visual_encoder/transformer.py)
# =============================================================================

class TransformerConfig:
    """Minimal config class for Transformer."""
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=2752,
        num_hidden_layers=16,
        num_attention_heads=8,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=10_000,
        rms_norm_eps=1e-5,
        rope_theta=20000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = apply_rotary_emb(
            query_states, key_states, seq_dim=2, freqs_cis=position_embeddings
        )

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Embeddings(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.resnet_block = ResnetBlock1d(config.hidden_size)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, config.hidden_size))

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> tuple:
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x = x.transpose(1, 2)
        if padding_mask is not None:
            padding_mask = F.pad(padding_mask, (1, 0), value=True)
        h = self.resnet_block(x, padding_mask=padding_mask)
        return h.transpose(1, 2), padding_mask


@dataclass
class BaseModelOutputWithPooling:
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None


class Transformer(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.layers = torch.nn.ModuleList([
            DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rope_embeddings = RotaryEmbedding(
            theta=max(10_000, 2 * config.max_position_embeddings),
            head_dim=config.hidden_size // config.num_attention_heads,
            max_seqlen=config.max_position_embeddings,
        )
        self.rope_embeddings.reset_parameters()
        self.output = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        inputs_embeds, attention_mask = self.embeddings(inputs_embeds, padding_mask=attention_mask)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None].bool()
        position_embeddings = self.rope_embeddings(seqlen=inputs_embeds.size(1))
        hidden_states = inputs_embeds
        
        # Store intermediate hidden states for deep supervision
        all_hidden_states = [] if return_hidden_states else None
        
        for layer_idx, encoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            if return_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output(hidden_states)

        output = BaseModelOutputWithPooling(
            last_hidden_state=hidden_states[:, 1:],
            pooler_output=hidden_states[:, 0],
        )
        
        if return_hidden_states:
            # all_hidden_states: list of (B, N+1, D) tensors, one per layer
            output.hidden_states = all_hidden_states
        
        return output


# =============================================================================
# Snake Activation
# =============================================================================

class Snake1d(nn.Module):
    """Snake activation function for 1D signals."""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * 0.5)

    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-8)) * torch.sin(self.alpha * x) ** 2


# =============================================================================
# DAC Encoder (reconstructed from checkpoint weights)
# =============================================================================

class DACResUnit(nn.Module):
    """DAC Encoder Residual Unit."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.snake1 = Snake1d(channels)
        self.snake2 = Snake1d(channels)

    def forward(self, x):
        h = self.snake1(x)
        h = self.conv1(h)
        h = self.snake2(h)
        h = self.conv2(h)
        return x + h


class DACEncoderBlock(nn.Module):
    """DAC Encoder Block with downsampling.
    
    Structure: snake(input) -> res_units(input) -> conv1(input->output, stride)
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.snake1 = Snake1d(in_channels)
        self.res_unit1 = DACResUnit(in_channels)
        self.res_unit2 = DACResUnit(in_channels)
        self.res_unit3 = DACResUnit(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=stride // 2
        )

    def forward(self, x):
        x = self.snake1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        x = self.conv1(x)
        return x


class DacEncoder(nn.Module):
    """
    DAC Encoder reconstructed from PE-AV checkpoint.
    
    Architecture:
        conv1: 1 -> 64, kernel=7
        block0: 64 -> 128, stride=2
        block1: 128 -> 256, stride=8
        block2: 256 -> 512, stride=10
        block3: 512 -> 1024, stride=12
        snake -> conv2: 1024 -> 1024, kernel=3
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.block = nn.ModuleList([
            DACEncoderBlock(64, 128, stride=2),
            DACEncoderBlock(128, 256, stride=8),
            DACEncoderBlock(256, 512, stride=10),
            DACEncoderBlock(512, 1024, stride=12),
        ])
        self.snake1 = Snake1d(1024)
        self.conv2 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, 1, T)
        x = self.conv1(x)
        for block in self.block:
            x = block(x)
        x = self.snake1(x)
        x = self.conv2(x)
        return x


class VAEBottleneck(nn.Module):
    def __init__(self, input_dim: int = 1024, bottleneck_dim: int = 128):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, bottleneck_dim * 2, kernel_size=1)
        self.out_proj = nn.Conv1d(bottleneck_dim, input_dim, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, _ = self.in_proj(z).chunk(2, dim=1)
        return mean


class DacEncoderVAE(nn.Module):
    """DAC VAE Encoder."""
    def __init__(self, hop_length: int = 1920, sampling_rate: int = 48000):
        super().__init__()
        self.encoder = DacEncoder()
        self.bottleneck = VAEBottleneck(1024, 128)
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            z = self.encoder(self._pad(waveform))
            encoded_frames = self.bottleneck(z)
        return encoded_frames

    def _pad(self, wavs):
        length = wavs.size(-1)
        if length % self.hop_length:
            p1d = (0, self.hop_length - (length % self.hop_length))
            return F.pad(wavs, p1d, "reflect")
        return wavs


# =============================================================================
# AudioEncoderConfig
# =============================================================================

class AudioEncoderConfig:
    def __init__(
        self,
        dac_vae_encoder: Optional[dict] = None,
        audio_transformer: Optional[dict] = None,
        **kwargs,
    ):
        default_dac = {
            "encoder_hidden_size": 64,
            "downsampling_ratios": [2, 8, 10, 12],
            "decoder_hidden_size": 1536,
            "n_codebooks": 16,
            "codebook_size": 1024,
            "codebook_dim": 128,
            "quantizer_dropout": 0,
            "sampling_rate": 48000,
        }
        dac_vae_encoder = dac_vae_encoder or default_dac
        audio_transformer = audio_transformer or {}
        self.dac_vae_encoder = dac_vae_encoder
        self.audio_transformer = TransformerConfig(**audio_transformer)


# =============================================================================
# Standalone AudioEncoder
# =============================================================================

@dataclass
class AudioOutput(BaseModelOutputWithPooling):
    audio_feature_padding_mask: Optional[torch.Tensor] = None
    dac_vae_features: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None  # intermediate layer outputs for deep supervision


class AudioEncoder(nn.Module):
    """
    Audio Encoder from PE-AV model.
    
    Architecture:
        1. DAC VAE Encoder: waveform -> codec features (128-dim)
        2. Data Projection: 128 -> 1024 dim
        3. Audio Transformer: 16-layer Transformer (hidden_size=1024)
    """
    
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        codebook_dim = config.dac_vae_encoder.get("codebook_dim", 128)
        self.data_proj = nn.Linear(codebook_dim, config.audio_transformer.hidden_size)
        self.dac_vae_encoder = DacEncoderVAE(
            hop_length=1920,
            sampling_rate=config.dac_vae_encoder.get("sampling_rate", 48000),
        )
        self.audio_transformer = Transformer(config.audio_transformer)

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> AudioOutput:
        """
        Args:
            input_values: Raw audio waveform, shape (B, 1, T), sampling_rate=48000
            padding_mask: Optional mask for waveform, shape (B, T)
            input_features: Optional pre-computed codec features, shape (B, T', 128)
            return_hidden_states: Whether to return intermediate layer hidden states
        
        Returns:
            AudioOutput with:
                - last_hidden_state: (B, T', 1024) frame-level features
                - pooler_output: (B, 1024) global audio embedding (CLS token)
                - audio_feature_padding_mask: (B, T') mask at codec level
                - dac_vae_features: (B, T', 128) raw codec features
                - hidden_states: list of (B, N+1, D) intermediate outputs (if requested)
        """
        if input_features is None:
            codec_features = self.dac_vae_encoder(input_values).transpose(1, 2)
            feature_padding_mask = None
            if padding_mask is not None:
                feature_padding_mask = padding_mask[:, :: self.dac_vae_encoder.hop_length]
        else:
            codec_features = input_features
            feature_padding_mask = padding_mask
        
        projected = self.data_proj(codec_features)
        outputs = self.audio_transformer(
            projected, 
            attention_mask=feature_padding_mask,
            return_hidden_states=return_hidden_states,
        )
        
        return AudioOutput(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            audio_feature_padding_mask=feature_padding_mask,
            dac_vae_features=codec_features,
            hidden_states=outputs.hidden_states if return_hidden_states else None,
        )


# =============================================================================
# LoRA (Low-Rank Adaptation)
# =============================================================================

class LoRALayer(nn.Module):
    """
    LoRA layer: wraps an existing Linear layer with low-rank adapters.

    Forward: h = W @ x + (alpha/r) * B @ A @ x
    where A (r x in_dim), B (out_dim x r) are trainable,
    W is frozen base weight.
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # LoRA matrices
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Initialize: A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output (no grad)
        base_output = self.base_layer(x)

        # LoRA path
        lora_output = F.linear(self.lora_dropout(x), self.lora_A)  # (..., r)
        lora_output = F.linear(lora_output, self.lora_B)           # (..., out_features)

        return base_output + lora_output * self.scaling

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into base layer and return the merged layer."""
        merged_weight = self.base_layer.weight.data + (
            self.lora_B @ self.lora_A
        ) * self.scaling

        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
        )
        merged.weight.data = merged_weight
        if self.base_layer.bias is not None:
            merged.bias.data = self.base_layer.bias.data.clone()

        return merged


class StandaloneAudioEncoder(nn.Module):
    """
    Standalone audio encoder with optional projection head and LoRA.
    """

    def __init__(
        self,
        config: AudioEncoderConfig,
        output_dim: Optional[int] = None,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        unfreeze_norm: bool = False,
    ):
        super().__init__()
        self.config = config
        self.audio_encoder = AudioEncoder(config)

        self.output_dim = output_dim
        if output_dim is not None:
            self.audio_head = nn.Sequential(
                nn.LayerNorm(config.audio_transformer.hidden_size, eps=1e-6),
                nn.Linear(config.audio_transformer.hidden_size, output_dim, bias=False),
            )
        else:
            self.audio_head = None

        # LoRA state
        self.use_lora = use_lora
        self.lora_config = None
        self.lora_layers: Dict[str, LoRALayer] = {}

        if use_lora:
            self._inject_lora(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                unfreeze_norm=unfreeze_norm,
            )
    
    def _inject_lora(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        unfreeze_norm: bool = False,
    ):
        """Inject LoRA adapters into the audio transformer's target Linear layers."""
        if target_modules is None:
            # Default: only q_proj and v_proj (most parameter-efficient)
            target_modules = ["q_proj", "v_proj"]#, "k_proj", "o_proj"]

        transformer = self.audio_encoder.audio_transformer
        matched = {}
        for name, module in transformer.named_modules():
            if any(name.endswith(t) for t in target_modules):
                if isinstance(module, nn.Linear):
                    matched[name] = module

        for name, module in matched.items():
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = transformer if parent_name == "" else transformer.get_submodule(parent_name)

            lora_layer = LoRALayer(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            setattr(parent, attr_name, lora_layer)
            self.lora_layers[name] = lora_layer

        lora_params = sum(p.numel() for layer in self.lora_layers.values()
                         for p in layer.parameters() if p.requires_grad)
        print(f"Added LoRA to {len(self.lora_layers)} modules")
        print(f"  Target modules: {target_modules}")
        print(f"  LoRA rank r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"  Unfreeze norm: {unfreeze_norm}")
        print(f"  LoRA params: {lora_params:,} ({lora_params/1e6:.3f}M)")
        print("LoRA injection finished.")

        # Optionally unfreeze norm layers (RMSNorm, LayerNorm, GroupNorm, BatchNorm)
        if unfreeze_norm:
            norm_params = 0
            for module in transformer.modules():
                if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, RMSNorm)):
                    for p in module.parameters():
                        p.requires_grad = True
                        norm_params += p.numel()
            print(f"Unfrozen Norm params: {norm_params:,} ({norm_params/1e6:.3f}M)")

        self.lora_config = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "unfreeze_norm": unfreeze_norm,
        }

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> dict:
        """
        Args:
            input_values: Raw audio waveform, shape (B, 1, T), sampling_rate=48000
            padding_mask: Optional mask, shape (B, T)
            input_features: Optional pre-computed codec features, shape (B, T', 128)
            return_hidden_states: Whether to return intermediate layer hidden states

        Returns:
            dict with:
                - frame_features: (B, T', 1024) frame-level features
                - pooler_output: (B, 1024) global audio embedding
                - embedding: (B, output_dim) projected embedding (if output_dim is set)
                - audio_feature_padding_mask: (B, T') mask at codec level
                - dac_vae_features: (B, T', 128) raw codec features
                - hidden_states: list of intermediate layer outputs (if requested)
        """
        audio_output = self.audio_encoder(
            input_values, 
            padding_mask=padding_mask, 
            input_features=input_features,
            return_hidden_states=return_hidden_states,
        )

        result = {
            "frame_features": audio_output.last_hidden_state,
            "pooler_output": audio_output.pooler_output,
            "audio_feature_padding_mask": audio_output.audio_feature_padding_mask,
            "dac_vae_features": audio_output.dac_vae_features,
        }
        
        if return_hidden_states:
            result["hidden_states"] = audio_output.hidden_states

        if self.audio_head is not None:
            result["embedding"] = self.audio_head(audio_output.pooler_output)

        return result
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        output_dim: Optional[int] = None,
        device: str = "cpu",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        unfreeze_norm: bool = False,
    ) -> "StandaloneAudioEncoder":
        """
        Load a StandaloneAudioEncoder from a PE-AV checkpoint or a saved standalone checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint directory containing config.json and model.safetensors
            output_dim: If provided, also load and use the audio_head projection
            device: Device to load the model on
        
        Returns:
            StandaloneAudioEncoder instance with loaded weights
        """
        # Load config
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Extract audio model config
        audio_model_config_dict = config_dict.get("audio_visual_model", {}).get("audio_model", {})
        if not audio_model_config_dict:
            audio_model_config_dict = config_dict.get("audio_model", {})
        
        config = AudioEncoderConfig(**audio_model_config_dict)

        # Create model WITHOUT LoRA first (so checkpoint loads correctly)
        model = cls(config=config, output_dim=output_dim)

        # Load weights
        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        state_dict = {}
        with safe_open(safetensors_path, framework="pt", device=device) as f:
            # Check if this is a standalone checkpoint (keys start with "audio_encoder.")
            # or an original PE-AV checkpoint (keys start with "audio_visual_model.audio_model.")
            sample_key = next(iter(f.keys()))
            if sample_key.startswith("audio_encoder."):
                # Standalone checkpoint: load directly
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            else:
                # Original PE-AV checkpoint: extract audio encoder weights
                prefix = "audio_visual_model.audio_model."
                for key in f.keys():
                    if key.startswith(prefix):
                        new_key = key.replace(prefix, "audio_encoder.")
                        state_dict[new_key] = f.get_tensor(key)

                # Load audio_head weights if output_dim is specified
                if output_dim is not None:
                    head_prefix = "audio_head."
                    for key in f.keys():
                        if key.startswith(head_prefix):
                            new_key = key.replace(head_prefix, "audio_head.")
                            state_dict[new_key] = f.get_tensor(key)

        model.load_state_dict(state_dict, strict=False)

        # Inject LoRA AFTER loading base weights
        if use_lora:
            model._inject_lora(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                unfreeze_norm=unfreeze_norm,
            )

        return model.to(device)

    def freeze_base_model(self):
        """Freeze all parameters except LoRA and norm (if unfreeze_norm)."""
        print("Freezing base model except LoRA and norm...")
        for name, param in self.named_parameters():
            # Keep LoRA parameters trainable
            if "lora_" in name:
                param.requires_grad = True
                continue
            # Keep norm parameters trainable if unfreeze_norm was set
            if self.lora_config and self.lora_config.get("unfreeze_norm", False):
                # Check if this param belongs to a norm layer
                module_name = name.rsplit(".", 1)[0]
                try:
                    module = self.get_submodule(module_name)
                    if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, RMSNorm)):
                        param.requires_grad = True
                        continue
                except AttributeError:
                    pass
            param.requires_grad = False
        

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for layer in self.lora_layers.values()
                         for p in layer.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        return self

    def save_pretrained(self, save_path: str):
        """Save the standalone audio encoder."""
        os.makedirs(save_path, exist_ok=True)
        
        config_dict = {
            "dac_vae_encoder": self.config.dac_vae_encoder,
            "audio_transformer": {
                "hidden_size": self.config.audio_transformer.hidden_size,
                "intermediate_size": self.config.audio_transformer.intermediate_size,
                "num_hidden_layers": self.config.audio_transformer.num_hidden_layers,
                "num_attention_heads": self.config.audio_transformer.num_attention_heads,
                "num_key_value_heads": self.config.audio_transformer.num_key_value_heads,
                "hidden_act": self.config.audio_transformer.hidden_act,
                "max_position_embeddings": self.config.audio_transformer.max_position_embeddings,
                "rms_norm_eps": self.config.audio_transformer.rms_norm_eps,
                "rope_theta": self.config.audio_transformer.rope_theta,
                "attention_bias": self.config.audio_transformer.attention_bias,
                "attention_dropout": self.config.audio_transformer.attention_dropout,
            },
        }
        if self.output_dim is not None:
            config_dict["output_dim"] = self.output_dim
        
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        from safetensors.torch import save_file
        state_dict = self.state_dict()
        save_file(state_dict, os.path.join(save_path, "model.safetensors"))


# =============================================================================
# Audio Processor
# =============================================================================

AudioInput = torch.Tensor | list[torch.Tensor] | str | list[str]


class AudioProcessor:
    """
    Audio processor for StandaloneAudioEncoder.
    
    Handles loading audio files and preparing inputs for the model.
    Expected sampling rate: 48000 Hz
    """
    
    def __init__(
        self,
        sampling_rate: int = 48_000,
        hop_length: int = 1920,
    ):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
    
    def _reflect_pad(self, wav: torch.Tensor) -> torch.Tensor:
        """Pad audio to be divisible by hop_length."""
        if wav.size(-1) % self.hop_length == 0:
            return wav
        pad_len = self.hop_length - (wav.size(-1) % self.hop_length)
        return F.pad(wav, (0, pad_len), mode="reflect")
    
    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio file and resample to target sampling rate."""
        try:
            from torchcodec.decoders import AudioDecoder
            ad = AudioDecoder(path, sample_rate=self.sampling_rate, num_channels=1)
            return ad.get_all_samples().data
        except ImportError:
            import torchaudio
            waveform, sr = torchaudio.load(path)
            if sr != self.sampling_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform
    
    def __call__(
        self,
        raw_audio: AudioInput,
        sampling_rate: Optional[int] = None,
    ) -> BatchFeature:
        """
        Process raw audio into model inputs.
        
        Args:
            raw_audio: Can be:
                - A file path (str)
                - A list of file paths (list[str])
                - A waveform tensor (torch.Tensor) of shape (1, T) or (channels, T)
                - A list of waveform tensors
            sampling_rate: Sampling rate of the input audio (if not from file).
                          Must match the model's sampling_rate (48000).
        
        Returns:
            BatchFeature with:
                - input_values: (B, 1, T) padded waveform
                - padding_mask: (B, T) boolean mask
        """
        from_file = False
        
        if isinstance(raw_audio, str):
            raw_audio = [raw_audio]
        
        if isinstance(raw_audio, (list, tuple)) and len(raw_audio) > 0 and isinstance(raw_audio[0], str):
            loaded = []
            for audio_file in raw_audio:
                loaded.append(self._load_audio(audio_file))
            raw_audio = loaded
            from_file = True
        
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"Model expects sampling_rate={self.sampling_rate}, "
                    f"but got {sampling_rate}. Please resample the audio."
                )
        elif not from_file:
            import warnings
            warnings.warn(
                f"It is recommended to pass sampling_rate={self.sampling_rate} "
                f"to ensure correct processing."
            )
        
        if isinstance(raw_audio, torch.Tensor):
            raw_audio = [raw_audio]
        
        padded = []
        for wav in raw_audio:
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)  # (T,) -> (1, T)
            elif wav.ndim == 2 and wav.shape[0] > 1:
                # Multiple channels: average to mono
                wav = wav.mean(dim=0, keepdim=True)
            elif wav.ndim > 2:
                raise ValueError(f"Expected shape (T,) or (channels, T), got {wav.shape}")
            padded.append(self._reflect_pad(wav).T)
        
        lengths = torch.tensor([x.size(0) for x in padded])
        input_values = pad_sequence(padded, batch_first=True).transpose(1, 2)
        padding_mask = torch.arange(lengths.max())[None] < lengths[:, None]
        
        return BatchFeature({
            "input_values": input_values,
            "padding_mask": padding_mask,
        })
