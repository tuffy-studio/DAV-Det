"""
AASIST-Style Backend for Variable-Length Audio Deepfake Detection

Inspired by AASIST (Graph Attention + Dual-Branch + Master Node),
but adapted for variable-length sequences using Self-Attention.

Key differences from original AASIST:
    - Self-Attention replaces GAT (naturally supports variable length + padding mask)
    - No fixed-size 2D convolutions or position encodings
    - No GraphPool (uses attention-based aggregation instead)
    - Maintains mask-aware processing throughout
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LightweightAttentionMap(nn.Module):
    """
    Generate attention weights for dual-branch feature extraction.
    Replaces AASIST's 1x1 conv attention map with a lightweight MLP.
    """
    def __init__(self, dim=1024, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SELU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
        )
        
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, T, D)
            padding_mask: (B, T), True for valid positions
        Returns:
            weights: (B, T, D), attention weights in [0, 1]
        """
        weights = torch.sigmoid(self.mlp(x))  # (B, T, D)
        if padding_mask is not None:
            weights = weights.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        return weights


class MaskedSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with padding mask support.
    Replaces GAT for variable-length sequences.
    """
    def __init__(self, dim=1024, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, N, D) where N is sequence length
            padding_mask: (B, N), True for valid positions
        Returns:
            out: (B, N, D)
            attn_weights: (B, num_heads, N, N) for visualization
        """
        B, N, D = x.shape
        
        # Project
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply padding mask
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return out, attn


class ChannelAttention(nn.Module):
    """
    Channel-wise attention: for each time frame, attend over feature dimensions.
    This is the spectral-branch equivalent in our variable-length setting.
    
    Instead of full self-attention over D (which is expensive and D is fixed at 1024),
    we use a lightweight SE-style attention.
    """
    def __init__(self, dim=1024, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.SELU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, T, D)
            padding_mask: (B, T)
        Returns:
            out: (B, T, D)
        """
        # Global average pooling over time (with mask)
        if padding_mask is not None:
            mask_float = padding_mask.unsqueeze(-1).float()
            avg = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)  # (B, D)
        else:
            avg = x.mean(dim=1)  # (B, D)
        
        # Generate channel weights
        weights = self.fc(avg)  # (B, D)
        
        # Apply channel weights
        out = x * weights.unsqueeze(1)  # (B, T, D)
        out = self.norm(out)
        
        return out


class AttentionPooling(nn.Module):
    """
    Attention-based pooling: learnable query aggregates variable-length sequence.
    """
    def __init__(self, dim=1024, num_queries=1, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(1, num_queries, dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, N, D)
            padding_mask: (B, N), True for valid positions
        Returns:
            pooled: (B, num_queries, D)
        """
        B = x.size(0)
        queries = self.query.expand(B, -1, -1)
        
        key_padding_mask = ~padding_mask if padding_mask is not None else None
        
        pooled, _ = self.cross_attn(
            queries, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        pooled = self.norm(pooled)
        
        return pooled


class AASISTStyleBackend(nn.Module):
    """
    AASIST-inspired backend that supports variable-length sequences.
    
    Architecture:
        1. Attention Map Generation (lightweight MLP)
        2. Dual-Branch Feature Extraction:
            - Temporal Branch: Self-Attention over time frames
            - Spectral Branch: Channel attention over feature dimensions
        3. Master Node Interaction: Cross-attention with cls_token
        4. Readout: cls + temporal_stats + spectral_stats
        5. MLP Classifier
    """
    def __init__(
        self,
        input_dim=1024,
        num_heads=8,
        attn_dropout=0.1,
        use_dual_path=True,
        use_attention_pooling=True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.use_dual_path = use_dual_path
        self.use_attention_pooling = use_attention_pooling
        
        # 1. Attention Map Generation
        self.attention_map = LightweightAttentionMap(dim=input_dim, hidden_dim=128)
        
        # 2. Dual-Branch Feature Extraction
        if use_dual_path:
            # Temporal Branch: Self-Attention over time frames
            self.temporal_attn = MaskedSelfAttention(input_dim, num_heads, attn_dropout)
            self.temporal_norm = nn.LayerNorm(input_dim)
            
            # Spectral Branch: Channel attention over feature dimensions
            self.spectral_attn = ChannelAttention(input_dim, reduction=16)
            
            # Pooling for each branch
            if use_attention_pooling:
                self.temporal_pool = AttentionPooling(input_dim, num_queries=1, num_heads=num_heads)
                self.spectral_pool = AttentionPooling(input_dim, num_queries=1, num_heads=num_heads)
        
        # 3. Master Node Interaction
        self.master_dim = input_dim
        if use_dual_path:
            self.master_cross_attn = nn.MultiheadAttention(
                input_dim, num_heads, dropout=attn_dropout, batch_first=True
            )
            self.master_norm = nn.LayerNorm(input_dim)
        
        # 4. Readout dimension calculation
        if use_dual_path:
            if use_attention_pooling:
                # master(1024) + temporal_pool(1024) + temporal_max(1024) + temporal_mean(1024)
                #               + spectral_pool(1024) + spectral_max(1024) + spectral_mean(1024)
                readout_dim = input_dim * 7
            else:
                # master(1024) + temporal_max(1024) + temporal_mean(1024)
                #               + spectral_max(1024) + spectral_mean(1024)
                readout_dim = input_dim * 5
        else:
            # master(1024) + frame_max(1024) + frame_mean(1024)
            readout_dim = input_dim * 3
        
        self.readout_dim = readout_dim
        
        # 5. MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, 512),
            nn.SELU(inplace=True),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
            nn.Linear(512, 1),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _masked_stats(self, x, padding_mask):
        """Compute masked max and mean."""
        if padding_mask is not None:
            x_masked = x.masked_fill(~padding_mask.unsqueeze(-1), float('-inf'))
            x_max, _ = x_masked.max(dim=1)
            x_max = torch.nan_to_num(x_max, nan=0.0)
            
            x_sum = (x * padding_mask.unsqueeze(-1).float()).sum(dim=1)
            x_mean = x_sum / padding_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x_max, _ = x.max(dim=1)
            x_mean = x.mean(dim=1)
        return x_max, x_mean
    
    def forward(
        self,
        frame_features: torch.Tensor,
        cls_token: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            frame_features: (B, T, D) frame-level features from transformer
            cls_token: (B, D) cls token from transformer
            padding_mask: (B, T), True for valid frame positions
        
        Returns:
            logits: (B,) raw logits
            prob: (B,) fake probability
            features: (B, readout_dim) readout features
        """
        B, T, D = frame_features.shape
        
        # 1. Generate attention weights
        attn_weights = self.attention_map(frame_features, padding_mask)  # (B, T, D)
        weighted_features = frame_features * attn_weights  # (B, T, D)
        
        if self.use_dual_path:
            # ===== Temporal Branch =====
            temporal_out, _ = self.temporal_attn(weighted_features, padding_mask)  # (B, T, D)
            temporal_out = self.temporal_norm(temporal_out + weighted_features)
            
            temporal_max, temporal_mean = self._masked_stats(temporal_out, padding_mask)
            
            if self.use_attention_pooling:
                temporal_pooled = self.temporal_pool(temporal_out, padding_mask).squeeze(1)
            
            # ===== Spectral Branch =====
            spectral_out = self.spectral_attn(weighted_features, padding_mask)  # (B, T, D)
            
            spectral_max, spectral_mean = self._masked_stats(spectral_out, padding_mask)
            
            if self.use_attention_pooling:
                spectral_pooled = self.spectral_pool(spectral_out, padding_mask).squeeze(1)
            
            # ===== Master Node Interaction =====
            if self.use_attention_pooling:
                branch_outputs = torch.stack([
                    temporal_pooled, temporal_max, temporal_mean,
                    spectral_pooled, spectral_max, spectral_mean
                ], dim=1)  # (B, 6, D)
            else:
                branch_outputs = torch.stack([
                    temporal_max, temporal_mean,
                    spectral_max, spectral_mean
                ], dim=1)  # (B, 4, D)
            
            master_query = cls_token.unsqueeze(1)  # (B, 1, D)
            master_out, _ = self.master_cross_attn(
                master_query, branch_outputs, branch_outputs,
                need_weights=False
            )
            master_out = self.master_norm(master_out.squeeze(1) + cls_token)
            
            # ===== Readout =====
            if self.use_attention_pooling:
                readout = torch.cat([
                    master_out,
                    temporal_pooled, temporal_max, temporal_mean,
                    spectral_pooled, spectral_max, spectral_mean,
                ], dim=-1)
            else:
                readout = torch.cat([
                    master_out,
                    temporal_max, temporal_mean,
                    spectral_max, spectral_mean,
                ], dim=-1)
        else:
            # Simple path
            frame_max, frame_mean = self._masked_stats(weighted_features, padding_mask)
            readout = torch.cat([cls_token, frame_max, frame_mean], dim=-1)
        
        # 5. Classifier
        logits = self.classifier(readout).squeeze(-1)
        prob = torch.sigmoid(logits)
        
        return logits, prob, readout


class SimpleDetectorWithAASISTBackend(nn.Module):
    """
    Drop-in replacement for SimpleDeepfakeDetector with AASIST-style backend.
    Maintains the same interface.
    """
    def __init__(
        self,
        feature_extractor,
        num_heads=8,
        attn_dropout=0.1,
        use_dual_path=True,
        use_attention_pooling=True,
        use_deep_supervision=False,
        num_supervision_layers=3,
    ):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.use_deep_supervision = use_deep_supervision
        self.num_supervision_layers = num_supervision_layers
        
        # Main backend
        self.backend = AASISTStyleBackend(
            input_dim=1024,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            use_dual_path=use_dual_path,
            use_attention_pooling=use_attention_pooling,
        )
        
        # Deep supervision
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.backend.readout_dim, 256),
                    nn.SELU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1),
                )
                for _ in range(num_supervision_layers)
            ])
            print(f"Deep supervision enabled: {num_supervision_layers} auxiliary heads")
        
        # Print parameter counts
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SimpleDetectorWithAASISTBackend:")
        print(f"  Total params: {total/1e6:.2f}M")
        print(f"  Trainable params: {trainable/1e6:.2f}M")
        print(f"  Frozen params: {(total-trainable)/1e6:.2f}M")
    
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        if_return_feature: bool = False,
    ):
        """
        Same interface as SimpleDeepfakeDetector.
        """
        features = self.feature_extractor(
            input_values=input_values,
            padding_mask=padding_mask,
            return_hidden_states=self.use_deep_supervision,
        )
        
        pooler_output = features["pooler_output"]
        frame_features = features["frame_features"]
        feature_padding_mask = features.get("audio_feature_padding_mask")
        
        # Main classification
        logits, prob, readout = self.backend(
            frame_features=frame_features,
            cls_token=pooler_output,
            padding_mask=feature_padding_mask,
        )
        
        result = {
            "logits": logits,
            "prob": prob,
        }
        
        # Deep supervision
        if self.use_deep_supervision and "hidden_states" in features:
            hidden_states = features["hidden_states"]
            num_layers = len(hidden_states)
            start_idx = num_layers - self.num_supervision_layers - 1
            selected_layers = hidden_states[start_idx:num_layers - 1]
            
            aux_logits = []
            for i, layer_output in enumerate(selected_layers):
                layer_cls = layer_output[:, 0]
                layer_frames = layer_output[:, 1:]
                
                _, _, layer_readout = self.backend(
                    frame_features=layer_frames,
                    cls_token=layer_cls,
                    padding_mask=feature_padding_mask,
                )
                aux_logit = self.aux_heads[i](layer_readout).squeeze(-1)
                aux_logits.append(aux_logit)
            
            result["aux_logits"] = aux_logits
        
        if if_return_feature:
            result["feature"] = readout
        
        return result
    
    @torch.no_grad()
    def predict(self, input_values, padding_mask=None):
        self.eval()
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
        clip_length: int = 144000,
        hop_length: int = 72000,
        aggregation: str = "mean",
    ) -> float:
        self.eval()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        total_len = waveform.shape[1]
        if total_len <= clip_length:
            padding_mask = torch.ones(1, total_len, dtype=torch.bool, device=waveform.device)
            return self.predict(waveform, padding_mask).item()
        
        probs = []
        for start in range(0, total_len - clip_length + 1, hop_length):
            chunk = waveform[:, start:start + clip_length]
            chunk_mask = torch.ones(1, clip_length, dtype=torch.bool, device=chunk.device)
            prob = self.predict(chunk, chunk_mask).item()
            probs.append(prob)
        
        last_start = start + hop_length
        if last_start < total_len:
            tail = waveform[:, last_start:]
            if tail.shape[1] >= clip_length // 2:
                tail_mask = torch.ones(1, tail.shape[1], dtype=torch.bool, device=tail.device)
                prob = self.predict(tail, tail_mask).item()
                probs.append(prob)
        
        if aggregation == "mean":
            return sum(probs) / len(probs)
        elif aggregation == "max":
            return max(probs)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    def save_checkpoint(self, path: str, config=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': config,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {path}")
        return checkpoint.get('config')
