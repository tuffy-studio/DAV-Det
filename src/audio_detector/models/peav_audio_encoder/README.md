# PE-AV Standalone Audio Encoder

从 [Perception Encoder (PE-AV)](https://github.com/facebookresearch/perception_models) 中提取的独立音频编码器，仅依赖音频输入，无需视频或文本分支。

## 模型架构

```
AudioEncoder
├── DAC VAE Encoder (waveform → 128-dim codec features)
│   ├── Conv1d: 1ch → 64ch
│   ├── 4个下采样块 (stride=2,8,10,12, 总下采样率=1920)
│   └── VAE Bottleneck: 1024 → 128
├── Data Projection: 128 → 1024
└── Audio Transformer (16层, hidden_size=1024)
    └── 输出: frame_features (B, T', 1024) + pooler_output (B, 1024)
```

## 参数量

| 组件 | 参数量 |
|------|--------|
| DAC VAE Encoder | 27.67 M |
| Audio Transformer | 209.76 M |
| Data Projection | 0.13 M |
| **总计** | **237.57 M** |

## 依赖

```bash
pip install torch safetensors transformers
# 可选：用于加载音频文件
pip install torchaudio  # 或 torchcodec
```

## 快速开始

```python
from audio_encoder import StandaloneAudioEncoder, AudioProcessor

# 加载模型（从 PE-AV checkpoint）
model = StandaloneAudioEncoder.from_pretrained(
    "/path/to/pe-av-base",  # 包含 config.json 和 model.safetensors 的目录
    output_dim=None,         # 可选：设为 1024 加载 projection head
    device="cuda",
)
model.eval()

# 处理音频
processor = AudioProcessor(sampling_rate=48000, hop_length=1920)

# 方式1：从文件路径加载
inputs = processor(["audio1.wav", "audio2.wav"])

# 方式2：从 tensor 输入（需指定采样率）
import torch
waveform = torch.randn(2, 48000 * 5)  # 5秒，48kHz
inputs = processor(waveform, sampling_rate=48000)

# 前向传播
with torch.no_grad():
    outputs = model(
        input_values=inputs["input_values"],    # (B, 1, T)
        padding_mask=inputs["padding_mask"],    # (B, T)
    )

# 输出
frame_features = outputs["frame_features"]      # (B, T', 1024) 帧级特征
pooler_output = outputs["pooler_output"]        # (B, 1024) 全局音频嵌入
dac_features = outputs["dac_vae_features"]      # (B, T', 128) DAC原始特征
```

## 输入输出说明

### 输入

| 参数 | 形状 | 说明 |
|------|------|------|
| `input_values` | `(B, 1, T)` | 原始音频波形，**采样率 48kHz**，单声道 |
| `padding_mask` | `(B, T)` | 可选，波形级别的 padding mask |
| `input_features` | `(B, T', 128)` | 可选，预计算的 codec features（跳过 DAC）|

### 输出

| 字段 | 形状 | 说明 |
|------|------|------|
| `frame_features` | `(B, T', 1024)` | 音频帧级特征（Transformer 输出）|
| `pooler_output` | `(B, 1024)` | 池化后的全局音频表示（CLS token）|
| `audio_feature_padding_mask` | `(B, T')` | codec 级别的 padding mask |
| `dac_vae_features` | `(B, T', 128)` | DAC VAE 输出的原始 codec features |
| `embedding` | `(B, output_dim)` | 若 `output_dim` 指定，则为投影后的嵌入 |

> 时间帧数 `T' = T / 1920`。例如：10秒音频（480,000采样点）→ 250帧。

## 保存提取后的模型

```python
# 保存为独立的 checkpoint（包含 config.json 和 model.safetensors）
model.save_pretrained("./my_audio_encoder")

# 之后可以直接加载
model = StandaloneAudioEncoder.from_pretrained("./my_audio_encoder")
```

## 许可证

遵循原仓库 [PE-AV](https://github.com/facebookresearch/perception_models) 的许可证。
