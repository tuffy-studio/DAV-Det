# DAV-Det: Technical Report

## 1. Introduction

This technical report describes the methodology, network architecture, data augmentation strategies, and training details of **DAV-Det**, a dual-branch deepfake detection system submitted to the IJCAI 2026 Deepfake Detection Competition.

The task is to determine whether a given video is real or fake. Since a video contains both audio and visual streams, we design two independent detectors:

- **Audio Detector**: detects audio deepfakes using the PE-AV audio encoder with an AASIST-style backend.
- **Video Detector**: detects visual deepfakes using DINOv3-ViT-L/16 with a GPS-DINO multi-granularity classifier.

The final decision is obtained by fusing the predicted probabilities from both modalities.

---

## 2. Problem Formulation

Given a video $v$, the goal is to predict its authenticity label $y \in \{0, 1\}$, where $y=1$ denotes fake and $y=0$ denotes real.

The video can be decomposed into an audio stream $a$ and a visual stream $x = \{x_1, x_2, \dots, x_T\}$, where $x_i$ is the $i$-th sampled frame. We train two independent classifiers:

- Audio classifier: $p_a = f_a(a) \in [0, 1]$
- Video classifier: $p_v = f_v(x) \in [0, 1]$

The fused fake probability is computed as:

$$
p_{\text{fused}} = \max(p_a, p_v)
$$

For the four-class task (RR, FF, FR, RF), we estimate the joint probability distribution:

$$
\begin{aligned}
P(\text{RR}) &= (1 - p_a)(1 - p_v) \\
P(\text{FF}) &= p_a \cdot p_v \\
P(\text{FR}) &= p_a \cdot (1 - p_v) \\
P(\text{RF}) &= (1 - p_a) \cdot p_v
\end{aligned}
$$

---

## 3. Audio Detector

### 3.1 Architecture Overview

The audio detector consists of two main components:

1. **Frontend**: PE-AV `StandaloneAudioEncoder` encodes raw 48kHz mono audio into frame-level features.
2. **Backend**: AASIST-style backend performs temporal and spectral attention for binary classification.

### 3.2 PE-AV Audio Encoder

The input waveform is denoted as $s \in \mathbb{R}^{1 \times T}$, where $T$ is the number of samples at 48kHz.

The encoder first applies a DAC-based VAE encoder to obtain codec features:

$$
Z = \text{DAC-Encoder}(s) \in \mathbb{R}^{T' \times 128}
$$

where $T' = T / 1920$ corresponds to a frame rate of 25 fps.

The features are then projected to 1024 dimensions and processed by a 16-layer decoder-only Transformer:

$$
H = \text{Transformer}(Z W + b) \in \mathbb{R}^{T' \times 1024}
$$

The encoder outputs:

- `frame_features`: $H \in \mathbb{R}^{T' \times 1024}$
- `pooler_output`: $\mathbf{h}_{\text{cls}} \in \mathbb{R}^{1024}$

### 3.3 AASIST-Style Backend

The backend extracts two types of representations:

- **Temporal Branch**: applies masked self-attention over time frames.
- **Spectral Branch**: applies channel-wise attention over feature dimensions.

Let $\mathbf{Q} = \mathbf{K} = \mathbf{V} = H$. The masked self-attention is defined as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

where $M$ is the attention mask derived from the padding mask.

A master node $\mathbf{c} \in \mathbb{R}^{1024}$ interacts with both branches via cross-attention. The readout vector is formed by concatenating:

- the master node
- temporal pooled / max / mean features
- spectral pooled / max / mean features

The readout vector is then fed into a classifier:

$$
\mathbf{r} = [\mathbf{c}; \mathbf{h}_{\text{temp}}^{\text{pool}}; \mathbf{h}_{\text{temp}}^{\text{max}}; \mathbf{h}_{\text{temp}}^{\text{mean}}; \mathbf{h}_{\text{spec}}^{\text{pool}}; \mathbf{h}_{\text{spec}}^{\text{max}}; \mathbf{h}_{\text{spec}}^{\text{mean}}]
$$

$$
p_a = \sigma(W_2 \cdot \text{LayerNorm}(\text{SELU}(W_1 \mathbf{r} + b_1)) + b_2)
$$

where $\sigma(\cdot)$ is the sigmoid function.

### 3.4 Deep Supervision

During training, the last $N$ Transformer layer outputs are also fed into the backend to produce auxiliary logits $\{\hat{y}^{(1)}, \hat{y}^{(2)}, \dots, \hat{y}^{(N)}\}$. The total loss is:

$$
\mathcal{L}_{\text{audio}} = \mathcal{L}_{\text{focal}}(y, \hat{y}) + \lambda_{\text{aux}} \sum_{i=1}^{N} \mathcal{L}_{\text{focal}}(y, \hat{y}^{(i)})
$$

where $\lambda_{\text{aux}}$ is the auxiliary loss weight.

### 3.5 LoRA Fine-Tuning

To reduce trainable parameters, Low-Rank Adaptation (LoRA) is injected into the query and value projection layers of the audio backbone. The update is:

$$
W' = W + \frac{\alpha}{r} B A
$$

where $W \in \mathbb{R}^{d \times d}$ is the pretrained weight, $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, $r$ is the LoRA rank, and $\alpha$ is the scaling factor. The audio detector uses $r=32$ and $\alpha=64$.

### 3.6 Loss Function

We use Sigmoid Focal Loss:

$$
\mathcal{L}_{\text{focal}}(y, \hat{y}) = -\alpha y (1 - p)^{\gamma_{+}} \log(p) - (1 - \alpha)(1 - y) p^{\gamma_{-}} \log(1 - p)
$$

where $p = \sigma(\hat{y})$, $\alpha$ balances positive and negative samples, and $\gamma_{+}$, $\gamma_{-}$ control the down-weighting of easy examples.

### 3.7 Data Augmentation

#### Temporal Augmentation

During training, each audio clip is randomly cropped with a duration jitter of up to $\pm 1$ second around the target clip length. Given a target clip length $L = 48{,}000 \times 3 = 144{,}000$ samples (3 seconds at 48kHz), the actual cropped length is:

$$
L' = L + \Delta, \quad \Delta \sim \mathcal{U}[-48{,}000, 48{,}000]
$$

If the audio is longer than $L'$, a random crop is applied; if shorter, zero-padding is applied in `collate_fn`.

#### Spectral and Codec Augmentations

The following augmentations are further applied to the cropped clip:

- Gaussian noise
- Pitch shift
- Synthetic reverberation
- MP3 compression
- Random volume scaling

Each training sample randomly applies $k \in [0, \text{num\_augment}]$ augmentations with intensity $i \in [1, \text{augment\_intensity}]$.

---

## 4. Video Detector

### 4.1 Architecture Overview

The video detector is built on DINOv3-ViT-L/16 as the backbone, followed by GPS-DINO for multi-granularity deepfake classification.

### 4.2 DINOv3 Backbone

Each sampled frame $x_i \in \mathbb{R}^{3 \times H \times W}$ is resized to $640 \times 640$ and patchified into tokens of size $16 \times 16$. The ViT-L/16 backbone extracts patch tokens, register tokens, and a CLS token from specified layers.

### 4.3 GPS-DINO Classifier

GPS-DINO (Global-Patch-Segment DINO) is the core classification head built on top of DINOv3. It performs deepfake detection at three complementary granularities: global image-level, local patch-level, and semantic segment-level. Each granularity is supervised at multiple DINOv3 layers (21, 22, 23, 24) when deep supervision is enabled.

#### 4.3.1 Multi-Layer Feature Extraction

The DINOv3 backbone returns features from the layers specified by `layer_indices`. By default, the model extracts outputs from layers 21, 22, 23, and 24. Let $\ell \in \{21, 22, 23, 24\}$ denote a layer index. For a batch of $B$ images, the output at layer $\ell$ consists of:

- CLS token: $\mathbf{c}^{(\ell)} \in \mathbb{R}^{B \times 1024}$
- Register tokens (not used for classification)
- Patch tokens: $\mathbf{P}^{(\ell)} \in \mathbb{R}^{B \times N \times 1024}$, where $N=1024$ for ViT-L/16 at $640 \times 640$ resolution

During inference, only the last layer (24) is used.

#### 4.3.2 Global Branch

The global branch directly classifies the normalized CLS token:

$$
\hat{y}_{\text{global}}^{(\ell)} = \text{MLP}_{\text{global}}^{(\ell)}\left(\text{LayerNorm}\left(\mathbf{c}^{(\ell)}\right)\right)
$$

where each $\text{MLP}_{\text{global}}^{(\ell)}$ is a small two-layer MLP with hidden size 256 and dropout 0.1.

#### 4.3.3 Patch Branch

The patch branch aggregates local patch tokens using `Patch_Classifier_Reducer`. This module first computes an importance score for each patch token:

$$
s_i = \text{MLP}_{\text{patch-score}}\left(\text{LayerNorm}\left(\mathbf{p}_i^{(24)}\right)\right) \in \mathbb{R}
$$

where $\mathbf{p}_i^{(24)} \in \mathbb{R}^{1024}$ is the $i$-th patch token at layer 24. The scores are converted to attention weights via softmax with temperature $\tau=0.07$:

$$
w_i = \frac{\exp(s_i / \tau)}{\sum_{j=1}^{N} \exp(s_j / \tau)}
$$

The aggregated patch feature is then:

$$
\mathbf{f}_{\text{patch}}^{(24)} = \sum_{i=1}^{N} w_i \mathbf{p}_i^{(24)} \in \mathbb{R}^{1024}
$$

The patch classifier produces:

$$
\hat{y}_{\text{patch}}^{(24)} = \text{MLP}_{\text{patch}}\left(\text{LayerNorm}\left(\mathbf{f}_{\text{patch}}^{(24)}\right)\right)
$$

**MIL auxiliary outputs.** The reducer additionally provides weak/strong multiple-instance-learning signals. Given `topk_ratio=0.05`, it selects the top $k = \max(1, \lfloor 0.05N \rfloor)$ patches with highest scores:

$$
\hat{y}_{\text{weak-patch}}^{(24)} = \frac{1}{k} \sum_{i \in \text{top-}k} s_i, \quad
\hat{y}_{\text{rest-patch}}^{(24)} = \frac{1}{N-k} \sum_{i \notin \text{top-}k} s_i
$$

These logits encourage the model to assign higher fake scores to the most suspicious patches.

#### 4.3.4 Segment Branch

The segment branch groups semantically similar patches into clusters and classifies the resulting segment prototypes. This captures forgery artifacts that span multiple adjacent patches (e.g., unnatural face boundaries, inconsistent lighting).

**Clustering.** To ensure stable clustering, patch tokens are obtained from the **frozen DINOv3 backbone without LoRA** (`use_lora=False`). Each patch token is L2-normalized, and pairwise cosine similarities are computed:

$$
S_{ij} = \frac{\mathbf{p}_i^{\top} \mathbf{p}_j}{\|\mathbf{p}_i\| \|\mathbf{p}_j\|}
$$

Agglomerative clustering with average linkage is applied to the distance matrix $D_{ij} = 1 - S_{ij}$, using a distance threshold of $1 - \tau$ where $\tau=0.9$. This yields a set of clusters:

$$
\mathcal{C} = \{C_1, C_2, \dots, C_K\}
$$

**Prototype generation.** For each cluster $C_k$, a segment prototype is computed by averaging the LoRA-augmented patch tokens belonging to that cluster:

$$
\mathbf{u}_k^{(24)} = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{p}_i^{(24)}
$$

The set of prototypes forms $\mathbf{U}^{(24)} \in \mathbb{R}^{K \times 1024}$.

**Segment classification.** The `Segment_Classifier_Reducer` applies the same attention-weighted aggregation as the patch reducer, but over segment prototypes instead of raw patches:

$$
\mathbf{f}_{\text{segment}}^{(24)} = \sum_{k=1}^{K} \tilde{w}_k \mathbf{u}_k^{(24)} \in \mathbb{R}^{1024}
$$

$$
\hat{y}_{\text{segment}}^{(24)} = \text{MLP}_{\text{segment}}\left(\text{LayerNorm}\left(\mathbf{f}_{\text{segment}}^{(24)}\right)\right)
$$

with `topk_ratio=0.1` for segments. MIL weak/rest logits are likewise computed over segment prototypes.

#### 4.3.5 Main Classifier

The main classifier combines global, patch, and segment features into a single comprehensive representation:

$$
\mathbf{f}_{\text{overall}}^{(24)} = \left[
\text{LayerNorm}\left(\mathbf{c}^{(24)}\right);
\text{LayerNorm}\left(\mathbf{f}_{\text{patch}}^{(24)}\right);
\text{LayerNorm}\left(\mathbf{f}_{\text{segment}}^{(24)}\right)
\right] \in \mathbb{R}^{3072}
$$

where $[\cdot; \cdot; \cdot]$ denotes concatenation. This 3072-dimensional vector is fed into a three-layer MLP with hidden sizes $[2048, 1024]$:

$$
\hat{y}_{\text{main}}^{(24)} = \text{MLP}_{\text{main}}\left(\mathbf{f}_{\text{overall}}^{(24)}\right)
$$

#### 4.3.6 Deep Supervision Branch Outputs

When `use_deep_supervision=True`, separate classifiers and reducers are instantiated for layers 21, 22, and 23 in addition to layer 24. The full set of training outputs includes:

| Output | Description |
|---|---|
| `main_logits_24` | Final fused prediction from layer 24 |
| `global_logits_21/22/23/24` | CLS-based predictions at each layer |
| `patch_logits_21/22/23/24` | Aggregated patch predictions at each layer |
| `segment_logits_21/22/23/24` | Aggregated segment predictions at each layer |
| `weak_patch_logits_21/22/23/24` | MIL top-k patch logits |
| `rest_patch_logits_21/22/23/24` | MIL rest-of-patches logits |
| `weak_segment_logits_21/22/23/24` | MIL top-k segment logits |
| `rest_segment_logits_21/22/23/24` | MIL rest-of-segments logits |



### 4.4 LoRA Fine-Tuning

The DINOv3-ViT-L/16 backbone is kept mostly frozen. LoRA is injected into the query and value projection layers of the Transformer blocks. The adapted weight is computed as:

$$
W' = W + \frac{\alpha}{r} B A
$$

where $W$ is the pretrained weight, $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, $r$ is the LoRA rank, and $\alpha$ is the scaling factor. The video detector uses $r=32$ and $\alpha=16$.

### 4.5 Training Losses

The video detector is trained with a combination of classification losses, MIL margin regularization, and consistency regularization. During each iteration, two views of the same image are generated: a strongly augmented "degraded" view and a weakly augmented "origin" view. Both views pass through the model, and losses are computed on both.

#### 4.5.1 Classification Losses

The primary classification losses are applied to the main, global, patch, and segment logits, as well as to the weak MIL logits:

$$
\mathcal{L}_{\text{cls}} = \mathcal{L}_{\text{main}} + \mathcal{L}_{\text{global}} + \mathcal{L}_{\text{patch}} + \mathcal{L}_{\text{segment}} + \mathcal{L}_{\text{weak-patch}} + \mathcal{L}_{\text{weak-segment}}
$$

Each term uses either Focal Loss or Cross-Entropy depending on the configuration. Deep supervision extends these losses to layers 21, 22, and 23.

#### 4.5.2 MIL Margin Regularization Loss

To enforce interpretable attention weights, a margin loss is applied between the weak (top-k) logits and the rest-of-tokens logits. Intuitively, for a fake sample, the most suspicious patches/segments should score higher than the remaining ones by at least a margin $m$.

For the patch branch at layer 24:

$$
\mathcal{L}_{\text{reg-patch}}^{(24)} = \frac{1}{B} \sum_{b=1}^{B} \max\left(0, m - y_b \cdot \left(\hat{y}_{\text{weak-patch},b}^{(24)} - \hat{y}_{\text{rest-patch},b}^{(24)}\right)\right)
$$

where $y_b \in \{0, 1\}$ is the label and $m=0.6$ is the margin. The same loss is applied to the segment branch:

$$
\mathcal{L}_{\text{reg-segment}}^{(24)} = \frac{1}{B} \sum_{b=1}^{B} \max\left(0, m - y_b \cdot \left(\hat{y}_{\text{weak-segment},b}^{(24)} - \hat{y}_{\text{rest-segment},b}^{(24)}\right)\right)
$$

This regularization is computed for both the degraded and origin views, and for all supervised layers (21, 22, 23, 24).

#### 4.5.3 Consistency Regularization Loss

The model encourages the representations extracted from the degraded view and the origin view to be similar. For each granularity $g \in \{\text{global}, \text{patch}, \text{segment}\}$, the cosine similarity between the two views is computed:

$$
\text{sim}_g = \frac{\mathbf{f}_g^{\text{degraded}} \cdot \mathbf{f}_g^{\text{origin}}}{\|\mathbf{f}_g^{\text{degraded}}\| \|\mathbf{f}_g^{\text{origin}}\|}
$$

The consistency loss is then:

$$
\mathcal{L}_{\text{consistency},g} = 1 - \text{sim}_g
$$

The total consistency loss aggregates all granularities and all supervised layers:

$$
\mathcal{L}_{\text{consistency}} = \sum_{\ell \in \{21,22,23,24\}} \sum_{g \in \{\text{global}, \text{patch}, \text{segment}\}} \mathcal{L}_{\text{consistency},g}^{(\ell)}
$$

#### 4.5.4 Total Training Loss

The overall video training objective is:

$$
\mathcal{L}_{\text{video}} = \mathcal{L}_{\text{cls}}^{\text{degraded}} + \mathcal{L}_{\text{cls}}^{\text{origin}} + \mathcal{L}_{\text{reg}}^{\text{degraded}} + \mathcal{L}_{\text{reg}}^{\text{origin}} + 0.05 \cdot \mathcal{L}_{\text{consistency}}
$$

where the consistency term is weighted by $0.05$.

### 4.6 Data Augmentation

Video training applies strong image augmentations:

- Geometric transforms (rotation, flipping, cropping)
- Blur and noise
- JPEG compression
- Color jitter
- Self-mixup
- Dynamic resolution: The training resolution is randomly sampled between 384 and 768 pixels and constrained to be divisible by the patch size 16.

For each training sample, both a degraded view(img_size=384~768) and a weakly augmented original view (img_size=512) are generated for consistency learning.

---

## 5. Training Details

### 5.1 Audio Detector

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LoRA learning rate | $10^{-4}$ |
| Head learning rate | $10^{-4}$ |
| Batch size | 8 |
| Clip length | 3s at 48kHz (144,000 samples) |
| LoRA rank $r$ | 32 |
| LoRA alpha $\alpha$ | 64 |
| Loss | Focal Loss |
| Scheduler | Warmup + Cosine decay |

### 5.2 Video Detector

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | $10^{-4}$ |
| Batch size | 128 with 16 gradient accumulation steps |
| Image size | training:512 inference:640 |
| LoRA rank $r$ | 32 |
| LoRA alpha $\alpha$ | 16 |
| Loss | Focal Loss|
| Scheduler | Warmup + Cosine decay |

---

## 6. Inference and Fusion

### 6.1 Audio Inference

The audio inference script splits each audio file into non-overlapping 3-second clips. Each clip is independently classified, and the final probability is aggregated by mean pooling.

### 6.2 Video Inference

The video inference script samples 16 frames per video. Frame-level probabilities are averaged to obtain the video-level probability.

### 6.3 Fusion

The binary fake probability is computed as:

$$
p_{\text{fake}} = \max(p_a, p_v)
$$

The four-class probabilities are:

| Class | Meaning | Probability |
|---|---|---|
| 0 (RR) | Real audio + Real video | $(1-p_a)(1-p_v)$ |
| 1 (FF) | Fake audio + Fake video | $p_a p_v$ |
| 2 (FR) | Fake audio + Real video | $p_a(1-p_v)$ |
| 3 (RF) | Real audio + Fake video | $(1-p_a)p_v$ |

---
