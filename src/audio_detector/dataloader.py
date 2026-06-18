"""
DataLoader for Simple Deepfake Detector

Training:
    - Audio > 3s: randomly crop to 3s
    - Audio < 3s: keep original length, zero-pad in collate_fn
    - Generate padding_mask for valid positions

Evaluation:
    - Audio > 3s: keep full length (will be chunked during inference)
    - Audio < 3s: keep original length
    - Generate padding_mask
"""

import os
import csv
import random
import tempfile
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List


# Label mapping
LABEL_MAP = {
    "real": 0,
    "fake": 1,
}


def snr_to_noise_factor(signal: torch.Tensor, snr_db: float) -> float:
    """Compute noise scaling factor for a given SNR (dB)."""
    signal_power = signal.pow(2).mean().item()
    if signal_power == 0:
        return 0.005
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    return np.sqrt(noise_power)


def add_reverb(waveform: torch.Tensor, sr: int, reverberance: float) -> torch.Tensor:
    """
    Add reverberation using a fast approximation with delayed attenuated copies.
    Much faster than full convolution with synthetic RIR.
    
    Args:
        waveform: (1, T) or (T,)
        sr: sampling rate
        reverberance: 0-100, controls decay time (higher = more reverb)
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    # Convert reverberance to delay parameters
    rt60 = 0.1 + (reverberance / 100.0) * 1.5  # 0.1s to 1.6s
    
    # Number of reflections and delay
    num_reflections = int(3 + reverberance / 20)  # 3 to 8 reflections
    delay_ms = 20 + reverberance * 0.5  # 20ms to 70ms base delay
    delay_samples = int(sr * delay_ms / 1000)
    
    # Mix dry and wet
    wet_ratio = 0.15 + (reverberance / 100.0) * 0.35  # 0.15 to 0.5
    
    # Generate wet signal with multiple delayed attenuated copies
    wet = torch.zeros_like(waveform)
    device = waveform.device
    
    for i in range(num_reflections):
        delay = delay_samples * (i + 1)
        if delay >= waveform.shape[1]:
            break
        # Attenuation increases with each reflection
        attenuation = (0.5 ** (i + 1)) * (1.0 - reverberance / 200.0)
        wet[:, delay:] += waveform[:, :-delay] * attenuation
    
    mixed = (1 - wet_ratio) * waveform + wet_ratio * wet
    
    return mixed


def apply_audio_compression(waveform: torch.Tensor, sr: int, bitrate: str) -> torch.Tensor:
    """
    Apply audio compression by encoding/decoding through MP3.
    
    Args:
        waveform: (1, T) or (T,)
        sr: sampling rate
        bitrate: e.g. '192k', '128k', '64k'
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    # Save to temp file, encode as MP3, then decode
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name
    
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
        tmp_mp3_path = tmp_mp3.name
    
    try:
        # Save as WAV first
        torchaudio.save(tmp_wav_path, waveform, sr)
        
        # Use ffmpeg to compress to MP3
        import subprocess
        cmd = [
            'ffmpeg', '-y', '-i', tmp_wav_path,
            '-ar', str(sr),
            '-b:a', bitrate,
            tmp_mp3_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0 or not os.path.exists(tmp_mp3_path) or os.path.getsize(tmp_mp3_path) == 0:
            # ffmpeg failed or not available, skip compression
            return waveform.squeeze(0)
        
        # Load back
        try:
            compressed, sr_loaded = torchaudio.load(tmp_mp3_path)
        except Exception:
            # MP3 file may be corrupted, skip compression
            return waveform.squeeze(0)
        
        # Resample if needed
        if sr_loaded != sr:
            compressed = torchaudio.functional.resample(compressed, sr_loaded, sr)
        
        # Trim/pad to original length
        if compressed.shape[1] > waveform.shape[1]:
            compressed = compressed[:, :waveform.shape[1]]
        elif compressed.shape[1] < waveform.shape[1]:
            pad_len = waveform.shape[1] - compressed.shape[1]
            compressed = torch.nn.functional.pad(compressed, (0, pad_len))
        
        return compressed.squeeze(0)
    
    finally:
        # Cleanup
        for p in [tmp_wav_path, tmp_mp3_path]:
            if os.path.exists(p):
                os.remove(p)


class ADDatasetSimple(Dataset):
    """
    ADD Dataset for simple binary classification.
    
    Supports two CSV formats:
    1. Old: name,label (relative path, needs audio_dir)
    2. New: file_path,label (absolute path, audio_dir optional)
    """
    def __init__(
        self,
        csv_path: str,
        audio_dir: str = "",
        sampling_rate: int = 48000,
        clip_length: Optional[int] = None,  # e.g., 48000 * 3 = 144000 for 3s
        mode: str = "train",  # "train" or "eval"
        augment: bool = False,
        augment_intensity: int = 3,  # 1-5, controls perturbation strength
        num_augment: int = 2,  # number of augmentations to apply each time (1-4)
    ):
        self.audio_dir = audio_dir
        self.sampling_rate = sampling_rate
        self.clip_length = clip_length  # in samples
        self.mode = mode
        self.augment = augment and (mode == "train")
        self.augment_intensity = max(1, min(5, augment_intensity))  # clamp to 1-5
        self.num_augment = max(1, min(4, num_augment))  # clamp to 1-4
        
        # Read CSV
        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support both old format (name) and new format (file_path)
                if 'file_path' in row:
                    # New format: absolute path
                    file_path = row['file_path']
                    # Extract filename for display
                    name = os.path.basename(file_path)
                else:
                    # Old format: relative path
                    file_path = os.path.join(self.audio_dir, row['name'])
                    name = row['name']
                
                # Support both string label and numeric label
                label_str = row.get('label', row.get('type', ''))
                if label_str in LABEL_MAP:
                    label = LABEL_MAP[label_str]
                else:
                    label = int(label_str)
                
                self.samples.append({
                    'name': name,
                    'file_path': file_path,
                    'label': label,
                })
        
        print(f"Loaded {len(self.samples)} samples from {csv_path} [{mode}]")
        
        # Label distribution
        labels = [s['label'] for s in self.samples]
        print(f"  real={labels.count(0)}, fake={labels.count(1)}")
        
        if self.augment:
            print(f"  Augmentation enabled (intensity: 1-{self.augment_intensity}, num: {self.num_augment})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio (use absolute path if available)
        audio_path = sample.get('file_path', os.path.join(self.audio_dir, sample['name']))
        
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            # Return a random other sample instead of corrupted one
            print(f"Warning: Failed to load {audio_path}, using random sample instead")
            random_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_idx)
        
        # Resample to target sampling rate
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Training mode: clip to fixed length
        if self.mode == "train" and self.clip_length is not None:
            random_length = random.randint(-self.sampling_rate, self.sampling_rate)
            clip_length = self.clip_length + random_length  # add up to ±1s random jitter
            if waveform.shape[1] > clip_length:
                # Random crop
                start = random.randint(0, waveform.shape[1] - clip_length)
                waveform = waveform[:, start:start + clip_length]
            # If shorter than clip_length, keep as is (will be padded in collate_fn)
        
        # Data augmentation (training only)
        if self.augment:
            waveform = self._augment(waveform)
        
        return {
            'audio': waveform.squeeze(0),  # (T,)
            'label': torch.tensor(sample['label'], dtype=torch.float),
            'name': sample['name'],
        }
    
    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation.
        
        Each call:
        - Randomly samples intensity from 1 to augment_intensity
        - Randomly selects num_augment augmentations from available types
        
        """
        # Random intensity for this sample
        intensity = random.randint(1, self.augment_intensity)
        
        # Define all augmentation types
        augmentations = []
        
        # --- 1. Gaussian Noise ---
        def aug_noise(wf):
            snr_values = {1: 40, 2: 30, 3: 20, 4: 15, 5: 10}
            snr_db = snr_values[intensity]
            noise_factor = snr_to_noise_factor(wf, snr_db)
            noise = torch.randn_like(wf) * noise_factor
            return wf + noise
        augmentations.append(('noise', aug_noise))
        
        # --- 2. Pitch Shift (fast numpy-based linear interpolation) ---
        def aug_pitch(wf):
            pitch_steps = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
            max_steps = pitch_steps[intensity]
            n_steps = random.uniform(-max_steps, max_steps)
            try:
                factor = 2 ** (-n_steps / 12.0)
                orig_len = wf.shape[-1]
                
                if wf.ndim == 2:
                    wf_np = wf.squeeze(0).numpy()
                else:
                    wf_np = wf.numpy()
                
                x_old = np.arange(orig_len)
                new_len = int(orig_len * factor)
                x_new = np.linspace(0, orig_len - 1, new_len)
                wf_shifted = np.interp(x_new, x_old, wf_np)
                
                wf_shifted = torch.from_numpy(wf_shifted).float()
                if wf.ndim == 2:
                    wf_shifted = wf_shifted.unsqueeze(0)
                
                if wf_shifted.shape[-1] > orig_len:
                    start = (wf_shifted.shape[-1] - orig_len) // 2
                    wf_shifted = wf_shifted[..., start:start + orig_len]
                elif wf_shifted.shape[-1] < orig_len:
                    pad_len = orig_len - wf_shifted.shape[-1]
                    wf_shifted = torch.nn.functional.pad(wf_shifted, (0, pad_len))
                
                return wf_shifted
            except Exception:
                return wf
        augmentations.append(('pitch', aug_pitch))
        
        # --- 3. Synthetic Reverberation ---
        def aug_reverb(wf):
            reverb_values = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}
            reverberance = reverb_values[intensity]
            wf_out = add_reverb(wf, self.sampling_rate, reverberance)
            if wf_out.ndim == 2:
                wf_out = wf_out.squeeze(0)
            return wf_out
        augmentations.append(('reverb', aug_reverb))
        
        # --- 4. Audio Compression ---
        def aug_compress(wf):
            bitrate_values = {1: '320k', 2: '256k', 3: '192k', 4: '128k', 5: '64k'}
            bitrate = bitrate_values[intensity]
            return apply_audio_compression(wf, self.sampling_rate, bitrate)
        augmentations.append(('compress', aug_compress))
        
        # Randomly select num_augment augmentations
        num_to_apply = random.randint(0, min(self.num_augment, len(augmentations)))
        selected = random.sample(augmentations, num_to_apply)
        
        # Apply selected augmentations in random order
        for name, aug_fn in selected:
            waveform = aug_fn(waveform)
        
        # Always apply random volume scaling (lightweight, always-on baseline)
        gain = random.uniform(0.8, 1.2)
        waveform = waveform * gain
        
        # Clamp to valid range
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for variable-length audio.
    Pads to max length in batch and generates padding_mask.
    """
    audios = [item['audio'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    names = [item['name'] for item in batch]
    
    # Get max length in this batch
    max_len = max(a.shape[0] for a in audios)
    
    # Pad and create masks
    padded_audios = []
    padding_masks = []
    for audio in audios:
        pad_len = max_len - audio.shape[0]
        if pad_len > 0:
            padded = torch.nn.functional.pad(audio, (0, pad_len), value=0)
            mask = torch.cat([
                torch.ones(audio.shape[0], dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
            ])
        else:
            padded = audio
            mask = torch.ones(audio.shape[0], dtype=torch.bool)
        padded_audios.append(padded)
        padding_masks.append(mask)
    
    return {
        'audio': torch.stack(padded_audios).unsqueeze(1),  # (B, 1, T)
        'padding_mask': torch.stack(padding_masks),  # (B, T)
        'label': labels,  # (B,)
        'name': names,
    }


def get_dataloader(
    csv_path: str,
    audio_dir: str,
    batch_size: int = 16,
    sampling_rate: int = 48000,
    clip_length: Optional[int] = None,
    mode: str = "train",
    augment: bool = False,
    augment_intensity: int = 3,
    num_augment: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create DataLoader.
    
    Args:
        csv_path: CSV label file path
        audio_dir: Audio file directory
        batch_size: Batch size
        sampling_rate: Target sampling rate
        clip_length: Max length in samples (None = no clipping)
        mode: "train" or "eval"
        augment: Whether to apply augmentation
        augment_intensity: Maximum intensity level 1-5 (each sample randomly picks 1~this)
        num_augment: Number of augmentations to apply per sample (1-4)
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
    """
    dataset = ADDatasetSimple(
        csv_path=csv_path,
        audio_dir=audio_dir,
        sampling_rate=sampling_rate,
        clip_length=clip_length,
        mode=mode,
        augment=augment,
        augment_intensity=augment_intensity,
        num_augment=num_augment,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )


if __name__ == "__main__":
    # Test
    train_loader = get_dataloader(
        csv_path='/data/data2/jielun/ADD/label/train.csv',
        audio_dir='/data/data2/jielun/ADD/train',
        batch_size=4,
        clip_length=48000 * 3,  # 3s
        mode="train",
        augment=True,
        augment_intensity=5,
        num_augment=2,
        num_workers=0,
    )
    
    for batch in train_loader:
        print("Audio shape:", batch['audio'].shape)
        print("Padding mask shape:", batch['padding_mask'].shape)
        print("Labels:", batch['label'])
        print("Names:", batch['name'])
        
        # Check lengths
        lengths = batch['padding_mask'].sum(dim=1)
        print("Actual lengths (samples):", lengths.tolist())
        print("Actual lengths (seconds):", (lengths / 48000).tolist())
        break
