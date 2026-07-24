import os
import sys
import argparse
import csv
import tempfile
import subprocess
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio

def _append_row(path: str, fieldnames: list, row: dict):
    """Append a single row to a CSV file (create with header if not exists)."""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# Fix numpy version compatibility before importing torch
import numpy as np
if not hasattr(np, '_core'):
    np._core = np.core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from audio_detector import build_detector


CLIP_LENGTH = 48000 * 3   # 3s @ 48kHz
HOP_LENGTH = 48000 * 3    # 3s @ 48kHz


def get_audio_path(file_path: str, sampling_rate: int = 48000) -> str:
    """Get audio path: if wav exists use it, else extract from video to same dir."""
    # If already audio file, return directly
    if file_path.endswith(('.wav', '.flac', '.mp3', '.ogg', '.m4a')):
        return file_path
    
    # Check if corresponding wav exists in same directory
    base_path = os.path.splitext(file_path)[0]
    wav_path = base_path + '.wav'
    
    if os.path.exists(wav_path):
        return wav_path
    
    # Extract audio from video to same directory
    print(f"Extracting audio from {file_path} -> {wav_path}")
    cmd = [
        'ffmpeg',
        '-y',
        '-i', file_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', str(sampling_rate),
        '-ac', '1',
        wav_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        if os.path.exists(wav_path):
            return wav_path
        else:
            raise RuntimeError(f"FFmpeg succeeded but {wav_path} not created")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout extracting audio from {file_path}")


def load_audio(file_path: str, sampling_rate: int = 48000):
    """Load and preprocess audio, return waveform (T,) or None on failure."""
    try:
        audio_path = get_audio_path(file_path, sampling_rate)
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None
    
    # Resample
    if sr != sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform.squeeze(0)  # (T,)


def split_into_chunks(waveform: torch.Tensor, clip_length: int, hop_length: int):
    """
    Split waveform into overlapping chunks.
    
    Args:
        waveform: (T,) tensor
        clip_length: chunk length in samples
        hop_length: hop size in samples
    
    Returns:
        List of (chunk, mask) tuples, where chunk is (clip_length,) and mask is (clip_length,)
    """
    total_len = waveform.shape[0]
    chunks = []
    
    if total_len <= clip_length:
        # Only one chunk, pad to clip_length
        pad_len = clip_length - total_len
        if pad_len > 0:
            chunk = torch.nn.functional.pad(waveform, (0, pad_len), value=0)
            mask = torch.cat([
                torch.ones(total_len, dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
            ])
        else:
            chunk = waveform
            mask = torch.ones(clip_length, dtype=torch.bool)
        chunks.append((chunk, mask))
    else:
        # Sliding window
        for start in range(0, total_len, hop_length):
            end = start + clip_length
            if end > total_len:
                # Last chunk: take the last clip_length samples
                start = total_len - clip_length
                end = total_len
                chunk = waveform[start:end]
                mask = torch.ones(clip_length, dtype=torch.bool)
                chunks.append((chunk, mask))
                break
            else:
                chunk = waveform[start:end]
                mask = torch.ones(clip_length, dtype=torch.bool)
                chunks.append((chunk, mask))
    
    return chunks


def inference_file(model, waveform: torch.Tensor, device: torch.device,
                   clip_length: int = CLIP_LENGTH, hop_length: int = HOP_LENGTH,
                   aggregation: str = "mean"):
    """
    Run inference on a single audio file by splitting into chunks and batching them.
    
    Args:
        model: detector model
        waveform: (T,) audio tensor
        device: torch device
        clip_length: chunk length in samples
        hop_length: hop size in samples
        aggregation: "mean" or "max"
    
    Returns:
        prob: float, aggregated fake probability (1.0 if all chunks fail)
        failed_chunks: int, number of failed chunks
    """
    chunks = split_into_chunks(waveform, clip_length, hop_length)
    
    if len(chunks) == 0:
        return 1.0, 0  # fallback for empty audio
    
    # Stack all chunks into a batch: (N, clip_length)
    audio_batch = torch.stack([c for c, _ in chunks]).unsqueeze(1).to(device)  # (N, 1, clip_length)
    mask_batch = torch.stack([m for _, m in chunks]).to(device)                # (N, clip_length)
    
    valid_logits = []
    failed_chunks = 0
    
    try:
        with torch.no_grad():
            outputs = model(audio_batch, mask_batch)
            logits = outputs['logits'].cpu().numpy()  # (N,)
            
            # Check for NaN/Inf in logits
            for i, z in enumerate(logits):
                if np.isnan(z) or np.isinf(z):
                    failed_chunks += 1
                    print(f"Warning: Chunk {i}/{len(chunks)} produced invalid logit ({z}), skipping")
                else:
                    valid_logits.append(z)
    except Exception as e:
        print(f"Warning: Batch inference failed: {e}. Falling back to chunk-by-chunk.")
        # Fallback: process chunk by chunk
        for i, (chunk, mask) in enumerate(chunks):
            try:
                chunk_batch = chunk.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T)
                mask_batch_i = mask.unsqueeze(0).to(device)               # (1, T)
                with torch.no_grad():
                    out = model(chunk_batch, mask_batch_i)
                    z = out['logits'].item()
                    if np.isnan(z) or np.isinf(z):
                        raise ValueError(f"Invalid logit: {z}")
                    valid_logits.append(z)
            except Exception as e2:
                failed_chunks += 1
                print(f"Warning: Chunk {i}/{len(chunks)} failed: {e2}")
    
    # Aggregate valid logits, then convert to prob
    if len(valid_logits) == 0:
        # All chunks failed
        print(f"Warning: All {len(chunks)} chunks failed, returning prob=1.0")
        return 1.0, failed_chunks
    
    if aggregation == "mean":
        # Average logits first, then sigmoid
        aggregated_logit = float(np.mean(valid_logits))
        prob = float(1.0 / (1.0 + np.exp(-aggregated_logit)))  # sigmoid
    elif aggregation == "max":
        # Max logits first, then sigmoid
        aggregated_logit = float(np.max(valid_logits))
        prob = float(1.0 / (1.0 + np.exp(-aggregated_logit)))  # sigmoid
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return prob, failed_chunks


def load_model(checkpoint_path: str, device: str, peav_checkpoint: str = None):
    """Load model from checkpoint or raw state dict.

    Supports two checkpoint formats:
      1. Full training checkpoint (dict with 'model_state_dict' and 'config')
      2. Raw state dict only (e.g. extracted by extract_model_state.py)

    For raw state dict, architecture flags are inferred from the keys and
    peav_checkpoint must be provided either via argument or in the config.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(raw, dict) and 'model_state_dict' in raw:
        checkpoint = raw
        config = checkpoint.get('config', {})
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        print("Detected full training checkpoint")
    else:
        # Raw state dict
        state_dict = raw
        config = {}
        epoch = 'unknown (raw state dict)'
        print("Detected raw model state dict")

    # Override peav_checkpoint if provided explicitly
    if peav_checkpoint is not None:
        config['peav_checkpoint'] = peav_checkpoint

    if 'peav_checkpoint' not in config:
        raise ValueError(
            "peav_checkpoint is required to build the model. "
            "Pass --peav_checkpoint or use a full checkpoint with config."
        )

    # Auto-detect backend type from state_dict keys
    has_backend = any('backend.' in k for k in state_dict.keys())
    if not has_backend:
        raise ValueError("Unsupported checkpoint: audio detector backend not found")

    # Infer architecture flags from state dict if not in config
    use_lora = config.get('use_lora', any('lora_' in k for k in state_dict.keys()))
    use_deep_supervision = config.get(
        'use_deep_supervision',
        any(k.startswith('aux_heads.') for k in state_dict.keys())
    )
    num_supervision_layers = config.get('num_supervision_layers', 0)
    if use_deep_supervision and num_supervision_layers == 0:
        # Infer from aux_heads.N.xxx keys
        aux_indices = [
            int(k.split('.')[1])
            for k in state_dict.keys()
            if k.startswith('aux_heads.')
        ]
        num_supervision_layers = max(aux_indices) + 1 if aux_indices else 3

    print("Detected audio detector backend checkpoint")
    model = build_detector(
        peav_checkpoint=config['peav_checkpoint'],
        use_lora=use_lora,
        lora_r=config.get('lora_r', 32),
        lora_alpha=config.get('lora_alpha', 64),
        lora_dropout=config.get('lora_dropout', 0.1),
        unfreeze_norm=config.get('unfreeze_norm', True),
        device=device,
        use_deep_supervision=use_deep_supervision,
        num_supervision_layers=num_supervision_layers,
        num_heads=config.get('num_heads', 8),
        attn_dropout=config.get('attn_dropout', 0.1),
    )

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint from epoch {epoch}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Inference from CSV file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (full checkpoint or raw state dict)')
    parser.add_argument('--peav_checkpoint', type=str, default=None,
                        help='Path to PE-AV base weights (required when loading raw state dict)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with file_path column')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'max'],
                        help='Aggregation method for chunk probabilities')
    parser.add_argument('--max_duration', type=float, default=20.0,
                        help='Maximum audio duration in seconds to process. Longer audios will be truncated.')
    parser.add_argument('--mode', type=str, default='infer', choices=['infer', 'eval'],
                        help='Mode: infer (no labels) or eval (with labels in input CSV)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output CSVs (skip already processed files)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(args.checkpoint, device, peav_checkpoint=args.peav_checkpoint)
    
    # Read CSV
    samples = []
    labels = {}  # file_path -> label (for eval mode)
    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = row.get('file_path', row.get('path', ''))
            if file_path:
                samples.append(file_path)
                if args.mode == 'eval':
                    label = row.get('label', '')
                    if label != '':
                        labels[file_path] = label
    
    print(f"Loaded {len(samples)} samples from {args.input}")
    if args.mode == 'eval':
        print(f"  Found {len(labels)} labels (eval mode)")
    
    # Resume: skip already processed files
    done_set = set()
    if args.resume:
        # Read pred.csv: extract file_name (basename) from file_path for consistent comparison
        done_pred = set()
        if os.path.exists(args.output):
            with open(args.output, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is not None:
                    for row in reader:
                        fp = row.get('file_path', row.get('path', ''))
                        if fp:
                            done_pred.add(os.path.basename(fp))
        
        done_set = done_pred
        print(f"  Resume mode: {len(done_set)} already processed, {len(samples) - len(done_set)} remaining")
    
    # Determine CSV fieldnames
    pred_fields = ['file_path', 'prob']
    
    # Inference: one file at a time, append to CSV immediately
    processed_count = 0
    skipped_count = 0
    failed_files = []  # Track files that completely failed to load
    files_with_failed_chunks = []  # Track files where some chunks failed
    max_samples = int(args.max_duration * 48000) if args.max_duration else None
    if max_samples:
        print(f"Max duration limit: {args.max_duration}s ({max_samples} samples)")
    
    for file_path in tqdm(samples, desc="Inferencing"):
        # Skip if already done (resume mode)
        if args.resume and os.path.basename(file_path) in done_set:
            skipped_count += 1
            continue
        
        waveform = load_audio(file_path)
        if waveform is None:
            prob = 1.0  # Failed samples get prob=1
            failed_files.append(file_path)
        else:
            if max_samples and waveform.shape[0] > max_samples:
                print(f"Truncating {file_path} from {waveform.shape[0]} samples to {max_samples} samples")
                waveform = waveform[:max_samples]
            prob, n_failed_chunks = inference_file(model, waveform, device,
                                                   clip_length=CLIP_LENGTH,
                                                   hop_length=HOP_LENGTH,
                                                   aggregation=args.aggregation)
            
            if n_failed_chunks > 0:
                files_with_failed_chunks.append((file_path, n_failed_chunks))
        
        # Append prediction to CSV immediately
        _append_row(args.output, pred_fields, {'file_path': file_path, 'prob': prob})
        
        processed_count += 1
    
    print(f"\nProcessed {processed_count} new samples, skipped {skipped_count} existing samples")
    
    # Report completely failed files
    if failed_files:
        print(f"WARNING: {len(failed_files)} files failed to load (prob set to 1.0)")
        failed_list_path = os.path.join(os.path.dirname(args.output), 'failed_files.txt')
        with open(failed_list_path, 'w') as f:
            for fp in failed_files:
                f.write(fp + '\n')
        print(f"Failed files list saved to: {failed_list_path}")
    
    # Report files with partial chunk failures
    if files_with_failed_chunks:
        print(f"WARNING: {len(files_with_failed_chunks)} files had failed chunks")
        chunk_fail_path = os.path.join(os.path.dirname(args.output), 'chunk_failures.txt')
        with open(chunk_fail_path, 'w') as f:
            for fp, n_failed in files_with_failed_chunks:
                f.write(f"{fp}\t{n_failed}\n")
        print(f"Chunk failure details saved to: {chunk_fail_path}")
    
    print(f"Predictions appended to {args.output}")
    
    # Print statistics from the full output file
    all_probs = []
    with open(args.output, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_probs.append(float(row['prob']))
    if all_probs:
        print(f"Prob stats: min={min(all_probs):.4f}, max={max(all_probs):.4f}, mean={sum(all_probs)/len(all_probs):.4f}")


if __name__ == "__main__":
    main()
