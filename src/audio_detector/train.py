import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from loss_function import sigmoid_focal_loss

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from audio_detector import build_detector
from dataloader import get_dataloader

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config['device'])
        self.clip_length = config.get('clip_length', 48000 * 3)
        
        # Multi-GPU / DataParallel
        self.use_dp = config.get('use_dp', False)
        self.gpu_ids = config.get('gpu_ids', None)
        
        # Mixed precision
        self.use_amp = config.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Build model
        self._build_model()
        
        # Build dataloaders
        self._build_dataloaders()
        
        # Build optimizer
        self._build_optimizer()
        
        # Loss function
        loss_type = config.get('loss_type', 'bce')
        if loss_type == 'focal':
            self.criterion = lambda logits, labels: sigmoid_focal_loss(
                logits, labels,
                alpha=config.get('focal_alpha', 0.5),
                gamma_pos=config.get('focal_gamma_pos', 2.0),
                gamma_neg=config.get('focal_gamma_neg', 2.0),
            )
            print(f"Using Focal Loss: alpha={config.get('focal_alpha', 0.5)}, "
                  f"gamma_pos={config.get('focal_gamma_pos', 2.0)}, "
                  f"gamma_neg={config.get('focal_gamma_neg', 2.0)}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            print("Using BCEWithLogitsLoss")
        
        # Training state
        self.epoch = 0
        self.best_eer = float('inf')
        self.global_step = 0
        
        # Gradient accumulation
        self.accum_steps = config.get('accum_steps', 1)
        if self.accum_steps > 1:
            print(f"Using gradient accumulation: {self.accum_steps} steps")
        
        if self.use_dp:
            print(f"Using DataParallel on GPUs: {self.gpu_ids}")
        if self.use_amp:
            print("Using Automatic Mixed Precision (AMP)")
    
    def _build_model(self):
        """Build model with audio detector backend."""
        print("Building model with audio detector backend...")

        self.model = build_detector(
            peav_checkpoint=self.config['peav_checkpoint'],
            use_lora=self.config['use_lora'],
            lora_r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=self.config['lora_dropout'],
            unfreeze_norm=self.config['unfreeze_norm'],
            device=self.device,
            use_deep_supervision=self.config.get('use_deep_supervision', True),
            num_supervision_layers=self.config.get('num_supervision_layers', 3),
            num_heads=self.config.get('num_heads', 8),
            attn_dropout=self.config.get('attn_dropout', 0.1),
        )
        
        # Wrap with DataParallel if requested
        if self.use_dp:
            if self.gpu_ids is not None:
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
            else:
                self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)
    
    def _build_dataloaders(self):
        """Build dataloaders."""
        print("Building dataloaders...")
        
        self.train_loader = get_dataloader(
            csv_path=self.config['train_csv'],
            audio_dir=self.config['train_audio_dir'],
            batch_size=self.config['batch_size'],
            sampling_rate=self.config['sampling_rate'],
            clip_length=self.clip_length,
            mode="train",
            augment=self.config.get('augment', True),
            augment_intensity=self.config.get('augment_intensity', 5),
            num_augment=self.config.get('num_augment', 2),
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        self.dev_loader = get_dataloader(
            csv_path=self.config['dev_csv'],
            audio_dir=self.config['dev_audio_dir'],
            batch_size=self.config['batch_size'],
            sampling_rate=self.config['sampling_rate'],
            clip_length=self.clip_length,
            mode="train",
            augment=False,
            shuffle=False,
            num_workers=self.config['num_workers'],
        )
    
    def _build_optimizer(self):
        """Build optimizer and scheduler."""
        # Separate parameters
        lora_params = []
        norm_params = []
        backend_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora_' in name:
                lora_params.append(param)
            elif 'norm' in name.lower() or 'ln' in name.lower() or 'bn' in name.lower():
                norm_params.append(param)
            elif 'backend' in name or 'classifier' in name or 'audio_head' in name or 'aux_heads' in name:
                backend_params.append(param)
            else:
                # Other trainable params
                backend_params.append(param)
        
        param_groups = []
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': self.config['lr_lora'],
                'weight_decay': self.config.get('weight_decay', 1e-4),
                'name': 'lora'
            })
        if norm_params:
            param_groups.append({
                'params': norm_params,
                'lr': self.config['lr_lora'],
                'weight_decay': 0.0,
                'name': 'norm'
            })
        if backend_params:
            param_groups.append({
                'params': backend_params,
                'lr': self.config['lr_head'],
                'weight_decay': self.config.get('weight_decay', 1e-4),
                'name': 'backend'
            })
        
        self.optimizer = optim.AdamW(param_groups)
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config['epochs']
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        print(f"Optimizer: AdamW")
        print(f"  LoRA lr: {self.config['lr_lora']}")
        print(f"  Backend lr: {self.config['lr_head']}")
        print(f"  Warmup steps: {warmup_steps}/{total_steps}")
    
    def train_epoch(self) -> dict:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        accum_steps = self.accum_steps
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            audio = batch['audio'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(audio, padding_mask)
                logits = outputs['logits']
                probs = outputs['prob']
                
                # Loss
                loss = self.criterion(logits, labels)
                
                # Deep supervision loss
                if 'aux_logits' in outputs:
                    aux_weight = self.config.get('aux_loss_weight', 0.5)
                    aux_loss = sum(self.criterion(aux_logit, labels) for aux_logit in outputs['aux_logits'])
                    loss = loss + aux_weight * aux_loss
                
                # Scale loss for gradient accumulation
                loss = loss / accum_steps
            
            # Backward with scaler if AMP is enabled
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Stats (use unscaled loss for display)
            display_loss = loss.item() * accum_steps
            total_loss += display_loss
            preds = (probs >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            num_batches += 1
            
            # Update progress bar
            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{display_loss:.4f}",
                'acc': f"{total_correct/total_samples:.4f}",
                'lr': f"{lr:.2e}",
            })
            
            # Only update weights after accumulating enough gradients
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('grad_clip', 1.0)
                )
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.global_step += 1
                self.optimizer.zero_grad()
        
            self.scheduler.step()
        
        return {
            'loss': total_loss / num_batches,
            'acc': total_correct / total_samples,
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate on a dataloader."""
        self.model.eval()
        
        all_probs = []
        all_labels = []
        all_names = []
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio = batch['audio'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            names = batch['name']
            
            # Forward with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(audio, padding_mask)
                logits = outputs['logits']
                probs = outputs['prob']
                
                # Loss
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_names.extend(names)
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        eer, threshold = self._compute_eer(all_probs, all_labels)
        auc = self._compute_auc(all_probs, all_labels)
        
        # Accuracy at 0.5 threshold
        preds = (all_probs >= 0.5).astype(int)
        acc = np.mean(preds == all_labels)
        
        # F1 scores
        f1_at_05 = self._compute_f1(all_probs, all_labels, threshold=0.5)
        f1_at_eer = self._compute_f1(all_probs, all_labels, threshold=threshold)
        
        return {
            'loss': total_loss / num_batches,
            'eer': eer,
            'auc': auc,
            'threshold': threshold,
            'acc': acc,
            'f1_at_05': f1_at_05,
            'f1_at_eer': f1_at_eer,
            'probs': all_probs,
            'labels': all_labels,
            'names': all_names,
        }
    
    def _compute_eer(self, scores: np.ndarray, labels: np.ndarray) -> tuple:
        """Compute Equal Error Rate."""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        return eer * 100, thresholds[eer_idx]
    
    def _compute_f1(self, scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
        """Compute F1 score at given threshold."""
        preds = (scores >= threshold).astype(int)
        
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return f1 * 100
    
    def _compute_auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute AUC."""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, scores) * 100
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Unwrap DataParallel if needed
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eer': self.best_eer,
            'config': self.config,
        }
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(os.path.dirname(path), 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint (resume training)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle DataParallel wrapper
        state_dict = checkpoint['model_state_dict']
        if hasattr(self.model, 'module'):
            # Current model is DP, loaded may or may not be
            if not any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
        else:
            # Current model is not DP, loaded may be DP
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_eer = checkpoint['best_eer']
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def load_pretrained(self, path: str):
        """Load only model weights (for fine-tuning from pretrained)."""
        print(f"Loading pretrained weights from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel wrapper
        if hasattr(self.model, 'module'):
            if not any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
        else:
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"  Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"    - {k}")
            if len(missing) > 5:
                print(f"    ... and {len(missing)-5} more")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"    - {k}")
            if len(unexpected) > 5:
                print(f"    ... and {len(unexpected)-5} more")
        
        print(f"Loaded pretrained weights successfully")
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config['epochs']} epochs...")
        print(f"  Device: {self.device}")
        if self.use_dp:
            print(f"  DataParallel: enabled")
        if self.use_amp:
            print(f"  AMP: enabled")
        print(f"  Clip length: {self.clip_length/48000:.1f}s")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Backend: audio detector backend")
        
        for epoch in range(self.epoch, self.config['epochs']):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            print(f"\nEpoch {epoch+1} - Train:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Acc:  {train_metrics['acc']:.4f}")
            
            # Evaluate
            if self.config.get('do_eval', False):
                dev_metrics = self.evaluate(self.dev_loader)
                print(f"\nEpoch {epoch+1} - Dev:")
                print(f"  Loss: {dev_metrics['loss']:.4f}")
                print(f"  EER:  {dev_metrics['eer']:.2f}%")
                print(f"  AUC:  {dev_metrics['auc']:.2f}%")
                print(f"  Acc:  {dev_metrics['acc']:.4f}")
                print(f"  F1@0.5: {dev_metrics['f1_at_05']:.2f}%")
                print(f"  F1@EER: {dev_metrics['f1_at_eer']:.2f}%")
                print(f"  Threshold: {dev_metrics['threshold']:.4f}")
                
                # Save checkpoint
                os.makedirs(self.config['save_dir'], exist_ok=True)
                checkpoint_path = os.path.join(
                    self.config['save_dir'],
                    f'audio_model.{epoch+1}.pt'
                )
                
                is_best = dev_metrics['eer'] < self.best_eer
                if is_best:
                    self.best_eer = dev_metrics['eer']
                
                self.save_checkpoint(checkpoint_path, is_best=is_best)
                print(f"  Best EER so far: {self.best_eer:.2f}%")
            else:
                os.makedirs(self.config['save_dir'], exist_ok=True)
                checkpoint_path = os.path.join(
                    self.config['save_dir'],
                    f'audio_model.{epoch+1}.pt'
                )
                self.save_checkpoint(checkpoint_path, is_best=False)
                print(f"  Saved checkpoint (no eval)")


def main():
    parser = argparse.ArgumentParser(description='Train Audio Deepfake Detector')
    
    # Data paths
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--train_audio_dir', type=str, default='')
    parser.add_argument('--dev_csv', type=str)
    parser.add_argument('--dev_audio_dir', type=str, default='')
    parser.add_argument('--peav_checkpoint', type=str)
    parser.add_argument('--save_dir', type=str, default='weights')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--lr_lora', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    
    # Model config
    parser.add_argument('--clip_length', type=int, default=48000*3)
    parser.add_argument('--sampling_rate', type=int, default=48000)
    parser.add_argument('--freeze_extractor', action='store_true', default=True)
    parser.add_argument('--no_freeze_extractor', action='store_true')
    parser.add_argument('--use_lora', action='store_true', default=True)
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--unfreeze_norm', action='store_true', default=True)
    
    # Audio detector backend config
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads for temporal branch (default: 8)')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='Dropout for attention layers (default: 0.1)')
    
    # Deep supervision
    parser.add_argument('--use_deep_supervision', action='store_true', default=False)
    parser.add_argument('--num_supervision_layers', type=int, default=3)
    parser.add_argument('--aux_loss_weight', type=float, default=0.5)
    
    # Loss function
    parser.add_argument('--loss_type', type=str, default='focal', choices=['bce', 'focal'])
    parser.add_argument('--focal_alpha', type=float, default=0.6)
    parser.add_argument('--focal_gamma_pos', type=float, default=2.0)
    parser.add_argument('--focal_gamma_neg', type=float, default=2.0)
    
    # Data config
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--augment_intensity', type=int, default=5)
    parser.add_argument('--num_augment', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Multi-GPU
    parser.add_argument('--use_dp', action='store_true', default=False,
                        help='Use DataParallel for multi-GPU training')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None,
                        help='GPU IDs to use for DataParallel (e.g., 0 1 2 3)')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--no_eval', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # Handle boolean flags
    if args.no_freeze_extractor:
        args.freeze_extractor = False
    if args.no_lora:
        args.use_lora = False

    config = vars(args)
    config['do_eval'] = not args.no_eval
    
    # Create trainer
    trainer = Trainer(config)
    
    # Load pretrained weights or resume training
    if args.pretrained:
        trainer.load_pretrained(args.pretrained)
    elif args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
