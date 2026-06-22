import os
import csv
import datetime
import time
import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.distributed as dist
from loss_fn import sigmoid_focal_loss, mil_margin_loss
import matplotlib.pyplot as plt
import math

from sklearn import metrics

def calculate_stats(output, target):
    """
    计算 ACC, AP, AUC，兼容二分类和多分类。
    output: [N, C]  target: [N, C]，C=1 也支持
    """
    classes_num = output.shape[1]
    stats = []

    # 二分类情况下（只有一个类别输出），使用 0.5 阈值判断准确率
    if classes_num == 1:
        preds = (output > 0.5).numpy().astype(int)
        acc = metrics.accuracy_score(target, preds)
    else:
        acc = metrics.accuracy_score(np.argmax(target, axis=1), np.argmax(output, axis=1))

    for k in range(classes_num):
        try:
            ap = metrics.average_precision_score(target[:, k], output[:, k])
            auc = metrics.roc_auc_score(target[:, k], output[:, k])
        except:
            ap, auc = -1, -1
            print(f"[Warning] Class {k} cannot compute AP or AUC (possibly no positive samples)")

        stats.append({
            'ap': ap,
            'auc': auc,
            'acc': acc
        })

    return stats

torch.autograd.set_detect_anomaly(False) # 若为True，则开启异常检测，追踪模型发散原因，但会影响训练速度

class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing LR Scheduler.
    
    Args:
        optimizer: torch optimizer.
        warmup_epochs: number of epochs for linear warmup.
        total_epochs: total number of training epochs.
        step_mode: 'epoch' or 'batch'. 
            - 'epoch': step once per epoch.
            - 'batch': step once per optimizer step (after accumulation).
        num_batches_per_epoch: number of batches (optimizer steps) per epoch. 
            Only needed when step_mode='batch'.
        eta_min: minimum learning rate.
        last_epoch: last epoch index for resuming training.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, step_mode='epoch',
                 num_batches_per_epoch=None, eta_min=1e-9, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.step_mode = step_mode
        self.num_batches_per_epoch = num_batches_per_epoch
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        if self.step_mode == 'batch':
            assert num_batches_per_epoch is not None, "num_batches_per_epoch must be provided when step_mode='batch'"
            self.warmup_steps = self.warmup_epochs * self.num_batches_per_epoch
            self.total_steps = self.total_epochs * self.num_batches_per_epoch
            # Convert epoch-based last_epoch to step-based current_step
            self.current_step = max(0, self.last_epoch) * self.num_batches_per_epoch
        else:
            self.current_step = self.last_epoch
            self.warmup_steps = self.warmup_epochs
            self.total_steps = self.total_epochs
        
        self._update_lr()
    
    def _get_lr(self, step, base_lr):
        if step < self.warmup_steps:
            # linear warmup
            return base_lr * (step + 1) / (self.warmup_steps + 1)
        else:
            # cosine annealing
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
    
    def _update_lr(self):
        for i, group in enumerate(self.optimizer.param_groups):
            lr = self._get_lr(self.current_step, self.base_lrs[i])
            group['lr'] = lr
    
    def step(self):
        self.current_step += 1
        self._update_lr()
    
    def state_dict(self):
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
        }
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']
        self._update_lr()

def save_data(csv_file, epoch, data, data_name):
    # 如果 CSV 文件不存在，就创建并写入表头
    try:
        # 打开 CSV 文件，追加模式（'a'）避免覆盖原数据
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # 如果文件为空，写入表头
            if file.tell() == 0:  # 检查文件是否为空
                writer.writerow(['epoch', data_name])  # 表头

            # 写入当前 epoch 和损失值
            writer.writerow([epoch, data])
    except Exception as e:
        print(f"Error while saving data: {e}")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, train_loader, train_sampler, args):
    """
    支持 GPS_DINO deep supervision (layers 21,22,23,24) 的训练函数。
    参照 train_GPS 编写，增加对 deep supervision 各层损失的计算和监控。
    训练流程：degraded + origin 双路 + consistency loss + deep supervision。
    """
    world_size = dist.get_world_size()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(True)

    total_loss_meter = AverageMeter()
    total_origin_loss_meter = AverageMeter()
    total_degraded_loss_meter = AverageMeter()
    total_consistency_loss_meter = AverageMeter()

    # main
    origin_main_loss_meter = AverageMeter()
    degraded_main_loss_meter = AverageMeter()

    # layer 24 meters
    origin_global_24_loss_meter = AverageMeter()
    degraded_global_24_loss_meter = AverageMeter()
    origin_patch_24_loss_meter = AverageMeter()
    degraded_patch_24_loss_meter = AverageMeter()
    origin_weak_patch_24_loss_meter = AverageMeter()
    degraded_weak_patch_24_loss_meter = AverageMeter()
    origin_segment_24_loss_meter = AverageMeter()
    degraded_segment_24_loss_meter = AverageMeter()
    origin_weak_segment_24_loss_meter = AverageMeter()
    degraded_weak_segment_24_loss_meter = AverageMeter()

    # deep supervision meters (21, 22, 23)
    origin_global_23_loss_meter = AverageMeter()
    degraded_global_23_loss_meter = AverageMeter()
    origin_patch_23_loss_meter = AverageMeter()
    degraded_patch_23_loss_meter = AverageMeter()
    origin_weak_patch_23_loss_meter = AverageMeter()
    degraded_weak_patch_23_loss_meter = AverageMeter()
    origin_segment_23_loss_meter = AverageMeter()
    degraded_segment_23_loss_meter = AverageMeter()
    origin_weak_segment_23_loss_meter = AverageMeter()
    degraded_weak_segment_23_loss_meter = AverageMeter()

    origin_global_22_loss_meter = AverageMeter()
    degraded_global_22_loss_meter = AverageMeter()
    origin_patch_22_loss_meter = AverageMeter()
    degraded_patch_22_loss_meter = AverageMeter()
    origin_weak_patch_22_loss_meter = AverageMeter()
    degraded_weak_patch_22_loss_meter = AverageMeter()
    origin_segment_22_loss_meter = AverageMeter()
    degraded_segment_22_loss_meter = AverageMeter()
    origin_weak_segment_22_loss_meter = AverageMeter()
    degraded_weak_segment_22_loss_meter = AverageMeter()

    origin_global_21_loss_meter = AverageMeter()
    degraded_global_21_loss_meter = AverageMeter()
    origin_patch_21_loss_meter = AverageMeter()
    degraded_patch_21_loss_meter = AverageMeter()
    origin_weak_patch_21_loss_meter = AverageMeter()
    degraded_weak_patch_21_loss_meter = AverageMeter()
    origin_segment_21_loss_meter = AverageMeter()
    degraded_segment_21_loss_meter = AverageMeter()
    origin_weak_segment_21_loss_meter = AverageMeter()
    degraded_weak_segment_21_loss_meter = AverageMeter()

    # consistency meters
    global_24_consistency_loss_meter = AverageMeter()
    patch_24_consistency_loss_meter = AverageMeter()
    segment_24_consistency_loss_meter = AverageMeter()

    # regularization meters (layer 24)
    origin_segment_loss_reg_meter = AverageMeter()
    degraded_segment_loss_reg_meter = AverageMeter()
    origin_patch_24_loss_reg_meter = AverageMeter()
    degraded_patch_24_loss_reg_meter = AverageMeter()

    # deep supervision regularization meters (21, 22, 23)
    origin_segment_loss_reg_23_meter = AverageMeter()
    degraded_segment_loss_reg_23_meter = AverageMeter()
    origin_patch_loss_reg_23_meter = AverageMeter()
    degraded_patch_loss_reg_23_meter = AverageMeter()
    origin_segment_loss_reg_22_meter = AverageMeter()
    degraded_segment_loss_reg_22_meter = AverageMeter()
    origin_patch_loss_reg_22_meter = AverageMeter()
    degraded_patch_loss_reg_22_meter = AverageMeter()
    origin_segment_loss_reg_21_meter = AverageMeter()
    degraded_segment_loss_reg_21_meter = AverageMeter()
    origin_patch_loss_reg_21_meter = AverageMeter()
    degraded_patch_loss_reg_21_meter = AverageMeter()

    # deep supervision consistency meters (21, 22, 23)
    global_23_consistency_loss_meter = AverageMeter()
    patch_23_consistency_loss_meter = AverageMeter()
    segment_23_consistency_loss_meter = AverageMeter()
    global_22_consistency_loss_meter = AverageMeter()
    patch_22_consistency_loss_meter = AverageMeter()
    segment_22_consistency_loss_meter = AverageMeter()
    global_21_consistency_loss_meter = AverageMeter()
    patch_21_consistency_loss_meter = AverageMeter()
    segment_21_consistency_loss_meter = AverageMeter()

    data_time_meter = AverageMeter()
    dnn_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    epoch = args.restart_epoch if args.restart else 1
    save_dir = args.save_dir

    model.to(device)

    trainables = [p for p in model.parameters() if p.requires_grad]
    trainables_ids = set(id(p) for p in trainables)
    if dist.get_rank() == 0:
        print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    
    base_params, head_params, token_head_params = [], [], []
    for name, param in model.named_parameters():
        if 'dinov3' in name and 'norm' not in name:
            base_params.append(param)
        elif 'reducer' in name and 'norm' not in name:
            token_head_params.append(param)
        elif 'norm' not in name:
            head_params.append(param)

    lora_params = [p for p in base_params if id(p) in trainables_ids]
    # 分离 dinov3 的 norm 和 head 的 norm
    dinov3_norm_params = [param for name, param in model.named_parameters() if 'norm' in name and 'dinov3' in name]
    head_norm_params = [param for name, param in model.named_parameters() if 'norm' in name and 'dinov3' not in name and 'reducer' not in name]
    reducer_norm_params = [param for name, param in model.named_parameters() if 'norm' in name and 'reducer' in name]

    if dist.get_rank() == 0:
        print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
        print('Total head parameter number is : {:.3f} million'.format(sum(p.numel() for p in head_params) / 1e6))
        print('Total lora parameter number is : {:.3f} million'.format(sum(p.numel() for p in lora_params) / 1e6))
        print('Total dinov3 norm parameter number is : {:.3f} million'.format(sum(p.numel() for p in dinov3_norm_params) / 1e6))
        print('Total head norm parameter number is : {:.3f} million'.format(sum(p.numel() for p in head_norm_params) / 1e6))
        print('Total token head parameter number is : {:.3f} million'.format(sum(p.numel() for p in token_head_params) / 1e6))
        print('Total reducer norm parameter number is : {:.3f} million'.format(sum(p.numel() for p in reducer_norm_params) / 1e6))

    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': args.lr, 'weight_decay': 5e-2},
        {'params': head_params, 'lr': args.lr * args.head_lr_ratio, 'weight_decay': 5e-2},
        {'params': token_head_params, 'lr': args.lr * args.token_head_lr_ratio, 'weight_decay': 5e-2},
        {'params': dinov3_norm_params, 'lr': args.lr, 'weight_decay': 0.0},
        {'params': head_norm_params, 'lr': args.lr * args.head_lr_ratio, 'weight_decay': 0.0},
        {'params': reducer_norm_params, 'lr': args.lr * args.token_head_lr_ratio, 'weight_decay': 0.0}
    ], betas=(0.95, 0.999))

    if dist.get_rank() == 0:
        print('lora lr, weight decay : ', optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['weight_decay'])
        print('head lr, weight decay : ', optimizer.param_groups[1]['lr'], optimizer.param_groups[1]['weight_decay'])
        print('token head lr, weight decay : ', optimizer.param_groups[2]['lr'], optimizer.param_groups[2]['weight_decay'])
        print('dinov3 norm lr, weight decay : ', optimizer.param_groups[3]['lr'], optimizer.param_groups[3]['weight_decay'])
        print('head norm lr, weight decay : ', optimizer.param_groups[4]['lr'], optimizer.param_groups[4]['weight_decay'])
        print('reducer norm lr, weight decay : ', optimizer.param_groups[5]['lr'], optimizer.param_groups[5]['weight_decay'])

    if args.restart and args.restart_epoch > 1:
        restart_scaler_path = os.path.join(args.checkpoint_root, f"scaler.{args.restart_epoch-1}.pth")
        restart_optimizer_path = os.path.join(args.checkpoint_root, f"optimizer.{args.restart_epoch-1}.pth")

    if args.restart and args.restart_epoch > 1:
        if os.path.exists(restart_optimizer_path):
            optimizer.load_state_dict(torch.load(restart_optimizer_path, map_location='cpu'))
            if dist.get_rank() == 0:
                print(f"Restart training from epoch {args.restart_epoch}, loaded optimizer state from {restart_optimizer_path}")
        else:
            if dist.get_rank() == 0:
                print(f"Restart epoch {args.restart_epoch} specified but no optimizer checkpoint found at {restart_optimizer_path}, starting training with new optimizer.")

    if args.restart and args.restart_epoch > 1:
        restart_scaler_path = os.path.join(args.checkpoint_root, f"scaler.{args.restart_epoch-1}.pth")
        restart_optimizer_path = os.path.join(args.checkpoint_root, f"optimizer.{args.restart_epoch-1}.pth")

    if args.restart and args.restart_epoch > 1:
        if os.path.exists(restart_optimizer_path):
            optimizer.load_state_dict(torch.load(restart_optimizer_path, map_location='cpu'))
            if dist.get_rank() == 0:
                print(f"Restart training from epoch {args.restart_epoch}, loaded optimizer state from {restart_optimizer_path}")
        else:
            if dist.get_rank() == 0:
                print(f"Restart epoch {args.restart_epoch} specified but no optimizer checkpoint found at {restart_optimizer_path}, starting training with new optimizer.")

    
    # scheduler
    if getattr(args, 'scheduler_step_mode', 'epoch') == 'batch':
        num_batches = len(train_loader)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.n_epochs,
            step_mode='batch',
            num_batches_per_epoch=num_batches,
            eta_min=1e-9,
            last_epoch=epoch-2 if args.if_new_epoch else epoch-1
        )
    else:
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.n_epochs,
            step_mode='epoch',
            eta_min=1e-9,
            last_epoch=epoch-2 if args.if_new_epoch else epoch-1
        )

    # scaler
    use_amp = args.use_amp
    scaler = GradScaler('cuda', enabled=use_amp)
    if args.restart and args.restart_epoch > 1:
        if os.path.exists(restart_scaler_path):
            scaler.load_state_dict(torch.load(restart_scaler_path, map_location='cpu'))
            if dist.get_rank() == 0:
                print(f"Restart training from epoch {args.restart_epoch}, loaded scaler state from {restart_scaler_path}")
        else:
            if dist.get_rank() == 0:
                print(f"Restart epoch {args.restart_epoch} specified but no scaler checkpoint found at {restart_scaler_path}, starting training with new scaler.")

    class_weights = torch.tensor([1.0, 1.0]).to(device)
    CE_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    if dist.get_rank() == 0:
        print(f"Start training model on {device}")
        print(f"current #epoch = {epoch}")
        print("start training...")

    start_time = datetime.datetime.now() + datetime.timedelta(hours=0)
    start_time_str = start_time.strftime("%Y_%m_%d_%H_%M")
    log_dir = os.path.join("./logs/finetuning/", start_time_str)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir

    optimizer.zero_grad()
    verbose = args.verbose

    # deep supervision loss weights (可配置)
    deep_loss_weight = getattr(args, 'deep_loss_weight', 0.5)

    while epoch < args.n_epochs + 1:
        epoch_start_time = time.time()
        A_main_predictions, A_global_predictions, A_patch_predictions, A_segment_predictions, A_targets = [], [], [], [], []
        model.train()
        if dist.get_rank() == 0:
            print('---------------')
            print(datetime.datetime.now())
            print(f"current #epoch = {epoch}")
            print('Epoch {0} learning rate: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        train_sampler.set_epoch(epoch)

        end = time.time()
        for i, (degraded_imgs, origin_imgs, labels) in enumerate(train_loader):
            print(i)
            iter_start = time.time()
            B = degraded_imgs.shape[0]
            data_time = iter_start - end
            data_time_meter.update(data_time, B)
            degraded_imgs = degraded_imgs.to(device)
            origin_imgs = origin_imgs.to(device)
            A_targets.append(labels)
            labels = labels.to(device)

            with autocast(device_type='cuda', enabled=use_amp):
                dnn_start = time.time()
                degraded_logits_dict, degraded_tokens_dict = model(degraded_imgs)
                origin_logits_dict, origin_tokens_dict = model(origin_imgs)
                dnn_time = time.time() - dnn_start
                dnn_time_meter.update(dnn_time, B)

                main_predictions = origin_logits_dict['main_logits_24'].to('cpu').detach()
                global_predictions = origin_logits_dict['global_logits_24'].to('cpu').detach()
                patch_predictions = origin_logits_dict['patch_logits_24'].to('cpu').detach()
                segment_predictions = origin_logits_dict['segment_logits_24'].to('cpu').detach()

                A_main_predictions.append(main_predictions)
                A_global_predictions.append(global_predictions)
                A_patch_predictions.append(patch_predictions)
                A_segment_predictions.append(segment_predictions)

                # ---------- layer 24 losses ----------
                if args.cls_loss == "focal":
                    degraded_main_loss = sigmoid_focal_loss(degraded_logits_dict['main_logits_24'], labels)
                    origin_main_loss = sigmoid_focal_loss(origin_logits_dict['main_logits_24'], labels)
                    degraded_global_24_loss = sigmoid_focal_loss(degraded_logits_dict['global_logits_24'], labels)
                    origin_global_24_loss = sigmoid_focal_loss(origin_logits_dict['global_logits_24'], labels)
                    degraded_patch_24_loss = sigmoid_focal_loss(degraded_logits_dict['patch_logits_24'], labels)
                    origin_patch_24_loss = sigmoid_focal_loss(origin_logits_dict['patch_logits_24'], labels)
                    degraded_segment_24_loss = sigmoid_focal_loss(degraded_logits_dict['segment_logits_24'], labels)
                    origin_segment_24_loss = sigmoid_focal_loss(origin_logits_dict['segment_logits_24'], labels)
                    degraded_weak_patch_24_loss = sigmoid_focal_loss(degraded_logits_dict['weak_patch_logits_24'], labels)
                    origin_weak_patch_24_loss = sigmoid_focal_loss(origin_logits_dict['weak_patch_logits_24'], labels)
                    degraded_weak_segment_24_loss = sigmoid_focal_loss(degraded_logits_dict['weak_segment_logits_24'], labels)
                    origin_weak_segment_24_loss = sigmoid_focal_loss(origin_logits_dict['weak_segment_logits_24'], labels)
                else:
                    degraded_main_loss = CE_loss_fn(degraded_logits_dict['main_logits_24'], labels)
                    origin_main_loss = CE_loss_fn(origin_logits_dict['main_logits_24'], labels)
                    degraded_global_24_loss = CE_loss_fn(degraded_logits_dict['global_logits_24'], labels)
                    origin_global_24_loss = CE_loss_fn(origin_logits_dict['global_logits_24'], labels)
                    degraded_patch_24_loss = CE_loss_fn(degraded_logits_dict['patch_logits_24'], labels)
                    origin_patch_24_loss = CE_loss_fn(origin_logits_dict['patch_logits_24'], labels)
                    degraded_segment_24_loss = CE_loss_fn(degraded_logits_dict['segment_logits_24'], labels)
                    origin_segment_24_loss = CE_loss_fn(origin_logits_dict['segment_logits_24'], labels)
                    degraded_weak_patch_24_loss = CE_loss_fn(degraded_logits_dict['weak_patch_logits_24'], labels)
                    origin_weak_patch_24_loss = CE_loss_fn(origin_logits_dict['weak_patch_logits_24'], labels)
                    degraded_weak_segment_24_loss = CE_loss_fn(degraded_logits_dict['weak_segment_logits_24'], labels)
                    origin_weak_segment_24_loss = CE_loss_fn(origin_logits_dict['weak_segment_logits_24'], labels)

                # ---------- layer 24 margin loss ----------
                origin_loss_segment_reg = mil_margin_loss(origin_logits_dict['weak_segment_logits_24'], origin_logits_dict['rest_segment_logits_24'], labels, margin=0.6)
                origin_loss_patch_reg = mil_margin_loss(origin_logits_dict['weak_patch_logits_24'], origin_logits_dict['rest_patch_logits_24'], labels, margin=0.6)
                degraded_loss_segment_reg = mil_margin_loss(degraded_logits_dict['weak_segment_logits_24'], degraded_logits_dict['rest_segment_logits_24'], labels, margin=0.6)
                degraded_loss_patch_reg = mil_margin_loss(degraded_logits_dict['weak_patch_logits_24'], degraded_logits_dict['rest_patch_logits_24'], labels, margin=0.6)

                # ---------- deep supervision losses (21, 22, 23) ----------
                def _compute_layer_loss(logits_dict, layer_idx, labels, loss_type):
                    prefix = f"_{layer_idx}"
                    if loss_type == "focal":
                        g_loss = sigmoid_focal_loss(logits_dict[f'global_logits{prefix}'], labels)
                        p_loss = sigmoid_focal_loss(logits_dict[f'patch_logits{prefix}'], labels)
                        s_loss = sigmoid_focal_loss(logits_dict[f'segment_logits{prefix}'], labels)
                        wp_loss = sigmoid_focal_loss(logits_dict[f'weak_patch_logits{prefix}'], labels)
                        ws_loss = sigmoid_focal_loss(logits_dict[f'weak_segment_logits{prefix}'], labels)
                    else:
                        g_loss = CE_loss_fn(logits_dict[f'global_logits{prefix}'], labels)
                        p_loss = CE_loss_fn(logits_dict[f'patch_logits{prefix}'], labels)
                        s_loss = CE_loss_fn(logits_dict[f'segment_logits{prefix}'], labels)
                        wp_loss = CE_loss_fn(logits_dict[f'weak_patch_logits{prefix}'], labels)
                        ws_loss = CE_loss_fn(logits_dict[f'weak_segment_logits{prefix}'], labels)
                    return g_loss, p_loss, s_loss, wp_loss, ws_loss

                degraded_global_23_loss, degraded_patch_23_loss, degraded_segment_23_loss, \
                    degraded_weak_patch_23_loss, degraded_weak_segment_23_loss = _compute_layer_loss(
                        degraded_logits_dict, 23, labels, args.cls_loss)
                origin_global_23_loss, origin_patch_23_loss, origin_segment_23_loss, \
                    origin_weak_patch_23_loss, origin_weak_segment_23_loss = _compute_layer_loss(
                        origin_logits_dict, 23, labels, args.cls_loss)

                degraded_global_22_loss, degraded_patch_22_loss, degraded_segment_22_loss, \
                    degraded_weak_patch_22_loss, degraded_weak_segment_22_loss = _compute_layer_loss(
                        degraded_logits_dict, 22, labels, args.cls_loss)
                origin_global_22_loss, origin_patch_22_loss, origin_segment_22_loss, \
                    origin_weak_patch_22_loss, origin_weak_segment_22_loss = _compute_layer_loss(
                        origin_logits_dict, 22, labels, args.cls_loss)

                degraded_global_21_loss, degraded_patch_21_loss, degraded_segment_21_loss, \
                    degraded_weak_patch_21_loss, degraded_weak_segment_21_loss = _compute_layer_loss(
                        degraded_logits_dict, 21, labels, args.cls_loss)
                origin_global_21_loss, origin_patch_21_loss, origin_segment_21_loss, \
                    origin_weak_patch_21_loss, origin_weak_segment_21_loss = _compute_layer_loss(
                        origin_logits_dict, 21, labels, args.cls_loss)

                # ---------- deep supervision margin loss ----------
                origin_loss_segment_reg_23 = mil_margin_loss(origin_logits_dict['weak_segment_logits_23'], origin_logits_dict['rest_segment_logits_23'], labels, margin=0.6)
                origin_loss_patch_reg_23 = mil_margin_loss(origin_logits_dict['weak_patch_logits_23'], origin_logits_dict['rest_patch_logits_23'], labels, margin=0.6)
                degraded_loss_segment_reg_23 = mil_margin_loss(degraded_logits_dict['weak_segment_logits_23'], degraded_logits_dict['rest_segment_logits_23'], labels, margin=0.6)
                degraded_loss_patch_reg_23 = mil_margin_loss(degraded_logits_dict['weak_patch_logits_23'], degraded_logits_dict['rest_patch_logits_23'], labels, margin=0.6)

                origin_loss_segment_reg_22 = mil_margin_loss(origin_logits_dict['weak_segment_logits_22'], origin_logits_dict['rest_segment_logits_22'], labels, margin=0.6)
                origin_loss_patch_reg_22 = mil_margin_loss(origin_logits_dict['weak_patch_logits_22'], origin_logits_dict['rest_patch_logits_22'], labels, margin=0.6)
                degraded_loss_segment_reg_22 = mil_margin_loss(degraded_logits_dict['weak_segment_logits_22'], degraded_logits_dict['rest_segment_logits_22'], labels, margin=0.6)
                degraded_loss_patch_reg_22 = mil_margin_loss(degraded_logits_dict['weak_patch_logits_22'], degraded_logits_dict['rest_patch_logits_22'], labels, margin=0.6)

                origin_loss_segment_reg_21 = mil_margin_loss(origin_logits_dict['weak_segment_logits_21'], origin_logits_dict['rest_segment_logits_21'], labels, margin=0.6)
                origin_loss_patch_reg_21 = mil_margin_loss(origin_logits_dict['weak_patch_logits_21'], origin_logits_dict['rest_patch_logits_21'], labels, margin=0.6)
                degraded_loss_segment_reg_21 = mil_margin_loss(degraded_logits_dict['weak_segment_logits_21'], degraded_logits_dict['rest_segment_logits_21'], labels, margin=0.6)
                degraded_loss_patch_reg_21 = mil_margin_loss(degraded_logits_dict['weak_patch_logits_21'], degraded_logits_dict['rest_patch_logits_21'], labels, margin=0.6)

                # ---------- total loss ----------
                degraded_deep_loss = (
                    degraded_global_23_loss + degraded_patch_23_loss + degraded_segment_23_loss
                    + degraded_weak_patch_23_loss + degraded_weak_segment_23_loss
                    + degraded_global_22_loss + degraded_patch_22_loss + degraded_segment_22_loss
                    + degraded_weak_patch_22_loss + degraded_weak_segment_22_loss
                    + degraded_global_21_loss + degraded_patch_21_loss + degraded_segment_21_loss
                    + degraded_weak_patch_21_loss + degraded_weak_segment_21_loss
                )
                origin_deep_loss = (
                    origin_global_23_loss + origin_patch_23_loss + origin_segment_23_loss
                    + origin_weak_patch_23_loss + origin_weak_segment_23_loss
                    + origin_global_22_loss + origin_patch_22_loss + origin_segment_22_loss
                    + origin_weak_patch_22_loss + origin_weak_segment_22_loss
                    + origin_global_21_loss + origin_patch_21_loss + origin_segment_21_loss
                    + origin_weak_patch_21_loss + origin_weak_segment_21_loss
                )

                degraded_deep_reg = (
                    degraded_loss_segment_reg_23 + degraded_loss_patch_reg_23
                    + degraded_loss_segment_reg_22 + degraded_loss_patch_reg_22
                    + degraded_loss_segment_reg_21 + degraded_loss_patch_reg_21
                )
                origin_deep_reg = (
                    origin_loss_segment_reg_23 + origin_loss_patch_reg_23
                    + origin_loss_segment_reg_22 + origin_loss_patch_reg_22
                    + origin_loss_segment_reg_21 + origin_loss_patch_reg_21
                )

                degraded_loss = (
                    degraded_main_loss + degraded_global_24_loss + degraded_patch_24_loss + degraded_segment_24_loss
                    + degraded_weak_patch_24_loss + degraded_weak_segment_24_loss
                ) + (degraded_loss_segment_reg + degraded_loss_patch_reg) + deep_loss_weight * (degraded_deep_loss + degraded_deep_reg)

                origin_loss = (
                    origin_main_loss + origin_global_24_loss + origin_patch_24_loss + origin_segment_24_loss
                    + origin_weak_patch_24_loss + origin_weak_segment_24_loss
                ) + (origin_loss_segment_reg + origin_loss_patch_reg) + deep_loss_weight * (origin_deep_loss + origin_deep_reg)

                # ---------- consistency loss (layer 24) ----------
                global_cos_sim_24 = F.cosine_similarity(origin_tokens_dict['cls_tokens_24'].detach(), degraded_tokens_dict['cls_tokens_24'], dim=-1)
                patch_cos_sim_24 = F.cosine_similarity(origin_tokens_dict['aggregated_patch_tokens_24'].detach(), degraded_tokens_dict['aggregated_patch_tokens_24'], dim=-1)
                segment_cos_sim_24 = F.cosine_similarity(origin_tokens_dict['aggregated_segment_tokens_24'].detach(), degraded_tokens_dict['aggregated_segment_tokens_24'], dim=-1)

                global_consistency_loss_24 = (1 - global_cos_sim_24.mean())
                patch_consistency_loss_24 = (1 - patch_cos_sim_24.mean())
                segment_consistency_loss_24 = (1 - segment_cos_sim_24.mean())

                # ---------- deep supervision consistency loss (21, 22, 23) ----------
                global_cos_sim_23 = F.cosine_similarity(origin_tokens_dict['cls_tokens_23'].detach(), degraded_tokens_dict['cls_tokens_23'], dim=-1)
                patch_cos_sim_23 = F.cosine_similarity(origin_tokens_dict['aggregated_patch_tokens_23'].detach(), degraded_tokens_dict['aggregated_patch_tokens_23'], dim=-1)
                segment_cos_sim_23 = F.cosine_similarity(origin_tokens_dict['aggregated_segment_tokens_23'].detach(), degraded_tokens_dict['aggregated_segment_tokens_23'], dim=-1)

                global_consistency_loss_23 = (1 - global_cos_sim_23.mean())
                patch_consistency_loss_23 = (1 - patch_cos_sim_23.mean())
                segment_consistency_loss_23 = (1 - segment_cos_sim_23.mean())

                global_cos_sim_22 = F.cosine_similarity(origin_tokens_dict['cls_tokens_22'].detach(), degraded_tokens_dict['cls_tokens_22'], dim=-1)
                patch_cos_sim_22 = F.cosine_similarity(origin_tokens_dict['aggregated_patch_tokens_22'].detach(), degraded_tokens_dict['aggregated_patch_tokens_22'], dim=-1)
                segment_cos_sim_22 = F.cosine_similarity(origin_tokens_dict['aggregated_segment_tokens_22'].detach(), degraded_tokens_dict['aggregated_segment_tokens_22'], dim=-1)

                global_consistency_loss_22 = (1 - global_cos_sim_22.mean())
                patch_consistency_loss_22 = (1 - patch_cos_sim_22.mean())
                segment_consistency_loss_22 = (1 - segment_cos_sim_22.mean())

                global_cos_sim_21 = F.cosine_similarity(origin_tokens_dict['cls_tokens_21'].detach(), degraded_tokens_dict['cls_tokens_21'], dim=-1)
                patch_cos_sim_21 = F.cosine_similarity(origin_tokens_dict['aggregated_patch_tokens_21'].detach(), degraded_tokens_dict['aggregated_patch_tokens_21'], dim=-1)
                segment_cos_sim_21 = F.cosine_similarity(origin_tokens_dict['aggregated_segment_tokens_21'].detach(), degraded_tokens_dict['aggregated_segment_tokens_21'], dim=-1)

                global_consistency_loss_21 = (1 - global_cos_sim_21.mean())
                patch_consistency_loss_21 = (1 - patch_cos_sim_21.mean())
                segment_consistency_loss_21 = (1 - segment_cos_sim_21.mean())

                consistency_loss = (
                    global_consistency_loss_24 + patch_consistency_loss_24 + segment_consistency_loss_24
                    +(global_consistency_loss_23 + patch_consistency_loss_23 + segment_consistency_loss_23
                    + global_consistency_loss_22 + patch_consistency_loss_22 + segment_consistency_loss_22
                    + global_consistency_loss_21 + patch_consistency_loss_21 + segment_consistency_loss_21)
                )
                loss = degraded_loss + origin_loss + 0.05 * consistency_loss

                accumulation_loss = loss / args.accumulation_steps

            scaler.scale(accumulation_loss).backward()

            if (i + 1) % args.accumulation_steps == 0 or (i + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if getattr(args, 'scheduler_step_mode', 'epoch') == 'batch':
                scheduler.step()

            # ========= loss meters update =========
            total_loss_meter.update(loss.item(), B)
            total_origin_loss_meter.update(origin_loss.item(), B)
            total_degraded_loss_meter.update(degraded_loss.item(), B)
            total_consistency_loss_meter.update(consistency_loss.item(), B)

            origin_main_loss_meter.update(origin_main_loss.item(), B)
            degraded_main_loss_meter.update(degraded_main_loss.item(), B)

            origin_global_24_loss_meter.update(origin_global_24_loss.item(), B)
            degraded_global_24_loss_meter.update(degraded_global_24_loss.item(), B)
            origin_patch_24_loss_meter.update(origin_patch_24_loss.item(), B)
            degraded_patch_24_loss_meter.update(degraded_patch_24_loss.item(), B)
            origin_weak_patch_24_loss_meter.update(origin_weak_patch_24_loss.item(), B)
            degraded_weak_patch_24_loss_meter.update(degraded_weak_patch_24_loss.item(), B)
            origin_segment_24_loss_meter.update(origin_segment_24_loss.item(), B)
            degraded_segment_24_loss_meter.update(degraded_segment_24_loss.item(), B)
            origin_weak_segment_24_loss_meter.update(origin_weak_segment_24_loss.item(), B)
            degraded_weak_segment_24_loss_meter.update(degraded_weak_segment_24_loss.item(), B)

            origin_global_23_loss_meter.update(origin_global_23_loss.item(), B)
            degraded_global_23_loss_meter.update(degraded_global_23_loss.item(), B)
            origin_patch_23_loss_meter.update(origin_patch_23_loss.item(), B)
            degraded_patch_23_loss_meter.update(degraded_patch_23_loss.item(), B)
            origin_weak_patch_23_loss_meter.update(origin_weak_patch_23_loss.item(), B)
            degraded_weak_patch_23_loss_meter.update(degraded_weak_patch_23_loss.item(), B)
            origin_segment_23_loss_meter.update(origin_segment_23_loss.item(), B)
            degraded_segment_23_loss_meter.update(degraded_segment_23_loss.item(), B)
            origin_weak_segment_23_loss_meter.update(origin_weak_segment_23_loss.item(), B)
            degraded_weak_segment_23_loss_meter.update(degraded_weak_segment_23_loss.item(), B)

            origin_global_22_loss_meter.update(origin_global_22_loss.item(), B)
            degraded_global_22_loss_meter.update(degraded_global_22_loss.item(), B)
            origin_patch_22_loss_meter.update(origin_patch_22_loss.item(), B)
            degraded_patch_22_loss_meter.update(degraded_patch_22_loss.item(), B)
            origin_weak_patch_22_loss_meter.update(origin_weak_patch_22_loss.item(), B)
            degraded_weak_patch_22_loss_meter.update(degraded_weak_patch_22_loss.item(), B)
            origin_segment_22_loss_meter.update(origin_segment_22_loss.item(), B)
            degraded_segment_22_loss_meter.update(degraded_segment_22_loss.item(), B)
            origin_weak_segment_22_loss_meter.update(origin_weak_segment_22_loss.item(), B)
            degraded_weak_segment_22_loss_meter.update(degraded_weak_segment_22_loss.item(), B)

            origin_global_21_loss_meter.update(origin_global_21_loss.item(), B)
            degraded_global_21_loss_meter.update(degraded_global_21_loss.item(), B)
            origin_patch_21_loss_meter.update(origin_patch_21_loss.item(), B)
            degraded_patch_21_loss_meter.update(degraded_patch_21_loss.item(), B)
            origin_weak_patch_21_loss_meter.update(origin_weak_patch_21_loss.item(), B)
            degraded_weak_patch_21_loss_meter.update(degraded_weak_patch_21_loss.item(), B)
            origin_segment_21_loss_meter.update(origin_segment_21_loss.item(), B)
            degraded_segment_21_loss_meter.update(degraded_segment_21_loss.item(), B)
            origin_weak_segment_21_loss_meter.update(origin_weak_segment_21_loss.item(), B)
            degraded_weak_segment_21_loss_meter.update(degraded_weak_segment_21_loss.item(), B)

            global_24_consistency_loss_meter.update(global_consistency_loss_24.item(), B)
            patch_24_consistency_loss_meter.update(patch_consistency_loss_24.item(), B)
            segment_24_consistency_loss_meter.update(segment_consistency_loss_24.item(), B)

            global_23_consistency_loss_meter.update(global_consistency_loss_23.item(), B)
            patch_23_consistency_loss_meter.update(patch_consistency_loss_23.item(), B)
            segment_23_consistency_loss_meter.update(segment_consistency_loss_23.item(), B)
            global_22_consistency_loss_meter.update(global_consistency_loss_22.item(), B)
            patch_22_consistency_loss_meter.update(patch_consistency_loss_22.item(), B)
            segment_22_consistency_loss_meter.update(segment_consistency_loss_22.item(), B)
            global_21_consistency_loss_meter.update(global_consistency_loss_21.item(), B)
            patch_21_consistency_loss_meter.update(patch_consistency_loss_21.item(), B)
            segment_21_consistency_loss_meter.update(segment_consistency_loss_21.item(), B)

            origin_segment_loss_reg_meter.update(origin_loss_segment_reg.item(), B)
            degraded_segment_loss_reg_meter.update(degraded_loss_segment_reg.item(), B)
            origin_patch_24_loss_reg_meter.update(origin_loss_patch_reg.item(), B)
            degraded_patch_24_loss_reg_meter.update(degraded_loss_patch_reg.item(), B)

            origin_segment_loss_reg_23_meter.update(origin_loss_segment_reg_23.item(), B)
            degraded_segment_loss_reg_23_meter.update(degraded_loss_segment_reg_23.item(), B)
            origin_patch_loss_reg_23_meter.update(origin_loss_patch_reg_23.item(), B)
            degraded_patch_loss_reg_23_meter.update(degraded_loss_patch_reg_23.item(), B)
            origin_segment_loss_reg_22_meter.update(origin_loss_segment_reg_22.item(), B)
            degraded_segment_loss_reg_22_meter.update(degraded_loss_segment_reg_22.item(), B)
            origin_patch_loss_reg_22_meter.update(origin_loss_patch_reg_22.item(), B)
            degraded_patch_loss_reg_22_meter.update(degraded_loss_patch_reg_22.item(), B)
            origin_segment_loss_reg_21_meter.update(origin_loss_segment_reg_21.item(), B)
            degraded_segment_loss_reg_21_meter.update(degraded_loss_segment_reg_21.item(), B)
            origin_patch_loss_reg_21_meter.update(origin_loss_patch_reg_21.item(), B)
            degraded_patch_loss_reg_21_meter.update(degraded_loss_patch_reg_21.item(), B)

            if np.isnan(total_loss_meter.avg):
                print("training diverged...")
                return

            if verbose == True and i % 20 == 0 and dist.get_rank() == 0:
                print(f"epoch: [{epoch}][{i}/{len(train_loader)}]", flush=True)
                per_sample_data_time = data_time_meter.avg
                per_sample_dnn_time = dnn_time_meter.avg
                per_sample_batch_time = batch_time_meter.avg
                remaining_iters_this_epoch = len(train_loader) - i
                eta_epoch = datetime.datetime.now() + datetime.timedelta(seconds=per_sample_batch_time * remaining_iters_this_epoch)
                print(f"per sample data time: {per_sample_data_time:.4f}s, per sample dnn time: {per_sample_dnn_time:.4f}s, estimated epoch finish: {eta_epoch.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"current learning rate: {optimizer.param_groups[0]['lr']}")
                print("===== Loss Information Average =====")
                print(f"Total Loss: {total_loss_meter.avg:.4f}")
                print(f"Origin Loss: {total_origin_loss_meter.avg:.4f}")
                print(f"Degraded Loss: {total_degraded_loss_meter.avg:.4f}")
                print(f"Consistency Loss: {total_consistency_loss_meter.avg:.4f}")

                print("---- Main ----")
                print(f"Origin Main Loss: {origin_main_loss_meter.avg:.4f}")
                print(f"Degraded Main Loss: {degraded_main_loss_meter.avg:.4f}")

                print("---- Layer 24 ----")
                print(f"Origin Global 24 Loss: {origin_global_24_loss_meter.avg:.4f}")
                print(f"Degraded Global 24 Loss: {degraded_global_24_loss_meter.avg:.4f}")
                print(f"Origin Patch 24 Loss: {origin_patch_24_loss_meter.avg:.4f}")
                print(f"Degraded Patch 24 Loss: {degraded_patch_24_loss_meter.avg:.4f}")
                print(f"Origin Weak Patch 24 Loss: {origin_weak_patch_24_loss_meter.avg:.4f}")
                print(f"Degraded Weak Patch 24 Loss: {degraded_weak_patch_24_loss_meter.avg:.4f}")
                print(f"Origin Segment 24 Loss: {origin_segment_24_loss_meter.avg:.4f}")
                print(f"Degraded Segment 24 Loss: {degraded_segment_24_loss_meter.avg:.4f}")
                print(f"Origin Weak Segment 24 Loss: {origin_weak_segment_24_loss_meter.avg:.4f}")
                print(f"Degraded Weak Segment 24 Loss: {degraded_weak_segment_24_loss_meter.avg:.4f}")

                print("---- Layer 23 ----")
                print(f"Origin Global 23 Loss: {origin_global_23_loss_meter.avg:.4f}")
                print(f"Degraded Global 23 Loss: {degraded_global_23_loss_meter.avg:.4f}")
                print(f"Origin Patch 23 Loss: {origin_patch_23_loss_meter.avg:.4f}")
                print(f"Degraded Patch 23 Loss: {degraded_patch_23_loss_meter.avg:.4f}")
                print(f"Origin Weak Patch 23 Loss: {origin_weak_patch_23_loss_meter.avg:.4f}")
                print(f"Degraded Weak Patch 23 Loss: {degraded_weak_patch_23_loss_meter.avg:.4f}")
                print(f"Origin Segment 23 Loss: {origin_segment_23_loss_meter.avg:.4f}")
                print(f"Degraded Segment 23 Loss: {degraded_segment_23_loss_meter.avg:.4f}")
                print(f"Origin Weak Segment 23 Loss: {origin_weak_segment_23_loss_meter.avg:.4f}")
                print(f"Degraded Weak Segment 23 Loss: {degraded_weak_segment_23_loss_meter.avg:.4f}")

                print("---- Layer 22 ----")
                print(f"Origin Global 22 Loss: {origin_global_22_loss_meter.avg:.4f}")
                print(f"Degraded Global 22 Loss: {degraded_global_22_loss_meter.avg:.4f}")
                print(f"Origin Patch 22 Loss: {origin_patch_22_loss_meter.avg:.4f}")
                print(f"Degraded Patch 22 Loss: {degraded_patch_22_loss_meter.avg:.4f}")
                print(f"Origin Weak Patch 22 Loss: {origin_weak_patch_22_loss_meter.avg:.4f}")
                print(f"Degraded Weak Patch 22 Loss: {degraded_weak_patch_22_loss_meter.avg:.4f}")
                print(f"Origin Segment 22 Loss: {origin_segment_22_loss_meter.avg:.4f}")
                print(f"Degraded Segment 22 Loss: {degraded_segment_22_loss_meter.avg:.4f}")
                print(f"Origin Weak Segment 22 Loss: {origin_weak_segment_22_loss_meter.avg:.4f}")
                print(f"Degraded Weak Segment 22 Loss: {degraded_weak_segment_22_loss_meter.avg:.4f}")

                print("---- Layer 21 ----")
                print(f"Origin Global 21 Loss: {origin_global_21_loss_meter.avg:.4f}")
                print(f"Degraded Global 21 Loss: {degraded_global_21_loss_meter.avg:.4f}")
                print(f"Origin Patch 21 Loss: {origin_patch_21_loss_meter.avg:.4f}")
                print(f"Degraded Patch 21 Loss: {degraded_patch_21_loss_meter.avg:.4f}")
                print(f"Origin Weak Patch 21 Loss: {origin_weak_patch_21_loss_meter.avg:.4f}")
                print(f"Degraded Weak Patch 21 Loss: {degraded_weak_patch_21_loss_meter.avg:.4f}")
                print(f"Origin Segment 21 Loss: {origin_segment_21_loss_meter.avg:.4f}")
                print(f"Degraded Segment 21 Loss: {degraded_segment_21_loss_meter.avg:.4f}")
                print(f"Origin Weak Segment 21 Loss: {origin_weak_segment_21_loss_meter.avg:.4f}")
                print(f"Degraded Weak Segment 21 Loss: {degraded_weak_segment_21_loss_meter.avg:.4f}")

                print("---- Regularization Loss ----")
                print(f"Origin Segment Loss Regularization 24: {origin_segment_loss_reg_meter.avg:.4f}")
                print(f"Degraded Segment Loss Regularization 24: {degraded_segment_loss_reg_meter.avg:.4f}")
                print(f"Origin Patch Loss Regularization 24: {origin_patch_24_loss_reg_meter.avg:.4f}")
                print(f"Degraded Patch Loss Regularization 24: {degraded_patch_24_loss_reg_meter.avg:.4f}")
                print(f"Origin Segment Loss Regularization 23: {origin_segment_loss_reg_23_meter.avg:.4f}")
                print(f"Degraded Segment Loss Regularization 23: {degraded_segment_loss_reg_23_meter.avg:.4f}")
                print(f"Origin Patch Loss Regularization 23: {origin_patch_loss_reg_23_meter.avg:.4f}")
                print(f"Degraded Patch Loss Regularization 23: {degraded_patch_loss_reg_23_meter.avg:.4f}")
                print(f"Origin Segment Loss Regularization 22: {origin_segment_loss_reg_22_meter.avg:.4f}")
                print(f"Degraded Segment Loss Regularization 22: {degraded_segment_loss_reg_22_meter.avg:.4f}")
                print(f"Origin Patch Loss Regularization 22: {origin_patch_loss_reg_22_meter.avg:.4f}")
                print(f"Degraded Patch Loss Regularization 22: {degraded_patch_loss_reg_22_meter.avg:.4f}")
                print(f"Origin Segment Loss Regularization 21: {origin_segment_loss_reg_21_meter.avg:.4f}")
                print(f"Degraded Segment Loss Regularization 21: {degraded_segment_loss_reg_21_meter.avg:.4f}")
                print(f"Origin Patch Loss Regularization 21: {origin_patch_loss_reg_21_meter.avg:.4f}")
                print(f"Degraded Patch Loss Regularization 21: {degraded_patch_loss_reg_21_meter.avg:.4f}")

                print("---- Consistency ----")
                print(f"Global Consistency 24 Loss: {global_24_consistency_loss_meter.avg:.4f}")
                print(f"Patch Consistency 24 Loss: {patch_24_consistency_loss_meter.avg:.4f}")
                print(f"Segment Consistency 24 Loss: {segment_24_consistency_loss_meter.avg:.4f}")
                print(f"Global Consistency 23 Loss: {global_23_consistency_loss_meter.avg:.4f}")
                print(f"Patch Consistency 23 Loss: {patch_23_consistency_loss_meter.avg:.4f}")
                print(f"Segment Consistency 23 Loss: {segment_23_consistency_loss_meter.avg:.4f}")
                print(f"Global Consistency 22 Loss: {global_22_consistency_loss_meter.avg:.4f}")
                print(f"Patch Consistency 22 Loss: {patch_22_consistency_loss_meter.avg:.4f}")
                print(f"Segment Consistency 22 Loss: {segment_22_consistency_loss_meter.avg:.4f}")
                print(f"Global Consistency 21 Loss: {global_21_consistency_loss_meter.avg:.4f}")
                print(f"Patch Consistency 21 Loss: {patch_21_consistency_loss_meter.avg:.4f}")
                print(f"Segment Consistency 21 Loss: {segment_21_consistency_loss_meter.avg:.4f}")

                print("===== Loss Information This Iteration =====")
                print(f"Total Loss: {total_loss_meter.val:.4f}")
                print(f"Origin Loss: {total_origin_loss_meter.val:.4f}")
                print(f"Degraded Loss: {total_degraded_loss_meter.val:.4f}")
                print(f"Consistency Loss: {total_consistency_loss_meter.val:.4f}")

                print("---- Main ----")
                print(f"Origin Main Loss: {origin_main_loss_meter.val:.4f}")
                print(f"Degraded Main Loss: {degraded_main_loss_meter.val:.4f}")

                print("---- Layer 24 ----")
                print(f"Origin Global 24 Loss: {origin_global_24_loss_meter.val:.4f}")
                print(f"Degraded Global 24 Loss: {degraded_global_24_loss_meter.val:.4f}")
                print(f"Origin Patch 24 Loss: {origin_patch_24_loss_meter.val:.4f}")
                print(f"Degraded Patch 24 Loss: {degraded_patch_24_loss_meter.val:.4f}")
                print(f"Origin Weak Patch 24 Loss: {origin_weak_patch_24_loss_meter.val:.4f}")
                print(f"Degraded Weak Patch 24 Loss: {degraded_weak_patch_24_loss_meter.val:.4f}")
                print(f"Origin Segment 24 Loss: {origin_segment_24_loss_meter.val:.4f}")
                print(f"Degraded Segment 24 Loss: {degraded_segment_24_loss_meter.val:.4f}")
                print(f"Origin Weak Segment 24 Loss: {origin_weak_segment_24_loss_meter.val:.4f}")
                print(f"Degraded Weak Segment 24 Loss: {degraded_weak_segment_24_loss_meter.val:.4f}")

                print("---- Layer 23 ----")
                print(f"Origin Global 23 Loss: {origin_global_23_loss_meter.val:.4f}")
                print(f"Degraded Global 23 Loss: {degraded_global_23_loss_meter.val:.4f}")
                print(f"Origin Patch 23 Loss: {origin_patch_23_loss_meter.val:.4f}")
                print(f"Degraded Patch 23 Loss: {degraded_patch_23_loss_meter.val:.4f}")
                print(f"Origin Weak Patch 23 Loss: {origin_weak_patch_23_loss_meter.val:.4f}")
                print(f"Degraded Weak Patch 23 Loss: {degraded_weak_patch_23_loss_meter.val:.4f}")
                print(f"Origin Segment 23 Loss: {origin_segment_23_loss_meter.val:.4f}")
                print(f"Degraded Segment 23 Loss: {degraded_segment_23_loss_meter.val:.4f}")
                print(f"Origin Weak Segment 23 Loss: {origin_weak_segment_23_loss_meter.val:.4f}")
                print(f"Degraded Weak Segment 23 Loss: {degraded_weak_segment_23_loss_meter.val:.4f}")

                print("---- Layer 22 ----")
                print(f"Origin Global 22 Loss: {origin_global_22_loss_meter.val:.4f}")
                print(f"Degraded Global 22 Loss: {degraded_global_22_loss_meter.val:.4f}")
                print(f"Origin Patch 22 Loss: {origin_patch_22_loss_meter.val:.4f}")
                print(f"Degraded Patch 22 Loss: {degraded_patch_22_loss_meter.val:.4f}")
                print(f"Origin Weak Patch 22 Loss: {origin_weak_patch_22_loss_meter.val:.4f}")
                print(f"Degraded Weak Patch 22 Loss: {degraded_weak_patch_22_loss_meter.val:.4f}")
                print(f"Origin Segment 22 Loss: {origin_segment_22_loss_meter.val:.4f}")
                print(f"Degraded Segment 22 Loss: {degraded_segment_22_loss_meter.val:.4f}")
                print(f"Origin Weak Segment 22 Loss: {origin_weak_segment_22_loss_meter.val:.4f}")
                print(f"Degraded Weak Segment 22 Loss: {degraded_weak_segment_22_loss_meter.val:.4f}")

                print("---- Layer 21 ----")
                print(f"Origin Global 21 Loss: {origin_global_21_loss_meter.val:.4f}")
                print(f"Degraded Global 21 Loss: {degraded_global_21_loss_meter.val:.4f}")
                print(f"Origin Patch 21 Loss: {origin_patch_21_loss_meter.val:.4f}")
                print(f"Degraded Patch 21 Loss: {degraded_patch_21_loss_meter.val:.4f}")
                print(f"Origin Weak Patch 21 Loss: {origin_weak_patch_21_loss_meter.val:.4f}")
                print(f"Degraded Weak Patch 21 Loss: {degraded_weak_patch_21_loss_meter.val:.4f}")
                print(f"Origin Segment 21 Loss: {origin_segment_21_loss_meter.val:.4f}")
                print(f"Degraded Segment 21 Loss: {degraded_segment_21_loss_meter.val:.4f}")
                print(f"Origin Weak Segment 21 Loss: {origin_weak_segment_21_loss_meter.val:.4f}")
                print(f"Degraded Weak Segment 21 Loss: {degraded_weak_segment_21_loss_meter.val:.4f}")

                print("---- Regularization Loss ----")
                print(f"Origin Segment Loss Regularization: {origin_segment_loss_reg_meter.val:.4f}")
                print(f"Degraded Segment Loss Regularization: {degraded_segment_loss_reg_meter.val:.4f}")
                print(f"Origin Patch Loss Regularization: {origin_patch_24_loss_reg_meter.val:.4f}")
                print(f"Degraded Patch Loss Regularization: {degraded_patch_24_loss_reg_meter.val:.4f}")

                print("---- Consistency ----")
                print(f"Global Consistency Loss: {global_24_consistency_loss_meter.val:.4f}")
                print(f"Patch Consistency Loss: {patch_24_consistency_loss_meter.val:.4f}")
                print(f"Segment Consistency Loss: {segment_24_consistency_loss_meter.val:.4f}")

            if args.save_model == True and dist.get_rank() == 0 and i%200 == 99:
                torch.save(model.module.state_dict(), "%s/model.%d.pth" % (save_dir, epoch))
                torch.save(optimizer.state_dict(), "%s/optimizer.%d.pth" % (save_dir, epoch))
                torch.save(scaler.state_dict(), "%s/scaler.%d.pth" % (save_dir, epoch))

            end = time.time()
            batch_time = end - iter_start
            batch_time_meter.update(batch_time, B)

        # ---------- epoch end stats ----------
        global_output = torch.cat(A_global_predictions)
        patch_output = torch.cat(A_patch_predictions)
        segment_output = torch.cat(A_segment_predictions)
        main_output = torch.cat(A_main_predictions)
        target = torch.cat(A_targets)

        if args.cls_loss == "ce":
            target = F.one_hot(target, num_classes=2).float()
            global_stats = calculate_stats(torch.softmax(global_output, dim=-1).cpu(), target.cpu())
            patch_stats = calculate_stats(torch.softmax(patch_output, dim=-1).cpu(), target.cpu())
            segment_stats = calculate_stats(torch.softmax(segment_output, dim=-1).cpu(), target.cpu())
        else:
            target = target.unsqueeze(1)
            main_stats = calculate_stats(torch.sigmoid(main_output.unsqueeze(1)).cpu(), target.cpu())
            global_stats = calculate_stats(torch.sigmoid(global_output.unsqueeze(1)).cpu(), target.cpu())
            patch_stats = calculate_stats(torch.sigmoid(patch_output.unsqueeze(1)).cpu(), target.cpu())
            segment_stats = calculate_stats(torch.sigmoid(segment_output.unsqueeze(1)).cpu(), target.cpu())

        if args.cls_loss == "ce":
            global_ap, global_auc, global_acc = global_stats[1]['ap'], global_stats[1]['auc'], global_stats[1]['acc']
            patch_ap, patch_auc, patch_acc = patch_stats[1]['ap'], patch_stats[1]['auc'], patch_stats[1]['acc']
            segment_ap, segment_auc, segment_acc = segment_stats[1]['ap'], segment_stats[1]['auc'], segment_stats[1]['acc']
            main_ap, main_auc, main_acc = main_stats[1]['ap'], main_stats[1]['auc'], main_stats[1]['acc']
        else:
            global_ap, global_auc, global_acc = global_stats[0]['ap'], global_stats[0]['auc'], global_stats[0]['acc']
            patch_ap, patch_auc, patch_acc = patch_stats[0]['ap'], patch_stats[0]['auc'], patch_stats[0]['acc']
            segment_ap, segment_auc, segment_acc = segment_stats[0]['ap'], segment_stats[0]['auc'], segment_stats[0]['acc']
            main_ap, main_auc, main_acc = main_stats[0]['ap'], main_stats[0]['auc'], main_stats[0]['acc']

        epoch_elapsed = time.time() - epoch_start_time
        print("============================================")
        print(f"Rank: {dist.get_rank()}")
        print(f"Finetuning epoch: {epoch} ")
        print(f"Epoch time: {epoch_elapsed:.2f}s ({epoch_elapsed/60:.2f}min)")
        print("Training finished")
        print("Main branch performance:")
        print("ACC: {:.6f}".format(main_acc))
        print("AUC: {:.6f}".format(main_auc))
        print("AP: {:.6f}".format(main_ap))
        print("Global branch performance:")
        print("ACC: {:.6f}".format(global_acc))
        print("AUC: {:.6f}".format(global_auc))
        print("AP: {:.6f}".format(global_ap))
        print("Patch branch performance:")
        print("ACC: {:.6f}".format(patch_acc))
        print("AUC: {:.6f}".format(patch_auc))
        print("AP: {:.6f}".format(patch_ap))
        print("Segment branch performance:")
        print("ACC: {:.6f}".format(segment_acc))
        print("AUC: {:.6f}".format(segment_auc))
        print("AP: {:.6f}".format(segment_ap))
        print("============================================")

        # 学习率调度器更新
        if getattr(args, 'scheduler_step_mode', 'epoch') == 'epoch':
            scheduler.step()

        if args.save_model == True and dist.get_rank() == 0:
            torch.save(model.module.state_dict(), "%s/model.%d.pth" % (save_dir, epoch))
            torch.save(optimizer.state_dict(), "%s/optimizer.%d.pth" % (save_dir, epoch))
            torch.save(scaler.state_dict(), "%s/scaler.%d.pth" % (save_dir, epoch))

        if dist.get_rank() == 0:
            save_data(f"{log_dir}/train_total_loss.csv", epoch=epoch, data=total_loss_meter.avg, data_name="train_total_loss")
            save_data(f"{log_dir}/train_origin_loss.csv", epoch=epoch, data=total_origin_loss_meter.avg, data_name="train_origin_loss")
            save_data(f"{log_dir}/train_degraded_loss.csv", epoch=epoch, data=total_degraded_loss_meter.avg, data_name="train_degraded_loss")
            save_data(f"{log_dir}/train_consistency_loss.csv", epoch=epoch, data=total_consistency_loss_meter.avg, data_name="train_consistency_loss")

        epoch += 1

        # 每个epoch重置计数类
        all_meters = [
            total_loss_meter,
            total_origin_loss_meter,
            total_degraded_loss_meter,
            total_consistency_loss_meter,

            origin_main_loss_meter,
            degraded_main_loss_meter,

            origin_global_24_loss_meter,
            degraded_global_24_loss_meter,
            origin_patch_24_loss_meter,
            degraded_patch_24_loss_meter,
            origin_weak_patch_24_loss_meter,
            degraded_weak_patch_24_loss_meter,
            origin_segment_24_loss_meter,
            degraded_segment_24_loss_meter,
            origin_weak_segment_24_loss_meter,
            degraded_weak_segment_24_loss_meter,

            origin_global_23_loss_meter,
            degraded_global_23_loss_meter,
            origin_patch_23_loss_meter,
            degraded_patch_23_loss_meter,
            origin_weak_patch_23_loss_meter,
            degraded_weak_patch_23_loss_meter,
            origin_segment_23_loss_meter,
            degraded_segment_23_loss_meter,
            origin_weak_segment_23_loss_meter,
            degraded_weak_segment_23_loss_meter,

            origin_global_22_loss_meter,
            degraded_global_22_loss_meter,
            origin_patch_22_loss_meter,
            degraded_patch_22_loss_meter,
            origin_weak_patch_22_loss_meter,
            degraded_weak_patch_22_loss_meter,
            origin_segment_22_loss_meter,
            degraded_segment_22_loss_meter,
            origin_weak_segment_22_loss_meter,
            degraded_weak_segment_22_loss_meter,

            origin_global_21_loss_meter,
            degraded_global_21_loss_meter,
            origin_patch_21_loss_meter,
            degraded_patch_21_loss_meter,
            origin_weak_patch_21_loss_meter,
            degraded_weak_patch_21_loss_meter,
            origin_segment_21_loss_meter,
            degraded_segment_21_loss_meter,
            origin_weak_segment_21_loss_meter,
            degraded_weak_segment_21_loss_meter,

            global_24_consistency_loss_meter,
            patch_24_consistency_loss_meter,
            segment_24_consistency_loss_meter,

            global_23_consistency_loss_meter,
            patch_23_consistency_loss_meter,
            segment_23_consistency_loss_meter,
            global_22_consistency_loss_meter,
            patch_22_consistency_loss_meter,
            segment_22_consistency_loss_meter,
            global_21_consistency_loss_meter,
            patch_21_consistency_loss_meter,
            segment_21_consistency_loss_meter,

            origin_segment_loss_reg_meter,
            degraded_segment_loss_reg_meter,

            origin_patch_24_loss_reg_meter,
            degraded_patch_24_loss_reg_meter,

            origin_segment_loss_reg_23_meter,
            degraded_segment_loss_reg_23_meter,
            origin_patch_loss_reg_23_meter,
            degraded_patch_loss_reg_23_meter,
            origin_segment_loss_reg_22_meter,
            degraded_segment_loss_reg_22_meter,
            origin_patch_loss_reg_22_meter,
            degraded_patch_loss_reg_22_meter,
            origin_segment_loss_reg_21_meter,
            degraded_segment_loss_reg_21_meter,
            origin_patch_loss_reg_21_meter,
            degraded_patch_loss_reg_21_meter,

            data_time_meter,
            dnn_time_meter,
            batch_time_meter
        ]

        for m in all_meters:
            m.reset()
