import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataloader import FineTuneDataset
from models.gps_dino import GPS_DINO
from training import train
# DDP 环境配置
import torch
import torch.distributed as dist
import os


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Finetune Stage')

    parser.add_argument('--data_train', type=str, help='path to train data csv')
    parser.add_argument('--data_val', type=str, help='path to val data csv')

    parser.add_argument('--restart', action='store_true', help='Whether to restart training.')
    parser.add_argument('--restart_epoch', default=0, type=int, help='Epoch to restart training from')
    parser.add_argument('--if_new_epoch', action='store_true')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='number of steps to accumulate gradients for before performing an optimizer step')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--gpu_num', default=4, type=int, help='number of GPUs')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--head_lr_ratio', default=10.0, type=float, help='learning rate ratio for the classification head compared to the backbone')
    parser.add_argument('--token_head_lr_ratio', default=0.1, type=float, help='learning rate ratio for the token classification head compared to the backbone')

    parser.add_argument('--n_epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--warmup_epochs', default=1, type=int, help='number of warmup epochs for lr scheduler')
    parser.add_argument('--scheduler_step_mode', default='epoch', type=str, choices=['epoch', 'batch'], help='lr scheduler step mode: epoch or batch')

    parser.add_argument('--use_amp', action='store_true', help='Whether to use mixed precision training.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print verbose training logs.')
    parser.add_argument('--cls_loss', default="focal", type=str, help='the loss function for classification head, can be "ce" or "focal"')
    parser.add_argument('--focal_alpha', default=0.6, type=float, help='alpha for sigmoid focal loss')
    parser.add_argument('--focal_gamma_pos', default=2.0, type=float, help='gamma for positive samples in sigmoid focal loss')
    parser.add_argument('--focal_gamma_neg', default=2.0, type=float, help='gamma for negative samples in sigmoid focal loss')

    parser.add_argument('--pretrain_path', type=str, help='path to pretrain model', default=None)
    parser.add_argument('--checkpoint_root', type=str, help='root directory to save checkpoints', default=None)
    parser.add_argument('--save_model', action='store_true', help='Whether to save model checkpoints.')
    parser.add_argument('--save_dir', default='checkpoints', type=str, help='directory to save checkpoints')
    parser.add_argument('--img_size', default=512, type=int, help='input image size for training')

    parser.add_argument('--backbone_configure', type=str, default='./dinov3-vitl16-pretrain-lvd1689m/',
                        help='path to DINOv3 ViT-L/16 pretrained backbone')
    parser.add_argument('--layer_indices', type=str, default='',
                        help='comma-separated layer indices, e.g., "24". Empty means use default.')
    parser.add_argument('--use_lora', action='store_true', default=True)
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--lora_r', default=32, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.1, type=float)
    parser.add_argument('--use_deep_supervision', action='store_true', default=True)
    parser.add_argument('--no_deep_supervision', action='store_true')
    parser.add_argument('--unfreeze_norm', action='store_true', default=True)
    parser.add_argument('--freeze_norm', action='store_true')

    args = parser.parse_args()
    
    # Handle boolean flags
    if args.no_lora:
        args.use_lora = False
    if args.no_deep_supervision:
        args.use_deep_supervision = False
    if args.freeze_norm:
        args.unfreeze_norm = False
    
    # Parse layer_indices
    if args.layer_indices:
        args.layer_indices = [int(x) for x in args.layer_indices.split(',')]
    else:
        args.layer_indices = None

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # 构造数据集
    train_dataset = FineTuneDataset(args.data_train, data_augment=True, mean=mean, std=std, if_normalize=True, img_size=args.img_size)

    args.ratio = train_dataset.get_real_fake_ratio()

    if dist.get_rank() == 0:
        print(f"Using Train: {len(train_dataset)}")
        print(f"real/fake samples ratio: {args.ratio}")
        print(f"Now using cls loss: {args.cls_loss}")

    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size//args.gpu_num,
        sampler=train_sampler,
        num_workers=args.num_workers//args.gpu_num,
        pin_memory=False,
        drop_last=True,
        collate_fn=train_dataset.collate_fn
    )

    # 构造模型并加载预训练权重
    ft_model = GPS_DINO(
        backbone_name=args.backbone_configure,
        layer_indices=args.layer_indices,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_deep_supervision=args.use_deep_supervision,
        unfreeze_norm=args.unfreeze_norm
    )

    # init model
    if args.restart and args.restart_epoch > 1:
        restart_path = os.path.join(args.checkpoint_root, f"model.{args.restart_epoch-1}.pth")
        if os.path.exists(restart_path):
            mdl_weight = torch.load(restart_path, map_location='cpu')
            miss, unexpected = ft_model.load_state_dict(mdl_weight, strict=False)
            if dist.get_rank() == 0:
                print("Missing: ", miss)
                print("Unexpected: ", unexpected)
                print(f"Restart training from epoch {args.restart_epoch}, loaded model from {restart_path}")
        else:
            if dist.get_rank() == 0:
                print(f"Restart epoch {args.restart_epoch} specified but no checkpoint found at {restart_path}, starting training from DINOv3 weights.")
    else:
        if args.pretrain_path and os.path.exists(args.pretrain_path):
            mdl_weight = torch.load(args.pretrain_path, map_location='cpu')
            miss, unexpected = ft_model.load_state_dict(mdl_weight, strict=False)
            if dist.get_rank() == 0:
                print("Missing: ", miss)
                print("Unexpected: ", unexpected)
                print('now load pretrain model from {:s}, missing keys: {:d}, unexpected keys: {:d}'.format(args.pretrain_path, len(miss), len(unexpected)))
        else:
            if dist.get_rank() == 0:
                print("Note you are finetuning a model from DINOv3 weights.")
        

    device = torch.device(f"cuda:{local_rank}")
    ft_model = ft_model.to(device)

    # 模型设置DDP
    ft_model = torch.nn.parallel.DistributedDataParallel(ft_model, device_ids=[local_rank], find_unused_parameters=True)

    # 开始训练
    if dist.get_rank() == 0:
        print("Now start training for %d epochs"%args.n_epochs)
    train(ft_model, train_loader, train_sampler, args)


if __name__ == '__main__':
    main()
