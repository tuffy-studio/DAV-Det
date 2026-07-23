import csv
import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import EvalDataset
from models.gps_dino import GPS_DINO


def evaluate(model, dataloader, args):
    """单图推理：输出 file_path, prob。"""
    model.eval()

    output_path = args.output_path
    write_mode = getattr(args, 'write_mode', 'batch')
    use_tta = args.if_test_time_augment

    header = ['file_path', 'prob']

    f = None
    writer = None
    if write_mode == 'stream':
        f = open(output_path, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(header)

    results = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Inferencing", ncols=100)
        for batch in pbar:
            if use_tta:
                file_paths, images, flip_images = batch
                images = images.to(args.device)
                flip_images = flip_images.to(args.device)
                main_logits, _, _, _ = model(images, is_training=False)
                flip_main_logits, _, _, _ = model(flip_images, is_training=False)
                # TTA：原图和翻转图的 logits 平均后再 sigmoid
                probs = torch.sigmoid((main_logits + flip_main_logits) / 2).cpu()
            else:
                file_paths, images = batch
                images = images.to(args.device)
                main_logits, _, _, _ = model(images, is_training=False)
                probs = torch.sigmoid(main_logits).cpu()

            for i in range(len(file_paths)):
                results.append([file_paths[i], float(probs[i])])

            pbar.set_postfix({"batch_size": len(file_paths)})

            if write_mode == 'stream' and writer is not None:
                writer.writerows(results)
                f.flush()
                results = []

    if write_mode == 'batch':
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)
    elif write_mode == 'stream' and f is not None:
        f.close()

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image-level Inference')
    parser.add_argument('--data_eval', required=True, type=str, help='input CSV with file_path in first column')
    parser.add_argument('--output_path', default='./output.csv', type=str)
    parser.add_argument('--pretrain_path', required=True, type=str, help='path to GPS_DINO checkpoint')
    parser.add_argument('--backbone_configure', required=True, type=str, help='path to DINOv3 weights')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--img_size', default=640, type=int)
    parser.add_argument('--write_mode', default='stream', type=str, choices=['batch', 'stream'])
    parser.add_argument('--if_test_time_augment', action='store_true', help='horizontal flip TTA')
    args = parser.parse_args()

    eval_dataset = EvalDataset(
        csv_file=args.data_eval,
        mode='inference',
        img_size=args.img_size,
        if_test_time_augment=args.if_test_time_augment,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False
    )

    print(f"Using Eval: {len(eval_dataset)}")

    # 加载模型
    ft_model = GPS_DINO(
        backbone_name=args.backbone_configure,
        layer_indices=[24],
        use_lora=True,
        lora_r=32,
        lora_alpha=16,
        use_deep_supervision=False
    )

    mdl_weight = torch.load(args.pretrain_path, map_location='cpu')
    miss, unexpected = ft_model.load_state_dict(mdl_weight, strict=False)
    print(f"Loaded checkpoint from {args.pretrain_path}")
    print(f"Missing keys: {len(miss)}, Unexpected keys: {len(unexpected)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    ft_model = ft_model.to(device)
    ft_model.eval()

    evaluate(ft_model, eval_loader, args)
