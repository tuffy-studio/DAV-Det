import os
import csv
import argparse
import random
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from tqdm import tqdm

from models.gps_dino import GPS_DINO


def sample_frames_from_video(
        video_path,
        num_frames=16,
        skip_head_tail=0.1
):
    """
    从视频中均匀采样若干帧
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    start_idx = int(total_frames * skip_head_tail)
    end_idx = int(total_frames * (1 - skip_head_tail))

    effective_frames = end_idx - start_idx

    if effective_frames < num_frames:
        start_idx = 0
        end_idx = total_frames
        effective_frames = total_frames

    frame_indices = np.linspace(
        start_idx,
        end_idx - 1,
        num_frames,
        dtype=int
    ).tolist()

    frames = []

    for idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(Image.fromarray(frame))

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Failed to extract frames: {video_path}")

    # 如果帧数不足，重复已有帧补齐到目标数量
    if len(frames) < num_frames:
        repeat_times = (num_frames + len(frames) - 1) // len(frames)
        frames = (frames * repeat_times)[:num_frames]

    return frames


# =========================================================
# 图像预处理
# =========================================================
def preprocess_frames(frames, img_size, mean, std):

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    tensors = [transform(img) for img in frames]

    return torch.stack(tensors, dim=0)


# =========================================================
# CPU 多线程读取视频
# =========================================================
def load_video(video_path, label, args):

    try:

        frames = sample_frames_from_video(
            video_path=video_path,
            num_frames=args.num_frames,
            skip_head_tail=0.1
        )

        images = preprocess_frames(
            frames,
            args.img_size,
            args.mean,
            args.std
        )

        return {
            'video_path': video_path,
            'label': label,
            'images': images
        }

    except Exception as e:

        return {
            'video_path': video_path,
            'label': label,
            'error': str(e)
        }


# =========================================================
# 多视频 Batch GPU 推理
# =========================================================
def inference_batch_videos(model, batch_data, args):

    valid_data = []
    error_results = []

    # =====================================================
    # 过滤错误视频
    # =====================================================
    for data in batch_data:

        if 'error' in data:

            error_results.append({
                'video_path': data['video_path'],
                'prob': -1,
            })

        else:
            valid_data.append(data)

    # 全部失败
    if len(valid_data) == 0:
        return error_results

    # =====================================================
    # 拼接所有视频帧
    # =====================================================
    all_images = torch.cat(
        [x['images'] for x in valid_data],
        dim=0
    )

    all_images = all_images.to(
        args.device,
        non_blocking=True
    )

    num_frames = args.num_frames

    # =====================================================
    # 分 batch 推理（防止 OOM）
    # =====================================================
    batch_size = args.frame_batch_size

    all_main_probs = []
    all_global_probs = []
    all_patch_probs = []
    all_segment_probs = []

    with torch.no_grad():

        for i in range(0, len(all_images), batch_size):

            batch = all_images[i:i + batch_size]

            main_logits, global_logits, patch_logits, segment_logits, visual_feature = model(
                batch,
                is_training=False,
                if_return_feature=True
            ) # the shape of visual_feature: [B, 1024*3]

            all_main_probs.append(
                torch.sigmoid(main_logits).cpu()
            )

            all_global_probs.append(
                torch.sigmoid(global_logits).cpu()
            )

            all_patch_probs.append(
                torch.sigmoid(patch_logits).cpu()
            )

            all_segment_probs.append(
                torch.sigmoid(segment_logits).cpu()
            )

            # 释放中间变量
            del main_logits, global_logits, patch_logits, segment_logits, batch

    # 清理 GPU 缓存
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()

    main_probs = torch.cat(all_main_probs)
    global_probs = torch.cat(all_global_probs)
    patch_probs = torch.cat(all_patch_probs)
    segment_probs = torch.cat(all_segment_probs)

    # =====================================================
    # 拆回每个视频
    # =====================================================
    results = []

    for i, data in enumerate(valid_data):

        s = i * num_frames
        e = (i + 1) * num_frames

        result = {
            'video_path': data['video_path'],
            'prob': main_probs[s:e].mean().item(),
        }

        results.append(result)

    # 加入错误结果
    results.extend(error_results)

    return results


# =========================================================
# 主函数
# =========================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_eval',
        required=True,
        type=str
    )

    parser.add_argument(
        '--output_path',
        default='./video_output.csv',
        type=str
    )

    parser.add_argument(
        '--pretrain_path',
        required=True,
        type=str
    )

    parser.add_argument(
        '--num_frames',
        default=16,
        type=int
    )

    parser.add_argument(
        '--frame_batch_size',
        default=64,
        type=int,
        help='image batch size'
    )

    parser.add_argument(
        '--video_batch_size',
        default=4,
        type=int,
        help='videos per GPU inference'
    )

    parser.add_argument(
        '--img_size',
        default=640,
        type=int
    )

    parser.add_argument(
        '--num_workers',
        default=8,
        type=int
    )

    parser.add_argument(
        '--mean',
        nargs=3,
        type=float,
        default=[0.485, 0.456, 0.406]
    )

    parser.add_argument(
        '--std',
        nargs=3,
        type=float,
        default=[0.229, 0.224, 0.225]
    )

    parser.add_argument(
        '--backbone_name',
        default='/data/data2/jielun/XPlainVerse/MM_2026_method/HiDINO/dinov3-vitl16-pretrain-lvd1689m/',
        type=str
    )

    parser.add_argument(
        '--mode',
        default='inference',
        choices=['eval', 'inference']
    )

    args = parser.parse_args()

    # =====================================================
    # device
    # =====================================================
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    args.device = device

    print(f'Using device: {device}')

    # =====================================================
    # 读取 CSV
    # =====================================================
    video_list = []

    with open(args.data_eval, 'r') as f:

        reader = csv.reader(f)

        header = next(reader)

        if args.mode == 'eval':

            for row in reader:
                video_list.append(
                    (row[0], int(row[1]))
                )

        else:

            for row in reader:
                video_list.append(
                    (row[0], None)
                )

    random.shuffle(video_list)

    print(f'Total videos in CSV: {len(video_list)}')

    # =====================================================
    # 断点恢复
    # =====================================================
    processed_videos = set()

    if os.path.exists(args.output_path):

        print(f'Found existing output: {args.output_path}')

        with open(args.output_path, 'r') as f:

            reader = csv.DictReader(f)

            for row in reader:
                processed_videos.add(
                    row['video_path']
                )

        print(f'Already processed: {len(processed_videos)}')

    # 过滤已完成
    video_list = [
        (v, l)
        for v, l in video_list
        if v not in processed_videos
    ]

    print(f'Remaining videos: {len(video_list)}')


    # =====================================================
    # 加载模型
    # =====================================================
    model = GPS_DINO(
        backbone_name=args.backbone_name,
        use_lora=True,
        lora_r=32,
        lora_alpha=16,
        use_deep_supervision=False
    )

    checkpoint = torch.load(
        args.pretrain_path,
        map_location='cpu'
    )

    miss, unexpected = model.load_state_dict(
        checkpoint,
        strict=False
    )

    print(
        f'Loaded checkpoint | '
        f'missing={len(miss)} '
        f'unexpected={len(unexpected)}'
    )

    model = model.to(device)

    model.eval()

    # =====================================================
    # CSV 输出
    # =====================================================
    output_header = [
        'video_path',
        'prob'
    ]

    write_header = not os.path.exists(args.output_path)

    f_out = open(
        args.output_path,
        'a',
        newline=''
    )

    writer = csv.DictWriter(
        f_out,
        fieldnames=output_header
    )

    if write_header:
        writer.writeheader()
        f_out.flush()

    # =====================================================
    # 流式读取 + 推理（控制内存）
    # =====================================================
    from collections import deque

    processed_count = 0
    batch_data = []
    prefetch = args.video_batch_size * 2

    pbar = tqdm(
        total=len(video_list),
        desc='Processing',
        ncols=120
    )

    with ThreadPoolExecutor(
            max_workers=args.num_workers
    ) as executor:

        future_queue = deque()
        video_iter = iter(video_list)

        # 预提交第一批任务
        for _ in range(prefetch):
            try:
                vp, lbl = next(video_iter)
                future_queue.append(
                    executor.submit(load_video, vp, lbl, args)
                )
            except StopIteration:
                break

        while future_queue:

            # 等待队列中第一个任务完成
            future = future_queue.popleft()

            try:
                data = future.result()

                # 补充一个新任务（保持并发数稳定）
                try:
                    vp, lbl = next(video_iter)
                    future_queue.append(
                        executor.submit(load_video, vp, lbl, args)
                    )
                except StopIteration:
                    pass

                batch_data.append(data)

                # =========================================
                # 达到 batch size 就推理
                # =========================================
                if len(batch_data) >= args.video_batch_size:

                    batch_results = inference_batch_videos(
                        model,
                        batch_data,
                        args
                    )

                    for result in batch_results:

                        writer.writerow(result)

                        f_out.flush()

                        processed_count += 1

                        pbar.set_postfix({
                            'prob':
                                f"{result['prob']:.4f}"
                        })

                        pbar.update(1)

                    # 清理 batch_data 释放内存
                    for d in batch_data:
                        if 'images' in d:
                            del d['images']
                    batch_data = []
                    gc.collect()

            except Exception as e:

                print(f'\nError: {e}')

                # 错误时也要补充新任务
                try:
                    vp, lbl = next(video_iter)
                    future_queue.append(
                        executor.submit(load_video, vp, lbl, args)
                    )
                except StopIteration:
                    pass

        # =============================================
        # 最后残余 batch
        # =============================================
        if len(batch_data) > 0:

            batch_results = inference_batch_videos(
                model,
                batch_data,
                args
            )

            for result in batch_results:

                writer.writerow(result)

                f_out.flush()

                processed_count += 1

                pbar.update(1)

            for d in batch_data:
                if 'images' in d:
                    del d['images']
            batch_data = []
            gc.collect()

    pbar.close()
    f_out.close()

    print(f'\nSaved to: {args.output_path}')
    print(f'Total processed: {processed_count}')


# =========================================================
# main
# =========================================================
if __name__ == '__main__':
    main()