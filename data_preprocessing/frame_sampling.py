import os
import cv2
import random
import pandas as pd
from tqdm import tqdm

# =====================================
# NOTE: 请修改save_root路径
# =====================================

input_csv = "./train_video_labels.csv" # 之前生成的CSV文件，包含video_path和label两列

save_root = "path_to_frame_save_directory" # 用于保存抽帧后的图片，目录结构为 save_root/label/video_name/frame.jpg

output_csv = "./train_frame_labels.csv" # 最终输出的CSV文件，包含frame_path和label两列，供后续视觉检测器训练使用

processed_txt = "./processed_videos.txt" # 记录已经处理过的视频路径，避免重复处理，任意路径即可

# =====================================
# 创建csv表头
# =====================================
if not os.path.exists(output_csv):

    pd.DataFrame(
        columns=["frame_path", "label"]
    ).to_csv(output_csv, index=False)

# =====================================
# 读取已经处理过的视频
# =====================================
processed_videos = set()

if os.path.exists(processed_txt):

    with open(processed_txt, "r") as f:

        for line in f:
            processed_videos.add(line.strip())

print(f"Already processed: {len(processed_videos)} videos")


# =====================================
# 均匀采样
# =====================================
def sample_frame_indices(total_frames, num_samples):

    if total_frames <= num_samples + 2:

        return list(
            range(1, min(total_frames - 1, num_samples + 1))
        )

    start = 1
    end = total_frames - 2

    interval = (end - start) / (num_samples - 1)

    indices = [
        int(start + interval * i)
        for i in range(num_samples)
    ]

    return indices


# =====================================
# 抽帧
# =====================================
def extract_frames(video_path, save_dir, num_frames):

    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(f"Failed to open: {video_path}")
        return []

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames < 3:

        print(f"Too short: {video_path}")

        cap.release()

        return []

    sample_indices = sample_frame_indices(
        total_frames,
        num_frames
    )

    saved_paths = []

    for i, frame_idx in enumerate(sample_indices):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx
        )

        ret, frame = cap.read()

        if not ret:
            continue

        # 随机后缀
        ext = random.choice([".jpg"])

        frame_name = f"{i:03d}{ext}"

        frame_path = os.path.join(
            save_dir,
            frame_name
        )

        # 保存
        if ext == ".jpg":

            cv2.imwrite(
                frame_path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

        else:

            cv2.imwrite(
                frame_path,
                frame,
                [cv2.IMWRITE_PNG_COMPRESSION, 3]
            )

        saved_paths.append(frame_path)

    cap.release()

    return saved_paths


# =====================================
# 主程序
# =====================================
df = pd.read_csv(input_csv)

for idx, row in tqdm(df.iterrows(), total=len(df)):

    video_path = row["video_path"]

    label = int(row["label"])

    # =====================================
    # 断点恢复
    # =====================================
    if video_path in processed_videos:
        continue

    # =====================================
    # fake抽 16 帧，real抽 8 帧
    # =====================================
    num_frames = 16 if label == 1 else 8

    # =====================================
    # 获取类别（直接从CSV的label读取，0或1）
    # =====================================
    category = str(label)

    # =====================================
    # 视频名
    # =====================================
    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    # =====================================
    # 保存目录
    # =====================================
    save_dir = os.path.join(
        save_root,
        category,
        video_name
    )

    try:

        # =====================================
        # 抽帧
        # =====================================
        frame_paths = extract_frames(
            video_path,
            save_dir,
            num_frames
        )

        # =====================================
        # 立即写csv
        # =====================================
        temp_df = pd.DataFrame([
            {
                "frame_path": p,
                "label": label
            }
            for p in frame_paths
        ])

        temp_df.to_csv(
            output_csv,
            mode="a",
            header=False,
            index=False
        )

        # =====================================
        # 写入已完成记录
        # =====================================
        with open(processed_txt, "a") as f:

            f.write(video_path + "\n")

        processed_videos.add(video_path)

    except Exception as e:

        print(f"Error processing {video_path}")
        print(e)

print("Done!")
