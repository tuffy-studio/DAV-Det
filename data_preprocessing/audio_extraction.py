import os
import subprocess
import csv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import Counter

# ======================================================
# NOTE: 请修改base_dir路径
# ======================================================

base_dir = "path_to_training_set_root" # 训练集根目录，包含 fake_fake、fake_real、real_fake、real_real 四个子目录
output_csv = "./train_audio_labels.csv"

folder_to_audio_label = {
    "fake_fake": 1,
    "fake_real": 0,
    "real_fake": 1,
    "real_real": 0,
}

# ======================================================
# ffmpeg 提取
# ======================================================

def extract_single(args):
    mp4_path, output_wav, label = args

    try:
        os.makedirs(os.path.dirname(output_wav), exist_ok=True)

        # 已存在且非空 -> 跳过
        if os.path.exists(output_wav) and os.path.getsize(output_wav) > 1000:
            return (output_wav, label, True, "skip")

        result = subprocess.run(
            [
                'ffmpeg',
                '-y',
                '-i', mp4_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                output_wav
            ],
            capture_output=True,
            text=True,
            timeout=300
        )

        # 提取成功
        if (
            result.returncode == 0
            and os.path.exists(output_wav)
            and os.path.getsize(output_wav) > 1000
        ):
            return (output_wav, label, True, "extract")

        return (mp4_path, label, False, "fail")

    except Exception:
        return (mp4_path, label, False, "fail")

# ======================================================
# 主函数
# ======================================================

def main():

    # --------------------------------------------------
    # 已有CSV -> 断点恢复
    # --------------------------------------------------

    processed_paths = set()

    if os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                processed_paths.add(row[0])

        print(f"Resume mode: loaded {len(processed_paths)} entries")

    # --------------------------------------------------
    # 初始化 CSV
    # --------------------------------------------------

    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_path', 'label'])

    extract_tasks = []

    # --------------------------------------------------
    # 扫描数据集
    # --------------------------------------------------

    for folder, label in folder_to_audio_label.items():

        folder_path = os.path.join(base_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        print(f"\nScanning: {folder}")

        audio_files = {}
        mp4_files = {}

        for root, dirs, files in os.walk(folder_path):

            for file in files:

                file_lower = file.lower()
                file_path = os.path.join(root, file)

                # 防止 basename 冲突
                rel_key = os.path.relpath(file_path, folder_path)
                rel_key = os.path.splitext(rel_key)[0]

                if file_lower.endswith(('.wav', '.flac')):
                    audio_files[rel_key] = file_path

                elif file_lower.endswith('.mp4'):
                    mp4_files[rel_key] = file_path

        # --------------------------------------------------
        # 匹配
        # --------------------------------------------------

        for rel_key, mp4_path in mp4_files.items():

            # 已有外部音频
            if rel_key in audio_files:

                audio_path = audio_files[rel_key]

                if audio_path not in processed_paths:

                    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([audio_path, label])

                    processed_paths.add(audio_path)

            else:

                mp4_dir = os.path.dirname(mp4_path)

                extract_dir = os.path.join(mp4_dir, "audio_extracted")

                output_wav = os.path.join(
                    extract_dir,
                    os.path.basename(rel_key) + ".wav"
                )

                # 已写入CSV -> 跳过
                if output_wav in processed_paths:
                    continue

                extract_tasks.append(
                    (mp4_path, output_wav, label)
                )

    # ======================================================
    # 多进程提取
    # ======================================================

    print(f"\nNeed extraction: {len(extract_tasks)}")

    n_workers = min(cpu_count(), 16)

    print(f"Using workers: {n_workers}")

    success_count = 0
    skip_count = 0
    fail_count = 0

    with Pool(n_workers) as pool:

        for result in tqdm(
            pool.imap_unordered(extract_single, extract_tasks),
            total=len(extract_tasks),
            desc="Extracting Audio"
        ):

            audio_path, label, success, status = result

            if success:

                # 实时写CSV
                with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([audio_path, label])

                processed_paths.add(audio_path)

                if status == "skip":
                    skip_count += 1
                else:
                    success_count += 1

            else:
                fail_count += 1

    # ======================================================
    # 最终统计
    # ======================================================

    print("\nDone.")
    print(f"Extract success : {success_count}")
    print(f"Already existed : {skip_count}")
    print(f"Failed          : {fail_count}")

    # 重新统计CSV标签
    labels = Counter()

    with open(output_csv, 'r', encoding='utf-8') as f:

        reader = csv.reader(f)
        next(reader)

        for row in reader:
            labels[int(row[1])] += 1

    print("\nLabel distribution:")
    print(f"label=0 : {labels[0]}")
    print(f"label=1 : {labels[1]}")

    print(f"\nCSV saved to:")
    print(output_csv)

# ======================================================
# 入口
# ======================================================

if __name__ == '__main__':
    main()