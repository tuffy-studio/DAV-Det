import os
import csv
from pathlib import Path

def generate_video_labels_csv(train_dir, output_csv):
    """
    遍历 train 目录，生成 video_path,label 的 CSV 文件。
    
    目录结构规则：
    - train/fake_fake/... -> label=1
    - train/fake_real/... -> label=1  
    - train/real_fake/... -> label=0
    - train/real_real/... -> label=0
    """
    train_path = Path(train_dir)
    
    # 类别到标签的映射
    category_to_label = {
        'fake_fake': 1,
        'fake_real': 1,
        'real_fake': 0,
        'real_real': 0
    }
    
    video_entries = []
    
    # 遍历 train 目录下的所有子目录
    for category_dir in train_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        category = category_dir.name
        if category not in category_to_label:
            print(f"Warning: Unknown category {category}, skipping...")
            continue
        
        label = category_to_label[category]
        
        # 递归查找该类别下的所有 .mp4 文件
        for video_file in category_dir.rglob("*.mp4"):
            video_path = str(video_file)
            video_entries.append((video_path, label))
    
    # 排序，保证输出顺序稳定
    video_entries.sort(key=lambda x: x[0])
    
    # 写入 CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_path', 'label'])
        for video_path, label in video_entries:
            writer.writerow([video_path, label])
    
    print(f"Done! Total videos: {len(video_entries)}")
    print(f"CSV saved to: {output_csv}")
    
    # 打印各类别统计
    for category, label in category_to_label.items():
        count = sum(1 for _, l in video_entries if l == label and category in [Path(p).parts[8] for p, _ in video_entries])
        print(f"  {category}: {count} videos (label={label})")

if __name__ == "__main__":
    # NOTE: 请修改以下路径
    train_dir = "path_to_training_set_root" # 训练集根目录，包含 fake_fake、fake_real、real_fake、real_real 四个子目录
    output_csv = "./train_video_labels.csv"
    
    generate_video_labels_csv(train_dir, output_csv)
