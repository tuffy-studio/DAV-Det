import os
import pandas as pd

# =========================================================
# 路径
# =========================================================
audio_csv = "./audio_prob.csv"
video_csv = "./video_prob.csv"

binary_txt = "binary.txt"
four_txt = "four_class.txt"

# =========================================================
# 读取csv
# =========================================================
audio_df = pd.read_csv(audio_csv)
video_df = pd.read_csv(video_csv)

# =========================================================
# 提取文件名
# =========================================================
audio_df["video_name"] = audio_df["file_path"].apply(os.path.basename)
video_df["video_name"] = video_df["file_path"].apply(os.path.basename)

# =========================================================
# 转dict
# prob = fake probability
# =========================================================
audio_dict = dict(zip(audio_df["video_name"], audio_df["prob"]))
video_dict = dict(zip(video_df["video_name"], video_df["prob"]))

# =========================================================
# 找公共视频
# =========================================================
common_videos = sorted(
    set(audio_dict.keys()) &
    set(video_dict.keys())
)

print(f"matched videos: {len(common_videos)}")

# =========================================================
# write binary.txt
# =========================================================
with open(binary_txt, "w") as f_bin:

    f_bin.write("video_path,0_pred,1_pred\n")

    for video_name in common_videos:

        # fake prob
        af = float(audio_dict[video_name])
        vf = float(video_dict[video_name])

        # real prob
        ar = 1.0 - af
        vr = 1.0 - vf

        fake_prob = max(af, vf)
        real_prob = 1 - fake_prob

        f_bin.write(
            f"{video_name} "
            f"{real_prob:.6f} "
            f"{fake_prob:.6f}\n"
        )

# =========================================================
# write four_class.txt
# =========================================================
with open(four_txt, "w") as f_four:

    f_four.write(
        "video_path,0_pred,1_pred,2_pred,3_pred\n"
    )

    for video_name in common_videos:

        af = float(audio_dict[video_name])
        vf = float(video_dict[video_name])

        ar = 1.0 - af
        vr = 1.0 - vf

        # RR FF FR RF
        rr = vr * ar
        ff = vf * af
        fr = vf * ar
        rf = vr * af

        f_four.write(
            f"{video_name} "
            f"{rr:.6f} "
            f"{ff:.6f} "
            f"{fr:.6f} "
            f"{rf:.6f}\n"
        )

print("done!")
print(f"saved: {binary_txt}")
print(f"saved: {four_txt}")