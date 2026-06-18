import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize

PRED_FILE = "four_class.txt"
GT_FILE = "four_class_gt.txt"

# ======================
# 读取
# ======================
pred_df = pd.read_csv(
    PRED_FILE,
    sep=r"\s+",
    header=None,
    names=[
        "video_path",
        "0_pred",
        "1_pred",
        "2_pred",
        "3_pred"
    ]
)

gt_df = pd.read_csv(
    GT_FILE,
    sep=r"\s+",
    header=None,
    names=["file_path", "label"]
)

df = pred_df.merge(
    gt_df,
    left_on="video_path",
    right_on="file_path",
    how="inner"
)

print(f"Matched samples: {len(df)}")

# ======================
# 指标
# ======================
score_cols = [
    "0_pred",
    "1_pred",
    "2_pred",
    "3_pred"
]

y_true = df["label"].values
y_score = df[score_cols].values
y_pred = np.argmax(y_score, axis=1)

acc = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

y_true_bin = label_binarize(
    y_true,
    classes=[0, 1, 2, 3]
)

auc = roc_auc_score(
    y_true_bin,
    y_score,
    multi_class="ovr",
    average="macro"
)

ap = average_precision_score(
    y_true_bin,
    y_score,
    average="macro"
)

print(f"ACC         : {acc:.6f}")
print(f"Macro F1    : {macro_f1:.6f}")
print(f"Weighted F1 : {weighted_f1:.6f}")
print(f"AUC         : {auc:.6f}")
print(f"AP          : {ap:.6f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report")
print(classification_report(y_true, y_pred, digits=4))