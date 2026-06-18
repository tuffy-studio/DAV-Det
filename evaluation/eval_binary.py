import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PRED_FILE = "binary.txt"
GT_FILE = "binary_gt.txt"

# ======================
# 读取
# ======================
pred_df = pd.read_csv(
    PRED_FILE,
    sep=r"\s+",
    header=None,
    names=["video_path", "0_pred", "1_pred"]
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
y_true = df["label"].values
y_score = df["1_pred"].values
y_pred = (y_score >= 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_score)
ap = average_precision_score(y_true, y_score)
f1 = f1_score(y_true, y_pred)

print(f"ACC : {acc:.6f}")
print(f"AUC : {auc:.6f}")
print(f"AP  : {ap:.6f}")
print(f"F1  : {f1:.6f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report")
print(classification_report(y_true, y_pred, digits=4))