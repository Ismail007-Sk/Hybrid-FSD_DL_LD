from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# ---------------- PATHS ----------------
DL_MODEL_PATH = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\Models\model2.pt"
# DATA_YAML     = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\SplitData\data.yaml"

TEST_LABELS   = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\SplitData\test\labels"
TEST_IMAGES   = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\SplitData\test\images"

# ---------------- LOAD MODEL ----------------
yolo = YOLO(DL_MODEL_PATH)

# ---------------- RUN PREDICTION ----------------
results = yolo.predict(
    source=TEST_IMAGES,
    save_txt=True,
    conf=0.5,
    verbose=False
)

PRED_LABELS = "runs/detect/predict/labels"

# ---------------- METRIC COMPUTATION ----------------
y_true = []
y_pred = []

for label_file in os.listdir(TEST_LABELS):
    gt_path = os.path.join(TEST_LABELS, label_file)
    pred_path = os.path.join(PRED_LABELS, label_file)

    # Ground truth
    gt_class = int(open(gt_path).read().split()[0])
    y_true.append(gt_class)

    # Prediction
    if os.path.exists(pred_path):
        pred_class = int(open(pred_path).read().split()[0])
    else:
        pred_class = 0  # no detection → fake
    y_pred.append(pred_class)

# ---------------- METRICS ----------------
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:\n", cm)
print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision : {precision_score(y_true, y_pred):.4f}")
print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
print(f"F1-score  : {f1_score(y_true, y_pred):.4f}")

# ----------------------------------
# Face Anti-Spoofing Metrics
# ----------------------------------
# Label convention:
# 0 → Fake (Attack)
# 1 → Real (Bona fide)

APCER = cm[0, 1] / cm[0].sum() if cm[0].sum() > 0 else 0.0
BPCER = cm[1, 0] / cm[1].sum() if cm[1].sum() > 0 else 0.0
HTER = (APCER + BPCER) / 2

print("\n===== Face Anti-Spoofing Metrics =====")
print(f"APCER : {APCER * 100:.2f}%")
print(f"BPCER : {BPCER * 100:.2f}%")
print(f"HTER  : {HTER * 100:.2f}%")