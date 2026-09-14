import cv2
import numpy as np
from ultralytics import YOLO
from skimage.feature import local_binary_pattern, hog
from LocalDescriptor.LPQ import lpq
from LocalDescriptor1.LTP import ltp
from LocalDescriptor1.BSIF import bsif
from LocalDescriptor1.WLD import wld
import joblib
import cvzone
from pathlib import Path

# ------------------ Paths ------------------
DL_MODEL_PATH  = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\Models\model2.pt"
LD_MODEL_PATH  = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor\Models\LDmodel2.pkl"
LD1_MODEL_PATH = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\Models\LD1model5.pkl"
LD1_SCALER_PATH = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\Models\LD1scaler5.pkl"

# ------------------ Load Models ------------------
yolo = YOLO(DL_MODEL_PATH)
ld_model = joblib.load(LD_MODEL_PATH)
ld1_model = joblib.load(LD1_MODEL_PATH)
ld1_scaler = joblib.load(LD1_SCALER_PATH)



# ------------------ Accuracy Counters ------------------
total_frames = 0
correct_dl = 0
correct_ld = 0
correct_ld1 = 0
correct_final = 0

# Webcam ground truth (you in front of camera)
GROUND_TRUTH = "real"



RESIZE_DIM = (128, 128)

# ------------------ Extract LBP + HOG + LPQ ------------------
def extract_ld1(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, RESIZE_DIM)

    lbp = local_binary_pattern(gray, 8, 1, 'uniform')
    lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)

    hog_feat, _ = hog(gray, pixels_per_cell=(16,16), cells_per_block=(2,2),
                      orientations=9, block_norm='L2-Hys', visualize=True)

    lpq_img = lpq(gray, win_size=3)
    lpq_hist, _ = np.histogram(lpq_img, bins=256, range=(0, 256))
    lpq_hist = lpq_hist / (lpq_hist.sum() + 1e-7)

    return np.concatenate([lbp_hist, lpq_hist, hog_feat]).reshape(1, -1)

# ------------------ Extract LTP + BSIF + WLD ------------------
def extract_ld2(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, RESIZE_DIM)

    _, ltp_hist = ltp(gray)
    _, bsif_hist = bsif(gray)
    _, wld_feat = wld(gray)

    feat = np.concatenate([ltp_hist, bsif_hist, wld_feat]).reshape(1, -1)
    return ld1_scaler.transform(feat)

# ------------------ Real-time Loop ------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo.predict(frame, stream=False)[0]

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        x, y, w, h = x1, y1, x2 - x1, y2 - y1

        head_crop = frame[y:y+h, x:x+w]
        if head_crop.size == 0:
            continue

        # YOLO prediction
        dl_pred = "fake"
        if int(cls) == 1:
            dl_pred = "real"

        # Face-only crop for LD
        fx = x + int(0.12 * w)
        fy = y + int(0.18 * h)
        fw = int(0.77 * w)
        fh = int(0.87 * h)

        face = frame[fy:fy+fh, fx:fx+fw]
        if face.size == 0:
            continue

        ld_pred = "real" if ld_model.predict(extract_ld1(face))[0] == 1 else "fake"
        ld1_pred = "real" if ld1_model.predict(extract_ld2(face))[0] == 1 else "fake"

        # Majority Voting
        votes = [dl_pred, ld_pred, ld1_pred]
        final_label = "REAL" if votes.count("real") >= 2 else "FAKE"
        color = (0,255,0) if final_label=="REAL" else (0,0,255)



        # ----- Accuracy calculation -----
        total_frames += 1

        if dl_pred == GROUND_TRUTH:
            correct_dl += 1
        if ld_pred == GROUND_TRUTH:
            correct_ld += 1
        if ld1_pred == GROUND_TRUTH:
            correct_ld1 += 1
        if final_label.lower() == GROUND_TRUTH:
            correct_final += 1

        # Compute accuracy %
        dl_acc = (correct_dl / total_frames) * 100
        ld_acc = (correct_ld / total_frames) * 100
        ld1_acc = (correct_ld1 / total_frames) * 100
        final_acc = (correct_final / total_frames) * 100



        # cvzone.cornerRect(frame, (x, y, w, h), colorC=color, colorR=color)
        # cvzone.putTextRect(frame, final_label, (x, y - 10), scale=1, thickness=1, colorR=color)

        cvzone.cornerRect(frame, (x, y, w, h), colorC=color, colorR=color)
        cvzone.putTextRect(frame, f"DL:{dl_pred}  LD:{ld_pred}  LD1:{ld1_pred} → {final_label}",
                            (x, y-10), scale=1, thickness=1, colorR=color)
        cv2.putText(frame,
                    f"ACC - DL:{dl_acc:.1f}%  LD:{ld_acc:.1f}%  LD1:{ld1_acc:.1f}%  FINAL:{final_acc:.1f}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

    cv2.imshow("Hybrid FSD", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
