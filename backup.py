import cv2
import numpy as np
from ultralytics import YOLO
from skimage.feature import local_binary_pattern, hog
from LocalDescriptor.LPQ import lpq
from LocalDescriptor1.LTP import ltp
from LocalDescriptor1.BSIF import bsif
from LocalDescriptor1.WDL import wld
import joblib
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from pathlib import Path

# ------------------ Paths ------------------
DL_MODEL_PATH  = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\Models\model2.pt"
LD_MODEL_PATH = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor\Models\LDmodel1.pkl"   # LBP+HOG+LPQ
LD1_MODEL_PATH = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\Models\LD1model1.pkl"  # LTP+BSIF+WDL
LD1_SCALER_PATH = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\Models\LD1scaler1.pkl"

# ------------------ Load Models ------------------
dl_model = YOLO(DL_MODEL_PATH)
ld_model = joblib.load(LD_MODEL_PATH)
ld1_model = joblib.load(LD1_MODEL_PATH)
ld1_scaler = joblib.load(LD1_SCALER_PATH)

# Resize must match datacollection
RESIZE_DIM = (128, 128)

detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

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
    feat = ld1_scaler.transform(feat)
    return feat

# ------------------ Real-time Loop ------------------
cap = cv2.VideoCapture(0)
CONF_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _, bboxs = detector.findFaces(frame, draw=False)

    if bboxs:
        for b in bboxs:
            x,y,w,h = b["bbox"]
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            # DL prediction
            dl_pred = "fake"
            r = dl_model(face, stream=False)[0]
            if len(r.boxes) > 0:
                cls = int(r.boxes[0].cls[0])
                dl_pred = "real" if cls == 1 else "fake"

            # LD1 prediction
            ld1_pred = "real" if ld_model.predict(extract_ld1(face))[0] == 1 else "fake"

            # LD2 prediction
            ld2_pred = "real" if ld1_model.predict(extract_ld2(face))[0] == 1 else "fake"

            # ------------------ Majority Voting ------------------
            votes = [dl_pred, ld1_pred, ld2_pred]
            final_label = "REAL" if votes.count("real") >= 2 else "FAKE"
            color = (0,255,0) if final_label=="REAL" else (0,0,255)

            cvzone.cornerRect(frame, (x,y,w,h), colorC=color, colorR=color)
            cvzone.putTextRect(frame, final_label, (x,y-10), scale=1, thickness=1, colorR=color)



            # --- Draw Fancy Box & Text (3-Model Fusion) ---
            # cvzone.cornerRect(frame, (x,y,w,h), colorC=color, colorR=color)
            #
            # cvzone.putTextRect(frame,
            #                    f"DL:{dl_pred}  LD1:{ld1_pred}  LD2:{ld2_pred}  →  {final_label}",
            #                    (x, max(35, y - 10)),
            #                    scale=1,
            #                    thickness=1,
            #                    colorR=color,
            #                    colorT=(0, 0, 0) if final_label == "REAL" else (255, 255, 255))

    cv2.imshow("HYBRID (DL + LD1 + LD2)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
