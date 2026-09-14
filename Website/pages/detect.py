import sys
sys.path.append(r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)")

from pathlib import Path
import streamlit as st
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
import time

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Detection", layout="wide")

# Session states
if "stop" not in st.session_state:
    st.session_state.stop = False

if "show_ld" not in st.session_state:
    st.session_state.show_ld = False     # Face preview

if "show_ld1" not in st.session_state:
    st.session_state.show_ld1 = False    # Descriptor preview

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------
st.markdown("<h2 style='color:white;'>Real-Time Hybrid Detection</h2>", unsafe_allow_html=True)

# ----------------------------------------------------
# LAYOUT → 3 windows side-by-side
# ----------------------------------------------------
# New layout: LD and LD1 stacked vertically
col_video, col_right = st.columns([4, 1])

video_box = col_video.empty()

col_ld = col_right.container()
col_ld1 = col_right.container()

ld_preview_box = col_ld.empty()
ld1_preview_box = col_ld1.empty()

with col_ld:
    if st.button("LD Preview", use_container_width=True):
        st.session_state.show_ld = not st.session_state.show_ld

with col_ld1:
    if st.button("LD1 Preview", use_container_width=True):
        st.session_state.show_ld1 = not st.session_state.show_ld1


# STOP button
if st.button("🛑 STOP"):
    st.session_state.stop = True
    st.switch_page("FrontEnd.py")

# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DL_MODEL_PATH = BASE_DIR / "DeepLearning" / "Models" / "model2.pt"
LD_MODEL_PATH = BASE_DIR / "LocalDescriptor" / "Models" / "LDmodel2.pkl"
LD1_MODEL_PATH = BASE_DIR / "LocalDescriptor1" / "Models" / "LD1model5.pkl"
LD1_SCALER_PATH = BASE_DIR / "LocalDescriptor1" / "Models" / "LD1scaler5.pkl"


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

# ----------------------------------------------------
# FEATURE EXTRACTION
# ----------------------------------------------------
def extract_ld(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, RESIZE_DIM)

    lbp = local_binary_pattern(gray, 8, 1, 'uniform')

    lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    lbp_hist = lbp_hist.astype("float32")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    hog_feat, _ = hog(gray, pixels_per_cell=(16,16), cells_per_block=(2,2),
                      orientations=9, block_norm='L2-Hys', visualize=True)

    lpq_img = lpq(gray, win_size=3)
    lpq_hist, _ = np.histogram(lpq_img, bins=256, range=(0, 256))
    lpq_hist = lpq_hist.astype("float32")
    lpq_hist /= (lpq_hist.sum() + 1e-7)

    return np.concatenate([lbp_hist, lpq_hist, hog_feat]).reshape(1, -1)



def extract_ld1(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, RESIZE_DIM)

    _, ltp_hist = ltp(gray)
    _, bsif_hist = bsif(gray)
    _, wld_feat = wld(gray)

    feat = np.concatenate([ltp_hist, bsif_hist, wld_feat]).reshape(1, -1)
    return ld1_scaler.transform(feat)

# ----------------------------------------------------
# LIVE VIDEO LOOP
# ----------------------------------------------------
cap = cv2.VideoCapture(0)

while True:

    if st.session_state.stop:
        break

    ret, frame = cap.read()
    if not ret:
        break

    results = yolo.predict(frame, stream=False)[0]

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):

        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1

        # YOLO head crop
        head = frame[y1:y2, x1:x2]
        if head.size == 0:
            continue

        # DL prediction
        dl_pred = "real" if int(cls) == 1 else "fake"

        # Face crop for LD and LD1
        fx = x1 + int(0.12 * w)
        fy = y1 + int(0.18 * h)
        fw = int(0.77 * w)
        fh = int(0.87 * h)
        face = frame[fy:fy+fh, fx:fx+fw]
        if face.size == 0:
            continue

        # LD prediction
        ld_pred = "real" if ld_model.predict(extract_ld(face))[0] == 1 else "fake"

        # LD1 prediction
        ld1_pred = "real" if ld1_model.predict(extract_ld1(face))[0] == 1 else "fake"

        # ---------------------------
        # LD PREVIEW WINDOW (LBP | HOG | LPQ)
        # ---------------------------
        if st.session_state.show_ld:

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, RESIZE_DIM)

            # ----- LBP -----
            lbp_map = local_binary_pattern(gray, 8, 1, 'uniform')
            lbp_disp = cv2.normalize(lbp_map, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

            # ----- HOG -----
            hog_feat, hog_img = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                                    orientations=9, visualize=True, block_norm='L2-Hys')
            hog_disp = cv2.normalize(hog_img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

            # ----- LPQ -----
            lpq_img = lpq(gray, win_size=3)
            lpq_disp = cv2.normalize(lpq_img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

            # ----- Combine -----
            ld_preview = np.hstack((lbp_disp, hog_disp, lpq_disp))

            ld_preview_box.image(ld_preview, channels="GRAY")

        else:
            ld_preview_box.empty()

        # ---------------------------
        # LD1 PREVIEW WINDOW (LTP | WLD | BSIF)
        # ---------------------------
        if st.session_state.show_ld1:

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, RESIZE_DIM)

            ltp_img, _ = ltp(gray)
            bsif_img, _ = bsif(gray)
            wld_disp, _ = wld(gray)

            ltp_disp = cv2.normalize(ltp_img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            bsif_disp = cv2.normalize(bsif_img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            wld_disp = cv2.normalize(wld_disp, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

            preview = np.hstack((ltp_disp, bsif_disp, wld_disp))
            ld1_preview_box.image(preview, channels="GRAY")

        else:
            ld1_preview_box.empty()

        # Majority voting
        final = "REAL" if [dl_pred, ld_pred, ld1_pred].count("real") >= 2 else "FAKE"
        color = (0,255,0) if final == "REAL" else (0,0,255)

        # ----- Accuracy calculation -----
        total_frames += 1

        if dl_pred == GROUND_TRUTH:
            correct_dl += 1
        if ld_pred == GROUND_TRUTH:
            correct_ld += 1
        if ld1_pred == GROUND_TRUTH:
            correct_ld1 += 1
        if final.lower() == GROUND_TRUTH:
            correct_final += 1

        # Compute accuracy %
        dl_acc = (correct_dl / total_frames) * 100
        ld_acc = (correct_ld / total_frames) * 100
        ld1_acc = (correct_ld1 / total_frames) * 100
        final_acc = (correct_final / total_frames) * 100

        # cvzone.cornerRect(frame, (x1,y1,w,h), colorC=color, colorR=color)
        # cvzone.putTextRect(frame, final, (x1, y1-10), scale=1, thickness=1, colorR=color)

        cvzone.cornerRect(frame, (x1, y1, w, h), colorC=color, colorR=color)
        cvzone.putTextRect(frame, f"DL:{dl_pred}  LD:{ld_pred}  LD1:{ld1_pred} → {final}",
                           (x1, y1 - 10), scale=1, thickness=1, colorR=color)
        cv2.putText(frame,
                    f"ACC - DL:{dl_acc:.1f}%  LD:{ld_acc:.1f}%  LD1:{ld1_acc:.1f}%  FINAL:{final_acc:.1f}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

    # Show main video
    video_box.image(frame, channels="BGR", width=600)
    time.sleep(0.001)

cap.release()
cv2.destroyAllWindows()
