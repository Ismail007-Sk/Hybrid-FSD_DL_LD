import sys
from pathlib import Path

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))


import threading
import av
import cv2
import joblib
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from skimage.feature import local_binary_pattern, hog
from LocalDescriptor.LPQ import lpq
from LocalDescriptor1.LTP import ltp
from LocalDescriptor1.BSIF import bsif
from LocalDescriptor1.WLD import wld
import cvzone



DL_MODEL_PATH = BASE_DIR / "DeepLearning" / "Models" / "model2.pt"
LD_MODEL_PATH = BASE_DIR / "LocalDescriptor" / "Models" / "LDmodel2.pkl"
LD1_MODEL_PATH = BASE_DIR / "LocalDescriptor1" / "Models" / "LD1model5.pkl"
LD1_SCALER_PATH = BASE_DIR / "LocalDescriptor1" / "Models" / "LD1scaler5.pkl"

# ----------------------------------------------------
# PAGE
# ----------------------------------------------------
st.set_page_config(page_title="Detection", layout="wide")

st.markdown(
    "<h2 style='color:white;'>Real-Time Hybrid Detection</h2>",
    unsafe_allow_html=True
)

# ----------------------------------------------------
# STATE
# ----------------------------------------------------
if "show_ld" not in st.session_state:
    st.session_state.show_ld = False

if "show_ld1" not in st.session_state:
    st.session_state.show_ld1 = False

# ----------------------------------------------------
# MODELS
# ----------------------------------------------------
@st.cache_resource
def load_models():
    return (
        YOLO(str(DL_MODEL_PATH)),
        joblib.load(str(LD_MODEL_PATH)),
        joblib.load(str(LD1_MODEL_PATH)),
        joblib.load(str(LD1_SCALER_PATH))
    )

yolo, ld_model, ld1_model, ld1_scaler = load_models()

RESIZE_DIM = (128, 128)
GROUND_TRUTH = "real"

# ----------------------------------------------------
# FEATURE EXTRACTION
# ----------------------------------------------------
def extract_ld(face):
    gray = cv2.resize(
        cv2.cvtColor(face, cv2.COLOR_BGR2GRAY),
        RESIZE_DIM
    )

    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    lbp_hist = lbp_hist.astype("float32")
    lbp_hist /= lbp_hist.sum() + 1e-7

    hog_feat, _ = hog(
        gray,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm="L2-Hys",
        visualize=True
    )

    lpq_img = lpq(gray, win_size=3)
    lpq_hist, _ = np.histogram(lpq_img, bins=256, range=(0, 256))
    lpq_hist = lpq_hist.astype("float32")
    lpq_hist /= lpq_hist.sum() + 1e-7

    return np.concatenate(
        [lbp_hist, lpq_hist, hog_feat]
    ).reshape(1, -1)


def extract_ld1(face):
    gray = cv2.resize(
        cv2.cvtColor(face, cv2.COLOR_BGR2GRAY),
        RESIZE_DIM
    )

    _, ltp_hist = ltp(gray)
    _, bsif_hist = bsif(gray)
    _, wld_feat = wld(gray)

    feat = np.concatenate(
        [ltp_hist, bsif_hist, wld_feat]
    ).reshape(1, -1)

    return ld1_scaler.transform(feat)

# ----------------------------------------------------
# VIDEO PROCESSOR
# ----------------------------------------------------
class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock = threading.Lock()
        self.total = 0
        self.dl_correct = 0
        self.ld_correct = 0
        self.ld1_correct = 0
        self.final_correct = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = yolo.predict(
            img,
            verbose=False
        )[0]

        for box, cls in zip(
            results.boxes.xyxy,
            results.boxes.cls
        ):
            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            w, h = x2 - x1, y2 - y1

            if w <= 0 or h <= 0:
                continue

            # DL
            dl_pred = "real" if int(cls) == 1 else "fake"

            # Face crop
            fx = x1 + int(0.12 * w)
            fy = y1 + int(0.18 * h)
            fw = int(0.77 * w)
            fh = int(0.87 * h)

            face = img[
                fy:fy + fh,
                fx:fx + fw
            ]

            if face.size == 0:
                continue

            # LD
            try:
                ld_pred = (
                    "real"
                    if ld_model.predict(extract_ld(face))[0] == 1
                    else "fake"
                )
            except Exception:
                ld_pred = "fake"

            # LD1
            try:
                ld1_pred = (
                    "real"
                    if ld1_model.predict(extract_ld1(face))[0] == 1
                    else "fake"
                )
            except Exception:
                ld1_pred = "fake"

            # Majority voting
            predictions = [dl_pred, ld_pred, ld1_pred]
            final = (
                "REAL"
                if predictions.count("real") >= 2
                else "FAKE"
            )

            color = (
                (0, 255, 0)
                if final == "REAL"
                else (0, 0, 255)
            )

            # Accuracy
            with self.lock:
                self.total += 1
                self.dl_correct += dl_pred == GROUND_TRUTH
                self.ld_correct += ld_pred == GROUND_TRUTH
                self.ld1_correct += ld1_pred == GROUND_TRUTH
                self.final_correct += final.lower() == GROUND_TRUTH

                total = self.total
                dl_acc = self.dl_correct / total * 100
                ld_acc = self.ld_correct / total * 100
                ld1_acc = self.ld1_correct / total * 100
                final_acc = self.final_correct / total * 100

            # Detection box
            cvzone.cornerRect(
                img,
                (x1, y1, w, h),
                colorC=color,
                colorR=color
            )

            cvzone.putTextRect(
                img,
                f"DL:{dl_pred} LD:{ld_pred} LD1:{ld1_pred} -> {final}",
                (x1, max(30, y1 - 10)),
                scale=1,
                thickness=1,
                colorR=color
            )

            # Accuracy
            cv2.putText(
                img,
                f"ACC DL:{dl_acc:.1f}% LD:{ld_acc:.1f}% "
                f"LD1:{ld1_acc:.1f}% FINAL:{final_acc:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# ----------------------------------------------------
# LAYOUT
# ----------------------------------------------------
col_video, col_right = st.columns([4, 1])

with col_right:

    if st.button("LD Preview", use_container_width=True):
        st.session_state.show_ld = not st.session_state.show_ld

    if st.button("LD1 Preview", use_container_width=True):
        st.session_state.show_ld1 = not st.session_state.show_ld1

col_video, col_right = st.columns([4, 1])

with col_video:
    ctx = webrtc_streamer(
        key="hybrid-face-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
        async_processing=True
    )
