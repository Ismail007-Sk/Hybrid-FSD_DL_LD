from LTP import ltp
from BSIF import bsif
from WLD import wld


import cv2
from cvzone.FaceDetectionModule import FaceDetector
from pathlib import Path
import numpy as np


# --- Resize size ---
RESIZE_DIM = (128, 128)

# --- Output dataset folders ---
BASE_DIR = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\Dataset"
REAL_DIR = Path(BASE_DIR, "Real")
FAKE_DIR = Path(BASE_DIR, "Fake")
for folder in [REAL_DIR, FAKE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Counters (continue your numbering)
count = {"real": 1100, "fake": 1300}

# --- Face Detector (MediaPipe) ---
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'r' to save REAL | 'f' to save FAKE | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_out = frame.copy()
    img, bboxs = detector.findFaces(frame, draw=False)

    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            if score > 0.80:
                # FACE ONLY (NO HAIR, NO EARS)
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

                # Grayscale + Resize
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, RESIZE_DIM)

                # --- LTP ---
                ltp_img, ltp_hist = ltp(gray)

                # --- BSIF ---
                bsif_img, bsif_hist = bsif(gray)

                # --- WLD ---
                wld_disp, wld_feature = wld(gray)  # already normalized vector

                # --- Feature Fusion ---
                features = np.concatenate([ltp_hist, bsif_hist, wld_feature])

                # --- Preview Window ---
                # Normalize maps for display
                ltp_disp = cv2.normalize(ltp_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                bsif_disp = cv2.normalize(bsif_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                wld_disp = wld_disp

                preview = np.hstack((ltp_disp, bsif_disp, wld_disp))
                cv2.imshow("LTP | BSIF | WLD Preview", preview)

                # Draw face box
                cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Key press actions
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                elif key == ord('r'):
                    np.save(str(REAL_DIR / f"real_{count['real']}.npy"), features)
                    count['real'] += 1
                    print(f"Saved Real #{count['real']}")

                elif key == ord('f'):
                    np.save(str(FAKE_DIR / f"fake_{count['fake']}.npy"), features)
                    count['fake'] += 1
                    print(f"Saved Fake #{count['fake']}")

    cv2.imshow("Webcam", img_out)

cap.release()
cv2.destroyAllWindows()
