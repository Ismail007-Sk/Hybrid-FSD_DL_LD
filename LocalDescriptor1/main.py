import cv2
import joblib
import numpy as np
from pathlib import Path
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

from LTP import ltp
from BSIF import bsif
from WLD import wld

# --- MODEL + SCALER PATHS ---
MODEL_PATH = Path(r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\Models\LD1model6.pkl.")
SCALER_PATH = Path(r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\Models\LD1scaler6.pkl")

clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Resize must match datacollection
RESIZE_DIM = (128, 128)

# Face Detector
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

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
                # --- FACE ONLY CROP (NO HAIR / NO EARS) ---
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, RESIZE_DIM)

                # --- LTP ---
                ltp_img, ltp_hist = ltp(gray)

                # --- BSIF ---
                bsif_img, bsif_hist = bsif(gray)

                # --- WLD ---
                wld_disp, wld_feature = wld(gray)

                # --- Feature Fusion ---
                features = np.concatenate([ltp_hist, bsif_hist, wld_feature]).reshape(1, -1)
                features = scaler.transform(features)

                # --- Prediction ---
                pred = clf.predict(features)[0]

                if pred == 1:
                    label = "Real"
                    box_color = (0, 255, 0)
                    text_color = (0, 0, 0)
                else:
                    label = "Fake"
                    box_color = (0, 0, 255)
                    text_color = (255, 255, 255)

                # # Draw face box + label
                # cv2.rectangle(img_out, (x, y), (x+w, y+h), color, 2)
                # cv2.putText(img_out, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Draw fancy bounding box
                cvzone.cornerRect(frame, (x, y ,w ,h), colorC=box_color, colorR=box_color)

                # Draw label
                cvzone.putTextRect(frame, label, (x, max(35, y)), scale=1, thickness=1, colorR=box_color,
                                   colorT=text_color)

                # # Preview textures (Optional)
                # ltp_disp = gray
                # bsif_disp = cv2.normalize(bsif_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                # wld_disp = cv2.normalize(wld_disp, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                # preview = np.hstack((ltp_disp, bsif_disp, wld_disp))
                # cv2.imshow("LTP | BSIF | WLD", preview)

    cv2.imshow("Face Verification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
