import cv2
import joblib
import numpy as np
from pathlib import Path
from skimage.feature import local_binary_pattern, hog
from LPQ import lpq  # Your LPQ module
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

# --- Paths ---
MODEL_PATH = Path(r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor\Models\LDmodel2.pkl")

# --- Load trained SVM model ---
clf = joblib.load(MODEL_PATH)

# --- Feature parameters (must match datacollect.py) ---
RADIUS = 1
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

HOG_PIXELS = (16, 16)
HOG_CELLS = (2, 2)
HOG_ORIENT = 9

LPQ_WINDOW = 3
RESIZE_DIM = (128, 128)  # must match datacollect.py

# --- Initialize cvzone FaceDetector (MediaPipe-based) ---
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# --- Start webcam ---
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

            if score > 0.5:
                # --- Face only (no hair, no ears) ---
                x_new, y_new, w_new, h_new = x, y, w, h
                head_crop = frame[y_new:y_new + h_new, x_new:x_new + w_new]

                if head_crop.size == 0:
                    continue

                gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray_head, RESIZE_DIM)

                # --- Extract features ---
                lbp = local_binary_pattern(gray_resized, N_POINTS, RADIUS, METHOD)
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
                lbp_hist = lbp_hist.astype(np.float32)
                lbp_hist /= (lbp_hist.sum() + 1e-7)

                hog_feat, _ = hog(
                    gray_resized,
                    pixels_per_cell=HOG_PIXELS,
                    cells_per_block=HOG_CELLS,
                    orientations=HOG_ORIENT,
                    block_norm='L2-Hys',
                    visualize=True
                )

                lpq_img = lpq(gray_resized, win_size=LPQ_WINDOW)
                lpq_hist, _ = np.histogram(lpq_img, bins=256, range=(0, 256))
                lpq_hist = lpq_hist.astype(np.float32)
                lpq_hist /= (lpq_hist.sum() + 1e-7)

                fused_feat = np.concatenate([lbp_hist, lpq_hist, hog_feat])

                if fused_feat.shape[0] != clf.n_features_in_:
                    print(f"Feature length mismatch: {fused_feat.shape[0]} != {clf.n_features_in_}")
                    continue

                pred = clf.predict([fused_feat])[0]

                # Set colors for visualization
                if pred == 1:
                    label = "Real"
                    box_color = (0, 255, 0)  # Green
                    text_color = (0, 0, 0)
                else:
                    label = "Fake"
                    box_color = (0, 0, 255)  # Red
                    text_color = (255, 255, 255)

                # Draw fancy bounding box
                cvzone.cornerRect(frame, (x_new, y_new, w_new, h_new), colorC=box_color, colorR=box_color)

                # Draw label
                cvzone.putTextRect(frame, label, (x_new, max(35, y_new)), scale=1, thickness=1, colorR=box_color, colorT=text_color)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
