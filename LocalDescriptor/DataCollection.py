import cv2
from cvzone.FaceDetectionModule import FaceDetector
from skimage.feature import local_binary_pattern, hog
from pathlib import Path
import numpy as np
from LPQ import lpq  # Custom LPQ module

# --- LBP parameters ---
RADIUS = 1
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

# --- HOG parameters ---
HOG_PIXELS = (16, 16)
HOG_CELLS = (2, 2)
HOG_ORIENT = 9

# --- LPQ parameters ---
LPQ_WINDOW_SIZE = 3
RESIZE_DIM = (128, 128)  # Resize dimension for consistent feature extraction

# --- Output dataset folders ---
BASE_DIR = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor\Dataset"
REAL_DIR = Path(BASE_DIR, "Real")
FAKE_DIR = Path(BASE_DIR, "Fake")
for folder in [REAL_DIR, FAKE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Counters
count = {"real":1785, "fake":1100}

# --- cvzone Face Detector (MediaPipe-based) ---
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'r' to save as real, 'f' to save as fake, 'q' to quit.")

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

            if score > 0.8:
                # --- Face only (no hair, no ears) ---
                x_new, y_new, w_new, h_new = x, y, w, h
                head_crop = frame[y_new:y_new + h_new, x_new:x_new + w_new]

                if head_crop.size == 0:
                    continue


                # Convert to grayscale and resize
                gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray_head, RESIZE_DIM)

                # --- LBP ---
                lbp = local_binary_pattern(gray_resized, N_POINTS, RADIUS, METHOD)
                lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
                lbp_hist = lbp_hist.astype(np.float32)
                lbp_hist /= (lbp_hist.sum() + 1e-7)
                lbp_img = (lbp / lbp.max() * 255).astype("uint8")

                # --- HOG ---
                hog_features, hog_img = hog(
                    gray_resized,
                    pixels_per_cell=HOG_PIXELS,
                    cells_per_block=HOG_CELLS,
                    orientations=HOG_ORIENT,
                    block_norm='L2-Hys',
                    visualize=True
                )
                hog_img = (hog_img / hog_img.max() * 255).astype("uint8")

                # --- LPQ ---
                lpq_img_raw = lpq(gray_resized, win_size=LPQ_WINDOW_SIZE)
                lpq_hist, _ = np.histogram(lpq_img_raw, bins=256, range=(0, 256))
                lpq_hist = lpq_hist.astype(np.float32)
                lpq_hist /= (lpq_hist.sum() + 1e-7)
                lpq_img_disp = ((lpq_img_raw - lpq_img_raw.min()) /
                                (lpq_img_raw.max() - lpq_img_raw.min()) * 255).astype("uint8")

                # --- Concatenate features ---
                features = np.concatenate([lbp_hist, lpq_hist, hog_features])

                # --- Display combined preview ---
                combined_preview = np.hstack((lbp_img, hog_img, lpq_img_disp))
                cv2.imshow("LBP | HOG | LPQ", combined_preview)

                # --- Draw expanded rectangle ---
                cv2.rectangle(img_out, (x_new, y_new), (x_new+w_new, y_new+h_new), (0, 255, 0), 2)

                # --- Handle key presses ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                elif key == ord('r'):
                    np.save(str(REAL_DIR / f"real_{count['real']}.npy"), features)
                    count['real'] += 1
                    print(f"Saved Real feature vector #{count['real']}")
                elif key == ord('f'):
                    np.save(str(FAKE_DIR / f"fake_{count['fake']}.npy"), features)
                    count['fake'] += 1
                    print(f"Saved Fake feature vector #{count['fake']}")



    cv2.imshow("Webcam", img_out)

cap.release()
cv2.destroyAllWindows()
