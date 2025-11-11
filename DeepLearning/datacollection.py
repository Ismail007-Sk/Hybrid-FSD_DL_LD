# Import required modules
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
import os

#cvzone FaceDetector (MediaPipe-based) ✅

# ------------------ Configuration ------------------
dataset_path = r'C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\Dataset'
categories = ['Real', 'Fake']
confidence = 0.8
save = True
blurThreshold = 35
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6

# ------------------ Create folders ------------------
for category in categories:
    img_folder = os.path.join(dataset_path, category)
    label_folder = os.path.join(dataset_path, category + "_labels")
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

# ------------------ Initialize ------------------
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

counts = {"Real": 2520, "Fake": 2050}
print("Press 'r' for Real, 'f' for Fake, 'q' to quit.")

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from webcam.")
            break

        imgOut = img.copy()
        img, bboxs = detector.findFaces(img, draw=False)

        if bboxs:
            for bbox in bboxs:
                x, y, w, h = bbox["bbox"]
                score = bbox["score"][0]

                if score > confidence:
                    # Offset adjustment
                    offsetW = (offsetPercentageW / 100) * w
                    offsetH = (offsetPercentageH / 100) * h
                    x = int(max(0, x - offsetW))
                    y = int(max(0, y - offsetH * 3))
                    w = int(w + offsetW * 2)
                    h = int(h + offsetH * 3.5)

                    # Face sharpness
                    imgFace = img[y:y + h, x:x + w]
                    blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

                    # Draw rectangle
                    cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(imgOut, f'Score:{int(score * 100)}% Blur:{blurValue}', (x, y - 20),
                                       scale=2, thickness=3)

                    # Normalized YOLO coordinates
                    ih, iw, _ = img.shape
                    xc, yc = x + w / 2, y + h / 2
                    xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                    wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                    xcn, ycn = min(1, xcn), min(1, ycn)
                    wn, hn = min(1, wn), min(1, hn)

                    # Wait for key input
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('r') and blurValue > blurThreshold:
                        counts["Real"] += 1
                        img_name = f"Real_{counts['Real']}.jpg"
                        label_name = f"Real_{counts['Real']}.txt"
                        cv2.imwrite(os.path.join(dataset_path, "Real", img_name), img)
                        with open(os.path.join(dataset_path, "Real_labels", label_name), "w") as f:
                            f.write(f"1 {xcn} {ycn} {wn} {hn}\n")
                        print(f"Saved Real: {counts['Real']}")

                    elif key == ord('f') and blurValue > blurThreshold:
                        counts["Fake"] += 1
                        img_name = f"Fake_{counts['Fake']}.jpg"
                        label_name = f"Fake_{counts['Fake']}.txt"
                        cv2.imwrite(os.path.join(dataset_path, "Fake", img_name), img)
                        with open(os.path.join(dataset_path, "Fake_labels", label_name), "w") as f:
                            f.write(f"0 {xcn} {ycn} {wn} {hn}\n")
                        print(f"Saved Fake: {counts['Fake']}")

                    elif key == ord('q'):
                        raise KeyboardInterrupt

        cv2.imshow("Data Collection", imgOut)

except KeyboardInterrupt:
    print("\n[INFO] Data collection stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam and windows released successfully.")
    print(f"Final counts → Real: {counts['Real']} | Fake: {counts['Fake']}")
