import cv2
import cvzone
from ultralytics import YOLO

# ------------------ Settings ------------------
#       C:\Users\Desktop\PycharmProjects\Face_Sp_Dt_DEEP(CNN)\models\11.pt
YOLO_MODEL_PATH = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\Models\model2.pt"
# C:\Users\Desktop\PycharmProjects\FSD_YOLO\Models\model1.pt

# Load YOLO model
yolo = YOLO(YOLO_MODEL_PATH)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO prediction
    results = yolo.predict(frame, stream=False)[0]

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1

        # Adjust bounding box
        y_new = max(0, y1 - int(0.02 * h))  # move top up
        h_new = min(frame.shape[0] - y_new, int(h * 1))  # increase height
        x_new = max(0, x1 - int(0.01 * w))  # move left
        w_new = min(frame.shape[1] - x_new, int(w * 1))  # increase width


        # Determine class label
        if int(cls) == 0:
            label = "Fake"
            box_color = (0, 0, 255)  # Red
            text_color = (255, 255, 255)
        else:
            label = "Real"
            box_color = (0, 255, 0)  # Green
            text_color = (0, 0, 0)

        # Draw fancy bounding box
        cvzone.cornerRect(frame, (x_new, y_new, w_new, h_new), colorC=box_color, colorR=box_color)

        # Draw label
        cvzone.putTextRect(frame, label, (max(0, x_new), max(35, y_new)),
                           scale=1, thickness=1, colorR=box_color, colorT=text_color)



    # Display
    cv2.imshow("Face Spoofing Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
