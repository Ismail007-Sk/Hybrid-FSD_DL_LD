import os
import cv2
import numpy as np
import joblib
from ultralytics import YOLO

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from skimage.feature import local_binary_pattern, hog
from LocalDescriptor.LPQ import lpq

from LocalDescriptor1.LTP import ltp
from LocalDescriptor1.BSIF import bsif
from LocalDescriptor1.WLD import wld


# ---------------- DATASET ----------------
DATASET_PATH = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (4Novel LD + DL)\Dataset"


# ---------------- MODEL PATHS ----------------
YOLO_MODEL_PATH = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\Models\model2.pt"

LD_BLOCK1_MODEL = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor\Models\LDmodel2.pkl"

LD_BLOCK2_MODEL = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\Models\LD1model7.pkl"

LD_BLOCK2_SCALER = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\Models\LD1scaler7.pkl"


# ---------------- LOAD MODELS ----------------
yolo = YOLO(YOLO_MODEL_PATH)

block1_model = joblib.load(LD_BLOCK1_MODEL)

block2_model = joblib.load(LD_BLOCK2_MODEL)
block2_scaler = joblib.load(LD_BLOCK2_SCALER)


# ---------------- PARAMETERS ----------------

RADIUS = 1
N_POINTS = 8 * RADIUS
METHOD = "uniform"

HOG_PIXELS = (16,16)
HOG_CELLS = (2,2)
HOG_ORIENT = 9

LPQ_WINDOW = 3

RESIZE_DIM = (128,128)


# ---------------- STORAGE ----------------

y_true = []
y_pred = []
y_prob = []


# ---------------- LOOP THROUGH DATASET ----------------

for label_name in ["fake","real"]:

    folder = os.path.join(DATASET_PATH,label_name)

    true_label = 1 if label_name=="real" else 0

    for img_name in os.listdir(folder):

        img_path = os.path.join(folder,img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue


        # ---------------- YOLO FACE DETECTION ----------------

        results = yolo.predict(image, conf=0.5, verbose=False)[0]

        if len(results.boxes)==0:
            continue


        box = results.boxes.xyxy[0]

        cls = int(results.boxes.cls[0])
        conf = float(results.boxes.conf[0])

        x1,y1,x2,y2 = map(int,box)

        face = image[y1:y2,x1:x2]

        if face.size==0:
            continue


        gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,RESIZE_DIM)


        # ---------------- YOLO PREDICTION ----------------

        dl_pred = 1 if cls==1 else 0
        dl_prob = conf if cls==1 else 1-conf


        # ================= BLOCK 1 =================
        # LBP + LPQ + HOG

        lbp = local_binary_pattern(gray,N_POINTS,RADIUS,METHOD)

        lbp_hist,_ = np.histogram(lbp.ravel(),bins=256,range=(0,256))
        lbp_hist = lbp_hist.astype(np.float32)
        lbp_hist /= (lbp_hist.sum()+1e-7)


        hog_feat,_ = hog(
            gray,
            pixels_per_cell=HOG_PIXELS,
            cells_per_block=HOG_CELLS,
            orientations=HOG_ORIENT,
            block_norm='L2-Hys',
            visualize=True
        )


        lpq_img = lpq(gray,win_size=LPQ_WINDOW)

        lpq_hist,_ = np.histogram(lpq_img,bins=256,range=(0,256))
        lpq_hist = lpq_hist.astype(np.float32)
        lpq_hist /= (lpq_hist.sum()+1e-7)


        block1_feat = np.concatenate([lbp_hist,lpq_hist,hog_feat]).reshape(1,-1)

        block1_prob = block1_model.predict_proba(block1_feat)[0][1]

        block1_pred = 1 if block1_prob>=0.5 else 0



        # ================= BLOCK 2 =================
        # LTP + BSIF + WLD

        ltp_img,ltp_hist = ltp(gray)
        bsif_img,bsif_hist = bsif(gray)
        wld_disp,wld_feat = wld(gray)

        block2_feat = np.concatenate([ltp_hist,bsif_hist,wld_feat]).reshape(1,-1)

        # APPLY SCALER
        block2_feat = block2_scaler.transform(block2_feat)

        block2_prob = block2_model.predict_proba(block2_feat)[0][1]

        block2_pred = 1 if block2_prob>=0.5 else 0



        # ================= MAJORITY VOTING =================

        votes = [dl_pred,block1_pred,block2_pred]

        final_label = 1 if sum(votes)>=2 else 0


        # probability (average used for AUC)

        final_prob = np.mean([dl_prob,block1_prob,block2_prob])


        y_true.append(true_label)
        y_pred.append(final_label)
        y_prob.append(final_prob)



# ---------------- METRICS ----------------

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

cm = confusion_matrix(y_true,y_pred)

acc = accuracy_score(y_true,y_pred)

prec = precision_score(y_true,y_pred)

rec = recall_score(y_true,y_pred)

f1 = f1_score(y_true,y_pred)

auc = roc_auc_score(y_true,y_prob)


TN,FP,FN,TP = cm.ravel()

APCER = FP/(FP+TN+1e-8)

BPCER = FN/(FN+TP+1e-8)

HTER = (APCER+BPCER)/2


# ---------------- PRINT RESULTS ----------------

print("\n===== Classification Results =====")

print("Confusion Matrix:\n",cm)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")


print("\n===== Face Anti-Spoofing Metrics =====")

print(f"APCER : {APCER*100:.2f}%")
print(f"BPCER : {BPCER*100:.2f}%")
print(f"HTER  : {HTER*100:.2f}%")