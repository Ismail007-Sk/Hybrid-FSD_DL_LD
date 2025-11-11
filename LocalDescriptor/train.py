import os
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime

# --- Dataset directories ---
BASE_DIR = Path(r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor\Dataset")
REAL_DIR = BASE_DIR / "Real"
FAKE_DIR = BASE_DIR / "Fake"

# --- Load all .npy feature files ---
real_files = list(REAL_DIR.glob("*.npy"))
fake_files = list(FAKE_DIR.glob("*.npy"))

X = []
y = []

for f in real_files:
    X.append(np.load(f))
    y.append(1)  # Real = 1

for f in fake_files:
    X.append(np.load(f))
    y.append(0)  # Fake = 0

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples ({len(real_files)} real, {len(fake_files)} fake)")

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# SplitData folder
SAVE_DIR = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor\SplitData"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save split data
np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)
print(f"Data saved successfully in {SAVE_DIR}")





# --- Train SVM model ---
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# --- Evaluate model ---
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- Save model ---
MODEL_DIR = Path(r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor\Models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = MODEL_DIR / f"LD_fused_SVM_{timestamp}.pkl"
joblib.dump(clf, model_path)

print(f"Model saved to {model_path}")
