import numpy as np
import joblib
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load test data
X_test = np.load(r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\SplitData\X_test.npy")
y_test = np.load(r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\SplitData\y_test.npy")

# Load scaler fitted on TRAIN data
scaler = joblib.load(
    r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\Models\LD1scaler7.pkl"
)

X_test = scaler.transform(X_test)  # ✅ correct

# Load trained model
model = joblib.load(
    r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor1\Models\LD1model7.pkl"
)

# Predict
y_pred = model.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
# Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()