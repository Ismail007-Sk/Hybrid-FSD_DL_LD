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
X_test = np.load(r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor\SplitData\X_test.npy")
y_test = np.load(r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor\SplitData\y_test.npy")

# Load trained model
model = joblib.load(r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\LocalDescriptor\Models\LDmodelBEST.pkl")

# Predict
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")

# Optional detailed report
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