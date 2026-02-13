# ==========================================================
# Hybrid IDS Simulation (CSV-Based)
# Part 2: Inference + Hybrid Logic
# ==========================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================================
# 1. Load Saved Model
# ==========================================================

print("Loading trained model...")

model_data = joblib.load("/home/kali/Documents/Intrusion-Detection-System-IDS-using-Neural-Networks/Hybrid-ids/Model/random_forest_model.pkl")

model = model_data["model"]
scaler = model_data["scaler"]
encoders = model_data["encoders"]

print("Model loaded successfully!\n")

# ==========================================================
# 2. Load Test Dataset
# ==========================================================

test_data_path = "/home/kali/Downloads/test.csv"
data = pd.read_csv(test_data_path)

print("Test Dataset Loaded.")
print("Shape:", data.shape)

# Rename class column
if "class" in data.columns:
    data.rename(columns={"class": "label"}, inplace=True)

# ==========================================================
# 3. Encode Categorical Columns (Same as Training)
# ==========================================================

categorical_cols = ["protocol_type", "service", "flag"]

for col in categorical_cols:
    if col in data.columns:
        le = encoders[col]
        data[col] = le.transform(data[col])

# Convert label to binary
data["label"] = data["label"].map({"normal": 0, "anomaly": 1})

# ==========================================================
# 4. Separate Features & Labels
# ==========================================================

X = data.drop("label", axis=1)
y_true = data["label"]

# Scale features using trained scaler
X_scaled = scaler.transform(X)

# ==========================================================
# 5. ML Prediction (Row-by-Row)
# ==========================================================

ml_predictions = []

print("\nStarting Row-by-Row Prediction...\n")

for i in range(len(X_scaled)):
    sample = X_scaled[i].reshape(1, -1)
    pred = model.predict(sample)

    # Convert Isolation Forest output
    if pred[0] == -1:
        ml_predictions.append(1)  # anomaly
    else:
        ml_predictions.append(0)  # normal

ml_predictions = np.array(ml_predictions)

# ==========================================================
# 6. Simulated Snort Detection (Signature-Based)
# ==========================================================

def simulate_snort(row):
    """
    Simple signature logic simulation
    Example rules:
    - High src_bytes + high dst_bytes
    - High serror_rate
    """

    if row["serror_rate"] > 0.5:
        return 1
    if row["src_bytes"] > 10000:
        return 1
    return 0


snort_predictions = data.apply(simulate_snort, axis=1).values

# ==========================================================
# 7. Hybrid OR Logic
# ==========================================================

hybrid_predictions = []

for ml, snort in zip(ml_predictions, snort_predictions):
    if ml == 1 or snort == 1:
        hybrid_predictions.append(1)
    else:
        hybrid_predictions.append(0)

hybrid_predictions = np.array(hybrid_predictions)

# ==========================================================
# 8. Evaluation
# ==========================================================

print("========== ML MODEL RESULTS ==========")
print(classification_report(y_true, ml_predictions))

print("\n========== SNORT SIMULATION RESULTS ==========")
print(classification_report(y_true, snort_predictions))

print("\n========== HYBRID MODEL RESULTS ==========")
print(classification_report(y_true, hybrid_predictions))

# ==========================================================
# 9. Confusion Matrix Visualization
# ==========================================================

cm = confusion_matrix(y_true, hybrid_predictions)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Hybrid IDS Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nHybrid IDS Simulation Completed Successfully!")
