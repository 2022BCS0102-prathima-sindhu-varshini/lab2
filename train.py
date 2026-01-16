import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "dataset/winequality-red.csv"
MODEL_PATH = "output/model/model.pkl"
RESULT_PATH = "output/results/metrics.json"

# -----------------------------
# Create output directories
# -----------------------------
os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH, sep=';')

# -----------------------------
# Feature selection
# -----------------------------
X = df.drop("quality", axis=1)
y = df["quality"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Pre-processing
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Model: Ridge Regression
# -----------------------------
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics (required)
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, MODEL_PATH)

# -----------------------------
# Save metrics
# -----------------------------
metrics = {
    "MSE": mse,
    "R2_score": r2
}

with open(RESULT_PATH, "w") as f:
    json.dump(metrics, f, indent=4)
