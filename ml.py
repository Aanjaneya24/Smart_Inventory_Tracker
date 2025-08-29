import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load Data
data = pd.read_csv("sensor_data.csv")  # Ensure this file has enough rows

# Prepare Features and Labels
X = data[['weight', 'light_intensity']]
y = data['order']

# Split Data (with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "smart_home_model.pkl")

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-Validation
scores = cross_val_score(model, X, y, cv=5)
print("\nCross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

print("Model trained and saved successfully!")