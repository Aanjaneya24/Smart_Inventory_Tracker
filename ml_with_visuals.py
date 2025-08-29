import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
data = pd.read_csv("sensor_data.csv")  # Ensure this file has enough rows

# Visualize Dataset Distribution
sns.scatterplot(data=data, x="weight", y="light_intensity", hue="order", palette="coolwarm", s=100)
plt.title("Dataset Distribution")
plt.xlabel("Weight")
plt.ylabel("Light Intensity")
plt.legend(title="Order")
plt.show()

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

# Visualize Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Cross-Validation
scores = cross_val_score(model, X, y, cv=5)
print("\nCross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

# Visualize Cross-Validation Scores
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='--', color='b')
plt.title("Cross-Validation Scores")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# Visualize Feature Importances
feature_names = ["weight", "light_intensity"]
importances = model.feature_importances_
plt.bar(feature_names, importances, color='skyblue')
plt.title("Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# Test the Model with New Data
test_data = pd.DataFrame([[1.5, 400], [2.5, 600], [3.0, 700]], columns=["weight", "light_intensity"])
predictions = model.predict(test_data)

# Visualize Test Data Predictions
plt.scatter(test_data["weight"], test_data["light_intensity"], c=predictions, cmap='coolwarm', s=100)
plt.title("Test Data Predictions")
plt.xlabel("Weight")
plt.ylabel("Light Intensity")
plt.colorbar(label="Predicted Class")
plt.show()

print("Prediction for test data:", predictions)