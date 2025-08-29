import joblib
import pandas as pd

# Load the saved model
model = joblib.load("smart_home_model.pkl")

# Test with new data
test_data = pd.DataFrame([[1.5, 400]], columns=["weight", "light_intensity"])  # Match feature names
prediction = model.predict(test_data)

print("Prediction for test data:", prediction)