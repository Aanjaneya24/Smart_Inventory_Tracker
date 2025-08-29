import joblib

# Load the saved model
model = joblib.load("smart_home_model.pkl")

# Print model details
print("Trained RandomForestClassifier Model:")
print(model)

# Print model parameters
print("\nModel Parameters:")
print(model.get_params())

# Print feature importances
print("\nFeature Importances:")
print(model.feature_importances_)