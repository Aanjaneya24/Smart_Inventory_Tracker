import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta, date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib # To save and load the model/scaler

# --- Configuration ---
NUM_INSTANCES = 1500
CONTAINER_HEIGHT_CM = 50.0
MAX_WEIGHT_KG = 20.0
IR_SENSOR_LEVEL_CM = CONTAINER_HEIGHT_CM * (3/4) # IR sensor triggers if item level > 1/4 height (distance < 3/4 height)
EMPTY_WEIGHT_THRESHOLD_KG = 0.5
LOW_WEIGHT_THRESHOLD_KG = MAX_WEIGHT_KG * 0.30 # e.g., below 30% weight is considered low
FULL_WEIGHT_THRESHOLD_KG = MAX_WEIGHT_KG * 0.80 # e.g., above 80% weight is considered full

# --- 1. Synthetic Data Generation ---
fake = Faker()

data = []
start_date = date(2023, 1, 1)

print("Generating synthetic data...")
for i in range(NUM_INSTANCES):
    # Determine status first to guide sensor readings
    status_choice = random.choices(['Full', 'Low', 'Empty'], weights=[0.4, 0.4, 0.2], k=1)[0] # Weighted distribution

    # Generate date
    # Generate date - ensure some temporal order for realism if needed later, but random for now
    # current_date = start_date + timedelta(days=random.randint(0, 365))
    current_date = fake.date_between(start_date=start_date, end_date=start_date + timedelta(days=400)) # Spread over ~1 year

    ultrasonic_distance_cm = 0.0
    ir_sensor_triggered = 0 # 0 = Below threshold, 1 = Above or at threshold
    load_cell_weight_kg = 0.0

    if status_choice == 'Full':
        # High item level -> Low distance
        ultrasonic_distance_cm = abs(random.gauss(5, 3)) # Close to the top, some variance
        ultrasonic_distance_cm = max(0, min(ultrasonic_distance_cm, CONTAINER_HEIGHT_CM * 0.3)) # Clamp values
        # Item level is above IR sensor
        ir_sensor_triggered = 1
        # High weight
        load_cell_weight_kg = random.uniform(FULL_WEIGHT_THRESHOLD_KG, MAX_WEIGHT_KG + 1.0) # Allow slightly over max
        load_cell_weight_kg = max(0, load_cell_weight_kg) # Ensure non-negative

    elif status_choice == 'Low':
        # Item level below IR sensor -> distance > IR level distance
        # Distance should be between IR level and max height
        ultrasonic_distance_cm = random.uniform(IR_SENSOR_LEVEL_CM + 1, CONTAINER_HEIGHT_CM * 0.9)
        ultrasonic_distance_cm = max(0, min(ultrasonic_distance_cm, CONTAINER_HEIGHT_CM)) # Clamp values
        # Item level is below IR sensor
        ir_sensor_triggered = 0
        # Medium weight
        load_cell_weight_kg = random.uniform(EMPTY_WEIGHT_THRESHOLD_KG + 0.1, LOW_WEIGHT_THRESHOLD_KG)
        load_cell_weight_kg = max(0, load_cell_weight_kg) # Ensure non-negative

    elif status_choice == 'Empty':
        # Item level at bottom -> Max distance (or near max)
        ultrasonic_distance_cm = abs(random.gauss(CONTAINER_HEIGHT_CM - 2 , 3)) # Near bottom, some variance
        ultrasonic_distance_cm = max(CONTAINER_HEIGHT_CM * 0.85, min(ultrasonic_distance_cm, CONTAINER_HEIGHT_CM + 2)) # Clamp near max
        # Item level is below IR sensor
        ir_sensor_triggered = 0
        # Negligible weight
        load_cell_weight_kg = abs(random.gauss(0.1, EMPTY_WEIGHT_THRESHOLD_KG * 0.5)) # Very low weight, some noise
        load_cell_weight_kg = max(0, min(load_cell_weight_kg, EMPTY_WEIGHT_THRESHOLD_KG)) # Clamp


    # Add slight inconsistencies/noise sometimes to make it harder
    if random.random() < 0.05: # 5% chance of slight sensor mismatch
        if status_choice == 'Full' and random.random() < 0.5:
             load_cell_weight_kg *= random.uniform(0.7, 0.9) # Make weight slightly lower than expected full
        elif status_choice == 'Low' and random.random() < 0.5:
             ultrasonic_distance_cm *= random.uniform(0.8, 1.2) # Make distance slightly off


    data.append({
        'date': current_date,
        'ultrasonic_distance_cm': round(ultrasonic_distance_cm, 2),
        'ir_sensor_triggered': ir_sensor_triggered, # Use 0/1 directly
        'load_cell_weight_kg': round(load_cell_weight_kg, 2),
        'status': status_choice
    })

df = pd.DataFrame(data)
# Sort by date potentially useful for simulation later, but not strictly needed for training
df = df.sort_values(by='date').reset_index(drop=True)

print(f"Generated {len(df)} instances.")
print(df.head())
print("\nData distribution:")
print(df['status'].value_counts())
print("\nData Description:")
print(df.describe())
print("\nNull values check:")
print(df.isnull().sum())

# Save generated data (optional)
df.to_csv("smart_container_data.csv", index=False)
print("\nSynthetic data saved to smart_container_data.csv")

# --- 2. Data Preprocessing ---
print("\nPreprocessing data...")

# Features (X) and Target (y)
# We drop 'date' for this classification model, as the status depends on current sensor readings,
# not the specific date itself in this context.
X = df[['ultrasonic_distance_cm', 'ir_sensor_triggered', 'load_cell_weight_kg']]
y = df['status']

# Encode the target variable ('Full', 'Low', 'Empty') into numerical labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Print mapping:
print("Label Encoding Mapping:")
for i, class_name in enumerate(le.classes_):
    print(f"{class_name} -> {i}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Create a pipeline including scaling and the classifier
# StandardScaler is important because features have different scales (cm, 0/1, kg)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')) # Added class_weight for imbalanced data if any
])


# --- 3. Model Training ---
print("\nTraining the model...")
pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- Model Evaluation ---
print("\nEvaluating the model...")
y_pred = pipeline.predict(X_test)

# Decode predictions back to original labels for reporting
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels, labels=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_))

# --- Save the pipeline (scaler + model) and label encoder ---
print("\nSaving model and label encoder...")
joblib.dump(pipeline, 'smart_container_model_pipeline.joblib')
joblib.dump(le, 'label_encoder.joblib')
print("Model and encoder saved.")


# --- 4. Real-time Classification & Alerting Simulation ---
print("\n--- Real-time Simulation ---")

# Load the saved pipeline and encoder (simulating a separate script/runtime)
loaded_pipeline = joblib.load('smart_container_model_pipeline.joblib')
loaded_le = joblib.load('label_encoder.joblib')

# Keep track of the previous status for alert generation
previous_status = None

def classify_and_alert(ultrasonic_reading, ir_reading, weight_reading):
    """
    Classifies container status based on sensor readings and generates alerts.
    Args:
        ultrasonic_reading (float): Distance reading from ultrasonic sensor (cm).
        ir_reading (int): IR sensor state (0 or 1).
        weight_reading (float): Weight reading from load cell (kg).

    Returns:
        tuple: (current_status_label, alert_message or None)
    """
    global previous_status # Use the global variable to track state

    # Create a DataFrame for the single input, matching training columns
    live_data = pd.DataFrame([[ultrasonic_reading, ir_reading, weight_reading]],
                             columns=['ultrasonic_distance_cm', 'ir_sensor_triggered', 'load_cell_weight_kg'])

    # Predict using the loaded pipeline (handles scaling and prediction)
    prediction_encoded = loaded_pipeline.predict(live_data)[0]

    # Decode the prediction back to the original label
    current_status_label = loaded_le.inverse_transform([prediction_encoded])[0]

    alert_message = None
    # Check for status changes and generate alerts
    if previous_status is not None:
        if previous_status == 'Full' and current_status_label == 'Low':
            alert_message = "ALERT: Container level is Low. Please schedule a refill."
        elif previous_status in ['Full', 'Low'] and current_status_label == 'Empty':
             alert_message = "CRITICAL ALERT: Container is Empty. Immediate refill required!"
        elif previous_status == 'Empty' and current_status_label == 'Low':
             alert_message = "INFO: Container refilled partially to Low status." # Optional info message
        elif previous_status in ['Empty', 'Low'] and current_status_label == 'Full':
             alert_message = "INFO: Container refilled to Full status." # Optional info message


    # Update the previous status for the next reading
    previous_status = current_status_label

    return current_status_label, alert_message

# --- Simulate receiving new sensor data ---
print("\nSimulating live sensor readings:")

# Example 1: Full Container
us_reading1, ir_reading1, wt_reading1 = 5.2, 1, 19.5
status1, alert1 = classify_and_alert(us_reading1, ir_reading1, wt_reading1)
print(f"Reading 1: US={us_reading1}cm, IR={ir_reading1}, WT={wt_reading1}kg -> Status: {status1}")
if alert1: print(f"   -> {alert1}")

# Example 2: Level drops to Low
us_reading2, ir_reading2, wt_reading2 = 40.1, 0, 4.8
status2, alert2 = classify_and_alert(us_reading2, ir_reading2, wt_reading2)
print(f"Reading 2: US={us_reading2}cm, IR={ir_reading2}, WT={wt_reading2}kg -> Status: {status2}")
if alert2: print(f"   -> {alert2}")

# Example 3: Stays Low
us_reading3, ir_reading3, wt_reading3 = 42.5, 0, 3.1
status3, alert3 = classify_and_alert(us_reading3, ir_reading3, wt_reading3)
print(f"Reading 3: US={us_reading3}cm, IR={ir_reading3}, WT={wt_reading3}kg -> Status: {status3}")
if alert3: print(f"   -> {alert3}")

# Example 4: Level drops to Empty
us_reading4, ir_reading4, wt_reading4 = 48.9, 0, 0.3
status4, alert4 = classify_and_alert(us_reading4, ir_reading4, wt_reading4)
print(f"Reading 4: US={us_reading4}cm, IR={ir_reading4}, WT={wt_reading4}kg -> Status: {status4}")
if alert4: print(f"   -> {alert4}")

# Example 5: Container Refilled to Full
us_reading5, ir_reading5, wt_reading5 = 3.5, 1, 20.1
status5, alert5 = classify_and_alert(us_reading5, ir_reading5, wt_reading5)
print(f"Reading 5: US={us_reading5}cm, IR={ir_reading5}, WT={wt_reading5}kg -> Status: {status5}")
if alert5: print(f"   -> {alert5}")

# Example 6: Rapid drop from Full to Empty (e.g., leak or bulk removal)
# Reset previous status to simulate this jump
previous_status = 'Full'
us_reading6, ir_reading6, wt_reading6 = 49.5, 0, 0.1
status6, alert6 = classify_and_alert(us_reading6, ir_reading6, wt_reading6)
print(f"\n-- Resetting Previous Status to Full for Test --")
print(f"Reading 6: US={us_reading6}cm, IR={ir_reading6}, WT={wt_reading6}kg -> Status: {status6}")
if alert6: print(f"   -> {alert6}")