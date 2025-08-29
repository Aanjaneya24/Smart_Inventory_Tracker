import serial
import csv
from datetime import datetime

# === CONFIGURE THESE ===
SERIAL_PORT = '/dev/tty.usbserial-0001' # Change this to your actual ESP32 port
BAUD_RATE = 115200

# === CSV FILE SETUP ===
filename = f"inventory_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["date", "ultrasonic_distance_cm", "ir_sensor_triggered", "load_cell_weight_kg", "status"])

    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            print(line)

            if "Weight:" in line and "Distance:" in line:
                try:
                    weight_grams = float(line.split("Weight: ")[1].split(" g")[0])
                    distance_cm = float(line.split("Distance: ")[1].split(" cm")[0])
                    weight_kg = round(weight_grams / 1000.0, 2)
                except:
                    continue

            elif "IR Sensor:" in line:
                ir_state = line.split(": ")[1]
                ir_triggered = 1 if ir_state == "Detected" else 0

                # === Classify status ===
                if weight_kg < 1.0 or distance_cm > 40:
                    status = "Empty"
                elif weight_kg < 10 or distance_cm > 20:
                    status = "Low"
                else:
                    status = "Full"

                # Write row
                today = datetime.now().strftime("%Y-%m-%d")
                writer.writerow([today, distance_cm, ir_triggered, weight_kg, status])
                csvfile.flush()  # Ensure data is written in real time

    except KeyboardInterrupt:
        print("Logging stopped by user.")
        ser.close()

