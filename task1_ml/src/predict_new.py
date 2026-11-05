import joblib
import pandas as pd

preprocessor = joblib.load(
    "task1_ml/models/preprocessor.joblib"
)

model = joblib.load(
    "task1_ml/models/tuned_random_forest.joblib"
)

print("Enter Machine Details")

machine_type = input("Machine Type (L/M/H): ").strip().upper()

if machine_type not in ["L", "M", "H"]:
    print("Invalid machine type. Enter L, M, or H.")
    exit()

air_temperature = float(input("Air Temperature (K): "))
process_temperature = float(input("Process Temperature (K): "))
rotational_speed = float(input("Rotational Speed (rpm): "))
torque = float(input("Torque (Nm): "))
tool_wear = float(input("Tool Wear (min): "))

new_machine = pd.DataFrame([{
    "Type": machine_type,
    "Air temperature [K]": air_temperature,
    "Process temperature [K]": process_temperature,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}])

processed_data = preprocessor.transform(new_machine)

prediction = model.predict(processed_data)[0]

probability = model.predict_proba(processed_data)[0][1]

print("\n========== PREDICTION RESULT ==========")

if prediction == 1:
    print("Prediction: MACHINE FAILURE")
else:
    print("Prediction: NO MACHINE FAILURE")

print(f"Failure Probability: {probability * 100:.2f}%")