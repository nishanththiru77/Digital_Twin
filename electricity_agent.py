import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

MODEL_PATH = "electricity_model.pkl"
DATA_PATH = "electricity_dataset.csv"

def train_electricity_model():
    """
    Load dataset, train model, and save it.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Please generate the dataset first.")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Features and target
    X = df[["Temperature", "Humidity", "Population", "Pump_Hours", "Month"]]
    y = df["Electricity_Consumption"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate metrics
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Save model
    joblib.dump(model, MODEL_PATH)

    return model, r2, mae

def electricity_agent(temperature, humidity, population, pump_usage, month):
    """
    Predict electricity consumption and evaluate risk.
    """
    # Convert categorical pump_usage to numeric pump_hours
    if pump_usage == "Low":
        pump_hours = 4
    elif pump_usage == "Medium":
        pump_hours = 8
    else:
        pump_hours = 12

    # Load model if exists, else train
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model, _, _ = train_electricity_model()

    input_data = pd.DataFrame(
        [[temperature, humidity, population, pump_hours, month]],
        columns=["Temperature", "Humidity", "Population", "Pump_Hours", "Month"]
    )

    # Predict electricity consumption
    predicted_consumption = float(model.predict(input_data)[0])
    
    # Assume transformer capacity
    transformer_capacity = 5000.0

    # Calculate utilization
    utilization_percent = (predicted_consumption / transformer_capacity) * 100

    # Risk Logic
    if utilization_percent >= 90:
        electricity_risk = "High Risk"
    elif utilization_percent >= 70:
        electricity_risk = "Medium Risk"
    else:
        electricity_risk = "Low Risk"

    return {
        "predicted_consumption": round(predicted_consumption, 2),
        "transformer_capacity": transformer_capacity,
        "utilization_percent": round(utilization_percent, 2),
        "electricity_risk": electricity_risk
    }