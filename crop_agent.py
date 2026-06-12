import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

MODEL_PATH = "crop_model.pkl"
DATA_PATH = "crop_dataset.csv"

def train_crop_model():
    """
    Load dataset, train model, and save it.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Please generate the dataset first.")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Features and target
    X = df[["Rainfall", "Temperature", "Soil_Moisture", "Fertilizer_Used"]]
    y = df["Crop_Yield"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate metrics
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Save model
    joblib.dump(model, MODEL_PATH)

    return model, r2, mae

def crop_agent(rainfall, temperature, soil_moisture, fertilizer_used):
    """
    Predict crop yield and evaluate risk.
    """
    # Load model if exists, else train
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model, _, _ = train_crop_model()

    input_data = pd.DataFrame(
        [[rainfall, temperature, soil_moisture, fertilizer_used]],
        columns=["Rainfall", "Temperature", "Soil_Moisture", "Fertilizer_Used"]
    )

    # Predict crop yield
    predicted_yield = float(model.predict(input_data)[0])
    
    # Assume expected yield
    expected_yield = 5000.0

    # Calculate yield percentage
    yield_percentage = (predicted_yield / expected_yield) * 100

    # Risk Logic
    if yield_percentage >= 80:
        crop_risk = "Low Risk"
    elif yield_percentage >= 50:
        crop_risk = "Medium Risk"
    else:
        crop_risk = "High Risk"

    return {
        "predicted_yield": round(predicted_yield, 2),
        "expected_yield": expected_yield,
        "yield_percentage": round(yield_percentage, 2),
        "crop_risk": crop_risk
    }