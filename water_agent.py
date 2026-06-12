import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Constants
MODEL_PATH = "water_model.pkl"
DATA_PATH = "rainwater_harvesting_dataset.csv"

def train_water_model():
    """
    Trains a RandomForest model using the village rainwater harvesting dataset.
    Returns the trained model, R2 score, and MAE.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Error: {DATA_PATH} not found. Please ensure the dataset is in the project folder.")

    # 1. Load the dataset
    df = pd.read_csv(DATA_PATH)
    
    # 2. Define Features and Target (Matched to actual CSV columns)
    # Note: Using underscores as found in the verified CSV structure
    features = ['Roof_Area(sqm)', 'Rainfall_mm', 'Storage_Tank_Capacity_Liters']
    target = 'Water_Collected_Liters'
    
    X = df[features]
    y = df[target]
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate Model
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    # 6. Save Model
    joblib.dump(model, MODEL_PATH)
    
    return model, r2, mae

def predict_water_storage(population, roof_area, rainfall, tank_capacity):
    """
    Predicts water collection and calculates risk levels for the village.
    """
    # Load model if exists, else train
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model, r2, mae = train_water_model()
    
    # Prepare input for prediction
    # Ensure column names match what the model was trained on
    input_data = pd.DataFrame([[roof_area, rainfall, tank_capacity]], 
                              columns=['Roof_Area(sqm)', 'Rainfall_mm', 'Storage_Tank_Capacity_Liters'])
    
    # Predict collected water
    predicted_water_collected = float(model.predict(input_data)[0])
    
    # Risk Logic (1 person requires 5 liters)
    required_water = population * 5
    water_deficit = required_water - predicted_water_collected
    
    if predicted_water_collected >= required_water:
        water_risk = "Low Risk"
    elif predicted_water_collected >= (0.5 * required_water):
        water_risk = "Medium Risk"
    else:
        water_risk = "High Risk"
        
    return {
        "predicted_water_storage": round(predicted_water_collected, 2),
        "required_water": round(required_water, 2),
        "water_deficit": round(water_deficit, 2) if water_deficit > 0 else 0,
        "water_risk": water_risk
    }
