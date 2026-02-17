from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_and_predict_baseline(ml_data):
    """
    Flexible Baseline ML Model
    Works even if cloud_cover column is missing.
    """

    # ⭐ Select only available columns safely
    feature_cols = ["solar_irradiance", "temperature", "hour"]

    if "cloud_cover" in ml_data.columns:
        feature_cols.append("cloud_cover")

    X = ml_data[feature_cols].values
    y = ml_data["solar_irradiance"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Predict next step
    last_row = X[-1].reshape(1, -1)
    last_row_scaled = scaler.transform(last_row)

    prediction = model.predict(last_row_scaled)[0]

    return prediction
