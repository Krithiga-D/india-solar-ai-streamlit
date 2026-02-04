from sklearn.linear_model import LinearRegression
import numpy as np

def train_and_predict_baseline(ml_data):
    """
    Trains a simple baseline ML model (Linear Regression)
    and predicts the next solar irradiance value.
    """

    # Use index as time step
    X = np.arange(len(ml_data)).reshape(-1, 1)
    y = ml_data["solar_irradiance"].values

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predict next time step
    next_step = np.array([[len(ml_data)]])
    prediction = model.predict(next_step)[0]

    return prediction
