import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def train_and_predict_lstm(ml_data, look_back=5):
    """
    Trains a simple LSTM model and predicts the next solar irradiance value.
    """

    values = ml_data["solar_irradiance"].values.reshape(-1, 1)

    # Scale data (very important for LSTM)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # Create sequences
    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i:i + look_back, 0])
        y.append(scaled[i + look_back, 0])

    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y)

    # Build LSTM model
    model = Sequential([
        LSTM(32, input_shape=(look_back, 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=8, verbose=0)

    # Predict next value
    last_sequence = scaled[-look_back:].reshape(1, look_back, 1)
    pred_scaled = model.predict(last_sequence, verbose=0)

    prediction = scaler.inverse_transform(pred_scaled)[0][0]

    return prediction
