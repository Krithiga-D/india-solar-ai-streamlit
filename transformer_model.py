import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_transformer_model(input_shape):
    """
    Transformer encoder for solar time-series forecasting
    """

    inputs = layers.Input(shape=input_shape)

    # Projection layer
    x = layers.Dense(64)(inputs)

    # ===== Transformer Encoder Block =====
    attention = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=32
    )(x, x)

    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)

    x_ff = layers.Dense(128, activation="relu")(x)
    x_ff = layers.Dense(64)(x_ff)

    x = layers.Add()([x, x_ff])
    x = layers.LayerNormalization()(x)

    # Pooling + Output
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation="relu")(x)  # ⭐ prevents negative solar

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# ⭐ NEW FUNCTION — TRAIN + PREDICT
def train_and_predict_transformer(ml_data):
    """
    Train transformer quickly on current data and predict next value
    """

    data = ml_data.values

    # Create simple sequence
    X = np.expand_dims(data[:-1], axis=0)
    y = np.array([data[-1][0]])  # target = last solar value

    model = build_transformer_model(
        input_shape=(X.shape[1], X.shape[2])
    )

    # 🔥 Quick training (fast for Streamlit)
    model.fit(X, y, epochs=10, verbose=0)

    prediction = model.predict(X)[0][0]

    return prediction
