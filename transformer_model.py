import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_transformer_model(input_shape):
    """
    Simple Transformer encoder for time-series forecasting
    """

    inputs = layers.Input(shape=input_shape)

    # Positional encoding (simple)
    x = layers.Dense(64)(inputs)

    # Transformer Encoder Block
    attention = layers.MultiHeadAttention(
        num_heads=4, key_dim=64
    )(x, x)

    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)

    x_ff = layers.Dense(128, activation="relu")(x)
    x_ff = layers.Dense(64)(x_ff)

    x = layers.Add()([x, x_ff])
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model
