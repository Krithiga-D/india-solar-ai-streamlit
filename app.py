import pandas as pd
import numpy as np
import streamlit as st
import datetime

from shap_explainer import shap_explain_baseline
from geocode import get_lat_lon_from_place
from india_places import india_places
from transformer_model import train_and_predict_transformer   # ⭐ UPDATED
from map_view import create_prediction_map
from streamlit_folium import st_folium

from lstm_model import train_and_predict_lstm
from baseline_model import train_and_predict_baseline
from data_preprocess import prepare_data_for_ml
from weather_fetch import fetch_realtime_solar_data

# =====================================================
# 🔒 CACHED MODEL FUNCTIONS
# =====================================================

@st.cache_data(show_spinner=False)
def run_baseline_cached(ml_data):
    return train_and_predict_baseline(ml_data)

@st.cache_data(show_spinner=False)
def run_lstm_cached(ml_data):
    return train_and_predict_lstm(ml_data)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="India Solar AI", layout="centered")

# -----------------------------
# SESSION STATE
# -----------------------------
if "show_results" not in st.session_state:
    st.session_state.show_results = False

st.title("🌤️India Solar Energy Forecast Platform")
st.write("✅ Streamlit is working correctly")

# -----------------------------
# Project Purpose
# -----------------------------
st.subheader("Project Purpose")
st.write("""
This project forecasts solar energy availability using
real-time data and AI models to support better
energy planning, storage, and selling decisions.
""")

st.success("Base application loaded successfully 🎉")

# -----------------------------
# Location Selection
# -----------------------------
st.subheader("📍 Select Location")

state = st.selectbox("State", options=list(india_places.keys()))
district = st.selectbox("District", options=india_places[state])

lat, lon = get_lat_lon_from_place(district, state)

if lat is None:
    st.error("❌ Location could not be resolved.")
    st.stop()

st.success(f"📡 Location resolved: {lat}, {lon}")

# -----------------------------
# Fetch Real-Time Data
# -----------------------------
st.subheader("🌤️ Real-Time Solar Data")

df = fetch_realtime_solar_data(lat, lon)
st.dataframe(df.head())

# -----------------------------
# Visualization
# -----------------------------
st.subheader("📈 Solar Irradiance Trend")
st.line_chart(df.set_index("time")["solar_irradiance"])

# -----------------------------
# Rule-Based Prediction
# -----------------------------
avg_irradiance = df["solar_irradiance"].mean()
predicted_next_day = avg_irradiance * 1.05

st.subheader("🔮 Next-Day Solar Prediction")
st.metric("Estimated Next-Day Irradiance (W/m²)", f"{predicted_next_day:.2f}")

# -----------------------------
# Energy Recommendation
# -----------------------------
st.subheader("⚡ Energy Recommendation")

if predicted_next_day > 500:
    st.success("💰 SELL electricity to grid")
else:
    st.warning("🔋 STORE electricity")

# -----------------------------
# Prepare ML Data
# -----------------------------
st.subheader("🧠 AI-Ready Data")

ml_data = prepare_data_for_ml(df)
st.dataframe(ml_data.head())

# -----------------------------
# 🚀 PREDICT BUTTON
# -----------------------------
if st.button("🚀 Predict Solar Energy"):
    st.session_state.show_results = True

# -----------------------------
# RUN MODELS
# -----------------------------
if st.session_state.show_results:

    # -------- Baseline ML --------
    st.subheader("🤖 Baseline ML Prediction")
    with st.spinner("🔮 Running Baseline ML model..."):
        ml_prediction = run_baseline_cached(ml_data)

    # -------- LSTM --------
    st.subheader("🧠 LSTM Prediction")
    with st.spinner("🧠 Running LSTM model..."):
        lstm_prediction = run_lstm_cached(ml_data)

    # -------- Transformer --------
    st.subheader("🤖 Transformer Prediction")
    with st.spinner("🤖 Running Transformer model..."):
        transformer_pred = train_and_predict_transformer(ml_data)

    # =====================================================
    # 🚨 SOLAR CANNOT BE NEGATIVE
    # =====================================================
    ml_prediction = max(0, ml_prediction)
    lstm_prediction = max(0, lstm_prediction)
    transformer_pred = max(0, transformer_pred)

    st.metric("Baseline ML (W/m²)", f"{ml_prediction:.2f}")
    st.metric("LSTM (W/m²)", f"{lstm_prediction:.2f}")
    st.metric("Transformer (W/m²)", f"{transformer_pred:.2f}")

    # -------- Map --------
    st.subheader("🗺️ Transformer Prediction on India Map")
    prediction_map = create_prediction_map(
        lat=lat,
        lon=lon,
        prediction=transformer_pred
    )
    st_folium(prediction_map, width=800, height=500)

    # -------- Model Comparison --------
    st.subheader("📊 Model Comparison")
    st.table({
        "Model": ["Baseline ML", "LSTM", "Transformer"],
        "Prediction (W/m²)": [
            round(ml_prediction, 2),
            round(lstm_prediction, 2),
            round(transformer_pred, 2)
        ]
    })

    # =====================================================
    # 🧠 SMART MODEL SELECTION
    # =====================================================
    predictions = {
        "Baseline ML": ml_prediction,
        "LSTM": lstm_prediction,
        "Transformer": transformer_pred
    }

    valid_preds = {k: v for k, v in predictions.items() if v >= 0}
    best_model = max(valid_preds, key=valid_preds.get)
    final_pred = valid_preds[best_model]

    # 🌙 NIGHT RULE
    current_hour = datetime.datetime.now().hour
    if current_hour < 6 or current_hour > 18:
        final_pred = 0

    st.subheader("🏆 Best Model Selected")
    st.success(best_model)
    st.metric("Final Prediction (W/m²)", f"{final_pred:.2f}")

    # -------- Daytime Filter --------
    df["time"] = pd.to_datetime(df["time"])
    daytime_df = df[
        (df["time"].dt.hour >= 6) &
        (df["time"].dt.hour <= 18)
    ]

    st.subheader("☀️ Daytime Data Used")
    st.dataframe(daytime_df.head())

    # -------- SHAP --------
    st.subheader("🧩 SHAP Explainability (Baseline ML)")
    ml_day = prepare_data_for_ml(daytime_df)
    shap_values, feature_names = shap_explain_baseline(ml_day)

    shap_importance = np.abs(shap_values.values).mean(axis=0)
    st.table({
        "Feature": feature_names,
        "Importance": shap_importance
    })

    # -------- Reset Button --------
    if st.button("🔄 Reset Prediction"):
        st.session_state.show_results = False
