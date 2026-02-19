import requests
import pandas as pd

def fetch_realtime_solar_data(lat, lon):
    """
    Fetch real-time solar and weather data from Open-Meteo API
    Added: cloud_cover feature for better AI prediction
    Streamlit Cloud Safe Version ✅
    """

    # 🌥️ Open-Meteo API URL
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=shortwave_radiation,temperature_2m,cloudcover"
        "&forecast_days=1"
    )

    try:
        # ✅ SAFE REQUEST (Important for Streamlit Cloud)
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # ✅ Safety check (avoid crash if API fails)
        if "hourly" not in data:
            print("API Error: No hourly data found")
            return pd.DataFrame()

        # Create dataframe
        df = pd.DataFrame({
            "time": data["hourly"]["time"],
            "solar_irradiance": data["hourly"]["shortwave_radiation"],
            "temperature": data["hourly"]["temperature_2m"],
            "cloud_cover": data["hourly"]["cloudcover"]
        })

        return df

    except requests.exceptions.RequestException as e:
        print("API Request Failed:", e)
        return pd.DataFrame()
