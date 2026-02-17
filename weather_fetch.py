import requests
import pandas as pd

def fetch_realtime_solar_data(lat, lon):
    """
    Fetch real-time solar and weather data from Open-Meteo API
    Added: cloud_cover feature for better AI prediction
    """

    # 🌥️ Added cloudcover in hourly request
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=shortwave_radiation,temperature_2m,cloudcover"
        "&forecast_days=1"
    )

    response = requests.get(url)
    data = response.json()

    # Create dataframe
    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "solar_irradiance": data["hourly"]["shortwave_radiation"],
        "temperature": data["hourly"]["temperature_2m"],
        "cloud_cover": data["hourly"]["cloudcover"]   # ⭐ NEW FEATURE
    })

    return df
