import requests
import pandas as pd

def fetch_realtime_solar_data(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=shortwave_radiation,temperature_2m"
        "&forecast_days=1"
    )

    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "solar_irradiance": data["hourly"]["shortwave_radiation"],
        "temperature": data["hourly"]["temperature_2m"]
    })

    return df
