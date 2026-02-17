import pandas as pd

def prepare_data_for_ml(df):
    """
    This function prepares solar data for AI models.
    - Converts time to datetime
    - Adds hour feature (important for solar patterns)
    - Selects relevant features
    - Handles missing values
    """

    # Work on a copy (safe practice)
    df_clean = df.copy()

    # Convert time column to datetime
    df_clean["time"] = pd.to_datetime(df_clean["time"])

    # 🌞 Add hour feature (VERY IMPORTANT)
    df_clean["hour"] = df_clean["time"].dt.hour

    # Select features for ML
    features = df_clean[[
        "solar_irradiance",
        "temperature",
        "hour"        # ⭐ NEW FEATURE
    ]]

    # Handle missing values
    features = features.fillna(method="ffill")

    return features
