import pandas as pd

def prepare_data_for_ml(df):
    """
    This function prepares solar data for AI models.
    - Converts time to datetime
    - Selects relevant features
    - Handles missing values
    """

    df_clean = df.copy()

    # Convert time column to datetime
    df_clean["time"] = pd.to_datetime(df_clean["time"])

    # Select features for ML
    features = df_clean[[
        "solar_irradiance",
        "temperature"
    ]]

    # Handle missing values (simple strategy)
    features = features.fillna(method="ffill")

    return features
