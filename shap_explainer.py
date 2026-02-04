import shap
import numpy as np
from sklearn.linear_model import LinearRegression

def shap_explain_baseline(ml_data):
    """
    Explains baseline ML model predictions using SHAP.
    """

    # Prepare data
    X = ml_data.values
    y = ml_data["solar_irradiance"].values

    model = LinearRegression()
    model.fit(X, y)

    # SHAP explainer
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    return shap_values, ml_data.columns
