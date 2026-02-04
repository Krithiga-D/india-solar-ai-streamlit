import folium

def create_prediction_map(lat, lon, prediction):
    """
    Create an India-centered map with Transformer prediction marker
    """

    # Center map near India
    m = folium.Map(
        location=[22.5, 79.0],
        zoom_start=5,
        tiles="cartodbpositron"
    )

    popup_text = f"""
    <b>Solar Prediction</b><br>
    Latitude: {lat:.4f}<br>
    Longitude: {lon:.4f}<br>
    Transformer Prediction: {prediction:.2f} W/m²
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=10,
        popup=popup_text,
        color="red",
        fill=True,
        fill_color="orange",
        fill_opacity=0.8
    ).add_to(m)

    return m
