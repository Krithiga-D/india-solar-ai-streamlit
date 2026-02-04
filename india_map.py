import folium
import json
import random

def create_india_solar_map(geojson_path):
    """
    Creates a demo India district-level solar heatmap.
    """

    with open(geojson_path, "r", encoding="utf-8") as f:
        india_geo = json.load(f)

    # Add demo solar values to each district
    for feature in india_geo["features"]:
        feature["properties"]["solar"] = random.randint(300, 800)

    m = folium.Map(location=[22.5, 79.0], zoom_start=5, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=india_geo,
        data=india_geo["features"],
        columns=["properties.solar", "properties.solar"],
        key_on="feature.properties.solar",
        fill_color="YlOrRd",
        fill_opacity=0.8,
        line_opacity=0.2,
        legend_name="Solar Irradiance (W/m²)"
    ).add_to(m)

    return m
