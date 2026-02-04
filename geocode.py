import requests

def get_lat_lon_from_place(district, state):
    """
    Robust geocoding:
    1) Try district + India
    2) Fallback to district + state + India
    """

    base_url = "https://geocoding-api.open-meteo.com/v1/search"

    # -------- Try 1: District + India --------
    query1 = f"{district}, India"
    params1 = {
        "name": query1,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    r1 = requests.get(base_url, params=params1)
    d1 = r1.json()

    if "results" in d1 and len(d1["results"]) > 0:
        return d1["results"][0]["latitude"], d1["results"][0]["longitude"]

    # -------- Try 2: District + State + India --------
    query2 = f"{district}, {state}, India"
    params2 = {
        "name": query2,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    r2 = requests.get(base_url, params=params2)
    d2 = r2.json()

    if "results" in d2 and len(d2["results"]) > 0:
        return d2["results"][0]["latitude"], d2["results"][0]["longitude"]

    return None, None
