import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAQ_API_KEY")

headers = {
    "X-API-Key": api_key
}

url = "https://api.openaq.org/v3/locations"

params = {
    "bbox": "-74.26,40.49,-73.70,40.92",
    "limit": 100
}

response = requests.get(url, headers=headers, params=params)
response.raise_for_status()

data = response.json()

nyc_locations = []

# New Jersey locations to exclude
nj_ids = [
    853, 857, 928, 971, 974, 984, 1122,
    1496, 2193, 387315, 391818, 2153518,
    2386747, 3103622, 3305066, 4419380,
    5066205, 5238263, 6268383, 6434257
]

for location in data["results"]:
    coordinates = location.get("coordinates")

    if coordinates is None:
        continue

    lat = coordinates["latitude"]
    lon = coordinates["longitude"]

    # Approximate NYC boundaries
    if 40.49 <= lat <= 40.92 and -74.26 <= lon <= -73.70:

        # Exclude New Jersey locations
        if location["id"] not in nj_ids:
            nyc_locations.append(location)

df = pd.DataFrame(nyc_locations)

# Save NYC locations to CSV
df.to_csv("nyc_only_locations.csv", index=False)

print(df)
print("Number of NYC locations:", len(df))