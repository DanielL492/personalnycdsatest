import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAQ_API_KEY")

headers = {
    "X-API-Key": api_key
}

url = "https://api.openaq.org/v3/locations"

response = requests.get(url, headers=headers)

print(response.status_code)
print(response.json())

import pandas as pd

data = response.json()

df = pd.DataFrame(data["results"])

print(df.head())