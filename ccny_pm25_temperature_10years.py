import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")
if not API_KEY:
    raise ValueError("Set OPENAQ_API_KEY in your .env file")

# CCNY station in Manhattan (OpenAQ location 384; PM2.5 sensor 673)
SENSOR_ID = 673
LAT, LON = 40.8197, -73.9481
START, END = "2016-04-01", "2026-03-31"  # 10 complete years
session = requests.Session()
session.headers.update({"X-API-Key": API_KEY})

# 1. Download daily PM2.5 readings from OpenAQ, one calendar year at a time.
records = []
for year in range(2016, 2027):
    year_start = max(START, f"{year}-01-01")
    year_end = min(END, f"{year}-12-31")
    if year_start > year_end:
        continue
    page = 1
    while True:
        url = f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/days"
        params = {
            "datetime_from": year_start,
            "datetime_to": year_end,
            "limit": 1000,
            "page": page,
        }
        response = session.get(url, params=params, timeout=40)
        response.raise_for_status()
        results = response.json().get("results", [])
        for item in results:
            period = item.get("period") or {}
            beginning = period.get("datetimeFrom") or {}
            timestamp = beginning.get("utc")
            if timestamp is not None and item.get("value") is not None:
                records.append({"date": timestamp, "pm25": item["value"]})
        if len(results) < 1000:
            break
        page += 1

if not records:
    raise RuntimeError("OpenAQ returned no PM2.5 daily data for this station and period")

pm = pd.DataFrame(records)
pm["date"] = pd.to_datetime(pm["date"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
pm = pm.drop_duplicates("date").set_index("date").sort_index()
pm = pm.loc[START:END]

# 2. Download gridded daily mean air temperature for the same coordinates.
weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START,
    "end_date": END,
    "daily": "temperature_2m_mean",
    "timezone": "America/New_York",
}
weather_response = requests.get(weather_url, params=weather_params, timeout=60)
weather_response.raise_for_status()
weather = weather_response.json()["daily"]
temp = pd.DataFrame({
    "date": pd.to_datetime(weather["time"]),
    "temperature_c": weather["temperature_2m_mean"],
}).set_index("date")

# 3. Monthly means are easier to read over a decade.
monthly_pm = pm["pm25"].resample("MS").mean()
monthly_temp = temp["temperature_c"].resample("MS").mean()
monthly = pd.concat([monthly_pm, monthly_temp], axis=1)
monthly.to_csv("ccny_pm25_temperature_10years.csv", index_label="month")

# 4. Plot both variables against time using separate y-axes.
fig, ax1 = plt.subplots(figsize=(13, 6))
line1, = ax1.plot(monthly.index, monthly["pm25"], label="PM2.5", linewidth=1.7)
ax1.set_ylabel("PM2.5 (µg/m³)")
ax1.set_xlabel("Date")
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
line2, = ax2.plot(monthly.index, monthly["temperature_c"], label="Temperature", linewidth=1.7, color="darkorange")
ax2.set_ylabel("Temperature (°C)")

ax1.legend([line1, line2], ["PM2.5", "Temperature"], loc="upper right")
ax1.set_title("CCNY, Manhattan: PM2.5 and Temperature (April 2016–March 2026)\nMonthly averages; temperature from Open-Meteo (modeled)")
fig.tight_layout()
fig.savefig("ccny_pm25_temperature_10years.png", dpi=200)
plt.show()
print("Saved CSV and graph. PM2.5 daily records:", len(pm))
