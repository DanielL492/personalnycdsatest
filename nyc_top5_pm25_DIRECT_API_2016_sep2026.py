"""NYC top 5 worst monthly PM2.5 stations, fetched DIRECTLY from the OpenAQ v3 API.

Inputs (same directory as this script):
    nyc_only_locations.csv   -- station IDs and PM2.5 sensor IDs (NOT measurements)
    .env                     -- OPENAQ_API_KEY=your_api_key

Outputs (nyc_top5_pm25_direct_api/):
    nyc_all_stations_monthly_pm25.csv   -- one row per NYC location/month; N/A when missing
    nyc_top5_highest_pm25_by_month.csv -- month-by-month ranking (up to 5 stations)
    graphs/nyc_top5_pm25_2016.png ... nyc_top5_pm25_2026.png

A single valid DAILY PM2.5 value is enough to compute a monthly mean.
No minimum coverage threshold. The count of observed days is exported.
Run with: python -u nyc_top5_pm25_DIRECT_API_2016_sep2026.py
Install once: pip install requests pandas matplotlib python-dotenv
"""

print("Launching direct OpenAQ API Top 5 PM2.5 program...", flush=True)

import ast
import calendar
import csv
import json
import math
import os
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Prevent Tkinter popups; write PNG files directly.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
STATIONS_CSV = ROOT / "nyc_only_locations.csv"
OUT = ROOT / "nyc_top5_pm25_direct_api"
CACHE = OUT / "cached_openaq_daily"
GRAPHS = OUT / "graphs"
API_ROOT = "https://api.openaq.org/v3"
START = date(2016, 1, 1)
END = date(2026, 9, 30)
TOP_N = 5
PAGE_LIMIT = 1000


def say(message):
    print(message, flush=True)


def parse_field(value, default):
    if isinstance(value, type(default)):
        return value
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, type(default)) else default
    except (TypeError, ValueError, SyntaxError):
        return default


def parse_date(value):
    if isinstance(value, dict):
        value = value.get("local") or value.get("utc")
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass
    return None


def load_nyc_stations():
    if not STATIONS_CSV.exists():
        raise FileNotFoundError(f"Missing {STATIONS_CSV}. Place the NYC locations CSV next to this .py file.")
    stations = []
    found_ids = set()
    with STATIONS_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            loc_id = int(row["id"])
            if loc_id in found_ids:
                continue
            found_ids.add(loc_id)
            sensors = parse_field(row.get("sensors"), [])
            pm_sensor_ids = sorted({
                int(sensor["id"])
                for sensor in sensors
                if isinstance(sensor, dict)
                and sensor.get("id") is not None
                and isinstance(sensor.get("parameter"), dict)
                and sensor["parameter"].get("name", "").lower() == "pm25"
            })
            stations.append({
                "id": loc_id,
                "name": (row.get("name") or f"Station {loc_id}").strip(),
                "sensor_ids": pm_sensor_ids,
                "first": parse_date(parse_field(row.get("datetimeFirst"), {})),
                "last": parse_date(parse_field(row.get("datetimeLast"), {})),
            })
    return stations


def chunks_for(station):
    # Avoid downloading periods before the monitor started or after it stopped.
    first_year = max(START.year, (station["first"] or START).year)
    last_year = min(END.year, (station["last"] or END).year)
    for first in range(first_year, last_year + 1, 3):
        start = max(date(first, 1, 1), START)
        end = min(date(min(first + 2, last_year), 12, 31), END)
        yield start, end


def get_openaq_json(session, url, params):
    for attempt in range(1, 3):
        try:
            say(f"      Requesting OpenAQ (attempt {attempt}/2)...")
            resp = session.get(url, params=params, timeout=(8, 25))
            say(f"      OpenAQ HTTP {resp.status_code}")
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"OpenAQ authentication/permission failed ({resp.status_code}). "
                    "Check OPENAQ_API_KEY in .env. " + resp.text[:180]
                )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                raise ValueError(f"Unexpected OpenAQ API response: {str(data)[:200]}")
            return data
        except RuntimeError:
            raise
        except (requests.RequestException, ValueError) as err:
            say(f"      Request failed: {err}")
            if attempt == 1:
                time.sleep(2)
    return None


def download_sensor_chunk(session, sensor_id, start, end):
    CACHE.mkdir(parents=True, exist_ok=True)
    cached_file = CACHE / f"pm25_sensor_{sensor_id}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if cached_file.exists():
        try:
            cached = json.loads(cached_file.read_text(encoding="utf-8"))
            if cached.get("complete") is True and isinstance(cached.get("rows"), list):
                say(f"    Sensor {sensor_id} {start.year}-{end.year}: cached ({len(cached['rows'])} days)")
                return cached["rows"]
        except (ValueError, OSError, AttributeError):
            pass

    all_rows = []
    page = 1
    while True:
        say(f"    Sensor {sensor_id}: {start} to {end}; API page {page}")
        payload = get_openaq_json(
            session, f"{API_ROOT}/sensors/{sensor_id}/days",
            {
                # Add an extra day for UTC/local-date boundaries, then filter locally.
                "datetime_from": f"{(start - timedelta(days=1)).isoformat()}T00:00:00Z",
                "datetime_to": f"{(end + timedelta(days=2)).isoformat()}T00:00:00Z",
                "limit": PAGE_LIMIT,
                "page": page,
            },
        )
        if payload is None:
            say(f"    API failure for sensor {sensor_id}; this period remains incomplete")
            return None
        results = payload["results"]
        say(f"      Received {len(results)} API daily readings")
        for reading in results:
            if not isinstance(reading, dict):
                continue
            period = reading.get("period") or {}
            start_time = period.get("datetimeFrom") or reading.get("datetime")
            measurement_day = parse_date(start_time)
            try:
                value = float(reading.get("value"))
            except (ValueError, TypeError):
                continue
            if measurement_day and start <= measurement_day <= end and math.isfinite(value) and value >= 0:
                all_rows.append({"date": measurement_day.isoformat(), "value": value})
        if len(results) < PAGE_LIMIT:
            break
        page += 1
        if page > 20:
            say(f"    Too many pages for sensor {sensor_id}; skipping incomplete chunk")
            return None

    tmp = cached_file.with_suffix(".tmp")
    tmp.write_text(json.dumps({"complete": True, "rows": all_rows}), encoding="utf-8")
    tmp.replace(cached_file)
    say(f"    Finished sensor {sensor_id}: {len(all_rows)} valid daily readings")
    return all_rows


def station_monthly_means(station, session):
    readings_by_day = defaultdict(list)
    failed_months = set()

    for sensor_id in station["sensor_ids"]:
        for start, end in chunks_for(station):
            chunk_rows = download_sensor_chunk(session, sensor_id, start, end)
            if chunk_rows is None:
                month_cursor = date(start.year, start.month, 1)
                while month_cursor <= end:
                    failed_months.add((month_cursor.year, month_cursor.month))
                    y = month_cursor.year + (month_cursor.month == 12)
                    m = (month_cursor.month % 12) + 1
                    month_cursor = date(y, m, 1)
                continue
            for row in chunk_rows:
                day = parse_date(row["date"])
                if day:
                    readings_by_day[day].append(row["value"])

    # Average duplicate/simultaneous sensors for each day before averaging months.
    daily_mean_by_month = defaultdict(list)
    for day, daily_values in readings_by_day.items():
        daily_mean_by_month[(day.year, day.month)].append(sum(daily_values) / len(daily_values))
    month_averages = {
        month: (sum(values) / len(values), len(values))
        for month, values in daily_mean_by_month.items()
    }
    return month_averages, failed_months


def make_graphs(top5):
    GRAPHS.mkdir(exist_ok=True, parents=True)
    for year in range(START.year, END.year + 1):
        say(f"Generating top-5 graphs for {year}...")
        last_month = END.month if year == END.year else 12
        months = list(range(1, last_month + 1))
        x = np.arange(len(months))
        fig, ax = plt.subplots(figsize=(18, 8))
        width = 0.16
        for rank in range(1, TOP_N + 1):
            values = []
            for month in months:
                found = top5[(top5.year == year) & (top5.month == month) & (top5["rank"] == rank)]
                values.append(float(found.iloc[0].pm25_mean_ug_m3) if len(found) else float("nan"))
            positions = x + (rank - (TOP_N + 1) / 2) * width
            ax.bar(positions, values, width, label=f"Rank {rank}")

        ax.set_xticks(x, [calendar.month_abbr[m] for m in months])
        ax.set_ylabel("Monthly mean PM2.5 (µg/m³)")
        ax.set_xlabel("Month")
        ax.set_title(f"5 Highest Monthly PM2.5 Monitoring Stations in NYC — {year}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Within-month rank")
        fig.text(.5, .02,
                 "Top 5 are ranked independently for each month; see ranking CSV for station names and days measured. "
                 "Even one measurement day qualifies.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        filepath = GRAPHS / f"nyc_top5_pm25_{year}.png"
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        say(f"  Saved {filepath.name}")

        # A second view displays the actual station names for each month.
        # This is more readable than 60 vertical name annotations.
        fig, axes = plt.subplots(4, 3, figsize=(20, 17))
        for month in range(1, 13):
            ax2 = axes.flat[month - 1]
            if month > last_month:
                ax2.axis("off")
                continue
            monthly = top5[(top5.year == year) & (top5.month == month)].sort_values("rank")
            ax2.set_title(calendar.month_name[month], fontsize=12)
            if monthly.empty:
                ax2.text(.5, .5, "No available PM2.5 data", ha="center", va="center",
                         transform=ax2.transAxes)
                ax2.set_yticks([])
                continue
            vals = monthly["pm25_mean_ug_m3"].astype(float).tolist()
            labels = [f"#{int(r['rank'])} {str(r['station'])[:28]}"
                      for _, r in monthly.iterrows()]
            y = np.arange(len(vals))
            ax2.barh(y, vals)
            ax2.set_yticks(y, labels, fontsize=8)
            ax2.invert_yaxis()
            ax2.set_xlabel("PM2.5 (µg/m³)", fontsize=9)
            ax2.grid(axis="x", alpha=.2)
            for j, (_, r) in enumerate(monthly.iterrows()):
                ax2.text(float(r["pm25_mean_ug_m3"]), j,
                         f" {float(r['pm25_mean_ug_m3']):.1f} ({int(r['pm25_days'])} days)",
                         va="center", fontsize=8)
            ax2.set_xlim(0, max(vals) * 1.5 if max(vals) > 0 else 1)
        fig.suptitle(f"NYC's 5 Highest Recorded Monthly Mean PM2.5 Stations — {year}", fontsize=17)
        fig.text(.5, .012, "Monthly rankings among stations with downloaded data; no minimum coverage. "
                 "Sensor availability varies by station and year.", ha="center", fontsize=10)
        fig.tight_layout(rect=(.03, .03, 1, .965))
        named_path = GRAPHS / f"nyc_top5_pm25_with_station_names_{year}.png"
        fig.savefig(named_path, dpi=135)
        plt.close(fig)
        say(f"  Saved {named_path.name}")


def main():
    say("START: Fetching NYC station-level PM2.5 directly from OpenAQ API")
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAQ_API_KEY not found. Create a .env file containing OPENAQ_API_KEY=your_key")

    stations = load_nyc_stations()
    say(f"Loaded {len(stations)} NYC stations; "
        f"{sum(bool(s['sensor_ids']) for s in stations)} list at least one PM2.5 sensor")
    say("Using OpenAQ API sensor /days endpoint for all actual measurements (not the previous CSV).")
    OUT.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"X-API-Key": api_key, "Accept": "application/json"})

    # All 54 stations and all 129 months, including N/A where there is no PM2.5.
    all_rows = []
    for index, station in enumerate(stations, 1):
        say(f"\n[{index}/{len(stations)}] {station['name']} (ID {station['id']}), "
            f"PM2.5 sensors: {station['sensor_ids'] or 'none'}")
        if station["sensor_ids"]:
            means, failures = station_monthly_means(station, session)
        else:
            means, failures = {}, set()
        for year in range(START.year, END.year + 1):
            last_month = END.month if year == END.year else 12
            for month in range(1, last_month + 1):
                result = means.get((year, month))
                failure = (year, month) in failures
                all_rows.append({
                    "year": year,
                    "month": month,
                    "month_name": calendar.month_abbr[month],
                    "location_id": station["id"],
                    "station": station["name"],
                    "pm25_mean_ug_m3": round(result[0], 4) if result and not failure else "N/A",
                    "pm25_days": result[1] if result and not failure else 0,
                    "data_status": (
                        "NO_PM25_SENSOR" if not station["sensor_ids"]
                        else "API_DOWNLOAD_FAILED" if failure
                        else "MEASURED" if result else "NO_READINGS"
                    ),
                })
        say(f"  Finished {station['name']}; saving progress...")
        pd.DataFrame(all_rows).to_csv(OUT / "nyc_all_stations_monthly_pm25.csv", index=False, na_rep="N/A")

    data = pd.DataFrame(all_rows)
    data["pm25_numeric"] = pd.to_numeric(data["pm25_mean_ug_m3"], errors="coerce")
    valid = data.dropna(subset=["pm25_numeric"])
    top5 = (
        valid.sort_values(["year", "month", "pm25_numeric", "location_id"],
                          ascending=[True, True, False, True])
             .groupby(["year", "month"], sort=True)
             .head(TOP_N)
             .copy()
    )
    top5["rank"] = top5.groupby(["year", "month"]).cumcount() + 1
    top5 = top5[["year", "month", "month_name", "rank", "location_id", "station",
                 "pm25_mean_ug_m3", "pm25_days"]]
    top5.to_csv(OUT / "nyc_top5_highest_pm25_by_month.csv", index=False, na_rep="N/A")
    say("Saved both CSV files")
    make_graphs(top5)
    failed_count = (data["data_status"] == "API_DOWNLOAD_FAILED").sum()
    if failed_count:
        say(f"WARNING: {failed_count} station-month rows could not be downloaded. "
            "Rankings reflect only successfully retrieved measurements; rerun to retry missing chunks.")
    say(f"DONE. All files are in: {OUT}")


if __name__ == "__main__":
    main()
