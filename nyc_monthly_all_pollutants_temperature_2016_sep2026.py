"""Create TWO monthly CSV files for every NYC monitoring location (2016–Sep 2026).

  1. nyc_monthly_air_pollutants_2016_sep2026.csv
     Every air-pollution parameter present in nyc_only_locations.csv,
     including PM2.5. One valid daily average is sufficient for a month.

  2. nyc_monthly_temperature_2016_sep2026.csv
     Monthly mean of modeled outdoor daily temperature from Open-Meteo.

Missing values are exactly 'N/A', never zero. All 54 input stations and all
129 months are represented, even if a station was not yet operating.
Modeled outdoor temperature is calculated for all years at each coordinate,
including dates before the pollution monitor was installed.

Requirements:
    pip install requests python-dotenv

Put this file, nyc_only_locations.csv, and .env in the SAME folder.
.env must contain OPENAQ_API_KEY=your_key_here

Run:
    python -u nyc_monthly_all_pollutants_temperature_2016_sep2026.py

Optional single-location test (puts output in its own folder):
    python -u nyc_monthly_all_pollutants_temperature_2016_sep2026.py --only-location 384

Successful API replies are cached in chunks, so stopping and restarting does
not repeat finished API downloads. CSVs are updated after each station.
"""

import argparse
import ast
import calendar
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
STATIONS_FILE = ROOT / "nyc_only_locations.csv"
START = date(2016, 1, 1)
END = date(2026, 9, 30)  # Never fetch October–December 2026.
API_URL = "https://api.openaq.org/v3"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
PAGE_SIZE = 1000
MISSING = "N/A"
# These are weather/environmental context, NOT air pollutants.
NON_POLLUTANTS = {"temperature", "relativehumidity", "humidity", "pressure", "windspeed", "winddirection"}
MONTH_NAMES = [calendar.month_abbr[i] for i in range(1, 13)]


def log(message):
    print(message, flush=True)


def parse_object(raw, expected):
    if isinstance(raw, expected):
        return raw
    if not raw or not isinstance(raw, str):
        return expected()
    try:
        data = ast.literal_eval(raw)
        return data if isinstance(data, expected) else expected()
    except (SyntaxError, ValueError, TypeError):
        try:
            data = json.loads(raw)
            return data if isinstance(data, expected) else expected()
        except (ValueError, TypeError):
            return expected()


def iso_date(value):
    if isinstance(value, dict):
        value = value.get("local") or value.get("utc")
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def numeric(value, *, allow_negative=False):
    try:
        number = float(value)
        if not (-1e20 < number < 1e20):
            return None
        if not allow_negative and number < 0:
            return None
        return number
    except (TypeError, ValueError, OverflowError):
        return None


def safe_identifier(s):
    s = str(s).replace("µ", "u").replace("μ", "u").replace("³", "3")
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def param_column(parameter):
    p = (parameter or {}).get("name", "").strip().lower()
    unit = (parameter or {}).get("units", "").strip()
    if not p or p in NON_POLLUTANTS:
        return None
    return f"{safe_identifier(p)}_{safe_identifier(unit)}" if unit else safe_identifier(p)


def read_stations():
    if not STATIONS_FILE.exists():
        raise FileNotFoundError(f"Missing station file: {STATIONS_FILE}")
    stations, pollutant_columns = [], set()
    ids = set()
    with STATIONS_FILE.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            station_id = int(row["id"])
            if station_id in ids:
                continue
            ids.add(station_id)
            coords = parse_object(row.get("coordinates"), dict)
            lat = numeric(coords.get("latitude"), allow_negative=True)
            lon = numeric(coords.get("longitude"), allow_negative=True)
            sensors = parse_object(row.get("sensors"), list)
            grouped = defaultdict(set)
            for sensor in sensors:
                if not isinstance(sensor, dict) or sensor.get("id") is None:
                    continue
                col = param_column(sensor.get("parameter"))
                if col:
                    grouped[col].add(int(sensor["id"]))
                    pollutant_columns.add(col)
            started = iso_date(parse_object(row.get("datetimeFirst"), dict))
            last = iso_date(parse_object(row.get("datetimeLast"), dict))
            stations.append({
                "id": station_id, "name": str(row.get("name") or "Unknown").strip(),
                "lat": lat, "lon": lon, "sensors": grouped,
                "start": started, "last": last,
            })
    # PM2.5 stays near the beginning; other pollutant categories are alphabetical.
    pollutant_columns = sorted(pollutant_columns, key=lambda c: (not c.startswith("pm25_"), c))
    return stations, pollutant_columns


def chunks_by_3_years(first_year, last_year):
    """Consecutive, nonoverlapping calendar blocks of <=3 years."""
    y = first_year
    while y <= last_year:
        end_y = min(y + 2, last_year)
        yield date(y, 1, 1), min(date(end_y, 12, 31), END)
        y = end_y + 1


def get_json(session, url, *, params, label):
    """Bounded network request; NO invisible automatic retries."""
    for attempt in (1, 2):
        response = None
        try:
            response = session.get(url, params=params, timeout=(8, 30))
            log(f"      {label}: HTTP {response.status_code} (attempt {attempt}/2)")
            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"API rejected access (HTTP {response.status_code}). "
                    f"Check your OpenAQ key and account. Response: {response.text[:200]}"
                )
            response.raise_for_status()
            return response.json()
        except RuntimeError:
            raise
        except (requests.RequestException, ValueError) as exc:
            log(f"      {label} error: {exc}")
            if response is not None:
                log(f"      Response body (first 180 chars): {response.text[:180]!r}")
            if attempt == 1:
                time.sleep(1)
    return None


def read_cache(path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("complete") is True and isinstance(cached.get("days"), list):
            return cached["days"]
    except (OSError, ValueError, AttributeError):
        pass
    return None


def write_cache(path, days):
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump({"complete": True, "days": days}, handle)
    temp.replace(path)


def fetch_sensor_days(session, cache, sensor_id, begin, end):
    """OpenAQ /days: daily pollution mean, using local date."""
    path = cache / f"openaq_sensor_{sensor_id}_{begin.year}_{end.year}.json"
    cached = read_cache(path)
    if cached is not None:
        log(f"    Sensor {sensor_id} {begin.year}–{end.year}: cache ({len(cached)} days)")
        return cached
    rows = []
    page = 1
    while True:
        log(f"    Sensor {sensor_id} {begin.year}–{end.year}, page {page} ...")
        payload = get_json(
            session, f"{API_URL}/sensors/{sensor_id}/days",
            params={
                "datetime_from": f"{begin.isoformat()}T00:00:00Z",
                "datetime_to": f"{(end + timedelta(days=1)).isoformat()}T00:00:00Z",
                "limit": PAGE_SIZE, "page": page,
            }, label=f"OpenAQ sensor {sensor_id} page {page}",
        )
        if payload is None or not isinstance(payload.get("results"), list):
            log(f"    Sensor {sensor_id}: request failed; this chunk will NOT be cached")
            return None
        results = payload["results"]
        log(f"      Received {len(results)} daily results")
        for item in results:
            period = item.get("period") or {}
            stamp = period.get("datetimeFrom") or item.get("datetime") or {}
            day = iso_date(stamp)
            val = numeric(item.get("value"))
            if day and begin <= day <= end and val is not None:
                rows.append({"date": day.isoformat(), "value": val})
        if len(results) < PAGE_SIZE:
            break
        page += 1
        if page > 30:
            log("    Too many result pages; skipping incomplete sensor chunk")
            return None
    # Possible duplicated daily rows: average at later stage by calendar day.
    write_cache(path, rows)
    return rows


def fetch_weather_days(session, cache, station, begin, end):
    """Open-Meteo modeled daily mean 2m outdoor temperature (not on-site sensor)."""
    station_id = station["id"]
    path = cache / f"openmeteo_temp_station_{station_id}_{begin.year}_{end.year}.json"
    cached = read_cache(path)
    if cached is not None:
        log(f"    Weather {begin.year}–{end.year}: cache ({len(cached)} days)")
        return cached
    log(f"    Weather {begin.isoformat()} to {end.isoformat()} ...")
    payload = get_json(
        session, WEATHER_URL,
        params={
            "latitude": station["lat"], "longitude": station["lon"],
            "start_date": begin.isoformat(), "end_date": end.isoformat(),
            "daily": "temperature_2m_mean", "temperature_unit": "celsius",
            "timezone": "America/New_York",
        }, label=f"Open-Meteo {station_id}",
    )
    if payload is None:
        return None
    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    values = daily.get("temperature_2m_mean") or []
    if not dates or len(dates) != len(values):
        log(f"    Weather format unexpected for station {station_id}; not caching")
        return None
    rows = []
    for stamp, value in zip(dates, values):
        day = iso_date(stamp)
        temp = numeric(value, allow_negative=True)
        if day and begin <= day <= end and temp is not None:
            rows.append({"date": day.isoformat(), "value": temp})
    log(f"      Received {len(rows)} valid temperature days")
    write_cache(path, rows)
    return rows


def months():
    for year in range(START.year, END.year + 1):
        max_month = END.month if year == END.year else 12
        for month in range(1, max_month + 1):
            yield year, month


def station_active_during(station, year, month):
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    began = station["start"] or START
    ended = station["last"] or END
    return first <= ended and last >= began


def monthly_mean_from_days(days, *, accept_negative=False):
    """Count distinct daily means; duplicates averaged once per day."""
    by_date = defaultdict(list)
    for item in days:
        day = iso_date(item.get("date"))
        value = numeric(item.get("value"), allow_negative=accept_negative)
        if day and START <= day <= END and value is not None:
            by_date[day].append(value)
    by_month = defaultdict(list)
    for day, values in by_date.items():
        daily_avg = sum(values) / len(values)
        by_month[(day.year, day.month)].append(daily_avg)
    return {ym: (sum(values) / len(values), len(values)) for ym, values in by_month.items()}


def summarize_station(station, pollutants, air_session, weather_session, cache):
    """Return air CSV rows and temperature CSV rows for one location."""
    supported = station["sensors"]
    all_pollutant_data = {}
    first = max(START.year, (station["start"] or START).year)
    last = min(END.year, (station["last"] or END).year)
    if station["lat"] is None or station["lon"] is None:
        log("    No coordinates: weather will be N/A")

    # For multiple sensors measuring the SAME pollutant at one station, first
    # average sensors on each date, THEN average those daily values per month.
    if first <= last:
        for pollutant, sensor_ids in sorted(supported.items()):
            by_day = defaultdict(list)
            successful_chunks = 0
            for sensor_id in sorted(sensor_ids):
                for begin, end in chunks_by_3_years(first, last):
                    data = fetch_sensor_days(air_session, cache, sensor_id, begin, end)
                    if data is None:
                        continue
                    successful_chunks += 1
                    for item in data:
                        day = iso_date(item.get("date"))
                        value = numeric(item.get("value"))
                        if day and START <= day <= END and value is not None:
                            by_day[day].append(value)
            averaged_daily = [
                {"date": day.isoformat(), "value": sum(values) / len(values)}
                for day, values in by_day.items()
            ]
            all_pollutant_data[pollutant] = monthly_mean_from_days(averaged_daily)
            log(f"    {pollutant}: {len(by_day)} measured days ({successful_chunks} cached/fetched chunks)")

    # Historical modeled weather is available for a location even in years
    # before its pollution monitor started operating. Keep all 129 months.
    temperature_daily = []
    if station["lat"] is not None and station["lon"] is not None:
        for begin, end in chunks_by_3_years(START.year, END.year):
            data = fetch_weather_days(weather_session, cache, station, begin, end)
            if data is not None:
                temperature_daily.extend(data)
    temp_monthly = monthly_mean_from_days(temperature_daily, accept_negative=True)

    pollutant_rows, temperature_rows = [], []
    for year, month in months():
        ym = (year, month)
        base = {
            "year": year, "month": month, "month_name": MONTH_NAMES[month - 1],
            "location_id": station["id"], "station": station["name"],
            "latitude": station["lat"] if station["lat"] is not None else MISSING,
            "longitude": station["lon"] if station["lon"] is not None else MISSING,
            "station_active": "Yes" if station_active_during(station, year, month) else "No",
        }
        air = dict(base)
        air["pollutants_supported"] = ", ".join(sorted(supported)) or MISSING
        for pollutant in pollutants:
            month_value = (all_pollutant_data.get(pollutant, {}).get(ym)
                           if air["station_active"] == "Yes" else None)
            air[pollutant] = round(month_value[0], 4) if month_value else MISSING
            air[f"{pollutant}_days"] = month_value[1] if month_value else 0
        pollutant_rows.append(air)

        temp = dict(base)
        weather = temp_monthly.get(ym)
        temp["temperature_mean_c"] = round(weather[0], 3) if weather else MISSING
        temp["temperature_days"] = weather[1] if weather else 0
        temp["temperature_source"] = "Open-Meteo historical modeled 2m temperature"
        temperature_rows.append(temp)
    return pollutant_rows, temperature_rows


def save_csv(path, rows, columns):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-location", type=int, help="Process one location ID as a quick test")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        raise SystemExit("Missing OPENAQ_API_KEY in .env. Put the .env file next to this script.")
    stations, pollutants = read_stations()
    if args.only_location is not None:
        stations = [s for s in stations if s["id"] == args.only_location]
        if not stations:
            raise SystemExit(f"Location {args.only_location} is not in nyc_only_locations.csv")
    folder_name = "nyc_monthly_air_quality_2016_sep2026"
    if args.only_location is not None:
        folder_name += f"_TEST_{args.only_location}"
    out = ROOT / folder_name
    cache = ROOT / "nyc_monthly_air_quality_api_cache"
    out.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)

    meta = ["year", "month", "month_name", "location_id", "station", "latitude", "longitude", "station_active"]
    air_cols = meta + ["pollutants_supported"]
    for col in pollutants:
        air_cols.extend([col, f"{col}_days"])
    temp_cols = meta + ["temperature_mean_c", "temperature_days", "temperature_source"]
    air_path = out / "nyc_monthly_air_pollutants_2016_sep2026.csv"
    temp_path = out / "nyc_monthly_temperature_2016_sep2026.csv"

    log(f"Loaded {len(stations)} locations; range {START} through {END}")
    log("Air-pollution columns: " + ", ".join(pollutants))
    log("NO minimum number of measured days: 1 valid day -> that month's mean.")
    log("N/A marks unavailable measurements; 0 days means no valid daily data.")
    log(f"Output directory: {out}")

    air_session = requests.Session()
    air_session.headers.update({"X-API-Key": key, "Accept": "application/json"})
    weather_session = requests.Session()
    weather_session.headers.update({"Accept": "application/json"})
    all_air, all_temps = [], []
    for index, station in enumerate(stations, 1):
        log(f"\n[{index}/{len(stations)}] {station['name']} (ID {station['id']})")
        air_rows, temp_rows = summarize_station(station, pollutants, air_session, weather_session, cache)
        all_air.extend(air_rows)
        all_temps.extend(temp_rows)
        # Save after each location, preventing lost progress if the user stops.
        save_csv(air_path, all_air, air_cols)
        save_csv(temp_path, all_temps, temp_cols)
        log(f"    Saved {len(all_air)} pollutant rows and {len(all_temps)} temperature rows")

    log("\nCOMPLETE: saved two CSV files:")
    log(f"  {air_path}")
    log(f"  {temp_path}")
    log(f"  Months: {sum(1 for _ in months())}; locations: {len(stations)}")
    log("Reminder: Open-Meteo temperature is modeled weather, not an OpenAQ sensor reading.")
    log("A missing monthly average is N/A, NOT evidence of clean air or zero pollution.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nStopped by user. CSVs already saved through the last completed station.")
        sys.exit(130)
