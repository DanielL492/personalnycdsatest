"""Monthly NYC PM2.5 + temperature comparison, 2016–September 2026.

Generates up to 11 yearly monthly bar charts, and optional station-level monthly
charts for all qualifying monitoring stations. Includes CSV with monthly data.
Requires pip install requests pandas matplotlib python-dotenv.
Run: python -u nyc_pm25_temperature_monthly_sep2026_no_merge.py
"""

import ast
import os
import time
from pathlib import Path
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")  # Works in VS Code terminals without a graphics window.
import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

START_YEAR = 2016
END_YEAR = 2026
# Fixed final date: include January–September 2026 only, with no October data.
# January–December remains available for all earlier years (2016–2025).
DATA_CUTOFF = date(2026, 9, 30)
print(f"Data period: {START_YEAR}-01-01 through {DATA_CUTOFF} (inclusive)", flush=True)
GOVERNMENT_STATIONS_ONLY = True   # Change to False to include AirGradient sites.
MIN_PM25_DAY_COVERAGE = 0.65      # Require >= 65% of days within each month.
MAX_STATIONS_PER_GRAPH = 6
MAKE_STATION_GRAPHS = True  # Generates a separate monthly chart per station/year.
# First run ONLY CCNY to verify that the program completes; set to None for all stations.
ONLY_LOCATION_ID = None

PAGE_SIZE = 1000

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "nyc_only_locations.csv"
OUT = ROOT / "pm25_temperature_monthly_graphs_through_sep_2026"
CACHE = OUT / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("OPENAQ_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing OPENAQ_API_KEY in your .env file.")
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Put nyc_only_locations.csv beside the script: {INPUT_CSV}")

nyc = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(nyc)} NYC monitoring locations.", flush=True)


def new_session():
    session = requests.Session()
    retries = Retry(total=2, backoff_factor=1.0,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"], respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


air_session = new_session()
air_session.headers.update({"X-API-Key": API_KEY, "Accept": "application/json"})
# Weather calls should not silently retry for minutes: fail clearly and keep going.
weather_session = requests.Session()
weather_session.headers.update({"Accept": "application/json"})


def parse_dict(value):
    if isinstance(value, dict):
        return value
    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, dict) else {}
    except (ValueError, SyntaxError, TypeError):
        return {}


def sensors_for_pm25(value):
    try:
        sensors = ast.literal_eval(value) if isinstance(value, str) else value
        return list(dict.fromkeys(int(s["id"]) for s in sensors
                                  if s.get("parameter", {}).get("name") == "pm25"
                                  and s.get("parameter", {}).get("units") == "µg/m³"))
    except (ValueError, SyntaxError, TypeError, KeyError):
        return []


def empty_daily(column):
    return pd.DataFrame(columns=["date", column])


def load_daily_csv(path, column):
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["date", column])[["date", column]]


def get_pm25(sensor_id):
    """OpenAQ daily means, reusing old 2016–2025 cache if present."""
    cache_file = CACHE / f"openaq_pm25_sensor_{sensor_id}_through_sep_2026.csv"
    if cache_file.exists():
        print(f"    PM2.5 sensor {sensor_id}: using cache", flush=True)
        return load_daily_csv(cache_file, "pm25")

    # Previous project's successful downloads can be reused, so only 2026
    # needs an additional API request when those older cached files exist.
    previous = ROOT / "pm25_temperature_monthly_graphs" / "cache" / f"openaq_pm25_sensor_{sensor_id}_2016_2025.csv"
    historical = load_daily_csv(previous, "pm25") if previous.exists() else empty_daily("pm25")
    request_start = date(2026, 1, 1) if not historical.empty else date(START_YEAR, 1, 1)
    request_end = DATA_CUTOFF + timedelta(days=1)  # exclusive API upper bound
    if request_start >= request_end:
        return historical

    rows = []
    for page in range(1, 100):
        print(f"    PM2.5 sensor {sensor_id}: OpenAQ page {page} ({request_start} to {DATA_CUTOFF})", flush=True)
        response = None
        try:
            response = air_session.get(
                f"https://api.openaq.org/v3/sensors/{sensor_id}/days",
                params={"datetime_from": f"{request_start}T00:00:00Z",
                        "datetime_to": f"{request_end}T00:00:00Z",
                        "limit": PAGE_SIZE, "page": page},
                timeout=(10, 35),
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"    OpenAQ error: {exc}", flush=True)
            if response is not None:
                print(f"    HTTP {response.status_code}: {response.text[:220]!r}", flush=True)
            # Don't cache partial results following a failure.
            return historical if not historical.empty else empty_daily("pm25")

        results = payload.get("results", [])
        print(f"    Retrieved {len(results)} daily readings", flush=True)
        for item in results:
            period = item.get("period") or {}
            stamp = period.get("datetimeFrom") or {}
            if isinstance(stamp, dict):
                stamp = stamp.get("local") or stamp.get("utc")
            if stamp and item.get("value") is not None:
                rows.append({"date": str(stamp)[:10], "pm25": item["value"]})
        if len(results) < PAGE_SIZE:
            break
        time.sleep(0.1)

    new_data = pd.DataFrame(rows, columns=["date", "pm25"])
    if not new_data.empty:
        new_data["date"] = pd.to_datetime(new_data["date"], errors="coerce")
        new_data["pm25"] = pd.to_numeric(new_data["pm25"], errors="coerce")
        new_data = new_data.dropna(subset=["date", "pm25"])
        new_data = new_data[new_data["pm25"] >= 0]
    combined = pd.concat([historical, new_data], ignore_index=True)
    if combined.empty:
        return empty_daily("pm25")
    combined = combined[combined["date"] <= pd.Timestamp(DATA_CUTOFF)]
    combined = combined.groupby("date", as_index=False)["pm25"].mean()
    combined.to_csv(cache_file, index=False)
    return combined


def get_temperature(location_id, lat, lon, pm_dates):
    """Open-Meteo daily modeled temperature, reusing previous archive cache."""
    cache_file = CACHE / f"openmeteo_temperature_{location_id}_through_sep_2026.csv"
    previous = ROOT / "pm25_temperature_monthly_graphs" / "cache" / f"openmeteo_temperature_{location_id}_2016_2025.csv"
    # Reuse cached periods, but don't mistake an interrupted partial cache for completion.
    if cache_file.exists():
        historical = load_daily_csv(cache_file, "temperature_c")
        print(f"    Temperature: found {len(historical)} cached daily readings", flush=True)
    elif previous.exists():
        historical = load_daily_csv(previous, "temperature_c")
    else:
        historical = empty_daily("temperature_c")
    pm_dates = pd.to_datetime(pm_dates, errors="coerce").dropna()
    if pm_dates.empty:
        return historical
    # Download only years where PM2.5 measurements actually exist. This avoids
    # huge 11-year Open-Meteo calls for stations that only report in 2026.
    pm_years = sorted({int(d.year) for d in pm_dates})
    already_cached_years = set(historical["date"].dt.year.unique()) if not historical.empty else set()
    years_needed = [yr for yr in pm_years if yr not in already_cached_years]
    if not years_needed:
        if not historical.empty:
            historical.to_csv(cache_file, index=False)
        print(f"    Temperature: existing historical cache covers PM2.5 years", flush=True)
        return historical

    print(f"    Temperature: need weather for {years_needed}", flush=True)
    parts = [historical]
    # Request up to three consecutive calendar years at once; each result is
    # cached immediately so an interrupted run won't lose previous downloads.
    groups = {}
    for yr in years_needed:
        groups.setdefault((yr - START_YEAR) // 3, []).append(yr)
    for group in groups.values():
        start = date(group[0], 1, 1)
        end = min(date(group[-1], 12, 31), DATA_CUTOFF)
        print(f"      Open-Meteo request {start} to {end} (timeout 25 seconds, no retries)...", flush=True)
        response = None
        try:
            response = weather_session.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={"latitude": lat, "longitude": lon,
                        "start_date": str(start), "end_date": str(end),
                        "daily": "temperature_2m_mean", "temperature_unit": "celsius",
                        "timezone": "America/New_York"},
                timeout=(8, 25),
            )
            print(f"      Open-Meteo HTTP {response.status_code}", flush=True)
            response.raise_for_status()
            payload = response.json()
            daily = payload.get("daily", {})
            dates = daily.get("time", [])
            temperatures = daily.get("temperature_2m_mean", [])
            if not dates or len(dates) != len(temperatures):
                raise ValueError(f"Invalid temperature response: {str(payload)[:160]}")
            part = pd.DataFrame({"date": pd.to_datetime(dates),
                                "temperature_c": pd.to_numeric(temperatures, errors="coerce")})
            part = part.dropna(subset=["date", "temperature_c"])
            parts.append(part)
            print(f"      Received {len(part)} temperature days", flush=True)
        except (requests.RequestException, ValueError) as exc:
            print(f"      Temperature unavailable for {start} to {end}: {exc}", flush=True)
            print("      Continuing with PM2.5; this period's temperature bars will be blank.", flush=True)
            if response is not None and not response.ok:
                print(f"      Response: {response.text[:200]!r}", flush=True)
        # Save successful data after each request, to resume quickly later.
        latest = pd.concat(parts, ignore_index=True)
        latest = latest.dropna(subset=["date", "temperature_c"])
        latest = latest[latest["date"] <= pd.Timestamp(DATA_CUTOFF)]
        latest = latest.drop_duplicates(subset="date", keep="last").sort_values("date")
        if not latest.empty:
            latest.to_csv(cache_file, index=False)
    if latest.empty:
        print(f"    No temperature data downloaded for location {location_id}; will leave temperature blank.", flush=True)
    print(f"    Finished temperature stage for location {location_id}", flush=True)
    return latest



# Each row below describes one station in one calendar month.
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PM_COL = "pm25_mean_ug_m3"
TEMP_COL = "temperature_mean_c"


def plot_month_bars(ax, month_values, color, y_label, title):
    """One bar per month; missing values remain missing (no false zeroes)."""
    positions = list(range(1, 13))
    present = month_values.reindex(positions)
    bars = ax.bar(positions, present.values, color=color, width=0.74)
    ax.set_xticks(positions, MONTH_LABELS)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return bars


def make_overview_plot(year, annual):
    """Two panels: station-averaged monthly PM2.5 and weather temperature."""
    monthly = annual.groupby("month", as_index=True).agg(
        pm25=(PM_COL, "mean"), temperature=(TEMP_COL, "mean"),
        stations=("location_id", "nunique"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    plot_month_bars(axes[0], monthly["pm25"], "steelblue", "PM2.5 (µg/m³)",
                    "Average PM2.5 across qualifying NYC monitoring stations")
    plot_month_bars(axes[1], monthly["temperature"], "darkorange", "Temperature (°C)",
                    "Average outdoor temperature on PM2.5 measurement days")
    axes[1].axhline(0, color="gray", lw=0.75)
    if not monthly.empty:
        for month, row in monthly.iterrows():
            axes[0].annotate(f"n={int(row['stations'])}", (month, row["pm25"]),
                             xytext=(0, 4), textcoords="offset points",
                             fontsize=7, ha="center", color="dimgray")
    partial_note = (f" (data through {DATA_CUTOFF:%b %d})"
                    if year == END_YEAR and DATA_CUTOFF < date(END_YEAR, 12, 31) else "")
    fig.suptitle(f"NYC PM2.5 and temperature — monthly overview, {year}{partial_note}\n"
                 "Equal weight per reporting station (NOT a neighborhood-wide average)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2)
    filename = OUT / f"nyc_pm25_temperature_monthly_{year}.png"
    fig.savefig(filename, dpi=165, bbox_inches="tight")
    plt.close(fig)
    print(f"Created {filename.name}", flush=True)
    return monthly.reset_index().assign(year=year)


def make_station_comparison_plot(year, annual):
    """Monthly grouped bar charts across up to N stations for both variables."""
    counts = (annual.groupby("station")["month"].nunique()
              .sort_values(ascending=False))
    chosen = counts.head(MAX_STATIONS_PER_GRAPH).index.tolist()
    if not chosen:
        return
    monthly = annual[annual["station"].isin(chosen)].copy()
    from numpy import arange
    x = arange(12)
    fig, axes = plt.subplots(2, 1, figsize=(19, 11), sharex=True)
    width = 0.82 / len(chosen)
    color_cycle = plt.get_cmap("tab10")
    for index, name in enumerate(chosen):
        by_month = monthly[monthly["station"] == name].set_index("month")
        positions = x - 0.41 + (index + 0.5) * width
        for axis, column in zip(axes, (PM_COL, TEMP_COL)):
            axis.bar(positions, by_month[column].reindex(range(1, 13)).values,
                     width=width * 0.94, label=name, color=color_cycle(index))
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    axes[0].set_title("PM2.5 by station and month")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_title("Outdoor temperature on PM2.5 measurement days")
    axes[1].axhline(0, lw=0.75, color="gray")
    axes[1].set_xticks(x, MONTH_LABELS)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
    axes[0].legend(ncol=3, fontsize=9, loc="upper right")
    partial_note = (f" (through {DATA_CUTOFF:%b %d})"
                    if year == END_YEAR and DATA_CUTOFF < date(END_YEAR, 12, 31) else "")
    fig.suptitle(f"NYC monthly PM2.5 and temperature: station comparison, {year}{partial_note}\n"
                 f"Up to {MAX_STATIONS_PER_GRAPH} stations with the most qualifying months", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2)
    filename = OUT / f"nyc_station_comparison_monthly_{year}.png"
    fig.savefig(filename, dpi=165, bbox_inches="tight")
    plt.close(fig)
    print(f"Created {filename.name}", flush=True)


def make_single_station_plot(year, station_id, name, annual):
    """Two charts per location: separate PM2.5 and temperature monthly bars."""
    by_month = annual.set_index("month")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    plot_month_bars(axes[0], by_month[PM_COL], "steelblue", "PM2.5 (µg/m³)",
                    "PM2.5 monthly mean")
    plot_month_bars(axes[1], by_month[TEMP_COL], "darkorange", "Temperature (°C)",
                    "Modeled outdoor temperature on measured PM2.5 days")
    axes[1].axhline(0, color="gray", lw=0.75)
    partial_note = (f" (through {DATA_CUTOFF:%b %d})"
                    if year == END_YEAR and DATA_CUTOFF < date(END_YEAR, 12, 31) else "")
    fig.suptitle(f"{name} (ID {station_id}) — {year}{partial_note}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2)
    folder = OUT / "station_charts"
    folder.mkdir(exist_ok=True)
    filename = folder / f"station_{station_id}_monthly_{year}.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    all_rows = []
    stations_found = 0
    for i, (_, station) in enumerate(nyc.iterrows(), 1):
        if ONLY_LOCATION_ID is not None and int(station["id"]) != ONLY_LOCATION_ID:
            continue
        if GOVERNMENT_STATIONS_ONLY and str(station.get("isMonitor", "")).lower() != "true":
            continue
        sensor_ids = sensors_for_pm25(station.get("sensors"))
        if not sensor_ids:
            continue
        stations_found += 1
        name = str(station["name"])
        station_id = int(station["id"])
        coords = parse_dict(station.get("coordinates"))
        print(f"[{i}/{len(nyc)}] {name}, PM2.5 sensors: {sensor_ids}", flush=True)
        if "latitude" not in coords or "longitude" not in coords:
            print("    No coordinates; skipping", flush=True)
            continue

        pm_parts = [get_pm25(sensor) for sensor in sensor_ids]
        pm_parts = [part for part in pm_parts if not part.empty]
        if not pm_parts:
            print("    No PM2.5 measurements; skipping", flush=True)
            continue
        pm = (pd.concat(pm_parts, ignore_index=True)
              .groupby("date", as_index=False)["pm25"].mean())
        temperature = get_temperature(station_id, float(coords["latitude"]),
                                      float(coords["longitude"]), pm["date"])
        print(f"    CHECKPOINT A: temperature function returned; PM2.5 rows={len(pm)}, weather rows={len(temperature)}", flush=True)
        # Avoid pd.merge() here. A date-to-temperature lookup is simpler and
        # prevents unexpected merge performance problems or row multiplication.
        print("    CHECKPOINT A1: building a date lookup (no pandas merge)...", flush=True)
        weather_by_date = {
            str(day)[:10]: float(temp)
            for day, temp in zip(temperature["date"].tolist(),
                                 temperature["temperature_c"].tolist())
            if pd.notna(day) and pd.notna(temp)
        }
        print(f"    CHECKPOINT A2: lookup ready for {len(weather_by_date)} days", flush=True)
        combined = pm.copy()
        combined["temperature_c"] = [
            weather_by_date.get(str(day)[:10], float("nan"))
            for day in combined["date"].tolist()
        ]
        print(f"    CHECKPOINT B: dates matched ({len(combined)} rows); calculating monthly averages...", flush=True)

        for year in range(START_YEAR, END_YEAR + 1):
            print(f"    CHECKPOINT C: calculating year {year}", flush=True)
            for month in range(1, 13):
                start = pd.Timestamp(year=year, month=month, day=1)
                if start > pd.Timestamp(DATA_CUTOFF):
                    continue  # Never draw future months.
                end = start + pd.offsets.MonthBegin(1)
                part = combined[(combined["date"] >= start) & (combined["date"] < end)]
                if part.empty:
                    continue
                days_in_month = (end - start).days
                days_available = min(days_in_month,
                    (pd.Timestamp(DATA_CUTOFF) - start).days + 1)
                if days_available <= 0:
                    continue
                measured_days = part["date"].nunique()
                matched_days = part["temperature_c"].notna().sum()
                coverage = measured_days / days_available
                all_rows.append({
                    "year": year, "month": month, "month_name": MONTH_LABELS[month - 1],
                    "location_id": station_id, "station": name,
                    "days_in_month": int(days_in_month),
                    "days_available": int(days_available),
                    "partial_month": bool(days_available < days_in_month),
                    "data_cutoff": str(DATA_CUTOFF),
                    "pm25_days": int(measured_days),
                    "pm25_coverage": round(coverage, 4),
                    "temperature_matching_days": int(matched_days),
                    PM_COL: round(part["pm25"].mean(), 3),
                    TEMP_COL: (round(part["temperature_c"].mean(), 3)
                               if matched_days else float("nan")),
                    "enough_pm25_data": bool(coverage >= MIN_PM25_DAY_COVERAGE),
                })

        print(f"    CHECKPOINT D: monthly calculations done; {len(all_rows)} station-month rows", flush=True)
        print(f"    Finished processing {name} (location {station_id}).", flush=True)
        if all_rows:
            # Save a rolling checkpoint so Ctrl+C doesn't erase the work done.
            pd.DataFrame(all_rows).to_csv(OUT / "partial_monthly_results.csv", index=False)

    print(f"Eligible government PM2.5 stations found: {stations_found}", flush=True)
    print("CHECKPOINT E: starting to generate charts and output CSVs", flush=True)
    if not all_rows:
        raise RuntimeError("No PM2.5 data retrieved. Look above for API errors.")

    summary = pd.DataFrame(all_rows).sort_values(["year", "month", "station"])
    summary_path = OUT / "nyc_pm25_temperature_monthly_through_sep_2026.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}", flush=True)
    (OUT / "partial_monthly_results.csv").unlink(missing_ok=True)

    usable = summary[summary["enough_pm25_data"]].copy()
    valid_path = OUT / "nyc_pm25_temperature_monthly_qualifying_only.csv"
    usable.to_csv(valid_path, index=False)
    if usable.empty:
        print("No station-months meet the coverage threshold."
              " Check the CSV or lower MIN_PM25_DAY_COVERAGE.", flush=True)
        return

    overview_rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        annual = usable[usable["year"] == year]
        if annual.empty:
            print(f"Year {year}: no qualifying station-months; skipping", flush=True)
            continue
        overview_rows.append(make_overview_plot(year, annual))
        make_station_comparison_plot(year, annual)
        if MAKE_STATION_GRAPHS:
            stations_year = annual.groupby(["location_id", "station"], sort=True)
            for (station_id, station_name), records in stations_year:
                make_single_station_plot(year, station_id, station_name, records)
            print(f"  Finished {len(stations_year)} station charts for {year}", flush=True)

    if overview_rows:
        overview = pd.concat(overview_rows, ignore_index=True)
        overview.to_csv(OUT / "monthly_reporting_station_averages.csv", index=False)
    print(f"\nDone! Graphs and CSVs are in: {OUT}", flush=True)
    print("Note: Monthly temperature is historical modeled weather at the station,"
          " averaged ONLY across days with PM2.5 data.", flush=True)
    print("These are station readings, not definitive neighborhood air-quality rankings.", flush=True)
    print(f"2026 includes January–September only (ends {DATA_CUTOFF}); October–December excluded.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user. Completed station rows are saved in partial_monthly_results.csv.", flush=True)
        raise SystemExit(130)
