"""
FloodLink – Live Flood Risk Evaluator (RAW + Linear)
Evaluates high-risk features from Citiesglobal.csv using Open-Meteo forecasts.

Now includes:
- Configurable forecast horizon (3h, 6h, 12h, etc.)
- Linear, unit-aware multipliers (rain unbounded; soil & RH clipped)
- RAW score only (no compression)
- Level-transition alerts only (Medium↔High, High↔Extreme; downgrades toggle)
- Single-file comparison (alerts_comparison.json)
- Rich Tweet Tracker (tweeted_alerts.json)
- BATCHED Open-Meteo calls (multiple locations per HTTP request)
"""

import os
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import tweepy
from requests.exceptions import RequestException, ReadTimeout, ConnectionError

# -------------------------------
# CONFIGURATION
# -------------------------------
CSV_PATH = "Citiesglobal.csv"
COMPARISON_PATH = "alerts_comparison.json"   # single source of truth
TWEET_LOG_PATH = "tweeted_alerts.json"       # map-ready tweet history

SLEEP_BETWEEN_CALLS = 0.1         # seconds between API calls (now: between BATCHES)
COMPARISON_HISTORY = 5            # or 10
TIMEZONE = "Europe/Madrid"
MAX_RETRIES = 1
TIMEOUT = 5                        # request timeout (s) per Open-Meteo call
FORECAST_HOURS = 6                 # 3, 6, 12, ...

# New: how many cities per Open-Meteo request
BATCH_SIZE = 50  # 50–100 is a good range; tune if needed

# --- Twitter config ---
TWITTER_ENABLED = os.getenv("TWITTER_ENABLED", "false").lower() == "true"
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_SECRET = os.getenv("TWITTER_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
MIN_SECONDS_BETWEEN_TWEETS = 30

# -------------------------------
# TUNABLE CONSTANTS (units!)
# -------------------------------
RISK_THRESHOLD = 8.5         # baseline FRisk cutoff from GIS layer

RAIN_UNIT_MM   = 100.0       # 100 mm → 1.0× rain multiplier
SOIL_MIN_MULT  = 0.95        # soil=0 -> 0.95×
SOIL_MAX_MULT  = 1.8         # soil=1 -> 1.8×
HUM_MIN_MULT   = 1.0         # RH=0% -> 1.0×
HUM_MAX_MULT   = 1.05        # RH=100% -> 1.05×
RAIN_CUTOFF_MM = 0.0         # set 0.5 to ignore drizzle; 0.0 keeps strict linearity

# RAW alert bands (tune later or learn from rolling percentiles)
RAW_LOW_MAX   = 5.0          # 0..5   -> Low
RAW_MED_MAX   = 15.0         # 5..15  -> Medium
RAW_HIGH_MAX  = 35.0         # 15..35 -> High
# >35 -> Extreme

# -------------------------------
# ALERT TRANSITION POLICY
# -------------------------------
TWEET_LEVELS = ["Medium", "High", "Extreme"]   # which levels are tweet-worthy at all
ALERT_ON_UPGRADES   = True                     # Medium→High, High→Extreme
ALERT_ON_DOWNGRADES = True                     # High→Medium, Extreme→High

LEVELS = ["None", "Low", "Medium", "High", "Extreme"]

# -------------------------------
# HELPER FUNCTIONS – WEATHER FETCH
# -------------------------------
def fetch_weather_batch(lat_list, lon_list):
    """
    Fetch Open-Meteo forecasts for a batch of locations.

    Returns:
        list of location JSON objects (same order as input lists),
        or None if the batch failed.
    """
    if not lat_list:
        return []

    # Build comma-separated coordinate strings
    lat_str = ",".join(f"{lat:.4f}" for lat in lat_list)
    lon_str = ",".join(f"{lon:.4f}" for lon in lon_list)

    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat_str,
        "longitude": lon_str,
        "hourly": "precipitation,relative_humidity_2m,soil_moisture_0_to_7cm",
        "forecast_days": 2,
        "timezone": TIMEZONE,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(base_url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()

            # Multi-location responses are lists; single location may be dict
            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                return data
            else:
                print("❓ Unexpected response type from Open-Meteo batch call.")
                return None

        except (ReadTimeout, ConnectionError):
            print(
                f"⚠️ Timeout/connection for batch of {len(lat_list)} locations "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            time.sleep(1.5 * attempt)
        except RequestException as e:
            print(f"❌ Request failed for batch of {len(lat_list)} locations: {e}")
            break

    print(f"🚫 Skipping batch of {len(lat_list)} locations after {MAX_RETRIES} failed attempts.")
    return None

# (Old per-point fetch_weather() kept for reference; no longer used in main)
def fetch_weather(lat, lon):
    """Legacy single-location fetch (unused with batching, kept for compatibility)."""
    base_url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=precipitation,relative_humidity_2m,soil_moisture_0_to_7cm"
        f"&forecast_days=2&timezone={TIMEZONE}"
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(base_url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except (ReadTimeout, ConnectionError):
            print(f"⚠️ Timeout/connection for {lat},{lon} (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(1.5 * attempt)
        except RequestException as e:
            print(f"❌ Request failed for {lat},{lon}: {e}")
            break
    print(f"🚫 Skipping {lat},{lon} after {MAX_RETRIES} failed attempts.")
    return None

# -------------------------------
# WEATHER INDICATORS
# -------------------------------
def compute_indicators(api_data):
    """
    Use the next FORECAST_HOURS starting at 'now' in the requested timezone.
    - precipitation: sum (mm)
    - RH / soil : average over window
    Open-Meteo soil moisture is m³/m³; useful range ~0–0.6; normalize to [0,1].
    """
    hourly = api_data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return 0.0, 0.0, 0.0

    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz).replace(minute=0, second=0, microsecond=0)

    # parse times (DatetimeIndex) — robust tz handling
    dt = pd.to_datetime(times, utc=True).tz_convert(tz)

    start_idx = next((i for i, t in enumerate(dt) if t >= now), 0)
    end_idx = start_idx + FORECAST_HOURS

    def window(key, default=0.0):
        arr = hourly.get(key, [])
        vals = []
        for v in arr[start_idx:end_idx]:
            vals.append(v if isinstance(v, (int, float)) and v is not None else default)
        # pad if short
        if len(vals) < FORECAST_HOURS:
            vals += [default] * (FORECAST_HOURS - len(vals))
        return vals

    rain_vals = window("precipitation", 0.0)
    rh_vals   = window("relative_humidity_2m", 0.0)
    soil_vals = window("soil_moisture_0_to_7cm", 0.0)

    rain_sum = float(sum(rain_vals))
    rh_avg   = float(sum(rh_vals) / FORECAST_HOURS)

    # Normalize soil: 0..0.6 → 0..1
    soil_norm = [min(max(x / 0.6, 0.0), 1.0) for x in soil_vals]
    soil_avg  = float(sum(soil_norm) / FORECAST_HOURS)

    return rain_sum, rh_avg, soil_avg

# -------------------------------
# LINEAR MULTIPLIERS
# -------------------------------
def rainfall_multiplier(rain_mm: float) -> float:
    return max(0.0, rain_mm / RAIN_UNIT_MM)

def soil_multiplier(soil_frac: float) -> float:
    s = max(0.0, min(1.0, soil_frac))
    return SOIL_MIN_MULT + s * (SOIL_MAX_MULT - SOIL_MIN_MULT)

def humidity_multiplier(rh_percent: float) -> float:
    rh = max(0.0, min(100.0, rh_percent))
    return HUM_MIN_MULT + (rh / 100.0) * (HUM_MAX_MULT - HUM_MIN_MULT)

# -------------------------------
# RISK MODEL (RAW ONLY)
# -------------------------------
def calculate_dynamic_risk_raw(base_risk: float, rain_mm: float, rh_percent: float, soil_frac: float):
    """
    Returns: (raw_score, level, r_mult, s_mult, h_mult)
    raw_score is linear in rain, soil, humidity (multiplicative across factors).
    """
    if rain_mm < RAIN_CUTOFF_MM:
        return 0.0, "None", 0.0, soil_multiplier(0.0), humidity_multiplier(0.0)

    r_mult = rainfall_multiplier(rain_mm)
    s_mult = soil_multiplier(soil_frac)
    h_mult = humidity_multiplier(rh_percent)

    raw_score = max(0.0, base_risk) * r_mult * s_mult * h_mult

    if raw_score == 0:
        level = "None"
    elif raw_score < RAW_LOW_MAX:
        level = "Low"
    elif raw_score < RAW_MED_MAX:
        level = "Medium"
    elif raw_score < RAW_HIGH_MAX:
        level = "High"
    else:
        level = "Extreme"

    return round(raw_score, 3), level, r_mult, s_mult, h_mult

# -------------------------------
# ALERT COMPARISON (level transitions only)
# -------------------------------
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"alerts": []}

def rotate_comparison_snapshots(max_history=COMPARISON_HISTORY):
    """
    Rotate alerts_comparison snapshots:

      alerts_comparison_{max_history-1}.json -> alerts_comparison_{max_history}.json
      ...
      alerts_comparison_1.json -> alerts_comparison_2.json
      alerts_comparison.json   -> alerts_comparison_1.json

    The new current run will then be written to alerts_comparison.json.
    """
    base = COMPARISON_PATH  # "alerts_comparison.json"

    # Shift numbered snapshots up: N-1 -> N, ..., 1 -> 2
    for i in range(max_history - 1, 0, -1):
        older = f"alerts_comparison_{i}.json"
        newer = f"alerts_comparison_{i + 1}.json"
        if os.path.exists(older):
            if os.path.exists(newer):
                os.remove(newer)
            os.replace(older, newer)

    # Move current base file to _1
    if os.path.exists(base):
        first_snapshot = "alerts_comparison_1.json"
        if os.path.exists(first_snapshot):
            os.remove(first_snapshot)
        os.replace(base, first_snapshot)

def build_alert_dict(alerts):
    return {(round(a["latitude"], 4), round(a["longitude"], 4)): a for a in alerts}

def compare_alerts(prev, curr):
    """
    Tweet when:
      • First time we see a site at a tweet-worthy level (Medium/High/Extreme)
      • Any UPGRADE into a tweet-worthy level (e.g., None→Medium, Low→Medium, Medium→High, High→Extreme)
      • (Optional) Downgrades if enabled
    """
    changes = []
    for key, c in curr.items():
        cur_lvl = c["dynamic_level"]

        # New site this run
        if key not in prev:
            if cur_lvl in TWEET_LEVELS:
                changes.append(("New", c))
            continue

        prev_lvl = prev[key]["dynamic_level"]
        if prev_lvl == cur_lvl:
            continue

        prev_i, cur_i = LEVELS.index(prev_lvl), LEVELS.index(cur_lvl)

        # Any upgrade into a tweet-worthy level
        if ALERT_ON_UPGRADES and cur_i > prev_i and cur_lvl in TWEET_LEVELS:
            changes.append(("Upgrade", c))
            continue

        # Downgrades from tweet-worthy levels (optional)
        if ALERT_ON_DOWNGRADES and cur_i < prev_i and prev_lvl in TWEET_LEVELS:
            changes.append(("Downgrade", c))

    return changes

# -------------------------------
# TWEET MANAGEMENT
# -------------------------------
def load_tweeted_alerts():
    if os.path.exists(TWEET_LOG_PATH):
        with open(TWEET_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_tweeted_alerts(tweeted):
    with open(TWEET_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(tweeted, f, indent=2, ensure_ascii=False)

def cleanup_tweeted_alerts(tweeted, valid_coords):
    """
    Keep only:
      - coordinates that still exist in the CSV, AND
      - entries NOT marked as resolved.

    Entries are marked resolved=True when a downgrade to Low/None happens.
    They stay in tweeted_alerts.json for that run, and are removed here on
    the following run.
    """
    cleaned = {}
    for k, v in tweeted.items():
        # k is "lat,lon" string
        if k not in valid_coords:
            continue
        # Drop entries that have already been marked as resolved
        if v.get("resolved", False):
            continue
        cleaned[k] = v

    if len(cleaned) < len(tweeted):
        print(f"🧹 Cleaned {len(tweeted) - len(cleaned)} outdated tweet entries.")
    return cleaned

def tweet_alert(change_type, alert):
    """Post a tweet for a new or transitioned flood alert."""
    lat, lon = alert["latitude"], alert["longitude"]
    level = alert["dynamic_level"]

    # 🎨 Emoji color map for risk level
    level_colors = {
        "None": "⚪",
        "Low": "⚪",
        "Medium": "🟢",
        "High": "🟠",
        "Extreme": "🔴"
    }

    # Get the appropriate color or default to ⚪
    color_emoji = level_colors.get(level, "⚪")

    tweet_text = (
        f"{color_emoji} Flood risk in "
        f"{', '.join([x for x in [alert.get('name','Location'), alert.get('country','')] if x])}.\n\n"
        f"{level} risk ({change_type})\n"
        f"Time: {FORECAST_HOURS} hours\n"
        f"Location ({lat:.2f}, {lon:.2f})\n\n"
        f"Rain: {alert[f'rain_{FORECAST_HOURS}h_mm']} mm\n"
        f"Soil moisture: {alert['soil_moisture_avg']:.2f}\n"
        f"Humidity: {alert['humidity_avg']}%\n"
    )

    print(f"🚨 Tweet → {tweet_text}\n")

    if not TWITTER_ENABLED:
        print("🧪 DRY RUN (tweet suppressed). Set TWITTER_ENABLED=true to send.")
        return

    try:
        client = tweepy.Client(
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_SECRET,
            wait_on_rate_limit=True,
        )
        client.create_tweet(text=tweet_text)
    except Exception as e:
        print(f"❌ Tweet failed: {e}")

# -------------------------------
# MAIN WORKFLOW
# -------------------------------
def main():
    print(f"🌧️ FloodLink Live Risk Evaluation started ({FORECAST_HOURS}-hour window)…")

    previous = load_json(COMPARISON_PATH)
    prev_alerts_dict = build_alert_dict(previous.get("alerts", []))
    tweeted_alerts = load_tweeted_alerts()

    df = pd.read_csv(CSV_PATH)
    high_risk = df[df["FRisk"] > RISK_THRESHOLD].copy()

    valid_coords = {f"{row['Latitude']:.4f},{row['Longitude']:.4f}" for _, row in df.iterrows()}
    tweeted_alerts = cleanup_tweeted_alerts(tweeted_alerts, valid_coords)

    alerts = []
    start_time = time.time()

    # Process high-risk features in batches
    num_high_risk = len(high_risk)
    print(f"📈 Evaluating {num_high_risk} high-risk locations in batches of {BATCH_SIZE}…")

    # Reset index so iloc + iterrows/play nicely
    high_risk = high_risk.reset_index(drop=True)

    for batch_start in range(0, num_high_risk, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_high_risk)
        batch = high_risk.iloc[batch_start:batch_end]

        lat_list = batch["Latitude"].astype(float).tolist()
        lon_list = batch["Longitude"].astype(float).tolist()

        # Fetch weather for the entire batch
        batch_weather = fetch_weather_batch(lat_list, lon_list)

        if not batch_weather:
            # Entire batch failed: reuse previous alert if available, else skip
            for (_, row), lat, lon in zip(batch.iterrows(), lat_list, lon_list):
                base_risk = float(row["FRisk"])
                name = str(row.get("ETIQUETA", f"id_{row['JOIN_ID']}"))
                country = str(row.get("Country", "")).strip()
                key = (round(lat, 4), round(lon, 4))
                prev_alert = prev_alerts_dict.get(key)

                if prev_alert:
                    print(
                        f"⚠️ Using previous alert for {name} "
                        f"[{lat:.4f},{lon:.4f}] due to batch API failure."
                    )
                    alerts.append(prev_alert)
                else:
                    print(
                        f"⚠️ No weather data and no previous alert for {name} "
                        f"[{lat:.4f},{lon:.4f}] – skipping this run."
                    )
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        # Sanity check: size mismatch between request and response
        if len(batch_weather) != len(lat_list):
            print(
                f"⚠️ Mismatch in batch response size: requested {len(lat_list)}, "
                f"received {len(batch_weather)}. Truncating to min length."
            )
        n = min(len(batch_weather), len(lat_list))

        # Normal path: we got fresh weather data for the first n locations
        for (idx, row), lat, lon, api_data in zip(
            batch.iloc[:n].iterrows(), lat_list[:n], lon_list[:n], batch_weather[:n]
        ):
            base_risk = float(row["FRisk"])
            name = str(row.get("ETIQUETA", f"id_{row['JOIN_ID']}"))
            country = str(row.get("Country", "")).strip()

            rain_sum, rh_avg, soil_avg = compute_indicators(api_data)
            raw_score, dyn_level, r_mult, s_mult, h_mult = calculate_dynamic_risk_raw(
                base_risk, rain_sum, rh_avg, soil_avg
            )

            alerts.append({
                "id": str(row["JOIN_ID"]),
                "country": country,
                "name": name,
                "latitude": float(lat),
                "longitude": float(lon),
                "base_risk": round(base_risk, 2),

                f"rain_{FORECAST_HOURS}h_mm": round(rain_sum, 2),
                "humidity_avg": round(rh_avg, 1),
                "soil_moisture_avg": round(soil_avg, 3),

                # Diagnostics for tuning
                "rain_mult": round(r_mult, 3),
                "soil_mult": round(s_mult, 3),
                "humidity_mult": round(h_mult, 3),

                "raw_dynamic_score": raw_score,
                "dynamic_level": dyn_level
            })

        # If the response was shorter than the batch, fall back for the remainder
        if len(batch) > n:
            for (_, row), lat, lon in zip(
                batch.iloc[n:].iterrows(), lat_list[n:], lon_list[n:]
            ):
                base_risk = float(row["FRisk"])
                name = str(row.get("ETIQUETA", f"id_{row['JOIN_ID']}"))
                country = str(row.get("Country", "")).strip()
                key = (round(lat, 4), round(lon, 4))
                prev_alert = prev_alerts_dict.get(key)

                if prev_alert:
                    print(
                        f"⚠️ Using previous alert for {name} "
                        f"[{lat:.4f},{lon:.4f}] due to partial batch response."
                    )
                    alerts.append(prev_alert)
                else:
                    print(
                        f"⚠️ No weather data and no previous alert for {name} "
                        f"[{lat:.4f},{lon:.4f}] – skipping this run."
                    )

        # Gentle pacing between batches
        time.sleep(SLEEP_BETWEEN_CALLS)

    # Persist current results
    result = {
        "timestamp": datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"),
        "source": "Open-Meteo",
        "forecast_window_hours": FORECAST_HOURS,
        "features_evaluated": len(alerts),
        "alerts": alerts
    }

    # Detect level-change events
    curr_alerts_dict = build_alert_dict(alerts)
    changes = compare_alerts(prev_alerts_dict, curr_alerts_dict)
    print(f"🔍 Detected {len(changes)} level-change events.")

    # 👉 Debug: list each transition with prev → current (plus key metrics)
    if changes:
        for change_type, a in changes:
            key = (round(a["latitude"], 4), round(a["longitude"], 4))
            prev_lvl = prev_alerts_dict.get(key, {}).get("dynamic_level", "None")
            print(
                "🛰️ "
                f"{a['name']} [{a['latitude']:.4f},{a['longitude']:.4f}]: "
                f"{prev_lvl} → {a['dynamic_level']} ({change_type}); "
                f"rain={a[f'rain_{FORECAST_HOURS}h_mm']} mm, "
                f"soil={a['soil_moisture_avg']:.3f}, RH={a['humidity_avg']}%"
            )
    else:
        print("ℹ️ No tweetable transitions this run (either steady level or below tweet-worthy).")

    last_tweet_ts = 0.0

    # Tweet + update tracker
    for change_type, alert in changes:
        key = f"{alert['latitude']:.4f},{alert['longitude']:.4f}"
        current_level = alert["dynamic_level"]
        last_entry = tweeted_alerts.get(key)

        # --- Downgrade gating logic ---
        if change_type == "Downgrade":
            # If we've never tweeted this location, ignore the downgrade
            if last_entry is None:
                print(
                    f"↘️ Skipping downgrade tweet for {key} "
                    f"({alert['name']}) – no prior tweet recorded."
                )
                continue

            # If the last tweeted level is already Low/None (i.e. not in TWEET_LEVELS),
            # we've already announced the downgrade for this alert cycle.
            last_level = last_entry.get("risk_level", "None")
            if last_level not in TWEET_LEVELS:
                print(
                    f"↘️ Skipping extra downgrade tweet for {key} "
                    f"({alert['name']}) – last tweeted level is already "
                    f"{last_level} (outside {TWEET_LEVELS})."
                )
                continue

        # -------------------------
        # Stream-wide rate limiting
        # -------------------------
        now_ts = time.time()
        if now_ts - last_tweet_ts < MIN_SECONDS_BETWEEN_TWEETS:
            time.sleep(MIN_SECONDS_BETWEEN_TWEETS - (now_ts - last_tweet_ts))

        # Send tweet (or DRY RUN printout)
        tweet_alert(change_type, alert)
        last_tweet_ts = time.time()

        # --- Update tweeted_alerts.json according to the new level ---
        if current_level in TWEET_LEVELS:
            # Still Medium / High / Extreme → keep or create/update entry
            tweeted_alerts[key] = {
                "country": alert.get("country", ""),
                "name": alert["name"],
                "risk_level": current_level,
                "latitude": alert["latitude"],
                "longitude": alert["longitude"],
                "rain_mm": alert[f"rain_{FORECAST_HOURS}h_mm"],
                "humidity": alert["humidity_avg"],
                "soil_moisture": alert["soil_moisture_avg"],
                "raw_dynamic_score": alert["raw_dynamic_score"],
                "last_updated": datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
            }
        else:
            # Downgrade into Low / None → keep it ONE more run as 'resolved'
            print(
                f"✅ Marking alert as resolved in tweet log: "
                f"{alert['name']} [{key}] (→ {current_level})"
            )

            tweeted_alerts[key] = {
                "country": alert.get("country", ""),
                "name": alert["name"],
                "risk_level": current_level,  # "Low" or "None"
                "latitude": alert["latitude"],
                "longitude": alert["longitude"],
                "rain_mm": alert[f"rain_{FORECAST_HOURS}h_mm"],
                "humidity": alert["humidity_avg"],
                "soil_moisture": alert["soil_moisture_avg"],
                "raw_dynamic_score": alert["raw_dynamic_score"],
                "last_updated": datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"),
                "resolved": True,  # <-- flag for next run's cleanup
            }

    save_tweeted_alerts(tweeted_alerts)

    # Rotate old comparison snapshots, then write the new one
    rotate_comparison_snapshots(COMPARISON_HISTORY)

    # Update comparison file
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"✅ Completed in {round((time.time() - start_time)/60, 1)} min. "
        f"Updated {COMPARISON_PATH} and {TWEET_LOG_PATH}."
    )

# -------------------------------
if __name__ == "__main__":
    main()
