"""
weather_client.py
=================
Fetch current / forecast weather for MLB stadiums using OpenWeatherMap.
Requires OPENWEATHER_API_KEY environment variable.

Wind direction is converted to game-relevant categories:
  "out_to_cf", "out_to_rf", "out_to_lf", "in_from_cf", "crosswind", "calm"
"""
from __future__ import annotations

import logging
import math
import os
from functools import lru_cache

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

OWM_BASE = "https://api.openweathermap.org/data/2.5"

# Approximate coordinates and CF orientation for each stadium
# CF orientation = compass bearing FROM home plate TO center field
STADIUM_META: dict[str, dict] = {
    "Fenway Park":              {"lat": 42.3467, "lon": -71.0972, "cf_bearing": 96},
    "Yankee Stadium":           {"lat": 40.8296, "lon": -73.9262, "cf_bearing": 300},
    "Camden Yards":             {"lat": 39.2839, "lon": -76.6217, "cf_bearing": 10},
    "Tropicana Field":          {"lat": 27.7682, "lon": -82.6534, "cf_bearing": 0,  "covered": True},
    "Globe Life Field":         {"lat": 32.7473, "lon": -97.0845, "cf_bearing": 5,  "covered": True},
    "Minute Maid Park":         {"lat": 29.7573, "lon": -95.3555, "cf_bearing": 25, "covered": True},
    "American Family Field":    {"lat": 43.0280, "lon": -87.9712, "cf_bearing": 5,  "covered": True},
    "Chase Field":              {"lat": 33.4453, "lon": -112.0667,"cf_bearing": 0,  "covered": True},
    "Rogers Centre":            {"lat": 43.6414, "lon": -79.3894, "cf_bearing": 330,"covered": True},
    "T-Mobile Park":            {"lat": 47.5914, "lon": -122.3325,"cf_bearing": 20, "covered": True},
    "loanDepot park":           {"lat": 25.7781, "lon": -80.2197, "cf_bearing": 350,"covered": True},
    "Coors Field":              {"lat": 39.7559, "lon": -104.9942,"cf_bearing": 13},
    "Dodger Stadium":           {"lat": 34.0739, "lon": -118.2400,"cf_bearing": 0},
    "Oracle Park":              {"lat": 37.7785, "lon": -122.3893,"cf_bearing": 110},
    "Petco Park":               {"lat": 32.7076, "lon": -117.1570,"cf_bearing": 315},
    "Wrigley Field":            {"lat": 41.9484, "lon": -87.6553, "cf_bearing": 355},
    "Great American Ball Park": {"lat": 39.0979, "lon": -84.5082, "cf_bearing": 352},
    "Citizens Bank Park":       {"lat": 39.9061, "lon": -75.1665, "cf_bearing": 350},
    "Kauffman Stadium":         {"lat": 39.0517, "lon": -94.4803, "cf_bearing": 5},
    "Guaranteed Rate Field":    {"lat": 41.8299, "lon": -87.6338, "cf_bearing": 346},
    "Comerica Park":            {"lat": 42.3390, "lon": -83.0485, "cf_bearing": 352},
    "Busch Stadium":            {"lat": 38.6226, "lon": -90.1928, "cf_bearing": 345},
    "Truist Park":              {"lat": 33.8908, "lon": -84.4678, "cf_bearing": 348},
    "Nationals Park":           {"lat": 38.8730, "lon": -77.0074, "cf_bearing": 355},
    "PNC Park":                 {"lat": 40.4469, "lon": -80.0057, "cf_bearing": 352},
    "Marlins Park":             {"lat": 25.7781, "lon": -80.2197, "cf_bearing": 350},
    "Citi Field":               {"lat": 40.7571, "lon": -73.8458, "cf_bearing": 2},
    "Progressive Field":        {"lat": 41.4962, "lon": -81.6852, "cf_bearing": 335},
    "Target Field":             {"lat": 44.9817, "lon": -93.2783, "cf_bearing": 336},
    "Angel Stadium":            {"lat": 33.8003, "lon": -117.8827,"cf_bearing": 355},
    "Oakland Coliseum":         {"lat": 37.7516, "lon": -122.2005,"cf_bearing": 360},
    "Sahlen Field":             {"lat": 42.8785, "lon": -78.8690, "cf_bearing": 345},
}

_DEFAULT_META = {"lat": 0, "lon": 0, "cf_bearing": 0}


@lru_cache(maxsize=64)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def _fetch_weather(lat: float, lon: float) -> dict:
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        log.warning("OPENWEATHER_API_KEY not set — weather features disabled.")
        return {}
    resp = requests.get(
        f"{OWM_BASE}/weather",
        params={
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "imperial",   # Fahrenheit, mph
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _wind_category(wind_deg: float, wind_speed: float, cf_bearing: int) -> str:
    """
    Classify wind relative to the field.
    'out' means toward the outfield (HR-friendly), 'in' means toward home plate.
    """
    if wind_speed < 3:
        return "calm"

    # Angle between wind direction and CF bearing
    # OWM wind_deg = direction FROM which wind blows (meteorological)
    # Wind blowing FROM 0° (N) means wind moves southward
    # "Out to CF" means wind is blowing TOWARD CF bearing
    wind_toward = (wind_deg + 180) % 360   # direction wind is moving toward
    diff = (wind_toward - cf_bearing + 360) % 360

    if diff <= 45 or diff >= 315:
        return "out_to_cf"
    elif 45 < diff <= 135:
        return "out_to_rf" if diff <= 90 else "crosswind"
    elif 135 < diff <= 225:
        return "in_from_cf"
    elif 225 < diff <= 315:
        return "out_to_lf" if diff >= 270 else "crosswind"
    return "crosswind"


def get_game_weather(venue: str) -> dict:
    """
    Return weather conditions for a venue relevant to HR probability.

    Returns
    -------
    dict with:
        temperature_f, humidity_pct, wind_speed_mph, wind_direction_deg,
        wind_category, covered, conditions (description string)
    """
    meta = STADIUM_META.get(venue, _DEFAULT_META)
    covered = meta.get("covered", False)

    default = {
        "temperature_f": 72.0,
        "humidity_pct": 50.0,
        "wind_speed_mph": 0.0,
        "wind_direction_deg": 0,
        "wind_category": "calm",
        "covered": covered,
        "conditions": "unknown",
    }

    if covered:
        default["conditions"] = "indoor (roof)"
        return default

    lat, lon = meta.get("lat", 0), meta.get("lon", 0)
    if lat == 0 and lon == 0:
        log.warning("No coordinates for venue: %s", venue)
        return default

    try:
        raw = _fetch_weather(lat, lon)
    except Exception as exc:
        log.warning("Weather fetch failed for %s: %s", venue, exc)
        return default

    if not raw:
        return default

    temp_f = raw.get("main", {}).get("temp", 72.0)
    humidity = raw.get("main", {}).get("humidity", 50.0)
    wind = raw.get("wind", {})
    wind_speed = wind.get("speed", 0.0)      # mph (imperial mode)
    wind_deg = wind.get("deg", 0)
    sky = raw.get("weather", [{}])[0].get("description", "clear")

    cat = _wind_category(wind_deg, wind_speed, meta.get("cf_bearing", 0))

    return {
        "temperature_f": round(float(temp_f), 1),
        "humidity_pct": round(float(humidity), 1),
        "wind_speed_mph": round(float(wind_speed), 1),
        "wind_direction_deg": int(wind_deg),
        "wind_category": cat,
        "covered": covered,
        "conditions": sky,
    }
