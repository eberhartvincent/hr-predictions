"""
weather_client.py
=================
Fetch current weather for MLB stadiums using OpenWeatherMap.
Requires OPENWEATHER_API_KEY environment variable.

Venue name resolution
---------------------
MLB Stats API returns full official venue names (e.g. "Oriole Park at
Camden Yards", "Angel Stadium of Anaheim").  STADIUM_META keys match those
exactly.  A fuzzy fallback using difflib handles any remaining mismatches
and logs a warning so you can add them to ALIASES.
"""
from __future__ import annotations

import difflib
import logging
import os
import time

import requests

log = logging.getLogger(__name__)

OWM_BASE = "https://api.openweathermap.org/data/2.5"

# ---------------------------------------------------------------------------
# Stadium metadata — keyed by EXACT MLB Stats API venue name.
# cf_bearing = compass direction FROM home plate TOWARD center field (degrees).
# covered   = retractable or fixed roof (weather ignored).
# ---------------------------------------------------------------------------
STADIUM_META: dict[str, dict] = {
    # AL East
    "Oriole Park at Camden Yards":    {"lat": 39.2839, "lon": -76.6217, "cf_bearing": 10},
    "Fenway Park":                    {"lat": 42.3467, "lon": -71.0972, "cf_bearing": 96},
    "Yankee Stadium":                 {"lat": 40.8296, "lon": -73.9262, "cf_bearing": 300},
    "Rogers Centre":                  {"lat": 43.6414, "lon": -79.3894, "cf_bearing": 330, "covered": True},
    "Tropicana Field":                {"lat": 27.7682, "lon": -82.6534, "cf_bearing": 0,   "covered": True},
    # AL Central
    "Guaranteed Rate Field":          {"lat": 41.8299, "lon": -87.6338, "cf_bearing": 346},
    "Progressive Field":              {"lat": 41.4962, "lon": -81.6852, "cf_bearing": 335},
    "Comerica Park":                  {"lat": 42.3390, "lon": -83.0485, "cf_bearing": 352},
    "Kauffman Stadium":               {"lat": 39.0517, "lon": -94.4803, "cf_bearing": 5},
    "Target Field":                   {"lat": 44.9817, "lon": -93.2783, "cf_bearing": 336},
    # AL West
    "Globe Life Field":               {"lat": 32.7473, "lon": -97.0845, "cf_bearing": 5,   "covered": True},
    "Minute Maid Park":               {"lat": 29.7573, "lon": -95.3555, "cf_bearing": 25,  "covered": True},
    "Angel Stadium of Anaheim":       {"lat": 33.8003, "lon": -117.8827,"cf_bearing": 355},
    "Oakland Coliseum":               {"lat": 37.7516, "lon": -122.2005,"cf_bearing": 360},
    "Sutter Health Park":             {"lat": 38.5787, "lon": -121.5001,"cf_bearing": 0},
    "T-Mobile Park":                  {"lat": 47.5914, "lon": -122.3325,"cf_bearing": 20,  "covered": True},
    # NL East
    "Nationals Park":                 {"lat": 38.8730, "lon": -77.0074, "cf_bearing": 355},
    "loanDepot park":                 {"lat": 25.7781, "lon": -80.2197, "cf_bearing": 350, "covered": True},
    "Citi Field":                     {"lat": 40.7571, "lon": -73.8458, "cf_bearing": 2},
    "Citizens Bank Park":             {"lat": 39.9061, "lon": -75.1665, "cf_bearing": 350},
    "Truist Park":                    {"lat": 33.8908, "lon": -84.4678, "cf_bearing": 348},
    # NL Central
    "Wrigley Field":                  {"lat": 41.9484, "lon": -87.6553, "cf_bearing": 355},
    "Great American Ball Park":       {"lat": 39.0979, "lon": -84.5082, "cf_bearing": 352},
    "American Family Field":          {"lat": 43.0280, "lon": -87.9712, "cf_bearing": 5,   "covered": True},
    "PNC Park":                       {"lat": 40.4469, "lon": -80.0057, "cf_bearing": 352},
    "Busch Stadium":                  {"lat": 38.6226, "lon": -90.1928, "cf_bearing": 345},
    # NL West
    "Chase Field":                    {"lat": 33.4453, "lon": -112.0667,"cf_bearing": 0,   "covered": True},
    "Coors Field":                    {"lat": 39.7559, "lon": -104.9942,"cf_bearing": 13},
    "Dodger Stadium":                 {"lat": 34.0739, "lon": -118.2400,"cf_bearing": 0},
    "Oracle Park":                    {"lat": 37.7785, "lon": -122.3893,"cf_bearing": 110},
    "Petco Park":                     {"lat": 32.7076, "lon": -117.1570,"cf_bearing": 315},
}

# Alternate / historical / shorthand names → canonical key above.
ALIASES: dict[str, str] = {
    "Camden Yards":                    "Oriole Park at Camden Yards",
    "Angel Stadium":                   "Angel Stadium of Anaheim",
    "loanDepot Park":                  "loanDepot park",
    "Loan Depot Park":                 "loanDepot park",
    "Marlins Park":                    "loanDepot park",
    "Oakland-Alameda County Coliseum": "Oakland Coliseum",
    "RingCentral Coliseum":            "Oakland Coliseum",
    "Rate field":                      "Chase Field",
    "Dodger Stadium":                  "UNIQLO Field at Dodger Stadium"
}

_KNOWN_NAMES = list(STADIUM_META.keys())

# Simple in-process cache — avoids duplicate OWM calls within one run.
# Keyed by (lat, lon) tuple.
_weather_cache: dict[tuple, dict] = {}


def _resolve_venue(venue: str) -> dict:
    """Exact → alias → fuzzy → default (with warnings)."""
    venue = venue.strip()
    if venue in STADIUM_META:
        return STADIUM_META[venue]
    canonical = ALIASES.get(venue)
    if canonical and canonical in STADIUM_META:
        log.debug("Venue alias resolved: %r -> %r", venue, canonical)
        return STADIUM_META[canonical]
    matches = difflib.get_close_matches(venue, _KNOWN_NAMES, n=1, cutoff=0.75)
    if matches:
        log.warning(
            "Venue %r fuzzy-matched to %r — add to ALIASES to silence this.",
            venue, matches[0],
        )
        return STADIUM_META[matches[0]]
    log.warning(
        "Unknown venue %r — weather disabled for this game. "
        "Add it to STADIUM_META in weather_client.py.",
        venue,
    )
    return {"lat": 0.0, "lon": 0.0, "cf_bearing": 0}


def _wind_category(wind_deg: float, wind_speed: float, cf_bearing: int) -> str:
    if wind_speed < 3:
        return "calm"
    wind_toward = (wind_deg + 180) % 360
    diff = (wind_toward - cf_bearing + 360) % 360
    if diff <= 45 or diff >= 315:
        return "out_to_cf"
    elif 45 < diff < 90:
        return "out_to_rf"
    elif 90 <= diff <= 135:
        return "crosswind"
    elif 135 < diff <= 225:
        return "in_from_cf"
    elif 225 < diff < 270:
        return "crosswind"
    elif 270 <= diff < 315:
        return "out_to_lf"
    return "crosswind"


def _fetch_owm(lat: float, lon: float) -> dict:
    """
    Call OWM current-weather endpoint with up to 3 retries.
    Raises requests.HTTPError on bad status (caller logs the code).
    """
    cache_key = (round(lat, 4), round(lon, 4))
    if cache_key in _weather_cache:
        return _weather_cache[cache_key]

    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY environment variable is not set.")

    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(
                f"{OWM_BASE}/weather",
                params={"lat": lat, "lon": lon, "appid": api_key, "units": "imperial"},
                timeout=10,
            )
            # Raise immediately so we can inspect the status code
            resp.raise_for_status()
            data = resp.json()
            _weather_cache[cache_key] = data
            return data
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if status == 401:
                # Invalid key — no point retrying
                raise requests.HTTPError(
                    f"OWM returned 401 Unauthorized. "
                    f"Check that OPENWEATHER_API_KEY is correct and has been activated "
                    f"(new keys can take up to 2 hours to activate at openweathermap.org).",
                    response=exc.response,
                ) from exc
            if status == 429:
                log.warning("OWM rate-limited (429) on attempt %d — waiting 5s.", attempt)
                time.sleep(5)
            else:
                log.warning("OWM HTTP %s on attempt %d.", status, attempt)
            last_exc = exc
        except Exception as exc:
            log.warning("OWM request error on attempt %d: %s", attempt, exc)
            last_exc = exc
            time.sleep(2 * attempt)

    raise last_exc  # type: ignore[misc]


def get_game_weather(venue: str) -> dict:
    """
    Return weather conditions for a venue relevant to HR probability.

    Returns dict with:
        temperature_f, humidity_pct, wind_speed_mph, wind_direction_deg,
        wind_category, covered, conditions
    """
    meta = _resolve_venue(venue)
    covered = meta.get("covered", False)

    default = {
        "temperature_f":      72.0,
        "humidity_pct":       50.0,
        "wind_speed_mph":     0.0,
        "wind_direction_deg": 0,
        "wind_category":      "calm",
        "covered":            covered,
        "conditions":         "indoor (roof)" if covered else "unknown",
    }

    if covered:
        return default

    lat, lon = meta.get("lat", 0.0), meta.get("lon", 0.0)
    if lat == 0.0 and lon == 0.0:
        return default

    if not os.environ.get("OPENWEATHER_API_KEY"):
        log.warning(
            "OPENWEATHER_API_KEY not set — weather features disabled. "
            "Add it to GitHub Actions secrets (Settings → Secrets → Actions)."
        )
        return default

    try:
        raw = _fetch_owm(lat, lon)
    except requests.HTTPError as exc:
        # Log the full human-readable message we built above
        log.warning("Weather fetch failed for %r: %s", venue, exc)
        return default
    except Exception as exc:
        log.warning("Weather fetch failed for %r: %s", venue, exc)
        return default

    temp_f   = float(raw.get("main", {}).get("temp", 72.0))
    humidity = float(raw.get("main", {}).get("humidity", 50.0))
    wind     = raw.get("wind", {})
    wind_spd = float(wind.get("speed", 0.0))
    wind_deg = int(wind.get("deg", 0))
    sky      = raw.get("weather", [{}])[0].get("description", "clear")
    cat      = _wind_category(wind_deg, wind_spd, meta.get("cf_bearing", 0))

    return {
        "temperature_f":      round(temp_f, 1),
        "humidity_pct":       round(humidity, 1),
        "wind_speed_mph":     round(wind_spd, 1),
        "wind_direction_deg": wind_deg,
        "wind_category":      cat,
        "covered":            covered,
        "conditions":         sky,
    }
