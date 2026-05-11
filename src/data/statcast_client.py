"""
statcast_client.py
==================
Pull Statcast (Baseball Savant) data via pybaseball.
Computes barrel rate, exit velocity, launch angle, and related
quality-of-contact metrics that are the strongest predictors of HR power.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Barrel definition (per Statcast):
# LA between 26–30° AND EV ≥ 98 mph, or higher LA paired with higher EV
BARREL_LA_MIN = 26
BARREL_LA_MAX = 30
BARREL_EV_BASE = 98.0

# Hard-hit threshold
HARD_HIT_EV = 95.0


def _safe_import_pybaseball():
    try:
        import pybaseball as pb  # noqa: F401
        return pb
    except ImportError:
        log.error("pybaseball not installed — Statcast features will be empty.")
        return None


@lru_cache(maxsize=32)
def _fetch_statcast_batter(player_id: int, start: str, end: str) -> pd.DataFrame:
    pb = _safe_import_pybaseball()
    if pb is None:
        return pd.DataFrame()
    try:
        pb.cache.enable()
        df = pb.statcast_batter(start, end, player_id)
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception as exc:
        log.warning("Statcast fetch failed for player %d: %s", player_id, exc)
        return pd.DataFrame()


def get_batter_statcast_metrics(
    player_id: int,
    season: int,
    days_back: int = 365,
) -> dict:
    """
    Return quality-of-contact metrics for a batter over the season window.

    Returns
    -------
    dict with keys:
        barrel_rate, hard_hit_rate, avg_exit_velocity,
        avg_launch_angle, hr_per_contact, fly_ball_rate,
        n_batted_balls   (sample size)
    """
    end_dt = date.today()
    start_dt = date(season, 3, 1)          # opening day earliest
    if (end_dt - start_dt).days > days_back:
        start_dt = end_dt - timedelta(days=days_back)

    df = _fetch_statcast_batter(
        player_id,
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
    )

    empty = {
        "barrel_rate": np.nan,
        "hard_hit_rate": np.nan,
        "avg_exit_velocity": np.nan,
        "avg_launch_angle": np.nan,
        "hr_per_contact": np.nan,
        "fly_ball_rate": np.nan,
        "n_batted_balls": 0,
    }

    if df.empty:
        return empty

    # Filter to batted balls only
    batted = df[df["type"] == "X"].copy()
    if len(batted) < 10:
        return empty

    ev = pd.to_numeric(batted["launch_speed"], errors="coerce")
    la = pd.to_numeric(batted["launch_angle"], errors="coerce")
    valid = batted[ev.notna() & la.notna()].copy()
    valid["ev"] = ev[ev.notna() & la.notna()]
    valid["la"] = la[ev.notna() & la.notna()]

    if len(valid) < 10:
        return empty

    n = len(valid)

    # Barrel: statcast definition (simplified)
    def is_barrel(row):
        e, l = row["ev"], row["la"]
        if l < BARREL_LA_MIN or l > 50:
            return False
        if l <= BARREL_LA_MAX:
            return e >= BARREL_EV_BASE
        # Above 30°: EV threshold rises 2 mph per degree
        required_ev = BARREL_EV_BASE + 2 * (l - BARREL_LA_MAX)
        return e >= required_ev

    valid["barrel"] = valid.apply(is_barrel, axis=1)
    barrel_rate = valid["barrel"].mean()
    hard_hit_rate = (valid["ev"] >= HARD_HIT_EV).mean()
    avg_ev = valid["ev"].mean()
    avg_la = valid["la"].mean()

    # Fly ball rate (LA > 10°)
    fly_ball_rate = (valid["la"] > 10).mean()

    # HR per batted ball in Statcast window
    hr_events = batted[batted["events"] == "home_run"]
    hr_per_contact = len(hr_events) / n

    return {
        "barrel_rate": round(float(barrel_rate), 4),
        "hard_hit_rate": round(float(hard_hit_rate), 4),
        "avg_exit_velocity": round(float(avg_ev), 2),
        "avg_launch_angle": round(float(avg_la), 2),
        "hr_per_contact": round(float(hr_per_contact), 4),
        "fly_ball_rate": round(float(fly_ball_rate), 4),
        "n_batted_balls": n,
    }


def get_pitcher_statcast_metrics(pitcher_id: int, season: int) -> dict:
    """
    Return HR-relevant pitching metrics from Statcast.
    Higher barrel_rate_allowed / hard_hit_rate_allowed = more HR-prone.
    """
    pb = _safe_import_pybaseball()
    empty = {
        "barrel_rate_allowed": np.nan,
        "hard_hit_rate_allowed": np.nan,
        "avg_ev_allowed": np.nan,
        "gb_rate": np.nan,
        "fb_rate": np.nan,
    }
    if pb is None:
        return empty

    try:
        pb.cache.enable()
        end_dt = date.today().strftime("%Y-%m-%d")
        start_dt = f"{season}-03-01"
        df = pb.statcast_pitcher(start_dt, end_dt, pitcher_id)
    except Exception as exc:
        log.warning("Statcast pitcher fetch failed %d: %s", pitcher_id, exc)
        return empty

    if df is None or df.empty:
        return empty

    batted = df[df["type"] == "X"].copy()
    if len(batted) < 10:
        return empty

    ev = pd.to_numeric(batted["launch_speed"], errors="coerce")
    la = pd.to_numeric(batted["launch_angle"], errors="coerce")
    valid = batted[ev.notna() & la.notna()].copy()
    valid["ev"] = ev[ev.notna() & la.notna()]
    valid["la"] = la[ev.notna() & la.notna()]

    if len(valid) < 10:
        return empty

    barrel_allowed = (valid["ev"] >= BARREL_EV_BASE) & (
        valid["la"].between(BARREL_LA_MIN, BARREL_LA_MAX)
    )
    hard_hit_allowed = valid["ev"] >= HARD_HIT_EV
    gb_rate = (valid["la"] < 10).mean()
    fb_rate = (valid["la"] > 25).mean()

    return {
        "barrel_rate_allowed": round(float(barrel_allowed.mean()), 4),
        "hard_hit_rate_allowed": round(float(hard_hit_allowed.mean()), 4),
        "avg_ev_allowed": round(float(valid["ev"].mean()), 2),
        "gb_rate": round(float(gb_rate), 4),
        "fb_rate": round(float(fb_rate), 4),
    }
