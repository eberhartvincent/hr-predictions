"""
statcast_client.py
==================
Fetches Statcast metrics from Baseball Savant leaderboard CSVs.
Fetches the leaderboard ONCE per season per process (not per player).

Threading safety
----------------
Multiple ThreadPoolExecutor workers call get_batter_statcast_metrics()
simultaneously. Without a lock, all workers race to fetch the leaderboard
before the cache is populated — causing 8+ concurrent Savant requests.
The threading.Lock ensures exactly one fetch happens.

Call preload_leaderboards(season) on the main thread before starting
any parallel worker pools to avoid any race at all.
"""
from __future__ import annotations

import io
import logging
import threading

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "hr-predictor/1.0 (github-actions; open-source)"}
_session = requests.Session()
_session.headers.update(HEADERS)

BARRELS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast"
    "?type=batter&year={year}&position=&team=&min=10&csv=true"
)
XSTATS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=batter&year={year}&position=&team=&min=10&csv=true"
)

_barrels_cache: dict[int, pd.DataFrame] = {}
_xstats_cache:  dict[int, pd.DataFrame] = {}
_lock = threading.Lock()    # prevents concurrent leaderboard fetches


def _f(val, default: float = np.nan) -> float:
    try:
        f = float(val)
        return f if (f == f) else default
    except (TypeError, ValueError):
        return default


def preload_leaderboards(season: int) -> None:
    """
    Call this on the main thread BEFORE spawning worker pools.
    Guarantees exactly one fetch per season — workers hit the cache only.
    """
    _load_leaderboards(season)
    log.info(
        "Savant leaderboards preloaded for %d — "
        "barrels: %d players, xstats: %d players",
        season,
        len(_barrels_cache.get(season, pd.DataFrame())),
        len(_xstats_cache.get(season, pd.DataFrame())),
    )


def _load_leaderboards(season: int) -> None:
    """Thread-safe leaderboard fetch. No-op if already cached."""
    with _lock:
        if season not in _barrels_cache:
            try:
                r = _session.get(BARRELS_URL.format(year=season), timeout=30)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
                _barrels_cache[season] = (
                    df.dropna(subset=["player_id"]).set_index("player_id")
                )
                log.info("Savant barrels/%d: %d players", season, len(_barrels_cache[season]))
            except Exception as exc:
                log.warning("Savant barrels/%d failed: %s", season, exc)
                _barrels_cache[season] = pd.DataFrame()

        if season not in _xstats_cache:
            try:
                r = _session.get(XSTATS_URL.format(year=season), timeout=30)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
                _xstats_cache[season] = (
                    df.dropna(subset=["player_id"]).set_index("player_id")
                )
                log.info("Savant xstats/%d: %d players", season, len(_xstats_cache[season]))
            except Exception as exc:
                log.warning("Savant xstats/%d failed: %s", season, exc)
                _xstats_cache[season] = pd.DataFrame()


def get_batter_statcast_metrics(player_id: int, season: int) -> dict:
    """
    Return Statcast metrics from the cached leaderboard.
    O(1) lookup — no network call if preload_leaderboards() was called first.
    """
    _load_leaderboards(season)   # no-op if already cached

    empty = {
        "barrel_pct":       np.nan,
        "barrel_pa":        np.nan,
        "exit_velocity":    np.nan,
        "launch_angle":     np.nan,
        "sweet_spot_pct":   np.nan,
        "hard_hit_pct":     np.nan,
        "xwoba":            np.nan,
        "xslg":             np.nan,
        "n_batted_balls":   0,
    }

    result = empty.copy()

    brl_df = _barrels_cache.get(season, pd.DataFrame())
    if not brl_df.empty and player_id in brl_df.index:
        row = brl_df.loc[player_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        result["barrel_pct"]     = _f(row.get("brl_percent"))
        result["barrel_pa"]      = _f(row.get("brl_pa"))
        result["exit_velocity"]  = _f(row.get("avg_hit_speed"))
        result["launch_angle"]   = _f(row.get("avg_hit_angle"))
        result["sweet_spot_pct"] = _f(row.get("anglesweetspotpercent"))
        result["hard_hit_pct"]   = _f(row.get("ev95percent"))
        result["n_batted_balls"] = int(_f(row.get("attempts"), 0))

    xs_df = _xstats_cache.get(season, pd.DataFrame())
    if not xs_df.empty and player_id in xs_df.index:
        row = xs_df.loc[player_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        result["xwoba"] = _f(row.get("est_woba"))
        result["xslg"]  = _f(row.get("est_slg"))

    return result


def get_pitcher_statcast_metrics(pitcher_id: int, season: int) -> dict:
    """Kept for backward compatibility — pitcher quality now handled via MLB API K%."""
    return {}
