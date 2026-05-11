"""
mlb_client.py
=============
Thin wrapper around the free MLB Stats API (statsapi.mlb.com).
No API key required.
"""
from __future__ import annotations

import time
import logging
from datetime import date
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

BASE    = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "hr-predictor/1.0 (github-actions)"}

# Games in these states will never be played — skip entirely.
# Final/completed games are NOT skipped so you can run historical predictions.
SKIP_STATES = {"C", "D", "U", "T"}   # Cancelled, Postponed, Suspended

# Informational — used for logging only
FINAL_STATES = {"F", "O", "FR", "FT", "FO"}


def _get(path: str, params: dict | None = None) -> dict:
    url  = f"{BASE}{path}"
    resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    time.sleep(0.12)
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_schedule(game_date: date) -> list[dict]:
    ds   = game_date.strftime("%Y-%m-%d")
    data = _get(
        "/schedule",
        params={
            "sportId": 1,
            "date":    ds,
            "hydrate": "probablePitcher,lineups,team,venue",
        },
    )
    games = [
        g
        for date_entry in data.get("dates", [])
        for g in date_entry.get("games", [])
    ]
    log.info("Found %d games on %s", len(games), ds)
    return games


def extract_matchups(games: list[dict], skip_final: bool = False) -> list[dict]:
    """
    Convert raw game dicts to structured matchup dicts.

    Parameters
    ----------
    skip_final : bool
        False (default) — include completed games so you can run
        predictions for any past date (useful for backtesting).
        True — skip finished games, used by the daily auto-advance logic.
    """
    matchups = []
    counts   = {"included": 0, "skipped_cancelled": 0, "skipped_final": 0}

    for g in games:
        status = g.get("status", {}).get("codedGameState", "?")
        detail = g.get("status", {}).get("detailedState", status)
        home   = g.get("teams", {}).get("home", {}).get("team", {}).get("name", "?")
        away   = g.get("teams", {}).get("away", {}).get("team", {}).get("name", "?")

        if status in SKIP_STATES:
            counts["skipped_cancelled"] += 1
            log.debug("Skipping %s @ %s — %s", away, home, detail)
            continue

        if skip_final and status in FINAL_STATES:
            counts["skipped_final"] += 1
            log.debug("Skipping finished game %s @ %s — %s", away, home, detail)
            continue

        is_final = status in FINAL_STATES
        if is_final:
            log.debug("Including completed game %s @ %s (historical mode)", away, home)

        venue  = g.get("venue", {}).get("name", "Unknown Venue")
        home_d = g.get("teams", {}).get("home", {})
        away_d = g.get("teams", {}).get("away", {})

        def pitcher_info(team_side: dict) -> dict | None:
            pp = team_side.get("probablePitcher")
            if not pp:
                return None
            return {
                "id":       pp["id"],
                "fullName": pp.get("fullName", "Unknown"),
                "throws":   pp.get("pitchHand", {}).get("code", "R"),
            }

        def lineup_ids(lineups: dict, side: str) -> list[int]:
            players = (lineups or {}).get(side, [])
            return [p["id"] for p in players if isinstance(p, dict) and "id" in p]

        lineups_raw = g.get("lineups", {}) or {}
        home_lu     = lineup_ids(lineups_raw, "homePlayers")
        away_lu     = lineup_ids(lineups_raw, "awayPlayers")

        matchups.append({
            "gamePk":           g["gamePk"],
            "venue":            venue,
            "home_team":        home_d.get("team", {}).get("name", ""),
            "away_team":        away_d.get("team", {}).get("name", ""),
            "home_team_id":     home_d.get("team", {}).get("id"),
            "away_team_id":     away_d.get("team", {}).get("id"),
            "home_pitcher":     pitcher_info(home_d),
            "away_pitcher":     pitcher_info(away_d),
            "home_lineup":      home_lu,
            "away_lineup":      away_lu,
            "lineup_confirmed": bool(home_lu or away_lu),
            "game_status":      detail,
            "is_final":         is_final,
        })
        counts["included"] += 1

    log.info(
        "Matchups: %d included, %d cancelled/postponed skipped%s",
        counts["included"],
        counts["skipped_cancelled"],
        f", {counts['skipped_final']} finished skipped" if counts["skipped_final"] else "",
    )
    return matchups


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_roster(team_id: int, roster_type: str = "active") -> list[dict]:
    data = _get(f"/teams/{team_id}/roster", params={"rosterType": roster_type})
    return data.get("roster", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_batter_stats(player_id: int, season: int) -> dict:
    data = _get(
        f"/people/{player_id}/stats",
        params={"stats": "season,career", "group": "hitting",
                "season": season, "sportId": 1},
    )
    result = {"season_stats": {}, "career_stats": {}}
    for split in data.get("stats", []):
        kind   = split.get("type", {}).get("displayName", "")
        splits = split.get("splits", [])
        if not splits:
            continue
        s = splits[0].get("stat", {})
        if kind == "season":
            result["season_stats"] = s
        elif kind == "career":
            result["career_stats"] = s
    return result


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_pitcher_stats(player_id: int, season: int) -> dict:
    data = _get(
        f"/people/{player_id}/stats",
        params={"stats": "season,career", "group": "pitching",
                "season": season, "sportId": 1},
    )
    result = {"season_stats": {}, "career_stats": {}}
    for split in data.get("stats", []):
        kind   = split.get("type", {}).get("displayName", "")
        splits = split.get("splits", [])
        if not splits:
            continue
        s = splits[0].get("stat", {})
        if kind == "season":
            result["season_stats"] = s
        elif kind == "career":
            result["career_stats"] = s
    return result


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_platoon_splits(player_id: int, season: int) -> dict:
    data = _get(
        f"/people/{player_id}/stats",
        params={"stats": "statSplits", "group": "hitting",
                "season": season, "sitCodes": "vr,vl", "sportId": 1},
    )
    result = {"vs_right": {}, "vs_left": {}}
    for split_group in data.get("stats", []):
        for sp in split_group.get("splits", []):
            code = sp.get("split", {}).get("code", "")
            if code == "vr":
                result["vs_right"] = sp.get("stat", {})
            elif code == "vl":
                result["vs_left"]  = sp.get("stat", {})
    return result


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_recent_games(player_id: int, season: int, last_n: int = 15) -> list[dict]:
    data = _get(
        f"/people/{player_id}/stats",
        params={"stats": "gameLog", "group": "hitting",
                "season": season, "sportId": 1},
    )
    games = []
    for split_group in data.get("stats", []):
        for sp in split_group.get("splits", []):
            games.append(sp.get("stat", {}))
    return games[:last_n]


def get_player_info(player_id: int) -> dict:
    data   = _get(f"/people/{player_id}", params={"hydrate": "currentTeam"})
    people = data.get("people", [])
    if not people:
        return {}
    p = people[0]
    return {
        "id":       p.get("id"),
        "fullName": p.get("fullName", ""),
        "bats":     p.get("batSide", {}).get("code", "R"),
        "position": p.get("primaryPosition", {}).get("abbreviation", ""),
        "team":     p.get("currentTeam", {}).get("name", ""),
    }
