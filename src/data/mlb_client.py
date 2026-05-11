"""
mlb_client.py
=============
Thin wrapper around the free MLB Stats API (statsapi.mlb.com).
No API key required.  Rate-limited by default to ~10 req/s.
"""
from __future__ import annotations

import time
import logging
from datetime import date, datetime
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

BASE = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "hr-predictor/1.0 (github-actions)"}


def _get(path: str, params: dict | None = None) -> dict:
    url = f"{BASE}{path}"
    resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    time.sleep(0.12)          # ~8 req/s — polite throttle
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_schedule(game_date: date) -> list[dict]:
    """Return list of game dicts for a given date including probable pitchers and lineups."""
    ds = game_date.strftime("%Y-%m-%d")
    data = _get(
        "/schedule",
        params={
            "sportId": 1,
            "date": ds,
            "hydrate": "probablePitcher,lineups,team,venue",
        },
    )
    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            games.append(g)
    log.info("Found %d games on %s", len(games), ds)
    return games


def extract_matchups(games: list[dict]) -> list[dict]:
    """
    For each game return a structured matchup dict:
      {
        gamePk, venue, home_team, away_team,
        home_pitcher: {id, fullName, throws},
        away_pitcher: {id, fullName, throws},
        home_lineup: [...player_ids],
        away_lineup: [...player_ids],
        lineup_confirmed: bool,
      }
    """
    matchups = []
    for g in games:
        status = g.get("status", {}).get("codedGameState", "")
        if status in ("F", "O"):          # skip finished/postponed
            continue

        venue = g.get("venue", {}).get("name", "Unknown Venue")
        home = g.get("teams", {}).get("home", {})
        away = g.get("teams", {}).get("away", {})

        def pitcher_info(team_side: dict) -> dict | None:
            pp = team_side.get("probablePitcher")
            if not pp:
                return None
            return {
                "id": pp["id"],
                "fullName": pp.get("fullName", "Unknown"),
                "throws": pp.get("pitchHand", {}).get("code", "R"),
            }

        def lineup_ids(lineups: dict, side: str) -> list[int]:
            players = lineups.get(side, []) if lineups else []
            return [p["id"] for p in players if isinstance(p, dict) and "id" in p]

        lineups_raw = g.get("lineups", {}) or {}
        home_lu = lineup_ids(lineups_raw, "homePlayers")
        away_lu = lineup_ids(lineups_raw, "awayPlayers")
        lineup_confirmed = bool(home_lu or away_lu)

        matchups.append(
            {
                "gamePk": g["gamePk"],
                "venue": venue,
                "home_team": home.get("team", {}).get("name", ""),
                "away_team": away.get("team", {}).get("name", ""),
                "home_team_id": home.get("team", {}).get("id"),
                "away_team_id": away.get("team", {}).get("id"),
                "home_pitcher": pitcher_info(home),
                "away_pitcher": pitcher_info(away),
                "home_lineup": home_lu,
                "away_lineup": away_lu,
                "lineup_confirmed": lineup_confirmed,
            }
        )
    return matchups


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_roster(team_id: int, roster_type: str = "active") -> list[dict]:
    """Return list of player dicts on a team's active roster."""
    data = _get(f"/teams/{team_id}/roster", params={"rosterType": roster_type})
    return data.get("roster", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_batter_stats(player_id: int, season: int) -> dict:
    """
    Return hitting stats for season + career for a batter.
    Keys: season_stats, career_stats (each a flat dict of counting stats).
    """
    data = _get(
        f"/people/{player_id}/stats",
        params={
            "stats": "season,career",
            "group": "hitting",
            "season": season,
            "sportId": 1,
        },
    )
    result = {"season_stats": {}, "career_stats": {}}
    for split in data.get("stats", []):
        kind = split.get("type", {}).get("displayName", "")
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
    """Return pitching stats (season + career) for a pitcher."""
    data = _get(
        f"/people/{player_id}/stats",
        params={
            "stats": "season,career",
            "group": "pitching",
            "season": season,
            "sportId": 1,
        },
    )
    result = {"season_stats": {}, "career_stats": {}}
    for split in data.get("stats", []):
        kind = split.get("type", {}).get("displayName", "")
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
    """
    Return vs-LHP and vs-RHP hitting splits for a batter.
    Keys: "vs_left", "vs_right" — each a stat dict.
    """
    data = _get(
        f"/people/{player_id}/stats",
        params={
            "stats": "statSplits",
            "group": "hitting",
            "season": season,
            "sitCodes": "vr,vl",
            "sportId": 1,
        },
    )
    result = {"vs_right": {}, "vs_left": {}}
    for split_group in data.get("stats", []):
        for sp in split_group.get("splits", []):
            code = sp.get("split", {}).get("code", "")
            if code == "vr":
                result["vs_right"] = sp.get("stat", {})
            elif code == "vl":
                result["vs_left"] = sp.get("stat", {})
    return result


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_recent_games(player_id: int, season: int, last_n: int = 15) -> list[dict]:
    """
    Return game-by-game hitting log for the last N games this season.
    Each entry is a flat stat dict with homeRuns, atBats, etc.
    """
    data = _get(
        f"/people/{player_id}/stats",
        params={
            "stats": "gameLog",
            "group": "hitting",
            "season": season,
            "sportId": 1,
        },
    )
    games = []
    for split_group in data.get("stats", []):
        for sp in split_group.get("splits", []):
            games.append(sp.get("stat", {}))
    # Most recent first
    return games[:last_n]


def get_player_info(player_id: int) -> dict:
    """Return basic player metadata (name, bats, position)."""
    data = _get(f"/people/{player_id}", params={"hydrate": "currentTeam"})
    people = data.get("people", [])
    if not people:
        return {}
    p = people[0]
    return {
        "id": p.get("id"),
        "fullName": p.get("fullName", ""),
        "bats": p.get("batSide", {}).get("code", "R"),
        "position": p.get("primaryPosition", {}).get("abbreviation", ""),
        "team": p.get("currentTeam", {}).get("name", ""),
    }
