"""
mlb_client.py
=============
Wrapper around the MLB Stats API (api-web.nhle.com / statsapi.mlb.com).

Performance optimization
------------------------
`get_all_batter_data()` replaces four separate API calls
(get_batter_stats + get_platoon_splits + get_recent_games + get_player_info)
with a single combined request, reducing API calls per batter from 4 → 1.

For 270 batters × 4 calls = 1,080 calls → 270 calls. At 20 workers this
drops the batter fetch phase from ~25s to ~7s wall time.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

BASE    = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "hr-predictor/1.0 (github-actions)"}

# Reuse a single session for connection pooling across all calls
_session = requests.Session()
_session.headers.update(HEADERS)

SKIP_STATES  = {"C", "D", "U", "T"}
FINAL_STATES = {"F", "O", "FR", "FT", "FO"}


def _get(path: str, params: dict | None = None) -> dict:
    url  = f"{BASE}{path}"
    resp = _session.get(url, params=params, timeout=20)
    resp.raise_for_status()
    time.sleep(0.10)   # polite throttle
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_schedule(game_date: date) -> list[dict]:
    ds   = game_date.strftime("%Y-%m-%d")
    data = _get("/schedule", params={
        "sportId": 1, "date": ds,
        "hydrate": "probablePitcher,lineups,team,venue",
    })
    games = [
        g
        for date_entry in data.get("dates", [])
        for g in date_entry.get("games", [])
    ]
    log.info("Found %d games on %s", len(games), ds)
    return games


def extract_matchups(games: list[dict], skip_final: bool = False) -> list[dict]:
    matchups = []
    skipped  = {}

    for g in games:
        status = g.get("status", {}).get("codedGameState", "?")
        detail = g.get("status", {}).get("detailedState", status)
        home   = g.get("teams", {}).get("home", {}).get("team", {}).get("name", "?")
        away   = g.get("teams", {}).get("away", {}).get("team", {}).get("name", "?")

        if status in SKIP_STATES:
            skipped[detail] = skipped.get(detail, 0) + 1
            continue

        is_final = status in FINAL_STATES
        if skip_final and is_final:
            skipped[detail] = skipped.get(detail, 0) + 1
            continue

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

    if skipped:
        log.info("Skipped %d game(s) — %s", sum(skipped.values()),
                 ", ".join(f"{v}× {k}" for k, v in sorted(skipped.items())))
    log.info("Actionable matchups: %d", len(matchups))
    return matchups


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_roster(team_id: int, roster_type: str = "active") -> list[dict]:
    data = _get(f"/teams/{team_id}/roster", params={"rosterType": roster_type})
    return data.get("roster", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_players_info_batch(player_ids: list[int]) -> dict[int, dict]:
    """
    Fetch biographical info for multiple players in ONE request.
    Batches up to 200 IDs per call. Returns {player_id: info_dict}.
    """
    results: dict[int, dict] = {}
    # API supports comma-separated personIds — batch in groups of 150
    for i in range(0, len(player_ids), 150):
        batch = player_ids[i:i+150]
        ids_str = ",".join(str(pid) for pid in batch)
        try:
            data = _get("/people", params={"personIds": ids_str, "hydrate": "currentTeam"})
            for p in data.get("people", []):
                pid = p.get("id")
                if pid:
                    results[int(pid)] = {
                        "id":       p.get("id"),
                        "fullName": p.get("fullName", ""),
                        "bats":     p.get("batSide", {}).get("code", "R"),
                        "position": p.get("primaryPosition", {}).get("abbreviation", ""),
                        "team":     p.get("currentTeam", {}).get("name", ""),
                    }
        except Exception as exc:
            log.warning("Batch player info failed (batch %d): %s", i, exc)
    return results


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_all_batter_data(player_id: int, season: int, recent_n: int = 15) -> dict:
    """
    Fetch season stats, career stats, platoon splits, AND game log
    in a SINGLE API request instead of four separate calls.

    Replaces: get_batter_stats + get_platoon_splits + get_recent_games
    Returns the same dict structure as _fetch_batter_bundle.
    """
    data = _get(
        f"/people/{player_id}/stats",
        params={
            "stats":    "season,career,statSplits,gameLog",
            "group":    "hitting",
            "season":   season,
            "sitCodes": "vr,vl",
            "sportId":  1,
        },
    )

    season_stats: dict = {}
    career_stats: dict = {}
    splits       = {"vs_right": {}, "vs_left": {}}
    recent_games : list[dict] = []

    for stat_group in data.get("stats", []):
        kind   = stat_group.get("type", {}).get("displayName", "")
        splist = stat_group.get("splits", [])

        if kind == "season" and splist:
            season_stats = splist[0].get("stat", {})

        elif kind == "career" and splist:
            career_stats = splist[0].get("stat", {})

        elif kind == "statSplits":
            for sp in splist:
                code = sp.get("split", {}).get("code", "")
                if code == "vr":
                    splits["vs_right"] = sp.get("stat", {})
                elif code == "vl":
                    splits["vs_left"]  = sp.get("stat", {})

        elif kind == "gameLog":
            recent_games = [sp.get("stat", {}) for sp in splist[:recent_n]]

    return {
        "stats":  {"season_stats": season_stats, "career_stats": career_stats},
        "splits": splits,
        "recent": recent_games,
    }


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


def get_player_info(player_id: int) -> dict:
    """Single-player info — use get_players_info_batch() for bulk fetching."""
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


def is_back_to_back_mlb(team_id: int, game_date: date) -> bool:
    """Return True if the team played yesterday."""
    yesterday = game_date - timedelta(days=1)
    try:
        data = _get("/schedule", params={
            "sportId": 1, "date": yesterday.strftime("%Y-%m-%d"),
            "teamId": team_id,
        })
        for d in data.get("dates", []):
            for g in d.get("games", []):
                state = g.get("status", {}).get("codedGameState", "")
                if state in FINAL_STATES:
                    return True
    except Exception:
        pass
    return False
