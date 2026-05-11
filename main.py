#!/usr/bin/env python3
"""
main.py
=======
Orchestrates the daily MLB home-run prediction pipeline.

Usage
-----
  python main.py                         # today's date, top_n from config
  python main.py --date 2024-08-10       # specific date
  python main.py --top-n 5              # override top_n
  python main.py --dry-run              # predict but don't send email
  python main.py --test-email           # send email with today's predictions
  python main.py --config my_config.yml # alternate config file
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timezone

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ── Lazy imports (keep startup fast for --help) ──────────────────────────────
def _import_pipeline():
    from src.data.mlb_client import (
        extract_matchups,
        get_batter_stats,
        get_pitcher_stats,
        get_platoon_splits,
        get_player_info,
        get_recent_games,
        get_roster,
        get_schedule,
    )
    from src.data.statcast_client import (
        get_batter_statcast_metrics,
        get_pitcher_statcast_metrics,
    )
    from src.data.weather_client import get_game_weather
    from src.models.predictor import ensemble_predict
    from src.notifications.email_sender import send_email

    return {
        "get_schedule": get_schedule,
        "extract_matchups": extract_matchups,
        "get_roster": get_roster,
        "get_player_info": get_player_info,
        "get_batter_stats": get_batter_stats,
        "get_pitcher_stats": get_pitcher_stats,
        "get_platoon_splits": get_platoon_splits,
        "get_recent_games": get_recent_games,
        "get_batter_statcast_metrics": get_batter_statcast_metrics,
        "get_pitcher_statcast_metrics": get_pitcher_statcast_metrics,
        "get_game_weather": get_game_weather,
        "ensemble_predict": ensemble_predict,
        "send_email": send_email,
    }


# ── Config loader ─────────────────────────────────────────────────────────────
def load_config(path: str = "config.yml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Allow env-var overrides
    top_n_env = os.environ.get("TOP_N")
    date_env = os.environ.get("PREDICT_DATE")
    if top_n_env:
        cfg.setdefault("prediction", {})["top_n"] = int(top_n_env)
    if date_env:
        cfg.setdefault("prediction", {})["date"] = date_env
    return cfg


# ── Roster fallback (when lineup not announced) ───────────────────────────────
def _probable_lineup(fn, team_id: int, season: int, min_games: int) -> list[int]:
    """
    Return player IDs of position players on the active roster who have
    played at least `min_games` this season (proxy for starters).
    """
    try:
        roster = fn["get_roster"](team_id)
    except Exception as exc:
        log.warning("Roster fetch failed for team %d: %s", team_id, exc)
        return []

    players = []
    for entry in roster:
        pos = entry.get("position", {}).get("abbreviation", "")
        if pos in ("SP", "RP", "P"):     # skip pitchers
            continue
        pid = entry.get("person", {}).get("id")
        if not pid:
            continue
        try:
            stats = fn["get_batter_stats"](pid, season)
            gp = int(stats["season_stats"].get("gamesPlayed", 0))
            if gp >= min_games:
                players.append(pid)
        except Exception:
            pass
    return players


# ── Core pipeline ─────────────────────────────────────────────────────────────
def run(config: dict, dry_run: bool = False) -> list[dict]:
    fn = _import_pipeline()
    pred_cfg = config.get("prediction", {})
    model_cfg = config.get("model", {})

    # Resolve date
    raw_date = pred_cfg.get("date", "today")
    if raw_date == "today":
        game_date = date.today()
    else:
        from dateutil.parser import parse as dparse
        game_date = dparse(raw_date).date()

    season = game_date.year
    top_n = int(pred_cfg.get("top_n", 10))
    min_pa = int(pred_cfg.get("min_pa_season", 30))
    min_games = int(pred_cfg.get("min_games_played", 10))

    log.info("── Running predictions for %s (top %d) ──", game_date, top_n)

    # 1. Schedule
    games = fn["get_schedule"](game_date)
    matchups = fn["extract_matchups"](games)
    log.info("Processing %d matchups", len(matchups))

    if not matchups:
        log.warning("No games found for %s. Exiting.", game_date)
        return []

    all_predictions: list[dict] = []
    lineups_confirmed = 0

    # 2. Iterate matchups
    for matchup in matchups:
        venue = matchup["venue"]
        weather = fn["get_game_weather"](venue)

        if matchup["lineup_confirmed"]:
            lineups_confirmed += 1

        for side in ("home", "away"):
            opp_side = "away" if side == "home" else "home"
            pitcher_info = matchup[f"{opp_side}_pitcher"]
            team_id = matchup[f"{side}_team_id"]
            team_name = matchup[f"{side}_team"]

            if pitcher_info is None:
                log.debug("No probable pitcher for %s in %s — skipping", side, venue)
                continue

            # Get pitcher Statcast metrics
            try:
                p_season = fn["get_pitcher_stats"](pitcher_info["id"], season)["season_stats"]
                p_career = fn["get_pitcher_stats"](pitcher_info["id"], season)["career_stats"]
            except Exception as exc:
                log.warning("Pitcher stats failed %s: %s", pitcher_info["fullName"], exc)
                p_season, p_career = {}, {}

            try:
                p_sc = fn["get_pitcher_statcast_metrics"](pitcher_info["id"], season)
            except Exception:
                p_sc = {}

            # Lineups
            lineup_ids = matchup[f"{side}_lineup"]
            if not lineup_ids:
                lineup_ids = _probable_lineup(fn, team_id, season, min_games)

            # 3. Per-batter predictions
            for player_id in lineup_ids:
                try:
                    info = fn["get_player_info"](player_id)
                    pos = info.get("position", "")
                    if pos in ("P", "SP", "RP"):
                        continue

                    b_stats = fn["get_batter_stats"](player_id, season)
                    b_season = b_stats["season_stats"]
                    b_career = b_stats["career_stats"]

                    # Filter by minimum PA
                    s_pa = int(b_season.get("plateAppearances", 0))
                    if s_pa < min_pa:
                        continue

                    splits = fn["get_platoon_splits"](player_id, season)
                    recent = fn["get_recent_games"](player_id, season, model_cfg.get("recent_form_games", 15))
                    sc_metrics = fn["get_batter_statcast_metrics"](player_id, season)

                    pred = fn["ensemble_predict"](
                        batter_season=b_season,
                        batter_career=b_career,
                        pitcher_season=p_season,
                        pitcher_career=p_career,
                        pitcher_throws=pitcher_info.get("throws", "R"),
                        batter_bats=info.get("bats", "R"),
                        platoon_splits=splits,
                        recent_games=recent,
                        statcast_metrics=sc_metrics,
                        venue=venue,
                        weather=weather,
                        config=config,
                    )

                    pred.update(
                        {
                            "player_id": player_id,
                            "player_name": info.get("fullName", "Unknown"),
                            "team": team_name,
                            "position": pos,
                            "bats": info.get("bats", "R"),
                            "pitcher": pitcher_info,
                            "venue": venue,
                            "weather": weather,
                            "game_pk": matchup["gamePk"],
                            "lineup_confirmed": matchup["lineup_confirmed"],
                        }
                    )
                    all_predictions.append(pred)

                except Exception as exc:
                    log.warning("Prediction failed for player %d: %s", player_id, exc)
                    continue

    # 4. Rank and trim
    all_predictions.sort(key=lambda x: x["hr_probability"], reverse=True)
    top_predictions = all_predictions[:top_n]

    log.info("Top %d predictions generated from %d candidates.", top_n, len(all_predictions))
    for i, p in enumerate(top_predictions, 1):
        log.info(
            "  %2d. %-25s %s  (conf: %s)",
            i, p["player_name"], p["hr_pct"], p["confidence_tier"]
        )

    # 5. Send email
    if not dry_run:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        email_cfg = config.get("email", {})
        subject_tmpl = email_cfg.get("subject", "⚾ Top {n} HR Predictions — {date}")

        sent = fn["send_email"](
            predictions=top_predictions,
            prediction_date=game_date.strftime("%A, %B %-d, %Y"),
            games_today=len(matchups),
            lineups_confirmed=lineups_confirmed,
            generated_at=now_utc,
            subject_template=subject_tmpl,
        )
        if sent:
            log.info("✅ Email delivered successfully.")
        else:
            log.error("❌ Email delivery failed.")
            sys.exit(1)
    else:
        log.info("Dry run — email not sent.")

    return top_predictions


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MLB HR Prediction Pipeline")
    p.add_argument("--config", default="config.yml", help="Path to config YAML")
    p.add_argument("--date", help="Override date (YYYY-MM-DD or 'today')")
    p.add_argument("--top-n", type=int, help="Override number of predictions")
    p.add_argument("--dry-run", action="store_true", help="Predict but don't send email")
    p.add_argument("--test-email", action="store_true", help="Send email (same as normal run)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    if args.date:
        config.setdefault("prediction", {})["date"] = args.date
    if args.top_n:
        config.setdefault("prediction", {})["top_n"] = args.top_n

    dry = args.dry_run and not args.test_email
    run(config, dry_run=dry)
