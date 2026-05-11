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
  python main.py --retrain              # retrain model then predict
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
# Suppress noisy matplotlib font manager logs from pybaseball dependency
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

log = logging.getLogger("main")


def _import_pipeline():
    from src.data.mlb_client import (
        extract_matchups, get_batter_stats, get_pitcher_stats,
        get_platoon_splits, get_player_info, get_recent_games,
        get_roster, get_schedule,
    )
    from src.data.statcast_client import (
        get_batter_statcast_metrics, get_pitcher_statcast_metrics,
    )
    from src.data.weather_client import get_game_weather
    from src.models.predictor import ensemble_predict
    from src.notifications.email_sender import send_email
    from src.models.model_registry import load as load_model

    return {
        "get_schedule":               get_schedule,
        "extract_matchups":           extract_matchups,
        "get_roster":                 get_roster,
        "get_player_info":            get_player_info,
        "get_batter_stats":           get_batter_stats,
        "get_pitcher_stats":          get_pitcher_stats,
        "get_platoon_splits":         get_platoon_splits,
        "get_recent_games":           get_recent_games,
        "get_batter_statcast_metrics":get_batter_statcast_metrics,
        "get_pitcher_statcast_metrics":get_pitcher_statcast_metrics,
        "get_game_weather":           get_game_weather,
        "ensemble_predict":           ensemble_predict,
        "send_email":                 send_email,
        "load_model":                 load_model,
    }


def load_config(path: str = "config.yml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    top_n_env  = os.environ.get("TOP_N")
    date_env   = os.environ.get("PREDICT_DATE")
    if top_n_env:
        cfg.setdefault("prediction", {})["top_n"] = int(top_n_env)
    if date_env:
        cfg.setdefault("prediction", {})["date"] = date_env
    return cfg


def _probable_lineup(fn, team_id: int, season: int, min_games: int) -> list[int]:
    try:
        roster = fn["get_roster"](team_id)
    except Exception as exc:
        log.warning("Roster fetch failed for team %d: %s", team_id, exc)
        return []
    players = []
    for entry in roster:
        pos = entry.get("position", {}).get("abbreviation", "")
        if pos in ("SP", "RP", "P"):
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


def run(config: dict, dry_run: bool = False) -> list[dict]:
    fn = _import_pipeline()
    pred_cfg  = config.get("prediction", {})
    model_cfg = config.get("model", {})

    # ── Date ─────────────────────────────────────────────────────────────────
    raw_date = pred_cfg.get("date", "today")
    if raw_date == "today":
        game_date = date.today()
    else:
        from dateutil.parser import parse as dparse
        game_date = dparse(raw_date).date()

    season = game_date.year
    top_n   = int(pred_cfg.get("top_n", 10))
    min_pa  = int(pred_cfg.get("min_pa_season", 30))
    min_games = int(pred_cfg.get("min_games_played", 10))

    # ── Load XGBoost model (graceful fallback if not trained yet) ─────────────
    xgb_model, xgb_meta = fn["load_model"]()
    if xgb_model is not None:
        log.info(
            "✅ XGBoost model loaded (R²=%.4f, trained on %d examples)",
            xgb_meta.get("cv_r2", 0), xgb_meta.get("n_training", 0),
        )
    else:
        log.info("ℹ️  No trained model found — using statistical model only.")
        log.info("   Run the 'Retrain HR Model' workflow to enable XGBoost.")

    log.info("── Running predictions for %s (top %d) ──", game_date, top_n)

    # ── Schedule ──────────────────────────────────────────────────────────────
    games    = fn["get_schedule"](game_date)
    matchups = fn["extract_matchups"](games)
    log.info("Processing %d matchups", len(matchups))

    if not matchups:
        log.warning("No games found for %s. Exiting.", game_date)
        return []

    all_predictions: list[dict] = []
    lineups_confirmed = 0

    # ── Per-game, per-batter predictions ─────────────────────────────────────
    for matchup in matchups:
        venue   = matchup["venue"]
        weather = fn["get_game_weather"](venue)

        if matchup["lineup_confirmed"]:
            lineups_confirmed += 1

        for side in ("home", "away"):
            opp_side    = "away" if side == "home" else "home"
            pitcher_info = matchup[f"{opp_side}_pitcher"]
            team_id      = matchup[f"{side}_team_id"]
            team_name    = matchup[f"{side}_team"]

            if pitcher_info is None:
                continue

            try:
                p_stats  = fn["get_pitcher_stats"](pitcher_info["id"], season)
                p_season = p_stats["season_stats"]
                p_career = p_stats["career_stats"]
            except Exception as exc:
                log.warning("Pitcher stats failed %s: %s", pitcher_info["fullName"], exc)
                p_season, p_career = {}, {}

            lineup_ids = matchup[f"{side}_lineup"] or _probable_lineup(
                fn, team_id, season, min_games
            )

            for player_id in lineup_ids:
                try:
                    info = fn["get_player_info"](player_id)
                    if info.get("position") in ("P", "SP", "RP"):
                        continue

                    b_stats  = fn["get_batter_stats"](player_id, season)
                    b_season = b_stats["season_stats"]
                    b_career = b_stats["career_stats"]

                    if int(b_season.get("plateAppearances", 0)) < min_pa:
                        continue

                    splits     = fn["get_platoon_splits"](player_id, season)
                    recent     = fn["get_recent_games"](player_id, season, model_cfg.get("recent_form_games", 15))
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
                        xgb_model=xgb_model,
                        xgb_meta=xgb_meta,
                    )

                    pred.update({
                        "player_id":        player_id,
                        "player_name":      info.get("fullName", "Unknown"),
                        "team":             team_name,
                        "position":         info.get("position", ""),
                        "bats":             info.get("bats", "R"),
                        "pitcher":          pitcher_info,
                        "venue":            venue,
                        "weather":          weather,
                        "game_pk":          matchup["gamePk"],
                        "lineup_confirmed": matchup["lineup_confirmed"],
                    })
                    all_predictions.append(pred)

                except Exception as exc:
                    log.warning("Prediction failed for player %d: %s", player_id, exc)

    # ── Rank and trim ─────────────────────────────────────────────────────────
    all_predictions.sort(key=lambda x: x["hr_probability"], reverse=True)
    top_predictions = all_predictions[:top_n]

    log.info("Top %d from %d candidates:", top_n, len(all_predictions))
    for i, p in enumerate(top_predictions, 1):
        src = p.get("rate_source", "?")
        log.info(
            "  %2d. %-25s %s  conf=%s  src=%s",
            i, p["player_name"], p["hr_pct"], p["confidence_tier"], src,
        )

    # ── Email ─────────────────────────────────────────────────────────────────
    if not dry_run:
        now_utc    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        email_cfg  = config.get("email", {})
        subject_t  = email_cfg.get("subject", "⚾ Top {n} HR Predictions — {date}")
        sent = fn["send_email"](
            predictions=top_predictions,
            prediction_date=game_date.strftime("%A, %B %-d, %Y"),
            games_today=len(matchups),
            lineups_confirmed=lineups_confirmed,
            generated_at=now_utc,
            subject_template=subject_t,
        )
        if sent:
            log.info("✅ Email delivered successfully.")
        else:
            log.error("❌ Email delivery failed.")
            sys.exit(1)
    else:
        log.info("Dry run — email not sent.")

    return top_predictions


def parse_args():
    p = argparse.ArgumentParser(description="MLB HR Prediction Pipeline")
    p.add_argument("--config",      default="config.yml")
    p.add_argument("--date",        help="YYYY-MM-DD or 'today'")
    p.add_argument("--top-n",       type=int)
    p.add_argument("--dry-run",     action="store_true")
    p.add_argument("--test-email",  action="store_true")
    p.add_argument("--retrain",     action="store_true", help="Retrain model before predicting")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = load_config(args.config)

    if args.date:
        config.setdefault("prediction", {})["date"] = args.date
    if args.top_n:
        config.setdefault("prediction", {})["top_n"] = args.top_n

    if args.retrain:
        log.info("── Retraining model before predictions ──")
        from src.models.train import train
        train()

    dry = args.dry_run and not args.test_email
    run(config, dry_run=dry)
