#!/usr/bin/env python3
"""
main.py — MLB HR Prediction Pipeline

Performance model
-----------------
Serial fetching for 270 batters × 4 API calls × 0.12 s sleep ≈ 11 min.
Parallel fetching with ThreadPoolExecutor(20 workers)             ≈ 1 min.

The pipeline runs in two phases:
  Phase 1 — Parallel data fetch
    Collect all unique player / pitcher IDs across every matchup,
    fetch each player's stats, splits, game log, and Statcast metrics
    concurrently, store in an in-memory cache.

  Phase 2 — Serial prediction
    Iterate matchups using only the cache — zero additional API calls.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
log = logging.getLogger("main")

# MLB API is I/O-bound and permissive — more workers = faster.
# Statcast leaderboard is preloaded and cached — lookup is O(1), no pool needed.
MLB_API_WORKERS = 30


def _import_pipeline():
    from src.data.mlb_client import (
        extract_matchups, get_all_batter_data, get_players_info_batch,
        get_pitcher_stats, get_roster, get_schedule,
    )
    from src.data.statcast_client import (
        get_batter_statcast_metrics, preload_leaderboards,
    )
    from src.data.weather_client import get_game_weather
    from src.models.predictor import ensemble_predict
    from src.notifications.email_sender import send_email
    from src.models.model_registry import load as load_model

    return {
        "get_schedule":                get_schedule,
        "extract_matchups":            extract_matchups,
        "get_roster":                  get_roster,
        "get_pitcher_stats":           get_pitcher_stats,
        "get_all_batter_data":         get_all_batter_data,
        "get_players_info_batch":      get_players_info_batch,
        "get_batter_statcast_metrics": get_batter_statcast_metrics,
        "preload_leaderboards":        preload_leaderboards,
        "get_game_weather":            get_game_weather,
        "ensemble_predict":            ensemble_predict,
        "send_email":                  send_email,
        "load_model":                  load_model,
    }


def load_config(path: str = "config.yml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("TOP_N"):
        n = int(os.environ["TOP_N"])
        for key in ("top_n", "top_n_hr", "top_n_tb", "top_n_hrbi", "top_n_rbi"):
            cfg.setdefault("prediction", {})[key] = n
    if os.environ.get("PREDICT_DATE"):
        cfg.setdefault("prediction", {})["date"] = os.environ["PREDICT_DATE"]
    return cfg


def _fetch_batter_bundle(fn, player_id: int, season: int, recent_n: int) -> dict:
    """
    ONE API call per player (was four).
    get_all_batter_data combines season + career + splits + game_log.
    """
    return fn["get_all_batter_data"](player_id, season, recent_n)


def _fetch_pitcher_bundle(fn, pitcher_id: int, season: int) -> dict:
    try:
        return fn["get_pitcher_stats"](pitcher_id, season)
    except Exception as exc:
        log.debug("Pitcher stats failed %d: %s", pitcher_id, exc)
        return {"season_stats": {}, "career_stats": {}}


def prefetch_all(
    fn,
    matchups: list[dict],
    season: int,
    recent_n: int,
) -> tuple[dict, dict, dict, dict]:
    """
    Phase 1: fetch all player and pitcher data in parallel.

    Returns
    -------
    batter_cache   : {player_id: {info, stats, splits, recent}}
    statcast_cache : {player_id: metrics_dict}
    pitcher_cache  : {pitcher_id: {season_stats, career_stats}}
    weather_cache  : {venue: weather_dict}
    """
    # Collect unique IDs
    all_batter_ids : set[int] = set()
    all_pitcher_ids: set[int] = set()
    all_venues     : set[str] = set()

    for m in matchups:
        all_venues.add(m["venue"])
        for side in ("home", "away"):
            opp = "away" if side == "home" else "home"
            p   = m[f"{opp}_pitcher"]
            if p:
                all_pitcher_ids.add(p["id"])
            for pid in m[f"{side}_lineup"]:
                all_batter_ids.add(pid)

    log.info(
        "Prefetching: %d batters, %d pitchers, %d venues …",
        len(all_batter_ids), len(all_pitcher_ids), len(all_venues),
    )

    # ── Preload Savant leaderboards on main thread (eliminates race) ──────
    # All worker threads then get O(1) cache hits — no network calls.
    fn["preload_leaderboards"](season)

    # ── Weather ───────────────────────────────────────────────────────────
    weather_cache: dict[str, dict] = {}
    for venue in all_venues:
        weather_cache[venue] = fn["get_game_weather"](venue)

    # ── Batch player info (1 call per 150 players, not 1 per player) ──────
    batter_info_cache: dict[int, dict] = fn["get_players_info_batch"](list(all_batter_ids))
    log.info("Player info batch fetched: %d players", len(batter_info_cache))

    # ── Pitchers + batters in parallel (both use same MLB API pool) ───────
    pitcher_cache: dict[int, dict] = {}
    batter_data_cache: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=MLB_API_WORKERS) as pool:
        # Submit pitchers
        pitcher_futures = {
            pool.submit(_fetch_pitcher_bundle, fn, pid, season): pid
            for pid in all_pitcher_ids
        }
        # Submit batters at the same time (4→1 API call each)
        batter_futures = {
            pool.submit(_fetch_batter_bundle, fn, pid, season, recent_n): pid
            for pid in all_batter_ids
        }
        all_futures = {**pitcher_futures, **batter_futures}
        pitcher_set = set(pitcher_futures.values())

        done = 0
        for fut in as_completed(all_futures):
            pid = all_futures[fut]
            try:
                result = fut.result()
                if pid in pitcher_set:
                    pitcher_cache[pid] = result
                else:
                    batter_data_cache[pid] = result
            except Exception as exc:
                if pid in pitcher_set:
                    log.warning("Pitcher prefetch failed %d: %s", pid, exc)
                    pitcher_cache[pid] = {"season_stats": {}, "career_stats": {}}
                else:
                    log.warning("Batter prefetch failed %d: %s", pid, exc)
                    batter_data_cache[pid] = {
                        "stats": {"season_stats": {}, "career_stats": {}},
                        "splits": {}, "recent": [],
                    }
            done += 1
            if done % 50 == 0:
                log.info("  MLB API: %d/%d total requests …", done, len(all_futures))

    log.info("Pitchers: %d | Batters: %d", len(pitcher_cache), len(batter_data_cache))

    # ── Merge batter info + batter data into unified cache ────────────────
    batter_cache: dict[int, dict] = {}
    for pid in all_batter_ids:
        info = batter_info_cache.get(pid, {})
        data = batter_data_cache.get(pid, {
            "stats": {"season_stats": {}, "career_stats": {}},
            "splits": {}, "recent": [],
        })
        batter_cache[pid] = {"info": info, **data}

    # ── Statcast — pure cache lookup, no network (leaderboard preloaded) ──
    statcast_cache: dict[int, dict] = {}
    for pid in all_batter_ids:
        statcast_cache[pid] = fn["get_batter_statcast_metrics"](pid, season)
    log.info("Statcast metrics: %d players (from cache)", len(statcast_cache))

    return batter_cache, statcast_cache, pitcher_cache, weather_cache


# ---------------------------------------------------------------------------
# Roster fallback (when lineup not announced)
# ---------------------------------------------------------------------------

def _probable_lineup(fn, team_id: int, season: int, min_games: int) -> list[int]:
    try:
        roster = fn["get_roster"](team_id)
    except Exception as exc:
        log.warning("Roster fetch failed team %d: %s", team_id, exc)
        return []
    players = []
    for entry in roster:
        if entry.get("position", {}).get("abbreviation", "") in ("SP", "RP", "P"):
            continue
        pid = entry.get("person", {}).get("id")
        if not pid:
            continue
        players.append(pid)
    return players


# ---------------------------------------------------------------------------
# Date resolution
# ---------------------------------------------------------------------------

def resolve_game_date(raw_date: str, fn: dict) -> date:
    if raw_date != "today":
        from dateutil.parser import parse as dparse
        return dparse(raw_date).date()

    today    = date.today()
    tomorrow = today + timedelta(days=1)

    games_today = fn["get_schedule"](today)
    matchups    = fn["extract_matchups"](games_today, skip_final=True)

    if matchups:
        return today

    if games_today:
        log.warning(
            "All %d game(s) on %s are already final — switching to %s.",
            len(games_today), today, tomorrow,
        )
        return tomorrow

    log.info("No games on %s — trying %s.", today, tomorrow)
    return tomorrow


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config: dict, dry_run: bool = False) -> list[dict]:
    fn        = _import_pipeline()
    pred_cfg  = config.get("prediction", {})
    model_cfg = config.get("model", {})

    # ── Model ─────────────────────────────────────────────────────────────
    xgb_models, xgb_metas = fn["load_model"]()
    loaded = [n for n, m in xgb_models.items() if m is not None]
    if loaded:
        log.info("✅ XGBoost models loaded: %s", loaded)
        for name in loaded:
            log.info("   %s — R²=%.4f, n=%d",
                     name, xgb_metas[name].get("cv_r2", 0),
                     xgb_metas[name].get("n_training", 0))
    else:
        log.info("ℹ️  No trained models — using statistical models only.")

    # ── Date ──────────────────────────────────────────────────────────────
    game_date = resolve_game_date(pred_cfg.get("date", "today"), fn)
    season    = game_date.year
    top_n     = int(pred_cfg.get("top_n", 10))   # legacy fallback
    top_n_hr   = int(pred_cfg.get("top_n_hr",   top_n))
    top_n_tb   = int(pred_cfg.get("top_n_tb",   top_n))
    top_n_hrbi = int(pred_cfg.get("top_n_hrbi", top_n))
    top_n_rbi  = int(pred_cfg.get("top_n_rbi",  top_n))
    min_pa    = int(pred_cfg.get("min_pa_season", 30))
    min_games = int(pred_cfg.get("min_games_played", 10))
    recent_n  = int(model_cfg.get("recent_form_games", 15))

    log.info("── Predictions for %s (top %d) ──", game_date, top_n)

    # ── Schedule ──────────────────────────────────────────────────────────
    games    = fn["get_schedule"](game_date)
    matchups = fn["extract_matchups"](games)

    if not matchups:
        log.error("No actionable games on %s.", game_date)
        return []

    # Fill in any missing lineups via roster before prefetch
    for m in matchups:
        for side in ("home", "away"):
            if not m[f"{side}_lineup"]:
                m[f"{side}_lineup"] = _probable_lineup(
                    fn, m[f"{side}_team_id"], season, min_games
                )

    # ── Phase 1: parallel data fetch ──────────────────────────────────────
    batter_cache, statcast_cache, pitcher_cache, weather_cache = prefetch_all(
        fn, matchups, season, recent_n
    )

    # ── Phase 2: predictions (cache only — no more API calls) ─────────────
    all_predictions: list[dict] = []
    lineups_confirmed = sum(1 for m in matchups if m["lineup_confirmed"])

    for matchup in matchups:
        venue   = matchup["venue"]
        weather = weather_cache.get(venue, {})

        for side in ("home", "away"):
            opp          = "away" if side == "home" else "home"
            is_home      = side == "home"
            pitcher_info = matchup[f"{opp}_pitcher"]
            team_name    = matchup[f"{side}_team"]

            if pitcher_info is None:
                continue

            p_data   = pitcher_cache.get(pitcher_info["id"], {})
            p_season = p_data.get("season_stats", {})
            p_career = p_data.get("career_stats", {})

            for player_id in matchup[f"{side}_lineup"]:
                b_data = batter_cache.get(player_id, {})
                info   = b_data.get("info", {})

                if info.get("position") in ("P", "SP", "RP"):
                    continue

                b_stats  = b_data.get("stats", {})
                b_season = b_stats.get("season_stats", {})
                b_career = b_stats.get("career_stats", {})

                if int(b_season.get("plateAppearances", 0)) < min_pa:
                    continue

                try:
                    pred = fn["ensemble_predict"](
                        batter_season=b_season,
                        batter_career=b_career,
                        pitcher_season=p_season,
                        pitcher_career=p_career,
                        pitcher_throws=pitcher_info.get("throws", "R"),
                        batter_bats=info.get("bats", "R"),
                        platoon_splits=b_data.get("splits", {}),
                        recent_games=b_data.get("recent", []),
                        statcast_metrics=statcast_cache.get(player_id, {}),
                        venue=venue,
                        weather=weather,
                        config=config,
                        is_home=is_home,
                        xgb_models=xgb_models,
                        xgb_metas=xgb_metas,
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
                        "is_home":          is_home,
                        "game_pk":          matchup["gamePk"],
                        "lineup_confirmed": matchup["lineup_confirmed"],
                    })
                    all_predictions.append(pred)
                except Exception as exc:
                    log.warning("Prediction failed player %d: %s", player_id, exc)

    # Sort into four independent ranked lists
    hr_preds   = sorted(all_predictions, key=lambda x: x["hr_probability"],       reverse=True)[:top_n_hr]
    tb_preds   = sorted(all_predictions, key=lambda x: x.get("expected_tb",   0), reverse=True)[:top_n_tb]
    hrbi_preds = sorted(all_predictions, key=lambda x: x.get("expected_hrbi", 0), reverse=True)[:top_n_hrbi]
    rbi_preds  = sorted(all_predictions, key=lambda x: x.get("expected_rbi",  0), reverse=True)[:top_n_rbi]

    log.info("HR %d | TB %d | H+R+RBI %d | RBI %d (from %d candidates)",
             len(hr_preds), len(tb_preds), len(hrbi_preds), len(rbi_preds), len(all_predictions))

    log.info("── 💥 HR ──")
    for i, p in enumerate(hr_preds, 1):
        log.info("  %2d. %-24s hr=%-6s tb=%-4s hrbi=%-4s rbi=%s",
                 i, p["player_name"], p["hr_pct"],
                 p.get("expected_tb","?"), p.get("expected_hrbi","?"), p.get("expected_rbi","?"))

    log.info("── 📊 Total Bases ──")
    for i, p in enumerate(tb_preds, 1):
        log.info("  %2d. %-24s tb=%-4s hr=%s",
                 i, p["player_name"], p.get("expected_tb","?"), p["hr_pct"])

    log.info("── ⭐ H+R+RBI ──")
    for i, p in enumerate(hrbi_preds, 1):
        log.info("  %2d. %-24s hrbi=%-4s h=%-4s r=%-4s rbi=%s",
                 i, p["player_name"], p.get("expected_hrbi","?"),
                 p.get("expected_h","?"), p.get("expected_r","?"), p.get("expected_rbi_hrbi","?"))

    log.info("── 🎯 RBI ──")
    for i, p in enumerate(rbi_preds, 1):
        log.info("  %2d. %-24s rbi=%-4s hrbi=%s",
                 i, p["player_name"], p.get("expected_rbi","?"), p.get("expected_hrbi","?"))

    # ── Email ─────────────────────────────────────────────────────────────
    if not dry_run:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        sent = fn["send_email"](
            hr_preds=hr_preds, tb_preds=tb_preds,
            hrbi_preds=hrbi_preds, rbi_preds=rbi_preds,
            date_str=game_date.strftime("%A, %B %-d, %Y"),
            games=len(matchups),
            lineups_confirmed=lineups_confirmed,
            ts=now_utc,
            subject_template=config.get("email", {}).get(
                "subject", "⚾ MLB Predictions — {date}"
            ),
        )
        if sent:
            log.info("✅ Email delivered.")
        else:
            log.error("❌ Email delivery failed.")
            sys.exit(1)
    else:
        log.info("Dry run — email not sent.")

    return {"hr": hr_preds, "tb": tb_preds, "hrbi": hrbi_preds, "rbi": rbi_preds}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MLB HR Prediction Pipeline")
    p.add_argument("--config",     default="config.yml")
    p.add_argument("--date",       help="YYYY-MM-DD or 'today'")
    p.add_argument("--top-n",      type=int)
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--test-email", action="store_true")
    p.add_argument("--retrain",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = load_config(args.config)
    if args.date:
        config.setdefault("prediction", {})["date"] = args.date
    if args.top_n:
        for key in ("top_n", "top_n_hr", "top_n_tb", "top_n_hrbi", "top_n_rbi"):
            config.setdefault("prediction", {})[key] = args.top_n
    if args.retrain:
        from src.models.train import train
        train()
    run(config, dry_run=args.dry_run and not args.test_email)
