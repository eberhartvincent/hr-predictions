"""
train.py
========
Weekly retraining for the XGBoost HR-rate talent estimator.

Three data sources, all free, all confirmed working from GitHub Actions:

  Source 1 — Baseball Savant barrels leaderboard (CSV)
    Confirmed columns: player_id, attempts, avg_hit_angle,
    anglesweetspotpercent, avg_hit_speed, ev95percent,
    barrels, brl_percent, brl_pa

  Source 2 — Baseball Savant expected-stats leaderboard (CSV)
    Confirmed columns: player_id, year, pa, bip, ba, slg,
    woba, est_woba (xwOBA)
    NOTE: no home_run column — HR comes from source 3.

  Source 3 — MLB Stats API  (already used by daily predictions)
    One GET per season — returns HR, PA, K%, BB%, ISO per player.
    player_id is MLBAM ID — matches Savant exactly.

Training: Year-N features → Year-(N+1) HR/PA rate (no look-ahead bias).
"""
from __future__ import annotations

import io
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "hr_model.json"
META_PATH  = MODEL_DIR / "feature_metadata.json"

MIN_PA = 150
SAVANT_HEADERS = {"User-Agent": "hr-predictor/1.0 (github-actions; open-source)"}
MLB_HEADERS    = {"User-Agent": "hr-predictor/1.0 (github-actions)"}

# ---------------------------------------------------------------------------
# Confirmed column names from live API responses (2025-05-11)
# ---------------------------------------------------------------------------

# Source 1: barrels leaderboard
BARRELS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast"
    "?type=batter&year={year}&position=&team=&min=50&csv=true"
)
BARRELS_FEATURE_MAP = {
    # canonical name      actual CSV column
    "barrel_pct":         "brl_percent",
    "barrel_pa":          "brl_pa",
    "exit_velocity":      "avg_hit_speed",
    "launch_angle":       "avg_hit_angle",
    "sweet_spot_pct":     "anglesweetspotpercent",
    "hard_hit_pct":       "ev95percent",
    "n_batted_balls":     "attempts",
}

# Source 2: expected-stats leaderboard
XSTATS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=batter&year={year}&position=&team=&min=50&csv=true"
)
XSTATS_FEATURE_MAP = {
    "xwoba":  "est_woba",   # expected wOBA — strong HR-talent signal
    "woba":   "woba",       # actual wOBA   — useful cross-check
    "xslg":   "est_slg",    # expected SLG
}

# Source 3: MLB Stats API
MLB_STATS_URL = "https://statsapi.mlb.com/api/v1/stats"

# Final XGBoost features (subset that's consistently available + HR-predictive)
FEATURES = [
    "barrel_pct",     # barrels / batted balls — #1 HR predictor
    "barrel_pa",      # barrels / PA — tighter signal
    "exit_velocity",  # raw power
    "hard_hit_pct",   # EV ≥ 95 mph %
    "launch_angle",   # avg trajectory
    "sweet_spot_pct", # optimal LA zone
    "xwoba",          # expected wOBA (absorbs EV + LA + contact type)
    "iso",            # isolated power (traditional power gauge)
    "k_pct",          # strikeout rate (non-linear HR correlation)
    "bb_pct",         # walk rate (pitch recognition)
]


# ---------------------------------------------------------------------------
# Source 1 — Baseball Savant barrels leaderboard
# ---------------------------------------------------------------------------

def fetch_barrels(year: int) -> pd.DataFrame:
    url = BARRELS_URL.format(year=year)
    try:
        r = requests.get(url, headers=SAVANT_HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        log.info("barrels/%d: %d rows", year, len(df))
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
        out = df[["player_id"]].copy()
        for canonical, col in BARRELS_FEATURE_MAP.items():
            out[canonical] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
        return out.dropna(subset=["player_id"])
    except Exception as exc:
        log.warning("barrels/%d failed: %s", year, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Source 2 — Baseball Savant expected-stats leaderboard
# ---------------------------------------------------------------------------

def fetch_xstats(year: int) -> pd.DataFrame:
    url = XSTATS_URL.format(year=year)
    try:
        r = requests.get(url, headers=SAVANT_HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        log.info("xstats/%d: %d rows, cols: %s", year, len(df), df.columns.tolist())
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
        out = df[["player_id"]].copy()
        # pa is in xstats — useful for the merge filter
        out["pa_xstats"] = pd.to_numeric(df.get("pa", np.nan), errors="coerce")
        for canonical, col in XSTATS_FEATURE_MAP.items():
            out[canonical] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
        return out.dropna(subset=["player_id"])
    except Exception as exc:
        log.warning("xstats/%d failed: %s", year, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Source 3 — MLB Stats API (one call per year)
# ---------------------------------------------------------------------------

def fetch_mlb_stats(year: int) -> pd.DataFrame:
    """
    Pull full-season hitting stats from the MLB Stats API.
    Returns player_id, hr, pa, k_pct, bb_pct, iso per qualified batter.
    """
    try:
        r = requests.get(
            MLB_STATS_URL,
            params={"stats": "season", "group": "hitting",
                    "season": year, "sportId": 1, "limit": 2000},
            headers=MLB_HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning("MLB API stats/%d failed: %s", year, exc)
        return pd.DataFrame()

    rows = []
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            player = split.get("player", {})
            s      = split.get("stat", {})
            pid = player.get("id")
            if not pid:
                continue

            pa  = float(s.get("plateAppearances") or 0)
            hr  = float(s.get("homeRuns")         or 0)
            so  = float(s.get("strikeOuts")        or 0)
            bb  = float(s.get("baseOnBalls")       or 0)
            slg_str = s.get("slugging", "0") or "0"
            avg_str = s.get("avg", "0")      or "0"

            try: slg = float(slg_str)
            except ValueError: slg = 0.0
            try: avg = float(avg_str)
            except ValueError: avg = 0.0

            k_pct = so / pa if pa > 0 else np.nan
            bb_pct = bb / pa if pa > 0 else np.nan
            iso   = slg - avg

            rows.append({
                "player_id": int(pid),
                "pa":        pa,
                "hr":        hr,
                "k_pct":     k_pct,
                "bb_pct":    bb_pct,
                "iso":       iso,
            })

    df = pd.DataFrame(rows)
    log.info("mlb_api/%d: %d batters", year, len(df))
    return df


# ---------------------------------------------------------------------------
# Assemble one season
# ---------------------------------------------------------------------------

def fetch_year(year: int) -> pd.DataFrame | None:
    brl  = fetch_barrels(year)
    xs   = fetch_xstats(year)
    mlb  = fetch_mlb_stats(year)

    if mlb.empty:
        log.warning("Year %d skipped — MLB API returned no data.", year)
        return None

    # MLB API is the authoritative source for HR + PA — start here
    base = mlb[mlb["pa"] >= MIN_PA].copy()
    base["season"] = year
    base["hr_pa"]  = base["hr"] / base["pa"]

    # Merge Savant barrels
    if not brl.empty:
        base = base.merge(brl, on="player_id", how="left")
    else:
        for col in BARRELS_FEATURE_MAP:
            base[col] = np.nan

    # Merge Savant xstats (xwOBA, wOBA, xSLG)
    if not xs.empty:
        base = base.merge(xs.drop(columns=["pa_xstats"], errors="ignore"),
                          on="player_id", how="left")
    else:
        for col in XSTATS_FEATURE_MAP:
            base[col] = np.nan

    log.info("Year %d: %d qualifying batters assembled.", year, len(base))
    return base


# ---------------------------------------------------------------------------
# Training pairs: Year-N → Year-(N+1) HR/PA
# ---------------------------------------------------------------------------

def build_training_pairs(
    stats: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    seasons = sorted(stats["season"].unique())
    all_X, all_y = [], []

    for i in range(len(seasons) - 1):
        yr_n, yr_np1 = seasons[i], seasons[i + 1]
        n   = stats[stats["season"] == yr_n].set_index("player_id")
        np1 = stats[stats["season"] == yr_np1].set_index("player_id")
        common = n.index.intersection(np1.index)

        if common.empty:
            log.warning("No common players %d→%d.", yr_n, yr_np1)
            continue

        avail = [f for f in features if f in n.columns]
        X_b = n.loc[common, avail].copy()
        y_b = np1.loc[common, "hr_pa"].copy()
        valid = X_b.notna().any(axis=1) & y_b.notna()

        all_X.append(X_b[valid])
        all_y.append(y_b[valid])
        log.info("Pair %d→%d: %d examples", yr_n, yr_np1, valid.sum())

    if not all_X:
        raise RuntimeError("No valid year-over-year training pairs built.")

    return (
        pd.concat(all_X).reset_index(drop=True),
        pd.concat(all_y).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train(years: list[int] | None = None) -> dict:
    import xgboost as xgb

    if years is None:
        years = [2022, 2023, 2024, 2025]

    MODEL_DIR.mkdir(exist_ok=True)

    frames = []
    for year in years:
        df = fetch_year(year)
        if df is not None:
            frames.append(df)
        time.sleep(0.5)   # be polite between years

    if not frames:
        raise RuntimeError("No data assembled — check network and API availability.")

    stats = pd.concat(frames, ignore_index=True)
    log.info("Total player-seasons: %d", len(stats))

    available = [f for f in FEATURES if f in stats.columns
                 and stats[f].notna().sum() >= 20]
    if len(available) < 2:
        raise RuntimeError(f"Too few usable features: {available}")
    log.info("Training on %d features: %s", len(available), available)

    X, y = build_training_pairs(stats, available)
    log.info("Training set: %d × %d", len(X), X.shape[1])

    medians = X.median()
    X = X.fillna(medians)

    n_splits = min(5, max(2, len(X) // 20))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    maes = []

    xgb_params = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=2.0, reg_alpha=0.5,
        objective="reg:squarederror",
        random_state=42, n_jobs=-1, verbosity=0,
    )

    for fold, (tr, val) in enumerate(kf.split(X), 1):
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = np.clip(m.predict(X.iloc[val]), 0, 0.15)
        oof[val] = p
        mae = mean_absolute_error(y.iloc[val], p)
        maes.append(mae)
        log.info("  Fold %d MAE: %.5f", fold, mae)

    cv_mae = float(np.mean(maes))
    cv_r2  = float(r2_score(y, oof))
    log.info("CV MAE: %.5f | CV R²: %.4f", cv_mae, cv_r2)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, y)
    model.save_model(str(MODEL_PATH))

    importance = dict(sorted(
        zip(available, model.feature_importances_.tolist()),
        key=lambda kv: -kv[1],
    ))

    metadata = {
        "features":           available,
        "feature_medians":    medians[available].to_dict(),
        "lg_hr_pa":           float(y.mean()),
        "training_years":     years,
        "n_training":         len(X),
        "cv_mae":             cv_mae,
        "cv_r2":              cv_r2,
        "feature_importance": importance,
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))

    log.info("=" * 50)
    log.info("Training complete.")
    log.info("  Samples  : %d", len(X))
    log.info("  CV MAE   : %.5f HR/PA", cv_mae)
    log.info("  CV R²    : %.4f", cv_r2)
    log.info("  LG HR/PA : %.4f", float(y.mean()))
    log.info("  Top features: %s", list(importance.keys())[:5])
    log.info("=" * 50)
    return metadata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    train()
