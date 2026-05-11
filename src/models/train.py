"""
train.py
========
Weekly retraining script for the XGBoost HR-rate talent estimator.

Data source
-----------
Baseball Savant public Statcast leaderboard CSVs (baseballsavant.mlb.com).
These work from GitHub Actions — unlike FanGraphs, which blocks cloud IPs.
No API key required.

Approach
--------
Year-N Statcast/contact features  →  Year-(N+1) HR/PA rate

No look-ahead bias. XGBoost learns the non-linear mapping from contact
quality → true HR talent better than hand-crafted weights.

Training years 2022→2023, 2023→2024, 2024→2025 ≈ 600 examples.

Output
------
  models/hr_model.json          XGBoost model (JSON — version-controlled)
  models/feature_metadata.json  Column names, league averages, eval metrics
"""
from __future__ import annotations

import io
import json
import logging
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

# ---------------------------------------------------------------------------
# Baseball Savant leaderboard URL
# min= accepts an integer (PA threshold) or "q" for qualified.
# Returns CSV with one row per batter.
# ---------------------------------------------------------------------------
SAVANT_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast"
    "?type=batter&year={year}&position=&team=&min={min_pa}&csv=true"
)

HEADERS = {"User-Agent": "hr-predictor/1.0 (github-actions; open-source)"}

# ---------------------------------------------------------------------------
# Canonical feature name → possible Baseball Savant CSV column names.
# We try each in order and use the first one found.
# ---------------------------------------------------------------------------
FEATURE_CANDIDATES: dict[str, list[str]] = {
    "barrel_pct":     ["brl_percent", "barrel_batted_rate", "brl_pa"],
    "exit_velocity":  ["exit_velocity_avg", "avg_exit_velocity", "launch_speed"],
    "hard_hit_pct":   ["hard_hit_percent", "hard_hit_pct"],
    "launch_angle":   ["launch_angle_avg", "avg_launch_angle"],   # proxy for FB rate
    "sweet_spot_pct": ["sweet_spot_percent", "sweet_spot_pct"],
    "xwoba":          ["est_woba", "xwoba", "expected_woba"],     # xwOBA — strong HR predictor
    "iso":            ["isolated_power", "iso"],
    "k_pct":          ["k_percent", "strikeout_percent", "k%"],
    "bb_pct":         ["bb_percent", "walk_percent", "bb%"],
}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _resolve_columns(df: pd.DataFrame, candidates: dict[str, list[str]]) -> dict[str, str]:
    """Return {canonical: actual_col} for columns present in df."""
    mapping = {}
    for canonical, options in candidates.items():
        for col in options:
            if col in df.columns:
                mapping[canonical] = col
                break
        if canonical not in mapping:
            log.debug("Feature %r not found (tried: %s)", canonical, options)
    return mapping


def fetch_savant_stats(year: int, min_pa: int = MIN_PA) -> pd.DataFrame | None:
    """
    Download one year of Statcast batting leaderboard from Baseball Savant.
    Returns a DataFrame or None on failure.
    """
    url = SAVANT_URL.format(year=year, min_pa=min_pa)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty:
            log.warning("Savant returned empty CSV for %d.", year)
            return None
        log.info("Fetched %d batters from Baseball Savant for %d.", len(df), year)
        log.debug("Columns: %s", df.columns.tolist())
        return df
    except Exception as exc:
        log.warning("Failed to fetch Savant data for %d: %s", year, exc)
        return None


def build_season_frame(df: pd.DataFrame, year: int, col_map: dict[str, str]) -> pd.DataFrame:
    """
    Convert a raw Savant CSV into a canonical feature DataFrame
    with columns: player_id, season, pa, hr, hr_pa, + feature columns.
    """
    # Identify the player ID and counting-stat columns
    id_col = next((c for c in ["player_id", "IDfg", "mlbam_id"] if c in df.columns), None)
    pa_col = next((c for c in ["pa", "PA"] if c in df.columns), None)
    hr_col = next((c for c in ["home_run", "HR", "hr"] if c in df.columns), None)

    if id_col is None or pa_col is None or hr_col is None:
        log.warning(
            "Year %d: missing id/pa/hr columns. Available: %s",
            year, df.columns.tolist(),
        )
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        pa = float(row.get(pa_col, 0) or 0)
        hr = float(row.get(hr_col, 0) or 0)
        if pa < MIN_PA:
            continue

        record: dict = {
            "player_id": int(row[id_col]),
            "season":    year,
            "pa":        pa,
            "hr":        hr,
            "hr_pa":     hr / pa,
        }
        for canonical, actual in col_map.items():
            val = row.get(actual, np.nan)
            try:
                record[canonical] = float(val)
            except (TypeError, ValueError):
                record[canonical] = np.nan

        rows.append(record)

    return pd.DataFrame(rows)


def fetch_all_seasons(years: list[int]) -> pd.DataFrame:
    """Fetch and combine Savant data for all requested years."""
    frames = []
    col_map: dict[str, str] = {}

    for year in years:
        df_raw = fetch_savant_stats(year)
        if df_raw is None:
            continue

        # Resolve column mapping from first successful year
        if not col_map:
            col_map = _resolve_columns(df_raw, FEATURE_CANDIDATES)
            log.info("Resolved feature columns: %s", col_map)

        df = build_season_frame(df_raw, year, col_map)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError(
            "No batting data fetched from Baseball Savant. "
            "Check network access to baseballsavant.mlb.com."
        )

    combined = pd.concat(frames, ignore_index=True)
    log.info("Total player-seasons: %d across years %s", len(combined), years)
    return combined, col_map


# ---------------------------------------------------------------------------
# Training pair construction
# ---------------------------------------------------------------------------

def build_training_pairs(
    stats: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) pairs: year-N features → year-(N+1) HR/PA rate.
    Matched on player_id across consecutive seasons.
    """
    seasons = sorted(stats["season"].unique())
    all_X, all_y = [], []

    for i in range(len(seasons) - 1):
        yr_n   = seasons[i]
        yr_np1 = seasons[i + 1]

        df_n   = stats[stats["season"] == yr_n].set_index("player_id")
        df_np1 = stats[stats["season"] == yr_np1].set_index("player_id")

        common = df_n.index.intersection(df_np1.index)
        if common.empty:
            log.warning("No common players between %d and %d.", yr_n, yr_np1)
            continue

        X_block = df_n.loc[common, feature_cols].copy()
        y_block = df_np1.loc[common, "hr_pa"].copy()

        valid = X_block.notna().any(axis=1) & y_block.notna()
        all_X.append(X_block[valid])
        all_y.append(y_block[valid])
        log.info("Pair %d→%d: %d training examples", yr_n, yr_np1, valid.sum())

    if not all_X:
        raise RuntimeError(
            "No valid year-over-year training pairs found. "
            "Need at least two consecutive seasons with overlapping players."
        )

    X = pd.concat(all_X).reset_index(drop=True)
    y = pd.concat(all_y).reset_index(drop=True)
    return X, y


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train(years: list[int] | None = None) -> dict:
    """Full training pipeline. Returns evaluation metadata dict."""
    import xgboost as xgb

    if years is None:
        years = [2022, 2023, 2024, 2025]

    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Fetch
    log.info("Fetching Statcast leaderboards from Baseball Savant for %s …", years)
    stats, col_map = fetch_all_seasons(years)

    # 2. Determine available features
    available_features = [
        col for col in FEATURE_CANDIDATES.keys()
        if col in stats.columns and stats[col].notna().sum() > 10
    ]
    if len(available_features) < 2:
        raise RuntimeError(
            f"Too few usable features ({available_features}). "
            "Check that Baseball Savant columns resolved correctly."
        )
    log.info("Training on %d features: %s", len(available_features), available_features)

    # 3. Training pairs
    X, y = build_training_pairs(stats, available_features)
    log.info("Training set: %d examples × %d features", len(X), X.shape[1])

    # 4. Impute with column medians
    medians = X.median()
    X = X.fillna(medians)

    # 5. Cross-validated evaluation
    kf = KFold(n_splits=min(5, len(X) // 20 or 2), shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    fold_maes = []

    xgb_params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = np.clip(m.predict(X.iloc[val_idx]), 0, 0.15)
        oof_preds[val_idx] = preds
        mae = mean_absolute_error(y.iloc[val_idx], preds)
        fold_maes.append(mae)
        log.info("  Fold %d MAE: %.5f", fold, mae)

    cv_mae = float(np.mean(fold_maes))
    cv_r2  = float(r2_score(y, oof_preds))
    log.info("CV MAE: %.5f | CV R²: %.4f", cv_mae, cv_r2)

    # 6. Final model on all data
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, y)

    importance = dict(sorted(
        zip(available_features, model.feature_importances_.tolist()),
        key=lambda kv: -kv[1],
    ))
    log.info("Feature importances: %s", importance)

    # 7. Save
    model.save_model(str(MODEL_PATH))

    metadata = {
        "features":           available_features,
        "feature_medians":    medians[available_features].to_dict(),
        "savant_col_map":     col_map,
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
    log.info("  Features : %s", available_features)
    log.info("  CV MAE   : %.5f HR/PA", cv_mae)
    log.info("  CV R²    : %.4f", cv_r2)
    log.info("  LG HR/PA : %.4f", float(y.mean()))
    log.info("=" * 50)

    return metadata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    train()
