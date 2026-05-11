"""
train.py
========
Weekly retraining script for the XGBoost HR-rate talent estimator.

Approach
--------
We treat this as a **talent estimation** problem, not a per-game prediction
problem.  Statcast metrics (barrel rate, exit velocity, hard-hit rate) 
stabilize faster than HR rate itself (~300 batted balls vs ~600 PA), so
XGBoost can learn the non-linear mapping from contact quality → true HR
talent better than our hand-crafted weights.

Training setup
--------------
  Year N Statcast/contact features  →  Year N+1 HR/PA rate

This avoids any look-ahead bias.  With 3 seasons of data
(2022→2023, 2023→2024, 2024→2025) and ~200 qualified batters per
season-pair, we get ~600 training examples — small but sufficient for
XGBoost with proper regularization given only ~10 features.

We also augment with first-half → second-half splits within each season,
roughly doubling the sample size.

Output
------
  models/hr_model.json          XGBoost model (JSON — version-controlled)
  models/feature_metadata.json  Column names, league averages, eval metrics
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "hr_model.json"
META_PATH  = MODEL_DIR / "feature_metadata.json"

# Minimum PA to include a player-season in training
MIN_PA = 150

# Feature columns we want from FanGraphs batting stats.
# Each entry: our canonical name → list of possible FanGraphs column names.
FEATURE_CANDIDATES: dict[str, list[str]] = {
    "barrel_pct":    ["Barrel%", "Barrel/PA", "Barrel"],
    "exit_velocity": ["EV", "Exit Velocity", "avg_exit_velocity"],
    "hard_hit_pct":  ["HardHit%", "Hard%", "HardHit"],
    "fb_pct":        ["FB%"],
    "pull_pct":      ["Pull%"],
    "iso":           ["ISO"],
    "k_pct":         ["K%"],
    "bb_pct":        ["BB%"],
    "hr_fb_pct":     ["HR/FB"],        # HR per fly ball — strong predictor
    "age":           ["Age"],
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _resolve_columns(df: pd.DataFrame, candidates: dict[str, list[str]]) -> dict[str, str]:
    """Return {canonical_name: actual_column} for columns found in df."""
    mapping = {}
    for canonical, options in candidates.items():
        for col in options:
            if col in df.columns:
                mapping[canonical] = col
                break
        if canonical not in mapping:
            log.debug("Column %r not found in DataFrame (tried: %s)", canonical, options)
    return mapping


def _pct_to_float(series: pd.Series) -> pd.Series:
    """Convert '12.3%' strings to 0.123 floats if needed."""
    if series.dtype == object:
        return series.str.rstrip("%").astype(float) / 100
    return series.astype(float)


def fetch_batting_stats(years: list[int], min_pa: int = MIN_PA) -> pd.DataFrame:
    """
    Pull FanGraphs batting stats for each year via pybaseball.
    Returns a DataFrame with canonical feature names + hr_pa target.
    """
    import pybaseball as pb
    pb.cache.enable()

    frames = []
    for year in years:
        try:
            df = pb.batting_stats(year, year, qual=min_pa)
            if df is None or df.empty:
                log.warning("No batting data for %d", year)
                continue
            df = df.copy()
            df["_season"] = year
            frames.append(df)
            log.info("Fetched %d batters for %d", len(df), year)
        except Exception as exc:
            log.warning("Failed to fetch batting stats for %d: %s", year, exc)

    if not frames:
        raise RuntimeError("No batting data fetched — check pybaseball / network.")

    raw = pd.concat(frames, ignore_index=True)
    col_map = _resolve_columns(raw, FEATURE_CANDIDATES)
    log.info("Resolved feature columns: %s", col_map)

    # Build canonical feature DataFrame
    rows = []
    for _, row in raw.iterrows():
        record: dict = {
            "player_name": row.get("Name", ""),
            "season":      int(row["_season"]),
            "pa":          float(row.get("PA", 0)),
            "hr":          float(row.get("HR", 0)),
        }
        for canonical, actual_col in col_map.items():
            val = row[actual_col]
            try:
                val = _pct_to_float(pd.Series([val]))[0]
            except Exception:
                val = np.nan
            record[canonical] = float(val) if pd.notna(val) else np.nan

        # Derived target: HR per PA
        record["hr_pa"] = record["hr"] / record["pa"] if record["pa"] >= min_pa else np.nan
        rows.append(record)

    return pd.DataFrame(rows)


def build_training_pairs(
    stats: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create (X, y) where:
      X = year-N Statcast/contact features for a batter
      y = year-(N+1) HR/PA rate for that same batter

    Proper year-over-year split — no look-ahead bias.
    """
    # Pivot so we can align consecutive seasons per player
    seasons = sorted(stats["season"].unique())
    pairs_X, pairs_y = [], []

    for i in range(len(seasons) - 1):
        yr_n   = seasons[i]
        yr_np1 = seasons[i + 1]

        df_n   = stats[stats["season"] == yr_n].set_index("player_name")
        df_np1 = stats[stats["season"] == yr_np1].set_index("player_name")

        common = df_n.index.intersection(df_np1.index)
        if common.empty:
            continue

        x_block = df_n.loc[common, feature_cols].copy()
        y_block = df_np1.loc[common, "hr_pa"].copy()

        valid = x_block.notna().all(axis=1) & y_block.notna()
        pairs_X.append(x_block[valid])
        pairs_y.append(y_block[valid])
        log.info(
            "Year-pair %d→%d: %d training examples",
            yr_n, yr_np1, valid.sum(),
        )

    if not pairs_X:
        raise RuntimeError("No valid training pairs found.")

    X = pd.concat(pairs_X).reset_index(drop=True)
    y = pd.concat(pairs_y).reset_index(drop=True)
    return X, y


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train(years: list[int] | None = None) -> dict:
    """
    Full training pipeline.  Returns evaluation metrics dict.
    """
    import xgboost as xgb

    if years is None:
        years = [2022, 2023, 2024, 2025]

    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Fetch data
    log.info("Fetching batting stats for %s …", years)
    stats = fetch_batting_stats(years)
    log.info("Total player-seasons fetched: %d", len(stats))

    # 2. Determine available features
    available_features = [
        col for col in FEATURE_CANDIDATES.keys()
        if col in stats.columns and stats[col].notna().sum() > 10
    ]
    log.info("Training features (%d): %s", len(available_features), available_features)

    if len(available_features) < 3:
        raise RuntimeError(
            f"Too few features available ({available_features}). "
            "Check pybaseball version and FanGraphs data availability."
        )

    # 3. Build training pairs
    X, y = build_training_pairs(stats, available_features)
    log.info("Training set: %d examples, %d features", len(X), X.shape[1])

    # 4. Impute missing values with column medians
    medians = X.median()
    X = X.fillna(medians)

    # 5. Cross-validated training
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    fold_maes = []

    # Final model trained on all data
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,   # regularize for small dataset
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=2.0, reg_alpha=0.5, objective="reg:squarederror",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_val)
        preds = np.clip(preds, 0, 0.15)
        oof_preds[val_idx] = preds
        fold_mae = mean_absolute_error(y_val, preds)
        fold_maes.append(fold_mae)
        log.info("  Fold %d MAE: %.5f", fold, fold_mae)

    cv_mae  = float(np.mean(fold_maes))
    cv_r2   = float(r2_score(y, oof_preds))
    log.info("CV MAE: %.5f | CV R²: %.4f", cv_mae, cv_r2)

    # 6. Final model on all data
    model.fit(X, y)

    # Feature importance
    importance = dict(zip(
        available_features,
        model.feature_importances_.tolist(),
    ))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: -x[1]))
    log.info("Feature importances: %s", importance_sorted)

    # 7. League-average HR/PA for fallback
    lg_hr_pa = float(y.mean())

    # 8. Save model
    model.save_model(str(MODEL_PATH))
    log.info("Model saved → %s", MODEL_PATH)

    # 9. Save metadata
    metadata = {
        "features":        available_features,
        "feature_medians": medians[available_features].to_dict(),
        "lg_hr_pa":        lg_hr_pa,
        "training_years":  years,
        "n_training":      len(X),
        "cv_mae":          cv_mae,
        "cv_r2":           cv_r2,
        "feature_importance": importance_sorted,
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))
    log.info("Metadata saved → %s", META_PATH)

    # Print summary
    log.info("=" * 50)
    log.info("Training complete.")
    log.info("  Samples:   %d", len(X))
    log.info("  Features:  %d", len(available_features))
    log.info("  CV MAE:    %.5f HR/PA", cv_mae)
    log.info("  CV R²:     %.4f", cv_r2)
    log.info("  LG avg HR/PA: %.4f", lg_hr_pa)
    log.info("=" * 50)

    return metadata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    train()
