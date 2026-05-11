"""
train.py
========
Weekly retraining script for the XGBoost HR-rate talent estimator.

Data sources (both from Baseball Savant — no auth required)
-----------------------------------------------------------
1. Statcast barrels/EV leaderboard  → barrel %, exit velo, launch angle,
                                       sweet spot %, hard hit %
2. Expected statistics leaderboard  → PA, HR, ISO, K%, BB%, xwOBA

Both are joined on player_id (MLBAM ID).

Approach: Year-N features → Year-(N+1) HR/PA  (no look-ahead bias)
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
HEADERS = {"User-Agent": "hr-predictor/1.0 (github-actions; open-source)"}

# ---------------------------------------------------------------------------
# Confirmed Baseball Savant column names (from live API response 2025-05-11)
# ---------------------------------------------------------------------------

# Endpoint 1: exit velo / barrels leaderboard
# https://baseballsavant.mlb.com/leaderboard/statcast?type=batter&year=X&min=Y&csv=true
# Confirmed columns: last_name, first_name, player_id, attempts, avg_hit_angle,
#   anglesweetspotpercent, max_hit_speed, avg_hit_speed, ev50, fbld, gb,
#   max_distance, avg_distance, avg_hr_distance, ev95plus, ev95percent,
#   barrels, brl_percent, brl_pa
BARRELS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast"
    "?type=batter&year={year}&position=&team=&min={min_pa}&csv=true"
)
BARRELS_COLS = {
    # canonical          actual Savant column
    "barrel_pct":        "brl_percent",        # barrels per batted ball
    "barrel_pa":         "brl_pa",             # barrels per PA (tighter signal)
    "exit_velocity":     "avg_hit_speed",      # average exit velo (mph)
    "launch_angle":      "avg_hit_angle",      # average launch angle (degrees)
    "sweet_spot_pct":    "anglesweetspotpercent",  # LA 8–32° sweet spot
    "hard_hit_pct":      "ev95percent",        # % batted balls ≥ 95 mph
    "n_batted_balls":    "attempts",           # sample size
}

# Endpoint 2: expected statistics leaderboard
# https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year=X&min=Y&csv=true
# Returns: player_id, pa, home_run, k_percent, bb_percent, isolated_power,
#   exit_velocity_avg, launch_angle_avg, sweet_spot_percent, barrel_batted_rate,
#   hard_hit_percent, est_woba (xwOBA), etc.
XSTATS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=batter&year={year}&position=&team=&min={min_pa}&csv=true"
)
XSTATS_COLS = {
    "pa":         ["pa", "PA"],
    "home_run":   ["home_run", "HR", "hr"],
    "iso":        ["isolated_power", "iso", "ISO"],
    "k_pct":      ["k_percent", "strikeout_percent", "k%"],
    "bb_pct":     ["bb_percent", "walk_percent", "bb%"],
    "xwoba":      ["est_woba", "xwoba", "expected_woba"],
    # Fallback Statcast metrics if barrels endpoint is missing something
    "ev_xstats":  ["exit_velocity_avg", "avg_exit_velocity"],
    "la_xstats":  ["launch_angle_avg", "avg_launch_angle"],
    "hh_xstats":  ["hard_hit_percent", "hard_hit_pct"],
    "brl_xstats": ["barrel_batted_rate", "barrel_pct"],
}

# Final feature list for XGBoost (subset that stabilizes well year-over-year)
FEATURES = [
    "barrel_pct",     # strongest single HR predictor
    "barrel_pa",      # barrels per PA (slightly different angle)
    "exit_velocity",  # raw power
    "hard_hit_pct",   # contact quality floor
    "launch_angle",   # trajectory (higher → more fly balls → more HR potential)
    "sweet_spot_pct", # optimal contact zone
    "xwoba",          # expected wOBA — absorbs multiple Statcast signals
    "iso",            # isolated power (SLG - AVG) — traditional power metric
    "k_pct",          # K% — useful non-linearity (extreme Ks can correlate with power)
    "bb_pct",         # BB% — plate discipline / pitch recognition
]


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_csv(url: str, label: str) -> pd.DataFrame | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty:
            log.warning("%s returned empty CSV.", label)
            return None
        log.info("%s: %d rows, columns: %s", label, len(df), df.columns.tolist())
        return df
    except Exception as exc:
        log.warning("Failed to fetch %s: %s", label, exc)
        return None


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df."""
    return next((c for c in candidates if c in df.columns), None)


# ---------------------------------------------------------------------------
# Per-year data assembly
# ---------------------------------------------------------------------------

def fetch_year(year: int, min_pa: int = MIN_PA) -> pd.DataFrame | None:
    """
    Fetch and join both Savant leaderboards for one year.
    Returns a tidy DataFrame keyed by player_id, or None on failure.
    """
    # ── Barrels leaderboard ──────────────────────────────────────────────────
    df_brl = _fetch_csv(
        BARRELS_URL.format(year=year, min_pa=50),   # lower threshold — filter on PA later
        f"barrels/{year}",
    )

    # ── Expected stats leaderboard ───────────────────────────────────────────
    df_xs = _fetch_csv(
        XSTATS_URL.format(year=year, min_pa=min_pa),
        f"xstats/{year}",
    )

    if df_xs is None:
        log.warning("Year %d skipped — no expected-stats data.", year)
        return None

    # Identify player_id column in each frame
    id_col_xs  = _first_col(df_xs,  ["player_id", "IDfg", "mlbam_id"])
    id_col_brl = _first_col(df_brl, ["player_id", "IDfg", "mlbam_id"]) if df_brl is not None else None

    if id_col_xs is None:
        log.warning("Year %d: no player_id in xstats. Columns: %s", year, df_xs.columns.tolist())
        return None

    # ── Extract target (PA + HR) from xstats ────────────────────────────────
    pa_col = _first_col(df_xs, XSTATS_COLS["pa"])
    hr_col = _first_col(df_xs, XSTATS_COLS["home_run"])

    if pa_col is None or hr_col is None:
        log.warning(
            "Year %d: missing pa/hr columns in xstats. Available: %s",
            year, df_xs.columns.tolist(),
        )
        return None

    df_xs = df_xs.copy()
    df_xs["_pid"]  = pd.to_numeric(df_xs[id_col_xs], errors="coerce")
    df_xs["_pa"]   = pd.to_numeric(df_xs[pa_col], errors="coerce")
    df_xs["_hr"]   = pd.to_numeric(df_xs[hr_col], errors="coerce")
    df_xs = df_xs[df_xs["_pa"] >= min_pa].copy()

    # ── Pull xstats features ─────────────────────────────────────────────────
    rows = []
    for _, row in df_xs.iterrows():
        pid = int(row["_pid"]) if pd.notna(row["_pid"]) else None
        if pid is None:
            continue
        pa = float(row["_pa"])
        hr = float(row["_hr"])

        rec: dict = {
            "player_id": pid,
            "season":    year,
            "pa":        pa,
            "hr":        hr,
            "hr_pa":     hr / pa,
        }

        # Extract xstats features
        for canonical, candidates in XSTATS_COLS.items():
            if canonical in ("pa", "home_run"):
                continue
            col = _first_col(df_xs, candidates if isinstance(candidates, list) else [candidates])
            if col and col in df_xs.columns:
                val = row.get(col, np.nan)
                try:
                    rec[canonical] = float(val)
                except (TypeError, ValueError):
                    rec[canonical] = np.nan
            else:
                rec[canonical] = np.nan

        rows.append(rec)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        log.warning("Year %d: no qualifying batters after PA filter.", year)
        return None

    # ── Merge barrels features ───────────────────────────────────────────────
    if df_brl is not None and id_col_brl is not None:
        df_brl = df_brl.copy()
        df_brl["_pid"] = pd.to_numeric(df_brl[id_col_brl], errors="coerce")

        brl_rows = {}
        for _, brow in df_brl.iterrows():
            pid = int(brow["_pid"]) if pd.notna(brow["_pid"]) else None
            if pid is None:
                continue
            rec = {}
            for canonical, actual in BARRELS_COLS.items():
                if actual in df_brl.columns:
                    try:
                        rec[canonical] = float(brow[actual])
                    except (TypeError, ValueError):
                        rec[canonical] = np.nan
                else:
                    rec[canonical] = np.nan
            brl_rows[pid] = rec

        for canonical in BARRELS_COLS:
            df_out[canonical] = df_out["player_id"].map(
                lambda pid, c=canonical: brl_rows.get(pid, {}).get(c, np.nan)
            )
    else:
        for canonical in BARRELS_COLS:
            df_out[canonical] = np.nan

    log.info("Year %d: %d qualifying batters assembled.", year, len(df_out))
    return df_out


# ---------------------------------------------------------------------------
# Training pair construction
# ---------------------------------------------------------------------------

def build_training_pairs(
    stats: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Year-N features → Year-(N+1) HR/PA rate, matched on player_id."""
    seasons  = sorted(stats["season"].unique())
    all_X, all_y = [], []

    for i in range(len(seasons) - 1):
        yr_n, yr_np1 = seasons[i], seasons[i + 1]
        df_n   = stats[stats["season"] == yr_n].set_index("player_id")
        df_np1 = stats[stats["season"] == yr_np1].set_index("player_id")
        common = df_n.index.intersection(df_np1.index)

        if common.empty:
            log.warning("No common players between %d and %d.", yr_n, yr_np1)
            continue

        X_block = df_n.loc[common, feature_cols].copy()
        y_block = df_np1.loc[common, "hr_pa"].copy()
        valid   = X_block.notna().any(axis=1) & y_block.notna()

        all_X.append(X_block[valid])
        all_y.append(y_block[valid])
        log.info("Pair %d→%d: %d training examples", yr_n, yr_np1, valid.sum())

    if not all_X:
        raise RuntimeError(
            "No valid year-over-year training pairs. "
            "Need at least two consecutive seasons with overlapping players."
        )

    return (
        pd.concat(all_X).reset_index(drop=True),
        pd.concat(all_y).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(years: list[int] | None = None) -> dict:
    """Full pipeline. Returns metadata dict."""
    import xgboost as xgb

    if years is None:
        years = [2022, 2023, 2024, 2025]

    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Fetch all years
    frames = []
    for year in years:
        df = fetch_year(year)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError(
            "No data fetched from Baseball Savant. "
            "Check network access to baseballsavant.mlb.com."
        )

    stats = pd.concat(frames, ignore_index=True)
    log.info("Total player-seasons: %d", len(stats))

    # 2. Determine usable features (enough non-null values)
    available = [
        f for f in FEATURES
        if f in stats.columns and stats[f].notna().sum() >= 20
    ]
    if len(available) < 2:
        raise RuntimeError(f"Too few usable features: {available}")
    log.info("Training features (%d): %s", len(available), available)

    # 3. Build pairs
    X, y = build_training_pairs(stats, available)
    log.info("Training set: %d examples × %d features", len(X), X.shape[1])

    # 4. Impute missing with column medians
    medians = X.median()
    X = X.fillna(medians)

    # 5. Cross-validated evaluation
    n_splits = min(5, max(2, len(X) // 20))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    fold_maes = []

    xgb_params = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=2.0, reg_alpha=0.5,
        objective="reg:squarederror",
        random_state=42, n_jobs=-1, verbosity=0,
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
        zip(available, model.feature_importances_.tolist()),
        key=lambda kv: -kv[1],
    ))

    # 7. Save
    model.save_model(str(MODEL_PATH))
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
    log.info("  Features : %s", available)
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
