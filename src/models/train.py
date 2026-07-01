"""
train.py
========
Trains FIVE XGBoost models from Baseball Savant + MLB Stats API data:
  1. hr_pa    → HR/PA rate            (existing)
  2. tb_pa    → Total Bases/PA rate   (new)
  3. h_pa     → Hits/PA rate          (new)
  4. r_game   → Runs/game rate        (new)
  5. rbi_game → RBI/game rate         (new)

All use year-N features → year-(N+1) target (no look-ahead bias).
Data sources:
  - Baseball Savant barrels leaderboard (barrel%, EV, launch angle, hard hit%)
  - Baseball Savant expected stats leaderboard (xwOBA, xSLG, ISO, K%, BB%)
  - MLB Stats API (HR, TB, H, R, RBI, GP for target computation)
"""
from __future__ import annotations

import io, json, logging, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MIN_PA    = 150
HEADERS   = {"User-Agent": "hr-predictor/1.0 (github-actions; open-source)"}

BARRELS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast"
    "?type=batter&year={year}&position=&team=&min=50&csv=true"
)
XSTATS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=batter&year={year}&position=&team=&min=50&csv=true"
)
MLB_STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
MLB_HEADERS   = {"User-Agent": "hr-predictor/1.0 (github-actions)"}

# ---------------------------------------------------------------------------
# Features per model — each uses the most predictive Statcast features
# for that specific outcome
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "hr": {
        "target":   "hr_pa",
        "clip_max": 0.15,
        "features": [
            "barrel_pct", "barrel_pa", "exit_velocity", "hard_hit_pct",
            "launch_angle", "sweet_spot_pct", "xwoba", "iso", "k_pct", "bb_pct",
        ],
    },
    "tb": {
        "target":   "tb_pa",
        "clip_max": 1.5,
        "features": [
            # xSLG is essentially "expected TB/AB from contact quality" — strongest feature
            "xslg", "barrel_pct", "exit_velocity", "hard_hit_pct",
            "launch_angle", "sweet_spot_pct", "iso", "bb_pct",
        ],
    },
    "h": {
        "target":   "h_pa",
        "clip_max": 0.6,
        "features": [
            # Hard contact and K avoidance drive hit rate
            "hard_hit_pct", "sweet_spot_pct", "exit_velocity",
            "k_pct", "bb_pct", "xwoba", "barrel_pct",
        ],
    },
    "r": {
        "target":   "r_game",
        "clip_max": 2.0,
        "features": [
            # Runs = getting on base + scoring; OBP proxies + power
            "xwoba", "bb_pct", "iso", "hard_hit_pct", "barrel_pct", "k_pct",
        ],
    },
    "rbi": {
        "target":   "rbi_game",
        "clip_max": 2.0,
        "features": [
            # RBI = power + contact quality; runners on base is context we can't model
            "barrel_pct", "exit_velocity", "hard_hit_pct", "iso",
            "xwoba", "k_pct", "sweet_spot_pct",
        ],
    },
}


# ---------------------------------------------------------------------------
# Safe column accessor
# ---------------------------------------------------------------------------
def _col(df: pd.DataFrame, *names: str, default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(float(default), index=df.index, dtype=float)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_barrels(year: int) -> pd.DataFrame:
    url = BARRELS_URL.format(year=year)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        log.info("Savant barrels/%d: %d rows", year, len(df))
        return df
    except Exception as exc:
        log.warning("Savant barrels/%d failed: %s", year, exc)
        return pd.DataFrame()


def fetch_xstats(year: int) -> pd.DataFrame:
    url = XSTATS_URL.format(year=year)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        log.info("Savant xstats/%d: %d rows, cols: %s", year, len(df), df.columns.tolist())
        return df
    except Exception as exc:
        log.warning("Savant xstats/%d failed: %s", year, exc)
        return pd.DataFrame()


def fetch_mlb_stats(year: int) -> pd.DataFrame:
    """
    Pull full-season hitting stats from MLB Stats API.
    Now includes hits, doubles, triples, runs, rbi in addition to HR/PA.
    """
    try:
        r = requests.get(
            MLB_STATS_URL,
            params={"stats": "season", "group": "hitting",
                    "season": year, "sportId": 1, "limit": 2000},
            headers=MLB_HEADERS, timeout=30,
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
            pid    = player.get("id")
            if not pid:
                continue

            pa  = float(s.get("plateAppearances", 0) or 0)
            ab  = float(s.get("atBats", 0) or 0)
            gp  = float(s.get("gamesPlayed", 1) or 1)
            hr  = float(s.get("homeRuns", 0) or 0)
            h   = float(s.get("hits", 0) or 0)
            d   = float(s.get("doubles", 0) or 0)
            t   = float(s.get("triples", 0) or 0)
            r   = float(s.get("runs", 0) or 0)
            rbi = float(s.get("rbi", 0) or 0)
            so  = float(s.get("strikeOuts", 0) or 0)
            bb  = float(s.get("baseOnBalls", 0) or 0)

            try: slg = float(s.get("slugging", "0") or 0)
            except ValueError: slg = 0.0
            try: avg = float(s.get("avg", "0") or 0)
            except ValueError: avg = 0.0

            # Derived targets
            tb = h + d + 2*t + 3*hr    # Total Bases

            rows.append({
                "player_id": int(pid),
                "pa":        pa,
                "ab":        ab,
                "gp":        gp,
                # Counting stats for target computation
                "hr":        hr,
                "h":         h,
                "tb":        tb,
                "r":         r,
                "rbi":       rbi,
                # Rate targets (year-(N+1) values become labels)
                "hr_pa":     hr  / pa if pa >= MIN_PA else np.nan,
                "tb_pa":     tb  / pa if pa >= MIN_PA else np.nan,
                "h_pa":      h   / pa if pa >= MIN_PA else np.nan,
                "r_game":    r   / gp if gp >= 10     else np.nan,
                "rbi_game":  rbi / gp if gp >= 10     else np.nan,
                # Features
                "k_pct":     so / pa if pa > 0 else np.nan,
                "bb_pct":    bb / pa if pa > 0 else np.nan,
                "iso":       slg - avg,
            })

    df = pd.DataFrame(rows)
    log.info("MLB API/%d: %d batters", year, len(df))
    return df


# ---------------------------------------------------------------------------
# Assemble one season — join all three sources on player_id
# ---------------------------------------------------------------------------
def fetch_year(year: int) -> pd.DataFrame | None:
    brl = fetch_barrels(year)
    xs  = fetch_xstats(year)
    mlb = fetch_mlb_stats(year)

    if mlb.empty:
        log.warning("Year %d: no MLB API data.", year)
        return None

    # Start from MLB stats (has all targets + basic features)
    base = mlb[mlb["pa"] >= MIN_PA].copy()
    base["season"] = year

    # ── Merge Savant barrels features ─────────────────────────────────────
    if not brl.empty:
        brl = brl.copy()
        brl["_pid"] = pd.to_numeric(brl.get("player_id", brl.get("IDfg")), errors="coerce")
        brl_map = brl.set_index("_pid")

        def _brl(col_name):
            return base["player_id"].map(
                pd.to_numeric(brl_map.get(col_name, pd.Series(dtype=float)),
                              errors="coerce")
            )

        base["barrel_pct"]    = _brl("brl_percent")
        base["barrel_pa"]     = _brl("brl_pa")
        base["exit_velocity"] = _brl("avg_hit_speed")
        base["launch_angle"]  = _brl("avg_hit_angle")
        base["sweet_spot_pct"]= _brl("anglesweetspotpercent")
        base["hard_hit_pct"]  = _brl("ev95percent")
    else:
        for col in ("barrel_pct","barrel_pa","exit_velocity",
                    "launch_angle","sweet_spot_pct","hard_hit_pct"):
            base[col] = np.nan

    # ── Merge Savant expected stats features ──────────────────────────────
    if not xs.empty:
        xs = xs.copy()
        xs["_pid"] = pd.to_numeric(xs.get("player_id", xs.get("IDfg")), errors="coerce")
        xs_map = xs.set_index("_pid")

        def _xs(col_name):
            return base["player_id"].map(
                pd.to_numeric(xs_map.get(col_name, pd.Series(dtype=float)),
                              errors="coerce")
            )

        base["xwoba"] = _xs("est_woba")
        base["xslg"]  = _xs("est_slg")    # KEY new feature for TB model
        # pa already in base from MLB API
    else:
        base["xwoba"] = np.nan
        base["xslg"]  = np.nan

    base = base.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    log.info("Year %d: %d qualifying batters assembled.", year, len(base))
    return base


# ---------------------------------------------------------------------------
# Training pair builder
# ---------------------------------------------------------------------------
def build_pairs(
    stats: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Year-N features → year-(N+1) target rate."""
    seasons = sorted(stats["season"].unique())
    all_X, all_y = [], []

    for i in range(len(seasons) - 1):
        yr_n, yr_np1 = seasons[i], seasons[i + 1]
        n   = stats[stats["season"] == yr_n].set_index("player_id")
        np1 = stats[stats["season"] == yr_np1].set_index("player_id")
        common = n.index.intersection(np1.index)
        if common.empty:
            continue

        avail   = [f for f in features if f in n.columns]
        X_block = n.loc[common, avail].copy()
        y_block = np1.loc[common, target].copy()
        valid   = X_block.notna().any(axis=1) & y_block.notna()

        all_X.append(X_block[valid])
        all_y.append(y_block[valid])
        log.info("  %s pair %d→%d: %d examples", target, yr_n, yr_np1, valid.sum())

    if not all_X:
        raise RuntimeError(f"No valid training pairs for {target}")
    return (
        pd.concat(all_X).reset_index(drop=True),
        pd.concat(all_y).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------
def _train_one(
    stats: pd.DataFrame,
    name: str,
    cfg: dict,
    years: list[int],
) -> tuple[object, dict]:
    import xgboost as xgb

    target   = cfg["target"]
    features = cfg["features"]
    clip_max = cfg["clip_max"]

    log.info("── Training %s model (target=%s) ──", name, target)

    available = [f for f in features if f in stats.columns
                 and stats[f].notna().sum() >= 20]
    if len(available) < 2:
        raise RuntimeError(f"{name}: too few features: {available}")

    X, y = build_pairs(stats, available, target)
    log.info("%s: %d examples × %d features", name, len(X), X.shape[1])

    medians = X.median()
    X = X.fillna(medians)

    n_splits = min(5, max(2, len(X) // 20))
    kf  = KFold(n_splits=n_splits, shuffle=True, random_state=42)
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
        p = np.clip(m.predict(X.iloc[val]), 0, clip_max)
        oof[val] = p
        maes.append(mean_absolute_error(y.iloc[val], p))

    cv_mae = float(np.mean(maes))
    cv_r2  = float(r2_score(y, oof))
    log.info("%s: CV MAE=%.5f  CV R²=%.4f", name, cv_mae, cv_r2)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, y)

    importance = dict(sorted(
        zip(available, model.feature_importances_.tolist()),
        key=lambda kv: -kv[1],
    ))

    metadata = {
        "model_name":      name,
        "target":          target,
        "features":        available,
        "feature_medians": medians[available].to_dict(),
        "lg_avg":          float(y.mean()),
        "training_years":  years,
        "n_training":      len(X),
        "cv_mae":          cv_mae,
        "cv_r2":           cv_r2,
        "feature_importance": importance,
    }
    return model, metadata


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def train(years: list[int] | None = None) -> dict:
    if years is None:
        years = [2022, 2023, 2024, 2025, 2026]

    MODEL_DIR.mkdir(exist_ok=True)

    # Fetch all years
    frames = []
    for year in years:
        df = fetch_year(year)
        if df is not None:
            frames.append(df)
        time.sleep(0.5)

    if not frames:
        raise RuntimeError("No data fetched.")

    stats = pd.concat(frames, ignore_index=True)
    log.info("Total player-seasons: %d across %s", len(stats), years)

    # Train all five models
    all_metadata = {}
    for name, cfg in MODEL_CONFIGS.items():
        try:
            model, meta = _train_one(stats, name, cfg, years)
        except RuntimeError as exc:
            log.error("Skipping %s model: %s", name, exc)
            continue

        model_path = MODEL_DIR / f"{name}_model.json"
        meta_path  = MODEL_DIR / f"{name}_metadata.json"
        model.save_model(str(model_path))
        meta_path.write_text(json.dumps(meta, indent=2))
        log.info("Saved %s → %s", name, model_path)
        all_metadata[name] = meta

    # Summary
    log.info("=" * 60)
    log.info("MLB Training complete — %d models.", len(all_metadata))
    for name, meta in all_metadata.items():
        log.info("  %-6s target=%-10s  R²=%.4f  MAE=%.5f  n=%d  top=%s",
                 name, meta["target"], meta["cv_r2"], meta["cv_mae"],
                 meta["n_training"],
                 list(meta["feature_importance"].keys())[:2])
    log.info("=" * 60)
    return all_metadata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    train()
