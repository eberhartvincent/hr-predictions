"""
model_registry.py
=================
Load all five MLB XGBoost models and map daily features to training features.

Feature mapping correctness is critical — mismatches between training
column names and daily inference values are the most common cause of
model degradation. Every mapping here is verified against:
  - Training: train.py column names from Savant CSV + MLB API
  - Inference: statcast_client.py + batter_season dict from MLB API
"""
from __future__ import annotations
import json, logging, math
from pathlib import Path
import numpy as np, pandas as pd

log = logging.getLogger(__name__)
MODEL_DIR = Path("models")
_NAMES    = ("hr", "tb", "h", "r", "rbi")

_models: dict[str, object] = {}
_metas:  dict[str, dict]   = {}


def load() -> tuple[dict, dict]:
    """Load all five models. Returns ({name: model}, {name: meta})."""
    global _models, _metas
    if _models:
        return _models, _metas

    try:
        import xgboost as xgb
    except ImportError:
        log.error("xgboost not installed.")
        return {n: None for n in _NAMES}, {n: {} for n in _NAMES}

    for name in _NAMES:
        model_path = MODEL_DIR / f"{name}_model.json"
        # Legacy HR path support
        if name == "hr" and not model_path.exists():
            model_path = MODEL_DIR / "hr_model.json"

        meta_path = MODEL_DIR / f"{name}_metadata.json"
        if name == "hr" and not meta_path.exists():
            meta_path = MODEL_DIR / "feature_metadata.json"

        if not model_path.exists() or not meta_path.exists():
            log.warning("No %s model at %s — stat baseline only.", name, model_path)
            _models[name] = None
            _metas[name]  = {}
            continue

        try:
            m = xgb.XGBRegressor()
            m.load_model(str(model_path))
            _models[name] = m
            _metas[name]  = json.loads(meta_path.read_text())
            log.info(
                "Loaded %s model — n=%d, R²=%.4f, target=%s",
                name, _metas[name].get("n_training", 0),
                _metas[name].get("cv_r2", 0), _metas[name].get("target", "?"),
            )
        except Exception as exc:
            log.warning("Failed to load %s model: %s", name, exc)
            _models[name] = None
            _metas[name]  = {}

    return _models, _metas


def _safe_float(val, default: float = np.nan) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _derive_season_features(season: dict) -> dict:
    """
    Compute derived features from MLB API season stats.
    The API returns raw counts (strikeOuts, baseOnBalls) not rates.
    We compute the rates here to match training feature names.
    """
    pa  = max(_safe_float(season.get("plateAppearances", 0), 0), 1)
    so  = _safe_float(season.get("strikeOuts", 0), 0)
    bb  = _safe_float(season.get("baseOnBalls", 0), 0)
    ab  = max(_safe_float(season.get("atBats", 0), 0), 1)

    # slugging and avg come back as decimal strings e.g. ".450"
    try:
        slg = float(str(season.get("slugging", "0")).lstrip(".") or 0)
        if slg > 1: slg = 0.0   # guard against malformed values
    except (ValueError, TypeError):
        slg = 0.0
    try:
        avg = float(str(season.get("avg", "0")).lstrip(".") or 0)
        if avg > 1: avg = 0.0
    except (ValueError, TypeError):
        avg = 0.0

    return {
        "k_pct":  so / pa,
        "bb_pct": bb / pa,
        "iso":    max(0.0, slg - avg),
    }


def predict_rate(
    model_name: str,
    statcast_metrics: dict,
    batter_season: dict,
    models: dict,
    metas: dict,
) -> float | None:
    """
    Predict a per-PA or per-game rate using the named XGBoost model.

    Feature resolution order:
      1. statcast_metrics  — from statcast_client.py (Savant leaderboard CSVs)
      2. derived season    — k_pct, bb_pct, iso computed from raw MLB API counts
      3. training median   — fallback when still missing
    """
    model = models.get(model_name)
    meta  = metas.get(model_name, {})
    if model is None or not meta:
        return None

    features: list[str] = meta.get("features", [])
    medians:  dict      = meta.get("feature_medians", {})
    clip_max = {"hr": 0.15, "tb": 1.5, "h": 0.6, "r": 2.0, "rbi": 2.0}.get(model_name, 1.0)

    # ── Statcast feature names (from statcast_client.py → training column names) ──
    # These are now consistent because statcast_client.py uses the same
    # Savant leaderboard CSVs as training.
    STATCAST_MAP = {
        "barrel_pct":     "barrel_pct",      # brl_percent in training + daily
        "barrel_pa":      "barrel_pa",        # brl_pa — NOW CORRECT (was mapped to barrel_rate before)
        "exit_velocity":  "exit_velocity",    # avg_hit_speed
        "launch_angle":   "launch_angle",     # avg_hit_angle
        "sweet_spot_pct": "sweet_spot_pct",   # anglesweetspotpercent
        "hard_hit_pct":   "hard_hit_pct",     # ev95percent
        "xwoba":          "xwoba",            # est_woba — NOW FETCHED DAILY
        "xslg":           "xslg",             # est_slg  — NOW FETCHED DAILY
    }

    # Derived season features (computed from raw MLB API counts)
    season_derived = _derive_season_features(batter_season)

    row = {}
    for feat in features:
        val = np.nan
        if feat in STATCAST_MAP:
            val = statcast_metrics.get(STATCAST_MAP[feat], np.nan)
        elif feat in season_derived:
            val = season_derived[feat]

        f = _safe_float(val)
        row[feat] = f if not np.isnan(f) else float(medians.get(feat, np.nan))

    X = pd.DataFrame([row])[features].fillna(pd.Series(medians))
    if X.isna().all(axis=None):
        return None

    pred = float(model.predict(X)[0])
    return float(np.clip(pred, 0, clip_max))
