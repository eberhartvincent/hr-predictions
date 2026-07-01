"""
model_registry.py
=================
Load and cache all five MLB XGBoost models:
  hr  → HR/PA rate
  tb  → TB/PA rate
  h   → H/PA rate
  r   → R/game rate
  rbi → RBI/game rate
"""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np, pandas as pd

log = logging.getLogger(__name__)
MODEL_DIR = Path("models")
_NAMES    = ("hr", "tb", "h", "r", "rbi")

_models: dict[str, object] = {}
_metas:  dict[str, dict]   = {}


def load() -> tuple[dict, dict]:
    """
    Load all five models. Returns ({name: model}, {name: meta}).
    Missing models return None — pipeline degrades gracefully.
    """
    global _models, _metas
    if _models:
        return _models, _metas

    try:
        import xgboost as xgb
    except ImportError:
        log.error("xgboost not installed.")
        return {n: None for n in _NAMES}, {n: {} for n in _NAMES}

    for name in _NAMES:
        # Support both old path (hr_model.json / feature_metadata.json) and new paths
        if name == "hr":
            model_path = MODEL_DIR / "hr_model.json"
            meta_path  = MODEL_DIR / "feature_metadata.json"
            # Also check new naming convention
            if not model_path.exists():
                model_path = MODEL_DIR / "hr_model.json"
        else:
            model_path = MODEL_DIR / f"{name}_model.json"
            meta_path  = MODEL_DIR / f"{name}_metadata.json"

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
                name,
                _metas[name].get("n_training", 0),
                _metas[name].get("cv_r2", 0),
                _metas[name].get("target", "?"),
            )
        except Exception as exc:
            log.warning("Failed to load %s model: %s", name, exc)
            _models[name] = None
            _metas[name]  = {}

    return _models, _metas


def predict_rate(
    model_name: str,
    statcast_metrics: dict,
    batter_season: dict,
    models: dict,
    metas: dict,
) -> float | None:
    """
    Predict a rate for a batter using the named model.
    Returns None if model unavailable or features insufficient.

    Input features are drawn from statcast_metrics (Savant data)
    and batter_season (MLB API) to match training feature sources.
    """
    model = models.get(model_name)
    meta  = metas.get(model_name, {})
    if model is None or not meta:
        return None

    features: list[str] = meta.get("features", [])
    medians:  dict      = meta.get("feature_medians", {})
    clip_max = {"hr": 0.15, "tb": 1.5, "h": 0.6, "r": 2.0, "rbi": 2.0}.get(model_name, 1.0)

    # Feature source mapping
    STATCAST_MAP = {
        "barrel_pct":     "barrel_rate",
        "barrel_pa":      "barrel_rate",     # approximation
        "exit_velocity":  "avg_exit_velocity",
        "launch_angle":   "avg_launch_angle",
        "sweet_spot_pct": "sweet_spot_pct",
        "hard_hit_pct":   "hard_hit_rate",
        "xwoba":          "xwoba",
        "xslg":           "xslg",
    }
    SEASON_MAP = {
        "k_pct":  "strikeoutPercentage",
        "bb_pct": "walkPercentage",
        "iso":    "isolatedPower",
    }

    row = {}
    for feat in features:
        val = np.nan
        if feat in STATCAST_MAP:
            val = statcast_metrics.get(STATCAST_MAP[feat], np.nan)
        elif feat in SEASON_MAP:
            val = batter_season.get(SEASON_MAP[feat], np.nan)

        try:
            f = float(val)
            row[feat] = f if np.isfinite(f) else float(medians.get(feat, np.nan))
        except (TypeError, ValueError):
            row[feat] = float(medians.get(feat, np.nan))

    X = pd.DataFrame([row])[features].fillna(pd.Series(medians))
    if X.isna().all(axis=None):
        return None

    pred = float(model.predict(X)[0])
    return float(np.clip(pred, 0, clip_max))
