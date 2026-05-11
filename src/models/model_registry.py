"""
model_registry.py
=================
Load and cache the trained XGBoost model + metadata.
Falls back gracefully to None if no model file exists yet
(first run before retrain.yml has ever executed).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/hr_model.json")
META_PATH  = Path("models/feature_metadata.json")

# Module-level cache — loaded once per process
_model = None
_meta: dict | None = None


def load() -> tuple[object | None, dict | None]:
    """
    Return (xgb_model, metadata).
    Both are None if the model hasn't been trained yet.
    """
    global _model, _meta

    if _model is not None:
        return _model, _meta

    if not MODEL_PATH.exists() or not META_PATH.exists():
        log.warning(
            "No trained model found at %s. "
            "Run the retrain workflow first, or trigger it manually. "
            "Falling back to statistical model only.",
            MODEL_PATH,
        )
        return None, None

    try:
        import xgboost as xgb
        m = xgb.XGBRegressor()
        m.load_model(str(MODEL_PATH))
        _model = m

        _meta = json.loads(META_PATH.read_text())
        log.info(
            "Loaded XGBoost model — trained on %d examples, CV R²=%.4f, features=%s",
            _meta.get("n_training", "?"),
            _meta.get("cv_r2", 0),
            _meta.get("features", []),
        )
        return _model, _meta

    except Exception as exc:
        log.warning("Failed to load model from %s: %s — using statistical model only.", MODEL_PATH, exc)
        _model, _meta = None, None
        return None, None


def predict_hr_rate(
    statcast_metrics: dict,
    batter_season_stats: dict,
    model,
    meta: dict,
) -> float | None:
    """
    Use the XGBoost model to predict a batter's HR/PA talent rate.

    Parameters
    ----------
    statcast_metrics    : dict from statcast_client.get_batter_statcast_metrics()
    batter_season_stats : dict from mlb_client.get_batter_stats()['season_stats']
    model               : loaded XGBRegressor
    meta                : metadata dict from load()

    Returns
    -------
    Predicted HR/PA rate (float), or None if features are insufficient.
    """
    import numpy as np
    import pandas as pd

    features: list[str] = meta["features"]
    medians:  dict      = meta["feature_medians"]

    # Map canonical feature names to available data sources
    SOURCE_MAP = {
        "barrel_pct":    ("statcast", "barrel_rate"),
        "exit_velocity": ("statcast", "avg_exit_velocity"),
        "hard_hit_pct":  ("statcast", "hard_hit_rate"),
        "fb_pct":        ("statcast", "fly_ball_rate"),
        "hr_fb_pct":     ("derived",  None),     # HR / (FB * contact) — derived below
        "pull_pct":      ("season",   "pullPercent"),
        "iso":           ("season",   "iso"),
        "k_pct":         ("season",   "strikeoutPercentage"),
        "bb_pct":        ("season",   "walkPercentage"),
        "age":           ("season",   "age"),
    }

    def _safe(val, default=np.nan):
        try:
            f = float(val)
            return f if (f == f) else default   # nan check
        except (TypeError, ValueError):
            return default

    row = {}
    for feat in features:
        source_type, source_key = SOURCE_MAP.get(feat, ("unknown", None))

        if source_type == "statcast" and source_key:
            row[feat] = _safe(statcast_metrics.get(source_key))
        elif source_type == "season" and source_key:
            row[feat] = _safe(batter_season_stats.get(source_key))
        elif source_type == "derived" and feat == "hr_fb_pct":
            # HR/FB = HR / estimated fly balls
            hr  = _safe(batter_season_stats.get("homeRuns", 0))
            ab  = _safe(batter_season_stats.get("atBats", 1))
            fb  = _safe(statcast_metrics.get("fly_ball_rate", np.nan))
            if fb > 0 and ab > 0:
                est_fb = ab * fb
                row[feat] = hr / est_fb if est_fb > 0 else np.nan
            else:
                row[feat] = np.nan
        else:
            row[feat] = np.nan

        # Fill missing with training median
        if np.isnan(row[feat]):
            row[feat] = medians.get(feat, np.nan)

    X = pd.DataFrame([row])[features]
    if X.isna().all(axis=None):
        return None

    X = X.fillna(pd.Series(medians))
    pred = float(model.predict(X)[0])
    return float(np.clip(pred, 0.001, 0.12))
