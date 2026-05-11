"""
predictor.py
============
Ensemble prediction model for daily HR probability.

Architecture
------------
Three complementary paths are combined:

1. **XGBoost talent model** (when model file exists)
   Predicts each batter's true HR/PA talent from Statcast quality-of-
   contact features (barrel rate, exit velocity, hard-hit rate, etc.)
   trained on year-N → year-(N+1) data to avoid look-ahead bias.
   This is the primary signal when the model is loaded.

2. **Statistical model** (always runs as baseline)
   A calibrated odds-ratio model using Bayesian-blended HR/PA rates,
   log5 pitcher adjustment, and multiplicative context factors.
   Used as the sole predictor when no trained model exists.

3. **Statcast quality-of-contact score** (supplements both above)
   An independent logistic score from barrel rate + EV + hard-hit rate
   that acts as a sanity check and confidence signal.

Game-level adjustments (park, weather, platoon, recent form) are applied
multiplicatively to whichever base HR rate is used — they are independent
of the talent estimator.

Final output: P(HR ≥ 1 in today's game) via Poisson distribution.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from src.features.engineer import (
    LG_HR_PER_PA,
    LG_PA_PER_GAME,
    LG_PITCHER_HR9,
    barrel_score,
    bayesian_blend,
    exit_velocity_score,
    hr_rate_from_stats,
    log5,
    park_factor,
    pitcher_hr_factor,
    platoon_adjustment,
    recent_form_factor,
    weather_factor,
    _safe,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pitcher HR/PA rate helper
# ---------------------------------------------------------------------------
def _pitcher_hr_pa_rate(p_season: dict, p_career: dict) -> float:
    s_hr = _safe(p_season.get("homeRuns", 0))
    s_ip = _safe(p_season.get("inningsPitched", 0))
    c_hr = _safe(p_career.get("homeRuns", 0))
    c_ip = _safe(p_career.get("inningsPitched", 0))
    pa_per_inn = 3.3
    s_pa = s_ip * pa_per_inn
    c_pa = c_ip * pa_per_inn
    s_rate = s_hr / s_pa if s_pa > 0 else LG_HR_PER_PA
    c_rate = c_hr / c_pa if c_pa > 0 else LG_HR_PER_PA
    blended = bayesian_blend(
        season_rate=s_rate, season_n=int(s_pa),
        career_rate=c_rate, career_n=int(c_pa),
        prior=LG_HR_PER_PA, prior_weight=400,
    )
    return float(np.clip(blended, 0.005, 0.10))


# ---------------------------------------------------------------------------
# Path 1 — XGBoost talent model
# ---------------------------------------------------------------------------
def xgboost_hr_rate(
    statcast_metrics: dict,
    batter_season: dict,
    model,
    meta: dict,
) -> float | None:
    """
    Use the trained XGBoost model to predict batter's HR/PA talent rate.
    Returns None if model unavailable or features insufficient.
    """
    from src.models.model_registry import predict_hr_rate
    try:
        rate = predict_hr_rate(statcast_metrics, batter_season, model, meta)
        return rate
    except Exception as exc:
        log.debug("XGBoost prediction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Path 2 — Statistical model
# ---------------------------------------------------------------------------
def statistical_hr_rate(
    batter_season: dict,
    batter_career: dict,
) -> tuple[float, int]:
    """Bayesian-blended batter HR/PA rate. Returns (rate, season_pa)."""
    return hr_rate_from_stats(batter_season, batter_career)


# ---------------------------------------------------------------------------
# Game-level adjustments (shared by all paths)
# ---------------------------------------------------------------------------
def game_adjustments(
    batter_rate: float,
    pitcher_season: dict,
    pitcher_career: dict,
    pitcher_throws: str,
    batter_bats: str,
    platoon_splits: dict,
    recent_games: list[dict],
    venue: str,
    weather: dict,
    config: dict,
) -> tuple[float, dict]:
    """
    Apply log5 pitcher matchup + multiplicative context factors.
    Returns (adjusted_rate_per_pa, factors_dict).
    """
    p_rate = _pitcher_hr_pa_rate(pitcher_season, pitcher_career)
    matchup_rate = log5(batter_rate, p_rate, LG_HR_PER_PA)

    pf   = park_factor(venue, config)
    wf   = weather_factor(weather, config)
    plat = platoon_adjustment(batter_bats, pitcher_throws, platoon_splits)
    rf   = recent_form_factor(
        recent_games,
        config.get("model", {}).get("recent_form_games", 15),
    )

    adj_rate = float(np.clip(matchup_rate * pf * wf * plat * rf, 1e-5, 0.30))

    factors = {
        "pitcher_hr_rate":    round(p_rate, 4),
        "matchup_rate":       round(matchup_rate, 4),
        "park_factor":        round(pf, 3),
        "weather_factor":     round(wf, 3),
        "platoon_factor":     round(plat, 3),
        "recent_form_factor": round(rf, 3),
        "adj_rate_per_pa":    round(adj_rate, 4),
    }
    return adj_rate, factors


# ---------------------------------------------------------------------------
# Path 3 — Statcast quality-of-contact score
# ---------------------------------------------------------------------------
def statcast_probability(metrics: dict) -> tuple[float, float]:
    """
    Logistic score from contact quality metrics.
    Returns (probability, confidence) where confidence ∈ [0, 1].
    """
    n  = metrics.get("n_batted_balls", 0)
    br = metrics.get("barrel_rate")
    ev = metrics.get("avg_exit_velocity")
    hh = metrics.get("hard_hit_rate")
    fb = metrics.get("fly_ball_rate")

    if n < 30 or br is None or math.isnan(float(br)):
        return 0.15, 0.0

    br = float(br); ev = float(ev or 88.3)
    hh = float(hh or 0.37); fb = float(fb or 0.35)

    score = (
        8.0 * br
        + 0.08 * (ev - 88)
        + 1.5  * hh
        + 1.0  * fb
        - 1.80
    )
    prob = 1.0 / (1.0 + math.exp(-score))
    confidence = float(np.clip((n - 30) / 200, 0, 1))
    return float(np.clip(prob, 0.01, 0.99)), confidence


# ---------------------------------------------------------------------------
# Main ensemble entry point
# ---------------------------------------------------------------------------
def ensemble_predict(
    batter_season: dict,
    batter_career: dict,
    pitcher_season: dict,
    pitcher_career: dict,
    pitcher_throws: str,
    batter_bats: str,
    platoon_splits: dict,
    recent_games: list[dict],
    statcast_metrics: dict,
    venue: str,
    weather: dict,
    config: dict,
    xgb_model=None,
    xgb_meta: dict | None = None,
) -> dict:
    """
    Full ensemble prediction for one batter in today's game.

    XGBoost model is used as the primary talent estimator when available;
    the statistical model is the fallback.  Game-level adjustments
    (pitcher, park, weather, platoon, recent form) are always applied.

    Returns dict with hr_probability, hr_pct, confidence_tier, factors, etc.
    """
    # ── Batter HR rate ──────────────────────────────────────────────────────
    stat_rate, season_pa = statistical_hr_rate(batter_season, batter_career)

    xgb_rate = None
    if xgb_model is not None and xgb_meta is not None:
        xgb_rate = xgboost_hr_rate(statcast_metrics, batter_season, xgb_model, xgb_meta)

    if xgb_rate is not None:
        # Blend: 65% XGBoost, 35% statistical (statistical stabilizes very
        # new players or those with sparse Statcast coverage)
        batter_rate = 0.65 * xgb_rate + 0.35 * stat_rate
        rate_source = "xgboost+statistical"
    else:
        batter_rate = stat_rate
        rate_source = "statistical"

    # ── Game-level adjustments ──────────────────────────────────────────────
    adj_rate, factors = game_adjustments(
        batter_rate, pitcher_season, pitcher_career,
        pitcher_throws, batter_bats, platoon_splits,
        recent_games, venue, weather, config,
    )

    # ── Poisson: P(≥1 HR) ───────────────────────────────────────────────────
    est_pa  = LG_PA_PER_GAME
    lam     = adj_rate * est_pa
    stat_prob = float(np.clip(1.0 - math.exp(-lam), 0, 0.999))

    # ── Statcast sanity check ───────────────────────────────────────────────
    sc_prob, sc_confidence = statcast_probability(statcast_metrics)

    # Blend Statcast check in at low weight (sanity check only)
    sc_w   = 0.15 * sc_confidence
    main_w = 1.0 - sc_w
    final_prob = float(np.clip(
        main_w * stat_prob + sc_w * sc_prob, 0, 0.999
    ))

    # ── Confidence tier ─────────────────────────────────────────────────────
    if xgb_rate is not None and season_pa >= 100 and sc_confidence > 0.4:
        tier = "High"
    elif season_pa >= 60 or sc_confidence > 0.2:
        tier = "Medium"
    else:
        tier = "Low"

    factors.update({
        "batter_hr_rate":     round(batter_rate, 4),
        "stat_hr_rate":       round(stat_rate, 4),
        "xgb_hr_rate":        round(xgb_rate, 4) if xgb_rate is not None else None,
        "rate_source":        rate_source,
        "lambda":             round(lam, 4),
        "expected_pa":        est_pa,
        "season_pa":          season_pa,
    })

    return {
        "hr_probability":      round(final_prob, 4),
        "hr_pct":              f"{final_prob * 100:.1f}%",
        "confidence_tier":     tier,
        "statistical_prob":    round(stat_prob, 4),
        "statcast_prob":       round(sc_prob, 4),
        "statcast_confidence": round(sc_confidence, 2),
        "rate_source":         rate_source,
        "factors":             factors,
        "statcast_metrics": {
            k: v for k, v in statcast_metrics.items()
            if k in ("barrel_rate", "hard_hit_rate", "avg_exit_velocity", "n_batted_balls")
        },
    }
