"""
predictor.py
============
Ensemble prediction model for daily HR probability.

Architecture
------------
Two complementary scoring paths are combined:

1. **Statistical model** (always runs)
   A calibrated odds-ratio model using:
   - Batter's blended HR/PA rate (Bayesian season+career blend)
   - Log5 adjustment for the opposing pitcher
   - Multiplicative adjustments: park factor, weather, platoon, recent form
   - Poisson probability: P(≥1 HR) = 1 − e^(−λ) where λ = adjusted_rate × est_PA

2. **Statcast quality-of-contact model** (runs when Statcast data available)
   Weighted combination of barrel rate, exit velocity, hard-hit rate, fly-ball rate
   → produces an independent HR "talent" score
   → calibrated to a probability using logistic transformation

The two paths are combined via a weighted blend (see config weights).
Final output is P(HR ≥ 1 in today's game) for each batter.
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
# Helper: pitcher HR/PA rate from pitcher stats
# ---------------------------------------------------------------------------
def _pitcher_hr_pa_rate(p_season: dict, p_career: dict) -> float:
    """Convert pitcher HR/9 to an approximate HR/PA-allowed rate."""
    s_hr = _safe(p_season.get("homeRuns", 0))
    s_ip = _safe(p_season.get("inningsPitched", 0))
    c_hr = _safe(p_career.get("homeRuns", 0))
    c_ip = _safe(p_career.get("inningsPitched", 0))

    # ~3.3 PA per inning as a baseline conversion
    pa_per_inn = 3.3
    s_pa = s_ip * pa_per_inn
    c_pa = c_ip * pa_per_inn

    s_rate = s_hr / s_pa if s_pa > 0 else LG_HR_PER_PA
    c_rate = c_hr / c_pa if c_pa > 0 else LG_HR_PER_PA

    blended = bayesian_blend(
        season_rate=s_rate,
        season_n=int(s_pa),
        career_rate=c_rate,
        career_n=int(c_pa),
        prior=LG_HR_PER_PA,
        prior_weight=400,
    )
    return float(np.clip(blended, 0.005, 0.10))


# ---------------------------------------------------------------------------
# Statistical model path
# ---------------------------------------------------------------------------
def statistical_probability(
    batter_season: dict,
    batter_career: dict,
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
    Returns (probability_of_hr, debug_factors_dict).
    """
    weights = config.get("model", {}).get("weights", {})

    # --- Batter rate ---
    batter_rate, season_pa = hr_rate_from_stats(batter_season, batter_career)

    # --- Pitcher rate (HR/PA allowed) ---
    p_rate = _pitcher_hr_pa_rate(pitcher_season, pitcher_career)

    # --- Log5 matchup probability per PA ---
    matchup_rate = log5(batter_rate, p_rate, LG_HR_PER_PA)

    # --- Multiplicative adjustments ---
    pf = park_factor(venue, config)
    wf = weather_factor(weather, config)
    plat = platoon_adjustment(batter_bats, pitcher_throws, platoon_splits)
    rf = recent_form_factor(recent_games, config.get("model", {}).get("recent_form_games", 15))

    adj_rate = matchup_rate * pf * wf * plat * rf
    adj_rate = float(np.clip(adj_rate, 1e-5, 0.30))

    # --- Poisson: P(≥1 HR) over estimated PA ---
    est_pa = LG_PA_PER_GAME
    lam = adj_rate * est_pa
    prob = 1.0 - math.exp(-lam)

    factors = {
        "batter_hr_rate": round(batter_rate, 4),
        "pitcher_hr_rate": round(p_rate, 4),
        "matchup_rate": round(matchup_rate, 4),
        "park_factor": round(pf, 3),
        "weather_factor": round(wf, 3),
        "platoon_factor": round(plat, 3),
        "recent_form_factor": round(rf, 3),
        "adj_rate_per_pa": round(adj_rate, 4),
        "expected_pa": est_pa,
        "lambda": round(lam, 4),
        "season_pa": season_pa,
    }
    return float(np.clip(prob, 0, 0.999)), factors


# ---------------------------------------------------------------------------
# Statcast quality-of-contact model path
# ---------------------------------------------------------------------------
def statcast_probability(metrics: dict) -> tuple[float, float]:
    """
    Returns (statcast_prob, confidence) where confidence is how much
    data backs the estimate (0→1 based on sample size).
    """
    n = metrics.get("n_batted_balls", 0)
    if n < 30:
        return 0.15, 0.0     # insufficient data — use prior, zero confidence

    br = metrics.get("barrel_rate", None)
    ev = metrics.get("avg_exit_velocity", None)
    hh = metrics.get("hard_hit_rate", None)
    fb = metrics.get("fly_ball_rate", None)

    if br is None or math.isnan(float(br if br is not None else float("nan"))):
        return 0.15, 0.0

    br = float(br)
    ev = float(ev) if ev is not None else 88.3
    hh = float(hh) if hh is not None else 0.37
    fb = float(fb) if fb is not None else 0.35

    # Linear combination → logit space → probability
    # Coefficients derived from Statcast/wRC+ research
    score = (
        8.0 * br           # barrel rate — strongest signal
        + 0.08 * (ev - 88) # exit velocity above avg
        + 1.5 * hh         # hard hit rate
        + 1.0 * fb         # fly ball rate
        - 1.80             # intercept (calibrated so avg player ≈ 0.18)
    )
    prob = 1.0 / (1.0 + math.exp(-score))

    confidence = float(np.clip((n - 30) / 200, 0, 1))
    return float(np.clip(prob, 0.01, 0.99)), confidence


# ---------------------------------------------------------------------------
# Main ensemble combiner
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
) -> dict:
    """
    Full ensemble prediction for one batter in today's game.

    Returns
    -------
    dict with:
        hr_probability (0-1), hr_pct (string), confidence_tier,
        stat_prob, statcast_prob, factors (debug info)
    """
    model_cfg = config.get("model", {})
    weights = model_cfg.get("weights", {})

    # Path 1: statistical
    stat_prob, factors = statistical_probability(
        batter_season, batter_career,
        pitcher_season, pitcher_career,
        pitcher_throws, batter_bats,
        platoon_splits, recent_games,
        venue, weather, config,
    )

    # Path 2: Statcast
    sc_prob, sc_confidence = statcast_probability(statcast_metrics)

    # Blend: weight Statcast by its confidence
    stat_w = 0.60
    sc_w = 0.40 * sc_confidence
    total_w = stat_w + sc_w

    if total_w < 0.01:
        final_prob = stat_prob
    else:
        final_prob = (stat_w * stat_prob + sc_w * sc_prob) / total_w

    final_prob = float(np.clip(final_prob, 0, 0.999))

    # Confidence tier based on sample size + Statcast availability
    season_pa = factors.get("season_pa", 0)
    if season_pa >= 150 and sc_confidence > 0.5:
        tier = "High"
    elif season_pa >= 80 or sc_confidence > 0.3:
        tier = "Medium"
    else:
        tier = "Low"

    return {
        "hr_probability": round(final_prob, 4),
        "hr_pct": f"{final_prob * 100:.1f}%",
        "confidence_tier": tier,
        "statistical_prob": round(stat_prob, 4),
        "statcast_prob": round(sc_prob, 4),
        "statcast_confidence": round(sc_confidence, 2),
        "factors": factors,
        "statcast_metrics": {
            k: v for k, v in statcast_metrics.items()
            if k in ("barrel_rate", "hard_hit_rate", "avg_exit_velocity", "n_batted_balls")
        },
    }
