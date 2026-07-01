"""
predictor.py
============
Ensemble prediction — returns HR, TB, H+R+RBI, and RBI predictions per batter.
"""
from __future__ import annotations
import logging, math
import numpy as np
from src.features.engineer import (
    LG_HR_PER_PA, LG_PA_PER_GAME,
    LG_TB_PER_PA, LG_H_PER_PA, LG_R_PER_GAME, LG_RBI_PER_GAME,
    bayesian_blend, hr_rate_from_stats,
    tb_rate_from_stats, h_rate_from_stats, r_rate_from_stats, rbi_rate_from_stats,
    log5, park_factor, pitcher_hr_factor, pitcher_hit_factor,
    platoon_adjustment, recent_form_factor, weather_factor,
    expected_tb_tonight, expected_hrbi_tonight,
    _safe,
)

log = logging.getLogger(__name__)


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
    """Return HR, TB, H+R+RBI, and RBI predictions for one batter."""
    model_cfg = config.get("model", {})
    est_pa    = float(model_cfg.get("lg_pa_per_game", LG_PA_PER_GAME))

    # ── Shared game-level adjustments ─────────────────────────────────────
    pf       = park_factor(venue, config)
    wf       = weather_factor(weather, config)
    plat     = platoon_adjustment(batter_bats, pitcher_throws, platoon_splits)
    rf       = recent_form_factor(recent_games, config.get("model", {}).get("recent_form_games", 15))
    p_hit_f  = pitcher_hit_factor(pitcher_season, pitcher_career)
    home_f   = 1.0   # home/away not tracked per-batter in MLB pipeline

    # ── HR ────────────────────────────────────────────────────────────────
    stat_rate, season_pa = hr_rate_from_stats(batter_season, batter_career)

    xgb_rate = None
    if xgb_model is not None and xgb_meta is not None:
        try:
            from src.models.model_registry import predict_hr_rate
            xgb_rate = predict_hr_rate(statcast_metrics, batter_season, xgb_model, xgb_meta)
        except Exception:
            pass

    if xgb_rate is not None:
        batter_rate = 0.65 * xgb_rate + 0.35 * stat_rate
        rate_source = "xgboost+statistical"
    else:
        batter_rate = stat_rate
        rate_source = "statistical"

    p_hr_rate   = _pitcher_hr_pa_rate(pitcher_season, pitcher_career)
    matchup_rate = log5(batter_rate, p_hr_rate, LG_HR_PER_PA)
    adj_hr_rate  = float(np.clip(matchup_rate * pf * wf * plat * rf, 1e-5, 0.30))
    hr_lam       = adj_hr_rate * est_pa
    hr_prob      = float(np.clip(1.0 - math.exp(-hr_lam), 0, 0.999))

    # Statcast sanity check
    sc_prob = 0.15
    n_bb = statcast_metrics.get("n_batted_balls", 0)
    br   = statcast_metrics.get("barrel_rate")
    if n_bb >= 30 and br is not None and br == br:
        br = float(br)
        ev = float(statcast_metrics.get("avg_exit_velocity") or 88.3)
        hh = float(statcast_metrics.get("hard_hit_rate") or 0.37)
        fb = float(statcast_metrics.get("fly_ball_rate") or 0.35)
        sc_score = 8.0*br + 0.08*(ev-88) + 1.5*hh + 1.0*fb - 1.80
        sc_prob  = float(np.clip(1.0/(1.0+math.exp(-sc_score)), 0.01, 0.99))
    sc_conf = float(np.clip((n_bb - 30)/200, 0, 1)) if n_bb >= 30 else 0.0
    sc_w    = 0.15 * sc_conf
    final_hr_prob = float(np.clip((1-sc_w)*hr_prob + sc_w*sc_prob, 0, 0.999))

    # ── Total Bases ────────────────────────────────────────────────────────
    tb_rate, _ = tb_rate_from_stats(batter_season, batter_career)
    exp_tb      = expected_tb_tonight(tb_rate, est_pa, pf, wf, p_hit_f, rf)
    # P(TB ≥ 2) via Poisson — useful for over/under framing
    tb_lam      = tb_rate * est_pa * pf * wf * p_hit_f * rf
    tb_lam      = float(np.clip(tb_lam, 0.001, 8.0))
    tb_prob_2   = float(np.clip(1.0 - math.exp(-tb_lam)*(1 + tb_lam), 0, 0.999))

    # ── H + R + RBI ────────────────────────────────────────────────────────
    h_rate   = h_rate_from_stats(batter_season, batter_career)
    r_rate   = r_rate_from_stats(batter_season, batter_career)
    rbi_rate = rbi_rate_from_stats(batter_season, batter_career)
    exp_h, exp_r, exp_rbi = expected_hrbi_tonight(
        h_rate, r_rate, rbi_rate, est_pa, p_hit_f, home_f, rf
    )
    exp_hrbi = round(exp_h + exp_r + exp_rbi, 2)

    # ── RBI ────────────────────────────────────────────────────────────────
    exp_rbi_only = float(np.clip(rbi_rate * p_hit_f * rf, 0, 3.0))

    # ── Confidence ────────────────────────────────────────────────────────
    if xgb_rate is not None and season_pa >= 100 and sc_conf > 0.4:
        tier = "High"
    elif season_pa >= 60 or sc_conf > 0.2:
        tier = "Medium"
    else:
        tier = "Low"

    factors = {
        "batter_hr_rate":    round(batter_rate, 4),
        "pitcher_hr_rate":   round(p_hr_rate, 4),
        "pitcher_hit_factor":round(p_hit_f, 3),
        "park_factor":       round(pf, 3),
        "weather_factor":    round(wf, 3),
        "platoon_factor":    round(plat, 3),
        "recent_form_factor":round(rf, 3),
        "adj_hr_rate":       round(adj_hr_rate, 4),
        "season_pa":         season_pa,
        "lambda_hr":         round(hr_lam, 4),
        "lambda_tb":         round(tb_lam, 4),
    }

    return {
        # ── HR ─────────────────────────────────────────────────────────────
        "hr_probability":    round(final_hr_prob, 4),
        "hr_pct":            f"{final_hr_prob*100:.1f}%",

        # ── Total Bases ────────────────────────────────────────────────────
        "expected_tb":       round(exp_tb, 2),
        "tb_prob_over_1_5":  round(tb_prob_2, 4),   # P(TB ≥ 2) ≈ P(over 1.5)
        "tb_pct":            f"{tb_prob_2*100:.1f}%",

        # ── H + R + RBI ────────────────────────────────────────────────────
        "expected_hrbi":     exp_hrbi,
        "expected_h":        round(exp_h, 2),
        "expected_r":        round(exp_r, 2),
        "expected_rbi_hrbi": round(exp_rbi, 2),

        # ── RBI ────────────────────────────────────────────────────────────
        "expected_rbi":      round(exp_rbi_only, 2),

        # ── Shared ─────────────────────────────────────────────────────────
        "confidence_tier":   tier,
        "rate_source":       rate_source,
        "statistical_prob":  round(hr_prob, 4),
        "statcast_prob":     round(sc_prob, 4),
        "statcast_confidence": round(sc_conf, 2),
        "factors":           factors,
        "statcast_metrics": {
            k: v for k, v in statcast_metrics.items()
            if k in ("barrel_rate","hard_hit_rate","avg_exit_velocity","n_batted_balls")
        },
    }
