"""
predictor.py
============
Ensemble prediction — five outputs per batter:
  HR probability, expected TB, expected H+R+RBI, expected RBI.

Each output blends XGBoost (when model available) with the statistical baseline.
Game-level adjustments (park, weather, pitcher, platoon, recent form) applied to all.
"""
from __future__ import annotations
import logging, math
import numpy as np
from src.features.engineer import (
    LG_HR_PER_PA, LG_PA_PER_GAME,
    LG_TB_PER_PA, LG_H_PER_PA, LG_R_PER_GAME, LG_RBI_PER_GAME,
    bayesian_blend, hr_rate_from_stats,
    tb_rate_from_stats, h_rate_from_stats,
    r_rate_from_stats, rbi_rate_from_stats,
    log5, park_factor, pitcher_hit_factor,
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


def _blend(xgb_rate, stat_rate, name: str) -> tuple[float, str]:
    """Blend XGBoost and statistical rates, return (blended_rate, source_label)."""
    if xgb_rate is not None:
        return 0.65 * xgb_rate + 0.35 * stat_rate, f"xgb+stat"
    return stat_rate, "statistical"


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
    xgb_models: dict | None = None,    # {name: model}
    xgb_metas:  dict | None = None,    # {name: meta}
    # Legacy single-model support
    xgb_model=None,
    xgb_meta: dict | None = None,
) -> dict:
    """Return all five batter predictions for tonight's game."""
    from src.models.model_registry import predict_rate

    # Handle legacy single-model interface
    if xgb_models is None:
        xgb_models = {"hr": xgb_model} if xgb_model else {}
    if xgb_metas is None:
        xgb_metas = {"hr": xgb_meta} if xgb_meta else {}

    model_cfg = config.get("model", {})
    est_pa    = float(model_cfg.get("lg_pa_per_game", LG_PA_PER_GAME))

    # ── Shared game-level factors ─────────────────────────────────────────
    pf      = park_factor(venue, config)
    wf      = weather_factor(weather, config)
    plat    = platoon_adjustment(batter_bats, pitcher_throws, platoon_splits)
    rf      = recent_form_factor(recent_games, model_cfg.get("recent_form_games", 15))
    p_hit_f = pitcher_hit_factor(pitcher_season, pitcher_career)
    home_f  = 1.0

    def _xgb(name: str) -> float | None:
        return predict_rate(name, statcast_metrics, batter_season, xgb_models, xgb_metas)

    # ── HR ────────────────────────────────────────────────────────────────
    stat_hr_rate, season_pa = hr_rate_from_stats(batter_season, batter_career)
    xgb_hr     = _xgb("hr")
    base_hr, src_hr = _blend(xgb_hr, stat_hr_rate, "hr")

    p_hr_rate    = _pitcher_hr_pa_rate(pitcher_season, pitcher_career)
    matchup_rate = log5(base_hr, p_hr_rate, LG_HR_PER_PA)
    adj_hr_rate  = float(np.clip(matchup_rate * pf * wf * plat * rf, 1e-5, 0.30))
    hr_lam       = adj_hr_rate * est_pa
    hr_prob      = float(np.clip(1.0 - math.exp(-hr_lam), 0, 0.999))

    # Statcast sanity check on HR
    n_bb = statcast_metrics.get("n_batted_balls", 0)
    br   = statcast_metrics.get("barrel_rate")
    sc_prob, sc_conf = 0.15, 0.0
    if n_bb >= 30 and br is not None and br == br:
        ev = float(statcast_metrics.get("avg_exit_velocity") or 88.3)
        hh = float(statcast_metrics.get("hard_hit_rate") or 0.37)
        fb = float(statcast_metrics.get("fly_ball_rate") or 0.35)
        sc_score = 8.0*float(br) + 0.08*(ev-88) + 1.5*hh + 1.0*fb - 1.80
        sc_prob  = float(np.clip(1/(1+math.exp(-sc_score)), 0.01, 0.99))
        sc_conf  = float(np.clip((n_bb-30)/200, 0, 1))
    sc_w = 0.15 * sc_conf
    final_hr_prob = float(np.clip((1-sc_w)*hr_prob + sc_w*sc_prob, 0, 0.999))

    # ── Total Bases ────────────────────────────────────────────────────────
    stat_tb_rate, _ = tb_rate_from_stats(batter_season, batter_career)
    xgb_tb          = _xgb("tb")
    base_tb, src_tb = _blend(xgb_tb, stat_tb_rate, "tb")
    exp_tb          = expected_tb_tonight(base_tb, est_pa, pf, wf, p_hit_f, rf)
    tb_lam          = float(np.clip(base_tb * est_pa * pf * wf * p_hit_f * rf, 0.001, 8.0))
    tb_prob_2       = float(np.clip(1.0 - math.exp(-tb_lam)*(1+tb_lam), 0, 0.999))

    # ── Hits ──────────────────────────────────────────────────────────────
    stat_h_rate = h_rate_from_stats(batter_season, batter_career)
    xgb_h       = _xgb("h")
    base_h, src_h = _blend(xgb_h, stat_h_rate, "h")

    # ── Runs ──────────────────────────────────────────────────────────────
    stat_r_rate = r_rate_from_stats(batter_season, batter_career)
    xgb_r       = _xgb("r")
    base_r, src_r = _blend(xgb_r, stat_r_rate, "r")

    # ── RBI ───────────────────────────────────────────────────────────────
    stat_rbi_rate = rbi_rate_from_stats(batter_season, batter_career)
    xgb_rbi       = _xgb("rbi")
    base_rbi, src_rbi = _blend(xgb_rbi, stat_rbi_rate, "rbi")

    # ── Expected H, R, RBI tonight ────────────────────────────────────────
    exp_h, exp_r, exp_rbi_hrbi = expected_hrbi_tonight(
        base_h, base_r, base_rbi, est_pa, p_hit_f, home_f, rf
    )
    exp_hrbi    = round(exp_h + exp_r + exp_rbi_hrbi, 2)
    exp_rbi_only = float(np.clip(base_rbi * p_hit_f * rf, 0, 3.0))

    # ── Confidence tier ───────────────────────────────────────────────────
    n_models_loaded = sum(1 for m in xgb_models.values() if m is not None)
    if n_models_loaded >= 3 and season_pa >= 100 and sc_conf > 0.4:
        tier = "High"
    elif season_pa >= 60 or sc_conf > 0.2 or n_models_loaded >= 1:
        tier = "Medium"
    else:
        tier = "Low"

    return {
        # HR
        "hr_probability":    round(final_hr_prob, 4),
        "hr_pct":            f"{final_hr_prob*100:.1f}%",
        # TB
        "expected_tb":       round(exp_tb, 2),
        "tb_prob_over_1_5":  round(tb_prob_2, 4),
        "tb_pct":            f"{tb_prob_2*100:.1f}%",
        # H+R+RBI
        "expected_hrbi":     exp_hrbi,
        "expected_h":        round(exp_h, 2),
        "expected_r":        round(exp_r, 2),
        "expected_rbi_hrbi": round(exp_rbi_hrbi, 2),
        # RBI
        "expected_rbi":      round(exp_rbi_only, 2),
        # Meta
        "confidence_tier":   tier,
        "rate_sources":      {"hr":src_hr,"tb":src_tb,"h":src_h,"r":src_r,"rbi":src_rbi},
        "rate_source":       src_hr,   # legacy field
        "statistical_prob":  round(hr_prob, 4),
        "statcast_prob":     round(sc_prob, 4),
        "statcast_confidence": round(sc_conf, 2),
        "factors": {
            "batter_hr_rate":     round(base_hr, 4),
            "pitcher_hr_rate":    round(p_hr_rate, 4),
            "pitcher_hit_factor": round(p_hit_f, 3),
            "park_factor":        round(pf, 3),
            "weather_factor":     round(wf, 3),
            "platoon_factor":     round(plat, 3),
            "recent_form_factor": round(rf, 3),
            "adj_hr_rate":        round(adj_hr_rate, 4),
            "season_pa":          season_pa,
        },
        "statcast_metrics": {
            k: v for k, v in statcast_metrics.items()
            if k in ("barrel_rate","hard_hit_rate","avg_exit_velocity","n_batted_balls")
        },
    }
