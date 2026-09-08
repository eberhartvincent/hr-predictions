"""
engineer.py — Feature engineering for MLB batter predictions.

Bug fixes in this version:
  - recent_form_factor now uses total bases (stable) instead of HR count (noisy)
  - pitcher_hit_factor now uses K% instead of ERA (defense-independent)
  - home_factor added for R and RBI sections
  - Config model.weights block removed (was documented but never used)
  - League averages updated to 2023-25 actuals
"""
from __future__ import annotations

import difflib as _difflib
import logging
import math
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# League average constants (2023-25 MLB)
LG_HR_PER_PA         = 0.031   # 3.1% — slightly down from 2019 peak
LG_BARREL_RATE       = 0.069
LG_HARD_HIT_RATE     = 0.370
LG_AVG_EXIT_VELOCITY = 88.3
LG_PA_PER_GAME       = 4.1
LG_SLG               = 0.411   # for total bases form calculation
LG_K_RATE_PITCHER    = 0.225   # pitcher K rate (SO / estimated PA)

# New constants for additional sections
LG_TB_PER_PA    = 0.358
LG_H_PER_PA     = 0.216
LG_R_PER_GAME   = 0.490
LG_RBI_PER_GAME = 0.490
LG_ERA          = 4.20

# Home advantage
HOME_BOOST = 1.05


def _safe(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def bayesian_blend(
    season_rate: float,
    season_n: int,
    career_rate: float,
    career_n: int,
    prior: float,
    prior_weight: int = 200,
) -> float:
    season_blend_n   = season_n + prior_weight
    season_blended   = (season_rate * season_n + prior * prior_weight) / season_blend_n
    if career_n < 50:
        return season_blended
    career_w = min(career_n / (career_n + 500), 0.40)
    return (1 - career_w) * season_blended + career_w * career_rate


def hr_rate_from_stats(season: dict, career: dict) -> tuple[float, int]:
    s_hr = _safe(season.get("homeRuns", 0))
    s_pa = _safe(season.get("plateAppearances", 0))
    c_hr = _safe(career.get("homeRuns", 0))
    c_pa = _safe(career.get("plateAppearances", 0))
    s_rate = s_hr / s_pa if s_pa > 0 else LG_HR_PER_PA
    c_rate = c_hr / c_pa if c_pa > 0 else LG_HR_PER_PA
    blended = bayesian_blend(s_rate, int(s_pa), c_rate, int(c_pa),
                              LG_HR_PER_PA, prior_weight=300)
    return float(np.clip(blended, 0, 0.15)), int(s_pa)


def pitcher_hr_pa_rate(p_season: dict, p_career: dict) -> float:
    s_hr = _safe(p_season.get("homeRuns", 0))
    s_ip = _safe(p_season.get("inningsPitched", 0))
    c_hr = _safe(p_career.get("homeRuns", 0))
    c_ip = _safe(p_career.get("inningsPitched", 0))
    pa_per_inn = 3.3
    s_pa = s_ip * pa_per_inn
    c_pa = c_ip * pa_per_inn
    s_rate = s_hr / s_pa if s_pa > 0 else LG_HR_PER_PA
    c_rate = c_hr / c_pa if c_pa > 0 else LG_HR_PER_PA
    blended = bayesian_blend(s_rate, int(s_pa), c_rate, int(c_pa),
                              LG_HR_PER_PA, prior_weight=400)
    return float(np.clip(blended, 0.005, 0.10))


def log5(p_batter: float, q_pitcher: float, lg_avg: float = LG_HR_PER_PA) -> float:
    eps = 1e-9
    p  = np.clip(p_batter,  eps, 1 - eps)
    q  = np.clip(q_pitcher, eps, 1 - eps)
    lg = np.clip(lg_avg,    eps, 1 - eps)
    num = p * q / lg
    den = num + (1 - p) * (1 - q) / (1 - lg)
    return float(num / den)


def platoon_adjustment(batter_bats: str, pitcher_throws: str, platoon_stats: dict) -> float:
    same_side_penalty = 0.88
    opp_side_boost    = 1.08
    if batter_bats == "S":
        return 1.0
    same_side  = batter_bats == pitcher_throws
    split_key  = "vs_right" if pitcher_throws == "R" else "vs_left"
    opp_key    = "vs_left"  if pitcher_throws == "R" else "vs_right"
    split_stats = platoon_stats.get(split_key, {})
    opp_stats   = platoon_stats.get(opp_key,   {})
    split_pa = _safe(split_stats.get("plateAppearances", 0))
    split_hr = _safe(split_stats.get("homeRuns", 0))
    opp_pa   = _safe(opp_stats.get("plateAppearances", 0))
    opp_hr   = _safe(opp_stats.get("homeRuns", 0))
    if split_pa >= 50 and opp_pa >= 50:
        overall_rate = (split_hr + opp_hr) / (split_pa + opp_pa)
        split_rate   = split_hr / split_pa if split_pa > 0 else LG_HR_PER_PA
        factor       = split_rate / overall_rate if overall_rate > 0 else (
            same_side_penalty if same_side else opp_side_boost)
        w_data   = min(min(split_pa, opp_pa) / 200, 0.6)
        expected = same_side_penalty if same_side else opp_side_boost
        return float(np.clip((1 - w_data) * expected + w_data * factor, 0.6, 1.5))
    return same_side_penalty if same_side else opp_side_boost


def recent_form_factor(recent_games: list[dict], window: int = 15) -> float:
    """
    BUG FIX: Was using HR count — too noisy (0-2 HRs in 15 games for most players).
    Now uses total bases (SLG proxy) — stable enough to carry real signal.
    """
    games = recent_games[:window]
    if len(games) < 3:
        return 1.0

    total_tb = 0.0
    total_ab = 0.0
    for g in games:
        h  = _safe(g.get("hits",     0))
        d  = _safe(g.get("doubles",  0))
        t  = _safe(g.get("triples",  0))
        hr = _safe(g.get("homeRuns", 0))
        ab = _safe(g.get("atBats",   0))
        total_tb += h + d + 2 * t + 3 * hr
        total_ab += ab

    if total_ab < 20:
        return 1.0

    recent_slg = total_tb / total_ab
    factor     = recent_slg / LG_SLG
    # Shrink toward 1.0 — recent form is real but noisy
    shrunk = 1.0 + 0.30 * (factor - 1.0)
    return float(np.clip(shrunk, 0.80, 1.25))


def weather_factor(weather: dict, config: dict) -> float:
    if weather.get("covered", False):
        return 1.0
    wcfg       = config.get("weather", {})
    temp       = _safe(weather.get("temperature_f", 72))
    wind_speed = _safe(weather.get("wind_speed_mph", 0))
    wind_cat   = weather.get("wind_category", "calm")
    temp_baseline = _safe(wcfg.get("temp_baseline_f", 70))
    temp_effect   = _safe(wcfg.get("temp_effect_per_10f", 0.012))
    wind_out      = _safe(wcfg.get("wind_out_boost", 0.10))
    wind_in       = _safe(wcfg.get("wind_in_penalty", -0.10))
    wind_threshold= _safe(wcfg.get("wind_threshold_mph", 10))
    temp_adj  = 1.0 + temp_effect * (temp - temp_baseline) / 10.0
    wind_adj  = 1.0
    if wind_speed >= wind_threshold:
        if wind_cat in ("out_to_cf", "out_to_rf", "out_to_lf"):
            wind_adj = 1.0 + wind_out * min(wind_speed / wind_threshold, 2.0)
        elif wind_cat == "in_from_cf":
            wind_adj = 1.0 + wind_in  * min(wind_speed / wind_threshold, 2.0)
    return float(np.clip(temp_adj * wind_adj, 0.7, 1.6))


def park_factor(venue: str, config: dict) -> float:
    factors  = config.get("parks", {}).get("factors", {})
    default  = float(factors.get("default", 1.0))
    venue    = (venue or "").strip()
    if venue in factors:
        return float(factors[venue])
    known   = [k for k in factors if k != "default"]
    matches = _difflib.get_close_matches(venue, known, n=1, cutoff=0.72)
    if matches:
        log.debug("Park factor fuzzy-matched %r → %r", venue, matches[0])
        return float(factors[matches[0]])
    return default


def home_factor(is_home: bool) -> float:
    """
    BUG FIX: Was hardcoded to 1.0 in predictor.py.
    Home teams score ~5% more runs. Applied to R and RBI sections.
    """
    return HOME_BOOST if is_home else 1.0


def pitcher_hit_factor(p_season: dict, p_career: dict) -> float:
    """
    BUG FIX: Was using ERA (affected by defense, noisy).
    Now uses K% — defense-independent, cleaner signal.

    High K rate (e.g. Gerrit Cole 35%) → factor < 1.0 → fewer hits projected.
    Low K rate (e.g. weak starter 15%)  → factor > 1.0 → more hits projected.
    """
    s_so = _safe(p_season.get("strikeOuts", 0))
    s_ip = _safe(p_season.get("inningsPitched", 0))
    c_so = _safe(p_career.get("strikeOuts", 0))
    c_ip = _safe(p_career.get("inningsPitched", 0))
    pa_per_inn = 3.3
    s_pa = s_ip * pa_per_inn
    c_pa = c_ip * pa_per_inn
    s_k_rate = s_so / s_pa if s_pa > 0 else LG_K_RATE_PITCHER
    c_k_rate = c_so / c_pa if c_pa > 0 else LG_K_RATE_PITCHER
    if s_ip < 10:
        blended_k = c_k_rate if c_ip > 30 else LG_K_RATE_PITCHER
    else:
        w = min(s_ip / (s_ip + 100), 0.7)
        blended_k = w * s_k_rate + (1 - w) * (c_k_rate if c_ip > 30 else LG_K_RATE_PITCHER)
    # Invert: high K pitcher → lower factor (harder to get hits off them)
    factor = LG_K_RATE_PITCHER / max(blended_k, 0.05)
    return float(np.clip(factor, 0.65, 1.50))


# ---------------------------------------------------------------------------
# TB, H, R, RBI rate functions
# ---------------------------------------------------------------------------

def _tb_from_stats_dict(s: dict) -> float:
    h  = _safe(s.get("hits",     0))
    d  = _safe(s.get("doubles",  0))
    t  = _safe(s.get("triples",  0))
    hr = _safe(s.get("homeRuns", 0))
    return h + d + 2 * t + 3 * hr


def tb_rate_from_stats(season: dict, career: dict) -> tuple[float, int]:
    s_tb = _tb_from_stats_dict(season)
    s_pa = _safe(season.get("plateAppearances", 0))
    c_tb = _tb_from_stats_dict(career)
    c_pa = _safe(career.get("plateAppearances", 0))
    s_rate = s_tb / s_pa if s_pa > 0 else LG_TB_PER_PA
    c_rate = c_tb / c_pa if c_pa > 0 else LG_TB_PER_PA
    blended = bayesian_blend(s_rate, int(s_pa), c_rate, int(c_pa),
                              LG_TB_PER_PA, prior_weight=400)
    return float(np.clip(blended, 0.0, 1.5)), int(s_pa)


def h_rate_from_stats(season: dict, career: dict) -> float:
    s_h  = _safe(season.get("hits", 0))
    s_pa = _safe(season.get("plateAppearances", 0))
    c_h  = _safe(career.get("hits", 0))
    c_pa = _safe(career.get("plateAppearances", 0))
    s_rate = s_h / s_pa if s_pa > 0 else LG_H_PER_PA
    c_rate = c_h / c_pa if c_pa > 0 else LG_H_PER_PA
    return float(np.clip(bayesian_blend(s_rate, int(s_pa), c_rate, int(c_pa),
                                        LG_H_PER_PA, prior_weight=400), 0.0, 0.6))


def r_rate_from_stats(season: dict, career: dict) -> float:
    s_r  = _safe(season.get("runs", 0))
    s_gp = _safe(season.get("gamesPlayed", 1))
    c_r  = _safe(career.get("runs", 0))
    c_gp = _safe(career.get("gamesPlayed", 1))
    s_rate = s_r / s_gp if s_gp > 0 else LG_R_PER_GAME
    c_rate = c_r / c_gp if c_gp > 0 else LG_R_PER_GAME
    return float(np.clip(bayesian_blend(s_rate, int(s_gp) * 4, c_rate,
                                        int(c_gp) * 4, LG_R_PER_GAME, prior_weight=300),
                         0.0, 2.0))


def rbi_rate_from_stats(season: dict, career: dict) -> float:
    s_rbi = _safe(season.get("rbi", 0))
    s_gp  = _safe(season.get("gamesPlayed", 1))
    c_rbi = _safe(career.get("rbi", 0))
    c_gp  = _safe(career.get("gamesPlayed", 1))
    s_rate = s_rbi / s_gp if s_gp > 0 else LG_RBI_PER_GAME
    c_rate = c_rbi / c_gp if c_gp > 0 else LG_RBI_PER_GAME
    return float(np.clip(bayesian_blend(s_rate, int(s_gp) * 4, c_rate,
                                        int(c_gp) * 4, LG_RBI_PER_GAME, prior_weight=300),
                         0.0, 2.0))


def expected_tb_tonight(tb_rate, est_pa, pf, wf, pitcher_hit_f, rf) -> float:
    return float(np.clip(tb_rate * est_pa * pf * wf * pitcher_hit_f * rf, 0.0, 8.0))


def expected_hrbi_tonight(h_rate, r_rate, rbi_rate, est_pa,
                           pitcher_hit_f, home_f, rf) -> tuple[float, float, float]:
    exp_h   = float(np.clip(h_rate   * est_pa * pitcher_hit_f * rf, 0, 4.0))
    exp_r   = float(np.clip(r_rate   * home_f * rf,                 0, 3.0))
    exp_rbi = float(np.clip(rbi_rate * pitcher_hit_f * rf,           0, 3.0))
    return exp_h, exp_r, exp_rbi


def barrel_score(metrics: dict) -> float:
    br = _safe(metrics.get("barrel_rate"), LG_BARREL_RATE)
    return float(np.clip(br / (LG_BARREL_RATE * 3), 0, 1))


def exit_velocity_score(metrics: dict) -> float:
    ev = _safe(metrics.get("avg_exit_velocity"), LG_AVG_EXIT_VELOCITY)
    return float(np.clip((ev - 78) / (105 - 78), 0, 1))
