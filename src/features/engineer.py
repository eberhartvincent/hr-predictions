"""
engineer.py
===========
Converts raw MLB API + Statcast data into normalized feature vectors
ready for the prediction model.

Feature philosophy
------------------
* Barrel rate is the single best Statcast predictor of HR power — it gets
  the highest weight.
* All rates use a Bayesian shrinkage blend of season + career data to
  stabilize small samples.
* Pitcher adjustments use a log-odds ratio (log5 extension): the expected
  outcome when a batter with rate p faces a pitcher who allows rate q.
* Weather and park effects are multiplicative adjustments layered on top.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# League average constants (2022-24 MLB average, update yearly)
LG_HR_PER_PA = 0.033          # ~3.3% PA result in HR
LG_BARREL_RATE = 0.069         # 6.9% of batted balls are barrels
LG_HARD_HIT_RATE = 0.370       # 37.0% hard-hit rate
LG_AVG_EXIT_VELOCITY = 88.3    # mph
LG_PITCHER_HR9 = 1.30          # league average HR/9 IP
LG_PA_PER_GAME = 4.1           # average PA per player per game


def _safe(val: Any, default: float = 0.0) -> float:
    """Convert to float, fallback to default on None/NaN/empty."""
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
    """
    Beta-binomial Bayesian shrinkage.
    Pulls season rate toward (career rate) using prior_weight pseudo-obs,
    then blends with career.
    """
    # Season blended toward league prior
    season_blend_n = season_n + prior_weight
    season_blended = (season_rate * season_n + prior * prior_weight) / season_blend_n

    if career_n < 50:
        return season_blended

    # Weight career more when more data is available
    career_w = min(career_n / (career_n + 500), 0.40)   # max 40% career
    return (1 - career_w) * season_blended + career_w * career_rate


def hr_rate_from_stats(season: dict, career: dict) -> tuple[float, int]:
    """
    Compute blended HR/PA rate using Bayesian shrinkage.
    Returns (blended_rate, season_pa).
    """
    s_hr = _safe(season.get("homeRuns", 0))
    s_pa = _safe(season.get("plateAppearances", 0))
    c_hr = _safe(career.get("homeRuns", 0))
    c_pa = _safe(career.get("plateAppearances", 0))

    s_rate = s_hr / s_pa if s_pa > 0 else LG_HR_PER_PA
    c_rate = c_hr / c_pa if c_pa > 0 else LG_HR_PER_PA

    blended = bayesian_blend(
        season_rate=s_rate,
        season_n=int(s_pa),
        career_rate=c_rate,
        career_n=int(c_pa),
        prior=LG_HR_PER_PA,
        prior_weight=300,
    )
    return blended, int(s_pa)


def pitcher_hr_factor(season: dict, career: dict) -> float:
    """
    Return a multiplicative factor for how HR-prone this pitcher is vs average.
    1.0 = league average. >1.0 = allows more HR.
    Uses HR/9 innings blended with career.
    """
    s_hr9 = _safe(season.get("homeRunsPer9Inn", season.get("hr9", LG_PITCHER_HR9)))
    c_hr9 = _safe(career.get("homeRunsPer9Inn", career.get("hr9", LG_PITCHER_HR9)))
    s_ip = _safe(season.get("inningsPitched", 0))
    c_ip = _safe(career.get("inningsPitched", 0))

    if s_ip < 5:
        blended_hr9 = c_hr9 if c_ip > 30 else LG_PITCHER_HR9
    else:
        w = min(s_ip / (s_ip + 200), 0.7)
        blended_hr9 = w * s_hr9 + (1 - w) * (c_hr9 if c_ip > 30 else LG_PITCHER_HR9)

    factor = blended_hr9 / LG_PITCHER_HR9
    return float(np.clip(factor, 0.5, 2.5))


def log5(p_batter: float, q_pitcher: float, lg_avg: float = LG_HR_PER_PA) -> float:
    """
    Bill James log5 formula: probability of event when batter with rate p
    faces pitcher who allows rate q, given league average lg_avg.

    log5(p, q) = (p*q/lg) / (p*q/lg + (1-p)*(1-q)/(1-lg))
    """
    eps = 1e-9
    p = np.clip(p_batter, eps, 1 - eps)
    q = np.clip(q_pitcher, eps, 1 - eps)
    lg = np.clip(lg_avg, eps, 1 - eps)

    num = p * q / lg
    den = num + (1 - p) * (1 - q) / (1 - lg)
    return float(num / den)


def platoon_adjustment(batter_bats: str, pitcher_throws: str, platoon_stats: dict) -> float:
    """
    Return a multiplicative platoon factor based on batter/pitcher handedness.
    Uses actual split HR rates if available, otherwise applies a fixed lookup.
    """
    # Fixed platoon penalties/bonuses (HR basis, from research)
    # Same-side matchup is disadvantageous for batter
    same_side_penalty = 0.88    # batter same side as pitcher: ~12% HR reduction
    opp_side_boost = 1.08       # opposite side: ~8% HR boost

    if batter_bats == "S":      # switch hitter — no platoon disadvantage
        return 1.0

    same_side = batter_bats == pitcher_throws

    # Try to refine using actual split data
    split_key = "vs_right" if pitcher_throws == "R" else "vs_left"
    split_stats = platoon_stats.get(split_key, {})
    opp_key = "vs_left" if pitcher_throws == "R" else "vs_right"
    opp_stats = platoon_stats.get(opp_key, {})

    split_pa = _safe(split_stats.get("plateAppearances", 0))
    split_hr = _safe(split_stats.get("homeRuns", 0))
    opp_pa = _safe(opp_stats.get("plateAppearances", 0))
    opp_hr = _safe(opp_stats.get("homeRuns", 0))

    if split_pa >= 50 and opp_pa >= 50:
        split_rate = split_hr / split_pa
        opp_rate = opp_hr / opp_pa
        # Blended with the fixed expectation
        expected = LG_HR_PER_PA * (same_side_penalty if same_side else opp_side_boost)
        w_data = min(min(split_pa, opp_pa) / 200, 0.6)
        overall_rate = (split_hr + opp_hr) / (split_pa + opp_pa)
        factor = split_rate / overall_rate if overall_rate > 0 else (same_side_penalty if same_side else opp_side_boost)
        return float(np.clip((1 - w_data) * (same_side_penalty if same_side else opp_side_boost) + w_data * factor, 0.6, 1.5))

    return same_side_penalty if same_side else opp_side_boost


def recent_form_factor(recent_games: list[dict], window: int = 15) -> float:
    """
    Compare recent HR rate to the expected average.
    Returns a multiplicative factor: 1.0 = average form.
    """
    games = recent_games[:window]
    if len(games) < 3:
        return 1.0

    total_ab = sum(_safe(g.get("atBats", 0)) for g in games)
    total_hr = sum(_safe(g.get("homeRuns", 0)) for g in games)
    if total_ab < 10:
        return 1.0

    recent_rate = total_hr / total_ab
    factor = recent_rate / LG_HR_PER_PA   # relative to league
    # Shrink heavily toward 1.0 — recent form is noisy
    shrunk = 1.0 + 0.25 * (factor - 1.0)
    return float(np.clip(shrunk, 0.8, 1.25))


def weather_factor(weather: dict, config: dict) -> float:
    """
    Multiplicative HR factor from temperature and wind.
    Returns 1.0 for covered stadiums.
    """
    if weather.get("covered", False):
        return 1.0

    wcfg = config.get("weather", {})
    temp = _safe(weather.get("temperature_f", 72))
    wind_speed = _safe(weather.get("wind_speed_mph", 0))
    wind_cat = weather.get("wind_category", "calm")
    temp_baseline = _safe(wcfg.get("temp_baseline_f", 70))
    temp_effect = _safe(wcfg.get("temp_effect_per_10f", 0.012))
    wind_out = _safe(wcfg.get("wind_out_boost", 0.10))
    wind_in = _safe(wcfg.get("wind_in_penalty", -0.10))
    wind_threshold = _safe(wcfg.get("wind_threshold_mph", 10))

    # Temperature: ball travels ~1.2% farther per 10°F above 70°F
    temp_adj = 1.0 + temp_effect * (temp - temp_baseline) / 10.0

    # Wind adjustment
    wind_adj = 1.0
    if wind_speed >= wind_threshold:
        if wind_cat in ("out_to_cf", "out_to_rf", "out_to_lf"):
            wind_adj = 1.0 + wind_out * min(wind_speed / wind_threshold, 2.0)
        elif wind_cat == "in_from_cf":
            wind_adj = 1.0 + wind_in * min(wind_speed / wind_threshold, 2.0)
        # crosswind: negligible effect

    total = float(np.clip(temp_adj * wind_adj, 0.7, 1.6))
    return total


def park_factor(venue: str, config: dict) -> float:
    """Return the park's HR factor from config."""
    factors = config.get("parks", {}).get("factors", {})
    return float(factors.get(venue, factors.get("default", 1.0)))


def barrel_score(metrics: dict) -> float:
    """
    Normalize barrel rate to a 0–1 score relative to league average.
    Capped at 3x league average.
    """
    br = _safe(metrics.get("barrel_rate"), LG_BARREL_RATE)
    if math.isnan(br):
        br = LG_BARREL_RATE
    return float(np.clip(br / (LG_BARREL_RATE * 3), 0, 1))


def exit_velocity_score(metrics: dict) -> float:
    """Normalize average exit velocity to a 0–1 score."""
    ev = _safe(metrics.get("avg_exit_velocity"), LG_AVG_EXIT_VELOCITY)
    if math.isnan(ev):
        ev = LG_AVG_EXIT_VELOCITY
    # Scale: 78 mph (low) → 0, 105 mph (elite) → 1
    return float(np.clip((ev - 78) / (105 - 78), 0, 1))


# ---------------------------------------------------------------------------
# Patch: replace park_factor with fuzzy-matching version
# (The original definition above is superseded by this one at import time
#  because Python uses the last definition of a name in a module.)
# ---------------------------------------------------------------------------
import difflib as _difflib   # noqa: E402 (late import OK — stdlib only)


def park_factor(venue: str, config: dict) -> float:  # type: ignore[no-redef]
    """
    Return the park's HR factor from config.
    Uses difflib fuzzy match so slight venue-name variants still resolve
    (e.g. 'Oriole Park at Camden Yards' ≈ 'Camden Yards').
    """
    factors: dict = config.get("parks", {}).get("factors", {})
    default = float(factors.get("default", 1.0))
    venue = (venue or "").strip()

    if venue in factors:
        return float(factors[venue])

    known = [k for k in factors if k != "default"]
    matches = _difflib.get_close_matches(venue, known, n=1, cutoff=0.72)
    if matches:
        log.debug("Park factor fuzzy-matched %r → %r", venue, matches[0])
        return float(factors[matches[0]])

    log.debug("Park factor not found for %r, using default %.2f", venue, default)
    return default


# ===========================================================================
# Additional rate functions for TB, H, R, RBI
# ===========================================================================

LG_TB_PER_PA    = 0.358
LG_H_PER_PA     = 0.216
LG_R_PER_GAME   = 0.490
LG_RBI_PER_GAME = 0.490
LG_ERA          = 4.20


def _tb_from_stats_dict(s: dict) -> float:
    """Compute total bases from counting stats dict."""
    hits  = _safe(s.get("hits", 0))
    d     = _safe(s.get("doubles", 0))
    t     = _safe(s.get("triples", 0))
    hr    = _safe(s.get("homeRuns", 0))
    # TB = 1B + 2×2B + 3×3B + 4×HR = hits + doubles + 2×triples + 3×HRs
    return hits + d + 2 * t + 3 * hr


def tb_rate_from_stats(season: dict, career: dict) -> tuple[float, int]:
    """Blended TB/PA rate."""
    s_tb = _tb_from_stats_dict(season)
    s_pa = _safe(season.get("plateAppearances", 0))
    c_tb = _tb_from_stats_dict(career)
    c_pa = _safe(career.get("plateAppearances", 0))

    s_rate = s_tb / s_pa if s_pa > 0 else LG_TB_PER_PA
    c_rate = c_tb / c_pa if c_pa > 0 else LG_TB_PER_PA

    blended = bayesian_blend(
        season_rate=s_rate, season_n=int(s_pa),
        career_rate=c_rate, career_n=int(c_pa),
        prior=LG_TB_PER_PA, prior_weight=400,
    )
    return float(np.clip(blended, 0.0, 1.5)), int(s_pa)


def h_rate_from_stats(season: dict, career: dict) -> float:
    """Blended H/PA rate."""
    s_h  = _safe(season.get("hits", 0))
    s_pa = _safe(season.get("plateAppearances", 0))
    c_h  = _safe(career.get("hits", 0))
    c_pa = _safe(career.get("plateAppearances", 0))

    s_rate = s_h / s_pa if s_pa > 0 else LG_H_PER_PA
    c_rate = c_h / c_pa if c_pa > 0 else LG_H_PER_PA

    return float(np.clip(bayesian_blend(
        season_rate=s_rate, season_n=int(s_pa),
        career_rate=c_rate, career_n=int(c_pa),
        prior=LG_H_PER_PA, prior_weight=400,
    ), 0.0, 0.6))


def r_rate_from_stats(season: dict, career: dict) -> float:
    """Blended R/game rate."""
    s_r  = _safe(season.get("runs", 0))
    s_gp = _safe(season.get("gamesPlayed", 1))
    c_r  = _safe(career.get("runs", 0))
    c_gp = _safe(career.get("gamesPlayed", 1))

    s_rate = s_r / s_gp if s_gp > 0 else LG_R_PER_GAME
    c_rate = c_r / c_gp if c_gp > 0 else LG_R_PER_GAME

    return float(np.clip(bayesian_blend(
        season_rate=s_rate, season_n=int(s_gp) * 4,
        career_rate=c_rate, career_n=int(c_gp) * 4,
        prior=LG_R_PER_GAME, prior_weight=300,
    ), 0.0, 2.0))


def rbi_rate_from_stats(season: dict, career: dict) -> float:
    """Blended RBI/game rate."""
    s_rbi = _safe(season.get("rbi", 0))
    s_gp  = _safe(season.get("gamesPlayed", 1))
    c_rbi = _safe(career.get("rbi", 0))
    c_gp  = _safe(career.get("gamesPlayed", 1))

    s_rate = s_rbi / s_gp if s_gp > 0 else LG_RBI_PER_GAME
    c_rate = c_rbi / c_gp if c_gp > 0 else LG_RBI_PER_GAME

    return float(np.clip(bayesian_blend(
        season_rate=s_rate, season_n=int(s_gp) * 4,
        career_rate=c_rate, career_n=int(c_gp) * 4,
        prior=LG_RBI_PER_GAME, prior_weight=300,
    ), 0.0, 2.0))


def pitcher_hit_factor(p_season: dict, p_career: dict) -> float:
    """
    Multiplicative factor for how hit-prone this pitcher is.
    1.0 = league average. >1.0 = more hits/TB/RBI allowed.
    Uses ERA relative to league average as a broad quality proxy.
    """
    s_era = _safe(p_season.get("era", LG_ERA))
    s_ip  = _safe(p_season.get("inningsPitched", 0))
    c_era = _safe(p_career.get("era", LG_ERA))
    c_ip  = _safe(p_career.get("inningsPitched", 0))

    if s_ip < 5:
        blended_era = c_era if c_ip > 30 else LG_ERA
    else:
        w = min(s_ip / (s_ip + 150), 0.7)
        blended_era = w * s_era + (1 - w) * (c_era if c_ip > 30 else LG_ERA)

    factor = blended_era / LG_ERA
    return float(np.clip(factor, 0.6, 1.6))


def expected_tb_tonight(
    tb_rate: float, est_pa: float,
    pf: float, wf: float, pitcher_hit_f: float, recent_f: float,
) -> float:
    """Expected total bases in tonight's game."""
    return float(np.clip(
        tb_rate * est_pa * pf * wf * pitcher_hit_f * recent_f,
        0.0, 8.0,
    ))


def expected_hrbi_tonight(
    h_rate: float, r_rate: float, rbi_rate: float,
    est_pa: float, pitcher_hit_f: float, home_f: float, recent_f: float,
) -> tuple[float, float, float]:
    """
    Expected H, R, RBI tonight. Returns (exp_h, exp_r, exp_rbi).
    Runs depend partly on lineup context so we apply a softer adjustment.
    """
    exp_h   = float(np.clip(h_rate * est_pa * pitcher_hit_f * recent_f, 0, 4.0))
    exp_r   = float(np.clip(r_rate * home_f * recent_f, 0, 3.0))
    exp_rbi = float(np.clip(rbi_rate * pitcher_hit_f * recent_f, 0, 3.0))
    return exp_h, exp_r, exp_rbi
