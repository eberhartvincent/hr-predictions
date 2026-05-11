"""
email_sender.py
===============
Sends the daily HR prediction report via SMTP (Gmail-compatible).

Required environment variables:
    EMAIL_SENDER      — from address (e.g. mybot@gmail.com)
    EMAIL_PASSWORD    — SMTP password / Gmail App Password
    EMAIL_RECIPIENTS  — comma-separated list of recipient addresses
    EMAIL_HOST        — (optional) SMTP host, default smtp.gmail.com
    EMAIL_PORT        — (optional) SMTP port, default 587
"""
from __future__ import annotations

import logging
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML template (inline — no Jinja required for this self-contained file)
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HR Predictions</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d1117;color:#e6edf3;font-family:'Syne',sans-serif;padding:0}
  .wrapper{max-width:680px;margin:0 auto;background:#0d1117}
  /* ── Header ── */
  .header{background:linear-gradient(135deg,#1a2332 0%,#0f1923 60%,#162032 100%);
          padding:36px 32px 28px;border-bottom:2px solid #21a96a}
  .header-top{display:flex;align-items:center;gap:12px;margin-bottom:6px}
  .logo{font-size:28px}
  .brand{font-size:13px;font-weight:600;letter-spacing:3px;text-transform:uppercase;
         color:#21a96a}
  .title{font-size:30px;font-weight:800;color:#ffffff;line-height:1.15;margin-bottom:4px}
  .subtitle{font-size:13px;color:#7d8590;font-family:'JetBrains Mono',monospace}
  .date-badge{display:inline-block;background:#21a96a18;border:1px solid #21a96a44;
              color:#21a96a;font-family:'JetBrains Mono',monospace;font-size:12px;
              padding:4px 12px;border-radius:20px;margin-top:10px}
  /* ── Summary bar ── */
  .summary{background:#161b22;padding:16px 32px;display:flex;gap:32px;
           border-bottom:1px solid #21262d}
  .stat-item{display:flex;flex-direction:column;gap:2px}
  .stat-label{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#7d8590}
  .stat-value{font-size:18px;font-weight:700;color:#e6edf3}
  /* ── Section header ── */
  .section-header{padding:20px 32px 12px;border-bottom:1px solid #21262d}
  .section-title{font-size:11px;font-weight:600;letter-spacing:3px;text-transform:uppercase;
                 color:#21a96a;margin-bottom:2px}
  /* ── Predictions list ── */
  .predictions{padding:0 32px}
  .player-card{border-bottom:1px solid #21262d1a;padding:20px 0;position:relative}
  .player-card:last-child{border-bottom:none}
  .card-top{display:flex;align-items:flex-start;gap:16px}
  .rank{width:36px;height:36px;border-radius:8px;display:flex;align-items:center;
        justify-content:center;font-size:13px;font-weight:800;flex-shrink:0;
        font-family:'JetBrains Mono',monospace}
  .rank-1{background:#f1c40f22;color:#f1c40f;border:1px solid #f1c40f44}
  .rank-2{background:#95a5a622;color:#bdc3c7;border:1px solid #95a5a644}
  .rank-3{background:#e67e2222;color:#e67e22;border:1px solid #e67e2244}
  .rank-other{background:#21262d;color:#7d8590;border:1px solid #30363d}
  .player-info{flex:1;min-width:0}
  .player-name{font-size:17px;font-weight:700;color:#e6edf3;margin-bottom:2px}
  .player-meta{font-size:12px;color:#7d8590;font-family:'JetBrains Mono',monospace}
  .prob-block{text-align:right;flex-shrink:0}
  .prob-pct{font-size:26px;font-weight:800;line-height:1;color:#21a96a}
  .prob-label{font-size:10px;color:#7d8590;letter-spacing:1px;text-transform:uppercase;margin-top:2px}
  /* ── Prob bar ── */
  .prob-bar-wrap{margin:10px 0 8px;height:4px;background:#21262d;border-radius:2px;overflow:hidden}
  .prob-bar{height:100%;border-radius:2px;
            background:linear-gradient(90deg,#21a96a,#3dd68c)}
  /* ── Factor chips ── */
  .factors{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
  .chip{font-family:'JetBrains Mono',monospace;font-size:10px;padding:3px 8px;
        border-radius:4px;display:inline-flex;align-items:center;gap:4px}
  .chip-park{background:#1d3a5c;color:#58a6ff;border:1px solid #1f6feb44}
  .chip-wind{background:#0d2818;color:#3fb950;border:1px solid #21a96a44}
  .chip-wind-bad{background:#3d1515;color:#f85149;border:1px solid #da363144}
  .chip-platoon{background:#2d1b4e;color:#d2a8ff;border:1px solid #8957e544}
  .chip-barrel{background:#3d2b00;color:#e3b341;border:1px solid #d2941344}
  .chip-ev{background:#1e2a1e;color:#56d364;border:1px solid #2ea04344}
  .chip-confidence-High{background:#0d2818;color:#3fb950;border:1px solid #2ea04344}
  .chip-confidence-Medium{background:#2d2200;color:#e3b341;border:1px solid #d2941344}
  .chip-confidence-Low{background:#3d1515;color:#f0883e;border:1px solid #e3631a44}
  /* ── Matchup row ── */
  .matchup{font-size:11px;color:#8b949e;font-family:'JetBrains Mono',monospace;
           margin-top:6px;padding:6px 10px;background:#161b22;border-radius:6px;
           border-left:2px solid #30363d}
  .matchup strong{color:#e6edf3}
  /* ── Footer ── */
  .footer{padding:28px 32px;border-top:1px solid #21262d;background:#0d1117}
  .footer-note{font-size:11px;color:#484f58;line-height:1.7;
               font-family:'JetBrains Mono',monospace}
  .footer-divider{height:1px;background:linear-gradient(90deg,transparent,#21a96a44,transparent);
                  margin:16px 0}
  .method-title{font-size:10px;font-weight:600;letter-spacing:2px;
                text-transform:uppercase;color:#21a96a;margin-bottom:8px}
  .method-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 24px}
  .method-item{font-size:10px;color:#484f58;font-family:'JetBrains Mono',monospace}
  .method-item span{color:#7d8590}
  a{color:#21a96a;text-decoration:none}
</style>
</head>
<body>
<div class="wrapper">
  <!-- HEADER -->
  <div class="header">
    <div class="header-top">
      <span class="logo">⚾</span>
      <span class="brand">MLB Analytics</span>
    </div>
    <div class="title">Home Run<br>Predictions</div>
    <div class="subtitle">Powered by Statcast · Log5 · Weather · Park Factors</div>
    <div class="date-badge">📅 {prediction_date}</div>
  </div>

  <!-- SUMMARY BAR -->
  <div class="summary">
    <div class="stat-item">
      <span class="stat-label">Players Ranked</span>
      <span class="stat-value">{top_n}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Games Today</span>
      <span class="stat-value">{games_today}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Avg Confidence</span>
      <span class="stat-value">{avg_prob}%</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Lineups Locked</span>
      <span class="stat-value">{lineups_confirmed}</span>
    </div>
  </div>

  <!-- PREDICTIONS -->
  <div class="section-header">
    <div class="section-title">Today's Top Picks</div>
  </div>

  <div class="predictions">
{player_cards}
  </div>

  <!-- FOOTER -->
  <div class="footer">
    <div class="method-title">Methodology</div>
    <div class="method-grid">
      <div class="method-item">🎯 <span>Barrel Rate (Statcast)</span></div>
      <div class="method-item">💨 <span>Exit Velocity</span></div>
      <div class="method-item">⚾ <span>Log5 Matchup (HR/PA)</span></div>
      <div class="method-item">🏟️ <span>Park HR Factors</span></div>
      <div class="method-item">🌤️ <span>Live Weather + Wind</span></div>
      <div class="method-item">🤝 <span>Platoon Splits</span></div>
      <div class="method-item">📈 <span>Recent Form (15G)</span></div>
      <div class="method-item">📊 <span>Bayesian Stat Blend</span></div>
    </div>
    <div class="footer-divider"></div>
    <div class="footer-note">
      Probabilities reflect P(≥1 HR) in today's game estimated via Poisson distribution.<br>
      Statcast data © Baseball Savant · Weather data © OpenWeatherMap<br>
      <strong>For informational purposes only.</strong> Generated at {generated_at} UTC.
    </div>
  </div>
</div>
</body>
</html>"""


_CARD_TEMPLATE = """    <div class="player-card">
      <div class="card-top">
        <div class="rank rank-{rank_class}">{rank}</div>
        <div class="player-info">
          <div class="player-name">{player_name}</div>
          <div class="player-meta">{team} · {position} · Bats {bats}</div>
          <div class="matchup">vs <strong>{pitcher_name}</strong> ({pitcher_hand}HP) · {venue}</div>
          <div class="prob-bar-wrap">
            <div class="prob-bar" style="width:{bar_pct}%"></div>
          </div>
          <div class="factors">
            {chips}
          </div>
        </div>
        <div class="prob-block">
          <div class="prob-pct">{hr_pct}</div>
          <div class="prob-label">HR Prob</div>
        </div>
      </div>
    </div>"""


def _chip(css_class: str, icon: str, text: str) -> str:
    return f'<span class="chip {css_class}">{icon} {text}</span>'


def _build_chips(pred: dict, weather: dict, venue: str) -> str:
    chips = []
    f = pred.get("factors", {})
    sc = pred.get("statcast_metrics", {})
    tier = pred.get("confidence_tier", "Medium")

    # Confidence
    chips.append(_chip(f"chip-confidence-{tier}", "◉", f"{tier} confidence"))

    # Barrel rate
    br = sc.get("barrel_rate")
    if br is not None and not (isinstance(br, float) and br != br):
        chips.append(_chip("chip-barrel", "🎯", f"Barrel {br*100:.1f}%"))

    # Exit velocity
    ev = sc.get("avg_exit_velocity")
    if ev is not None and not (isinstance(ev, float) and ev != ev):
        chips.append(_chip("chip-ev", "💥", f"EV {ev:.1f} mph"))

    # Park factor
    pf = f.get("park_factor", 1.0)
    if pf >= 1.05:
        chips.append(_chip("chip-park", "🏟️", f"Park +{(pf-1)*100:.0f}%"))
    elif pf <= 0.95:
        chips.append(_chip("chip-park", "🏟️", f"Park {(pf-1)*100:.0f}%"))

    # Weather
    wf = f.get("weather_factor", 1.0)
    wind_cat = weather.get("wind_category", "calm")
    temp = weather.get("temperature_f", 72)
    if wf >= 1.06:
        desc = f"Wind out {weather.get('wind_speed_mph',0):.0f}mph" if "out" in wind_cat else f"Warm {temp:.0f}°F"
        chips.append(_chip("chip-wind", "💨", desc))
    elif wf <= 0.94:
        chips.append(_chip("chip-wind-bad", "🌬️", f"Wind in {weather.get('wind_speed_mph',0):.0f}mph"))

    # Platoon
    plat = f.get("platoon_factor", 1.0)
    if plat >= 1.05:
        chips.append(_chip("chip-platoon", "↔️", "Platoon adv"))
    elif plat <= 0.92:
        chips.append(_chip("chip-platoon", "↔️", "Same hand"))

    return "\n            ".join(chips)


def build_html(
    predictions: list[dict],
    prediction_date: str,
    games_today: int,
    lineups_confirmed: int,
    generated_at: str,
) -> str:
    cards = []
    for i, pred in enumerate(predictions, 1):
        rank_class = str(i) if i <= 3 else "other"
        prob = pred["hr_probability"]
        bar_pct = min(int(prob * 100 * 2.5), 100)   # scale up for visual impact

        weather = pred.get("weather", {})
        venue = pred.get("venue", "")
        chips_html = _build_chips(pred, weather, venue)
        pitcher = pred.get("pitcher", {}) or {}

        card = _CARD_TEMPLATE.format(
            rank=i,
            rank_class=rank_class,
            player_name=pred.get("player_name", "Unknown"),
            team=pred.get("team", ""),
            position=pred.get("position", ""),
            bats=pred.get("bats", "R"),
            pitcher_name=pitcher.get("fullName", "Unknown"),
            pitcher_hand=pitcher.get("throws", "R"),
            venue=venue,
            bar_pct=bar_pct,
            chips=chips_html,
            hr_pct=pred.get("hr_pct", "0.0%"),
        )
        cards.append(card)

    avg_prob = (
        sum(p["hr_probability"] for p in predictions) / len(predictions) * 100
        if predictions else 0
    )

    html = _HTML_TEMPLATE.format(
        prediction_date=prediction_date,
        top_n=len(predictions),
        games_today=games_today,
        avg_prob=f"{avg_prob:.1f}",
        lineups_confirmed=lineups_confirmed,
        player_cards="\n".join(cards),
        generated_at=generated_at,
    )
    return html


def send_email(
    predictions: list[dict],
    prediction_date: str,
    games_today: int,
    lineups_confirmed: int,
    generated_at: str,
    subject_template: str = "⚾ Top {n} HR Predictions — {date}",
) -> bool:
    sender = os.environ.get("EMAIL_SENDER", "")
    password = os.environ.get("EMAIL_PASSWORD", "")
    recipients_raw = os.environ.get("EMAIL_RECIPIENTS", "")
    host = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
    port = int(os.environ.get("EMAIL_PORT", "587"))

    if not sender or not password or not recipients_raw:
        log.error("EMAIL_SENDER, EMAIL_PASSWORD, or EMAIL_RECIPIENTS not set.")
        return False

    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    subject = subject_template.format(n=len(predictions), date=prediction_date)

    html_body = build_html(
        predictions, prediction_date, games_today, lineups_confirmed, generated_at
    )

    # Plain-text fallback
    lines = [f"MLB Home Run Predictions — {prediction_date}", "=" * 50]
    for i, p in enumerate(predictions, 1):
        pitcher = p.get("pitcher", {}) or {}
        lines.append(
            f"{i:2}. {p['player_name']:25s} {p['hr_pct']:>6}  vs {pitcher.get('fullName','?')}"
        )
    text_body = "\n".join(lines)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"MLB HR Predictor <{sender}>"
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        log.info("Email sent to %d recipient(s): %s", len(recipients), subject)
        return True
    except Exception as exc:
        log.error("Failed to send email: %s", exc)
        return False
