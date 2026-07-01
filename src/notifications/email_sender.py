"""
email_sender.py
===============
Four-section email: HR · Total Bases · H+R+RBI · RBI
Each section independently ranked.
Uses %%PLACEHOLDER%% templating to avoid CSS curly-brace conflicts.
"""
from __future__ import annotations
import logging, os, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)

_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif}
.wrapper{max-width:680px;margin:0 auto}
.header{background:linear-gradient(135deg,#0f1f2e,#0d1117);padding:32px;border-bottom:2px solid #21a96a}
.brand{font-size:11px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#21a96a}
.title{font-size:28px;font-weight:800;color:#fff;margin:4px 0}
.sub{font-size:12px;color:#7d8590;font-family:monospace}
.badge{display:inline-block;background:#21a96a14;border:1px solid #21a96a44;color:#21a96a;
       font-family:monospace;font-size:11px;padding:3px 10px;border-radius:20px;margin-top:8px}
.summary{background:#161b22;padding:14px 32px;display:flex;gap:24px;flex-wrap:wrap;
         border-bottom:1px solid #21262d}
.si{display:flex;flex-direction:column;gap:2px}
.sl{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#7d8590}
.sv{font-size:16px;font-weight:700;color:#e6edf3}
/* Section headers */
.sec-hr   {padding:18px 32px 10px;border-bottom:1px solid #21262d;border-top:3px solid #21a96a;margin-top:4px}
.sec-tb   {padding:18px 32px 10px;border-bottom:1px solid #21262d;border-top:3px solid #4493f8;margin-top:4px}
.sec-hrbi {padding:18px 32px 10px;border-bottom:1px solid #21262d;border-top:3px solid #d2a8ff;margin-top:4px}
.sec-rbi  {padding:18px 32px 10px;border-bottom:1px solid #21262d;border-top:3px solid #e3b341;margin-top:4px}
.t-hr   {font-size:14px;font-weight:700;color:#21a96a;margin-bottom:2px}
.t-tb   {font-size:14px;font-weight:700;color:#4493f8;margin-bottom:2px}
.t-hrbi {font-size:14px;font-weight:700;color:#d2a8ff;margin-bottom:2px}
.t-rbi  {font-size:14px;font-weight:700;color:#e3b341;margin-bottom:2px}
.sec-sub{font-size:11px;color:#7d8590;font-family:monospace}
.preds{padding:0 32px}
/* Cards */
.card{border-bottom:1px solid #1a2030;padding:16px 0}
.card:last-child{border-bottom:none}
.ct{display:flex;align-items:flex-start;gap:14px}
.rk{width:32px;height:32px;border-radius:7px;display:flex;align-items:center;justify-content:center;
    font-size:12px;font-weight:800;flex-shrink:0;font-family:monospace}
.r1-hr  {background:#21a96a22;color:#21a96a;border:1px solid #21a96a44}
.r2-hr  {background:#21a96a14;color:#3dd68c;border:1px solid #21a96a30}
.r3-hr  {background:#0d2818;color:#3dd68c;border:1px solid #21a96a20}
.r1-tb  {background:#4493f822;color:#4493f8;border:1px solid #4493f844}
.r2-tb  {background:#4493f814;color:#79c0ff;border:1px solid #4493f830}
.r3-tb  {background:#1d3a5c;color:#79c0ff;border:1px solid #4493f820}
.r1-hrbi{background:#d2a8ff22;color:#d2a8ff;border:1px solid #d2a8ff44}
.r2-hrbi{background:#d2a8ff14;color:#c4a0f8;border:1px solid #d2a8ff30}
.r3-hrbi{background:#2d1b4e;color:#c4a0f8;border:1px solid #d2a8ff20}
.r1-rbi {background:#e3b34122;color:#e3b341;border:1px solid #e3b34144}
.r2-rbi {background:#e3b34114;color:#d29922;border:1px solid #e3b34130}
.r3-rbi {background:#3d2b00;color:#d29922;border:1px solid #e3b34120}
.ro{background:#21262d;color:#7d8590;border:1px solid #30363d}
.pi{flex:1;min-width:0}
.pn{font-size:15px;font-weight:700;color:#e6edf3;margin-bottom:1px}
.pm{font-size:11px;color:#7d8590;font-family:monospace}
.mu{font-size:11px;color:#8b949e;margin-top:4px;padding:4px 8px;background:#161b22;
    border-radius:5px;border-left:2px solid #30363d;font-family:monospace}
.mu strong{color:#e6edf3}
.bar-w{margin:7px 0 5px;height:3px;background:#21262d;border-radius:2px;overflow:hidden}
.bar-hr  {height:100%;border-radius:2px;background:linear-gradient(90deg,#21a96a,#3dd68c)}
.bar-tb  {height:100%;border-radius:2px;background:linear-gradient(90deg,#4493f8,#79c0ff)}
.bar-hrbi{height:100%;border-radius:2px;background:linear-gradient(90deg,#8957e5,#d2a8ff)}
.bar-rbi {height:100%;border-radius:2px;background:linear-gradient(90deg,#e3b341,#f0c060)}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px}
.chip{font-family:monospace;font-size:10px;padding:2px 6px;border-radius:4px;display:inline-flex;align-items:center;gap:3px}
.c-b{background:#1d3a5c;color:#58a6ff;border:1px solid #1f6feb33}
.c-g{background:#0d2818;color:#3fb950;border:1px solid #2ea04333}
.c-r{background:#3d1515;color:#f85149;border:1px solid #da363133}
.c-a{background:#3d2b00;color:#e3b341;border:1px solid #d2941333}
.c-p{background:#2d1b4e;color:#d2a8ff;border:1px solid #8957e533}
.c-hi {background:#0d2818;color:#3fb950;border:1px solid #2ea04333}
.c-med{background:#2d2200;color:#e3b341;border:1px solid #d2941333}
.c-low{background:#3d1515;color:#f0883e;border:1px solid #e3631a33}
.pb{text-align:right;flex-shrink:0;min-width:64px}
.v-hr  {font-size:22px;font-weight:800;line-height:1;color:#21a96a}
.v-tb  {font-size:22px;font-weight:800;line-height:1;color:#4493f8}
.v-hrbi{font-size:22px;font-weight:800;line-height:1;color:#d2a8ff}
.v-rbi {font-size:22px;font-weight:800;line-height:1;color:#e3b341}
.pl{font-size:9px;color:#7d8590;letter-spacing:1px;text-transform:uppercase;margin-top:2px}
/* Footer */
.footer{padding:24px 32px;border-top:1px solid #21262d}
.fn{font-size:11px;color:#484f58;line-height:1.7;font-family:monospace}
.fd{height:1px;background:linear-gradient(90deg,transparent,#21a96a44,transparent);margin:12px 0}
.mg{display:grid;grid-template-columns:1fr 1fr;gap:3px 20px}
.mi{font-size:10px;color:#484f58;font-family:monospace}
.mi span{color:#7d8590}
.mt{font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#21a96a;margin-bottom:6px}
</style></head><body><div class="wrapper">
<div class="header">
  <div class="brand">MLB Analytics</div>
  <div class="title">Daily Batter<br>Predictions</div>
  <div class="sub">HR · Total Bases · H+R+RBI · RBI · Statcast · Log5</div>
  <div class="badge">⚾ %%DATE%%</div>
</div>
<div class="summary">
  <div class="si"><span class="sl">Games</span><span class="sv">%%GAMES%%</span></div>
  <div class="si"><span class="sl">Players</span><span class="sv">%%PLAYERS%%</span></div>
  <div class="si"><span class="sl">Avg HR Prob</span><span class="sv">%%AVGHR%%%</span></div>
  <div class="si"><span class="sl">Lineups</span><span class="sv">%%LINEUPS%%</span></div>
</div>

<div class="sec-hr">
  <div class="t-hr">💥 Top %%N_HR%% — Home Run Probability</div>
  <div class="sec-sub">P(HR ≥ 1) · Statcast barrel rate · Log5 matchup · Park + weather</div>
</div>
<div class="preds">%%HR_CARDS%%</div>

<div class="sec-tb">
  <div class="t-tb">📊 Top %%N_TB%% — Total Bases</div>
  <div class="sec-sub">Expected TB tonight · SLG-based rate · Pitcher hit factor</div>
</div>
<div class="preds">%%TB_CARDS%%</div>

<div class="sec-hrbi">
  <div class="t-hrbi">⭐ Top %%N_HRBI%% — H + R + RBI</div>
  <div class="sec-sub">Expected combined fantasy total · Hits + Runs + RBI</div>
</div>
<div class="preds">%%HRBI_CARDS%%</div>

<div class="sec-rbi">
  <div class="t-rbi">🎯 Top %%N_RBI%% — RBI</div>
  <div class="sec-sub">Expected RBI tonight · Season rate · Pitcher quality adjustment</div>
</div>
<div class="preds">%%RBI_CARDS%%</div>

<div class="footer">
  <div class="mt">Methodology</div>
  <div class="mg">
    <div class="mi">🎯 <span>Barrel Rate (Statcast)</span></div>
    <div class="mi">💥 <span>Exit Velocity</span></div>
    <div class="mi">⚾ <span>Log5 HR Matchup</span></div>
    <div class="mi">🏟️ <span>Park HR Factors</span></div>
    <div class="mi">🌤️ <span>Live Weather + Wind</span></div>
    <div class="mi">📊 <span>TB/PA + SLG rate</span></div>
    <div class="mi">🤖 <span>XGBoost + Bayesian Blend</span></div>
    <div class="mi">📐 <span>Poisson P(≥1 HR)</span></div>
  </div>
  <div class="fd"></div>
  <div class="fn">Statcast © Baseball Savant · Weather © OpenWeatherMap<br>
  <strong>For informational purposes only.</strong> Generated %%TS%% UTC.</div>
</div></div></body></html>"""

_CARD = """<div class="card"><div class="ct">
  <div class="rk %%RC%%">%%RANK%%</div>
  <div class="pi">
    <div class="pn">%%NAME%%</div>
    <div class="pm">%%TEAM%% · %%POS%% · Bats %%BATS%%</div>
    <div class="mu">vs <strong>%%PITCHER%%</strong> (%%HAND%%HP) · %%VENUE%%</div>
    <div class="bar-w"><div class="%%BAR_CLS%%" style="width:%%BAR%%%%"></div></div>
    <div class="chips">%%CHIPS%%</div>
  </div>
  <div class="pb">
    <div class="%%VAL_CLS%%">%%PRIMARY%%</div>
    <div class="pl">%%LABEL%%</div>
  </div>
</div></div>"""


def _sub(t: str, d: dict) -> str:
    for k, v in d.items():
        t = t.replace(f"%%{k}%%", str(v))
    return t


def _chip(cls: str, icon: str, text: str) -> str:
    return f'<span class="chip {cls}">{icon} {text}</span>'


def _shared_chips(pred: dict, section: str) -> str:
    chips = []
    f    = pred.get("factors", {})
    sc   = pred.get("statcast_metrics", {})
    tier = pred.get("confidence_tier", "Medium")
    cls  = {"High":"c-hi","Medium":"c-med","Low":"c-low"}.get(tier,"c-med")
    chips.append(_chip(cls, "◉", tier))

    br = sc.get("barrel_rate")
    if br and br == br:
        chips.append(_chip("c-a", "🎯", f"Barrel {float(br)*100:.1f}%"))

    ev = sc.get("avg_exit_velocity")
    if ev and ev == ev:
        chips.append(_chip("c-g", "💥", f"EV {float(ev):.1f}mph"))

    pf = float(f.get("park_factor", 1.0))
    if pf >= 1.05:
        chips.append(_chip("c-b", "🏟️", f"Park +{(pf-1)*100:.0f}%"))
    elif pf <= 0.95:
        chips.append(_chip("c-r", "🏟️", f"Park {(pf-1)*100:.0f}%"))

    wf = float(f.get("weather_factor", 1.0))
    if wf >= 1.06:
        chips.append(_chip("c-g", "💨", "Wind out"))
    elif wf <= 0.94:
        chips.append(_chip("c-r", "🌬️", "Wind in"))

    phf = float(f.get("pitcher_hit_factor", 1.0))
    if section in ("tb", "hrbi", "rbi"):
        if phf >= 1.10:
            chips.append(_chip("c-g", "⚾", "Hit-prone P"))
        elif phf <= 0.90:
            chips.append(_chip("c-r", "⚾", "Stingy P"))

    if section == "tb":
        chips.append(_chip("c-b", "📊", f"HR {pred.get('hr_pct','')}"))
    elif section == "hrbi":
        chips.append(_chip("c-g", "💥", f"HR {pred.get('hr_pct','')}"))
        chips.append(_chip("c-b", "📊", f"TB {pred.get('expected_tb','')}"))
    elif section == "rbi":
        chips.append(_chip("c-b", "📊", f"TB {pred.get('expected_tb','')}"))

    return "\n".join(chips)


def _rank_cls(i: int, section: str) -> str:
    sfx = {"hr":"hr","tb":"tb","hrbi":"hrbi","rbi":"rbi"}.get(section,"hr")
    return {1:f"r1-{sfx}",2:f"r2-{sfx}",3:f"r3-{sfx}"}.get(i,"ro")


def _build_section(predictions: list[dict], section: str) -> str:
    bar_cls = {
        "hr":"bar-hr","tb":"bar-tb","hrbi":"bar-hrbi","rbi":"bar-rbi"
    }[section]
    val_cls = {
        "hr":"v-hr","tb":"v-tb","hrbi":"v-hrbi","rbi":"v-rbi"
    }[section]
    cards = []
    for i, p in enumerate(predictions, 1):
        if section == "hr":
            primary = p.get("hr_pct", "0%")
            label   = "HR Prob"
            bar     = min(int(p.get("hr_probability", 0) * 100 * 2.5), 100)
        elif section == "tb":
            primary = f"{p.get('expected_tb','?')} TB"
            label   = "Exp Total Bases"
            bar     = min(int(float(p.get("expected_tb", 0)) / 4.0 * 100), 100)
        elif section == "hrbi":
            primary = f"{p.get('expected_hrbi','?')}"
            label   = "Exp H+R+RBI"
            bar     = min(int(float(p.get("expected_hrbi", 0)) / 4.0 * 100), 100)
        else:  # rbi
            primary = f"{p.get('expected_rbi','?')} RBI"
            label   = "Exp RBI"
            bar     = min(int(float(p.get("expected_rbi", 0)) / 2.0 * 100), 100)

        pitcher = p.get("pitcher", {}) or {}
        cards.append(_sub(_CARD, {
            "RANK":    str(i), "RC": _rank_cls(i, section),
            "NAME":    p.get("player_name","Unknown"),
            "TEAM":    p.get("team",""),
            "POS":     p.get("position",""),
            "BATS":    p.get("bats","R"),
            "PITCHER": pitcher.get("fullName","Unknown"),
            "HAND":    pitcher.get("throws","R"),
            "VENUE":   p.get("venue",""),
            "BAR_CLS": bar_cls, "BAR": str(bar),
            "CHIPS":   _shared_chips(p, section),
            "VAL_CLS": val_cls,
            "PRIMARY": primary, "LABEL": label,
        }))
    return "\n".join(cards)


def build_html(
    hr_preds: list[dict], tb_preds: list[dict],
    hrbi_preds: list[dict], rbi_preds: list[dict],
    date_str: str, games: int, lineups_confirmed: int, ts: str,
) -> str:
    total = len(set(
        p.get("player_id") for p in hr_preds + tb_preds + hrbi_preds + rbi_preds
        if p.get("player_id")
    ))
    avg_hr = sum(p.get("hr_probability",0) for p in hr_preds) / len(hr_preds) * 100 if hr_preds else 0

    return _sub(_HTML, {
        "DATE":      date_str, "GAMES": str(games),
        "PLAYERS":   str(total), "AVGHR": f"{avg_hr:.1f}",
        "LINEUPS":   str(lineups_confirmed),
        "N_HR":      str(len(hr_preds)),
        "N_TB":      str(len(tb_preds)),
        "N_HRBI":    str(len(hrbi_preds)),
        "N_RBI":     str(len(rbi_preds)),
        "HR_CARDS":  _build_section(hr_preds,   "hr"),
        "TB_CARDS":  _build_section(tb_preds,   "tb"),
        "HRBI_CARDS":_build_section(hrbi_preds, "hrbi"),
        "RBI_CARDS": _build_section(rbi_preds,  "rbi"),
        "TS":        ts,
    })


def send_email(
    hr_preds: list[dict], tb_preds: list[dict],
    hrbi_preds: list[dict], rbi_preds: list[dict],
    date_str: str, games: int, lineups_confirmed: int, ts: str,
    subject_template: str = "⚾ MLB Predictions — {date}",
) -> bool:
    sender  = os.environ.get("EMAIL_SENDER","")
    pw      = os.environ.get("EMAIL_PASSWORD","")
    recip_r = os.environ.get("EMAIL_RECIPIENTS","")
    host    = os.environ.get("EMAIL_HOST","smtp.gmail.com")
    port    = int(os.environ.get("EMAIL_PORT","587"))

    if not sender or not pw or not recip_r:
        log.error("EMAIL_SENDER, EMAIL_PASSWORD, or EMAIL_RECIPIENTS not set.")
        return False

    recipients = [r.strip() for r in recip_r.split(",") if r.strip()]
    total      = len(set(p.get("player_id") for p in hr_preds+tb_preds+hrbi_preds+rbi_preds))
    subject    = subject_template.format(n=total, date=date_str)
    html_body  = build_html(hr_preds, tb_preds, hrbi_preds, rbi_preds,
                            date_str, games, lineups_confirmed, ts)

    lines = [f"MLB Predictions — {date_str}", "="*55,
             f"\n💥 TOP HR"]
    for i,p in enumerate(hr_preds,1):
        pitcher = p.get("pitcher",{}) or {}
        lines.append(f"  {i:2}. {p['player_name']:25s} {p['hr_pct']:>6}  vs {pitcher.get('fullName','?')}")
    lines.append(f"\n📊 TOP TOTAL BASES")
    for i,p in enumerate(tb_preds,1):
        lines.append(f"  {i:2}. {p['player_name']:25s} {p.get('expected_tb','?')} exp TB")
    lines.append(f"\n⭐ TOP H+R+RBI")
    for i,p in enumerate(hrbi_preds,1):
        lines.append(f"  {i:2}. {p['player_name']:25s} {p.get('expected_hrbi','?')} exp")
    lines.append(f"\n🎯 TOP RBI")
    for i,p in enumerate(rbi_preds,1):
        lines.append(f"  {i:2}. {p['player_name']:25s} {p.get('expected_rbi','?')} exp RBI")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"MLB Predictor <{sender}>"
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText("\n".join(lines), "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(host, port) as s:
            s.ehlo(); s.starttls(); s.login(sender, pw)
            s.sendmail(sender, recipients, msg.as_string())
        log.info("Email sent (%d HR, %d TB, %d H+R+RBI, %d RBI predictions).",
                 len(hr_preds), len(tb_preds), len(hrbi_preds), len(rbi_preds))
        return True
    except Exception as exc:
        log.error("Failed to send email: %s", exc)
        return False
