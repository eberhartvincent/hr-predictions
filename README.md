# ⚾ MLB Daily Home Run Predictor

A production-grade daily HR prediction system powered by **Statcast**, **MLB Stats API**, **weather data**, and a calibrated statistical ensemble model — delivered straight to your inbox every morning via GitHub Actions.

---

## How It Works

The prediction engine combines **two complementary models**:

### 1. Statistical Model (Log5 + Bayesian blending)
Estimates each batter's probability of hitting ≥1 HR using:
- **Batter HR rate** — Bayesian blend of season + career HR/PA (stabilizes small samples)
- **Log5 matchup** — adjusts for the specific pitcher's HR-allowed rate vs. league average
- **Park factor** — per-stadium HR coefficients (Coors = 1.38, Oracle = 0.88, etc.)
- **Weather** — live temperature (ball flight), wind speed & direction vs. CF alignment
- **Platoon split** — batter/pitcher handedness advantage using actual split HR rates
- **Recent form** — last 15-game HR rate vs. expected (shrunk toward mean to reduce noise)
- Final probability via **Poisson distribution**: `P(≥1 HR) = 1 − e^(−λ)` where `λ = adj_rate × est_PA`

### 2. Statcast Quality-of-Contact Model
Independently scores each batter using:
- **Barrel rate** — the single strongest predictor of HR power (highest weight)
- **Average exit velocity** — raw power proxy
- **Hard-hit rate** (≥95 mph) — contact quality floor
- **Fly-ball rate** — ball trajectory prerequisite for HR

The two models are **blended** by confidence — Statcast weight scales with sample size (minimum 30 batted balls).

---

## Project Structure

```
hr-predictions/
├── .github/
│   └── workflows/
│       └── hr_predictions.yml   ← GitHub Actions workflow
├── src/
│   ├── data/
│   │   ├── mlb_client.py        ← MLB Stats API (free, no key)
│   │   ├── statcast_client.py   ← Statcast via pybaseball
│   │   └── weather_client.py    ← OpenWeatherMap API
│   ├── features/
│   │   └── engineer.py          ← Feature engineering + math
│   ├── models/
│   │   └── predictor.py         ← Ensemble prediction engine
│   └── notifications/
│       └── email_sender.py      ← HTML email builder + SMTP sender
├── config.yml                   ← All tunable settings
├── main.py                      ← CLI entrypoint + pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/hr-predictions.git
cd hr-predictions
pip install -r requirements.txt
```

### 2. Set up secrets

Go to **GitHub → Settings → Secrets → Actions** and add:

| Secret | Description |
|--------|-------------|
| `EMAIL_SENDER` | Gmail address to send from |
| `EMAIL_PASSWORD` | [Gmail App Password](https://myaccount.google.com/apppasswords) (not your real password) |
| `EMAIL_RECIPIENTS` | Comma-separated recipient addresses |
| `OPENWEATHER_API_KEY` | Free key from [openweathermap.org](https://openweathermap.org/api) |

> **Gmail App Password setup:** Google Account → Security → 2-Step Verification → App Passwords → generate one for "Mail"

### 3. Test locally

```bash
# Dry run (no email sent)
python main.py --dry-run

# Specific date
python main.py --date 2024-08-15 --top-n 5 --dry-run

# Send a test email with today's predictions
python main.py --test-email
```

### 4. Run manually in GitHub Actions

Go to **Actions → ⚾ HR Predictions → Run workflow** and fill in:
- **Date**: `today` or `2024-08-15`
- **Top N**: number of predictions
- **Dry run**: `true` to skip email

---

## Configuration (`config.yml`)

```yaml
prediction:
  top_n: 10          # Players in daily email
  date: "today"      # Override with YYYY-MM-DD
  min_pa_season: 30  # Min PA to qualify (filters April call-ups)

model:
  weights:
    barrel_rate: 0.25      # Statcast barrel % — strongest HR signal
    hr_rate: 0.20          # Batter's blended HR/PA rate
    pitcher_factor: 0.18   # Pitcher's HR-allowed profile
    exit_velocity: 0.15    # Average exit velocity
    park_factor: 0.10      # Ballpark HR factor
    weather_factor: 0.07   # Temperature + wind
    platoon_factor: 0.03   # Handedness matchup
    recent_form: 0.02      # Last 15 games

  recent_form_games: 15    # Window for form calculation
```

---

## Schedule

The workflow runs daily at **11:30 AM ET** — timed so that:
- Most lineups are announced for 1 PM ET games
- Night game lineups get early probability estimates

---

## Data Sources

| Source | Data | Cost |
|--------|------|------|
| MLB Stats API | Schedule, lineups, rosters, stats, splits | Free (no key) |
| Baseball Savant (via pybaseball) | Statcast barrel rate, EV, launch angle | Free |
| OpenWeatherMap | Live weather at stadium coordinates | Free (1000 calls/day) |

---

## Accuracy Notes

HR prediction is inherently uncertain — even the best hitters only go deep ~4-5% of their PA. This model estimates **relative probability** across players, not absolute certainty. Confidence tiers (High / Medium / Low) reflect sample size and data availability.

A player with a 25% HR probability is meaningfully more likely to go deep than one at 12%, even though neither is a guarantee.
