# Primeira Liga Match Predictor

A statistical model for predicting outcomes of Portuguese Primeira Liga matches, built from scratch in Python. The project covers the full data-science workflow: data ingestion, exploratory analysis, model fitting, empirical evaluation against a bookmaker baseline, calibration diagnostics, model upgrades, and deployment as an interactive web app.

**Live demo:** [your-app-name.streamlit.app](https://your-app-name.streamlit.app)

---

## What it does

Given any two Primeira Liga teams, the model produces:

- Calibrated probabilities for home win, draw, away win
- Expected goals for each side
- A full scoreline distribution (most likely results, 0-0 through 5-5)
- "Fair" decimal odds, with adjustable bookmaker margin
- Side-by-side comparison against Bet365's historical odds
- Expected value of each outcome at Bet365's prices

---

## Results

The model was evaluated on the 2024-25 Primeira Liga season (240 held-out matches with available bookmaker odds).

| Model                          | Log-loss | Brier  | Accuracy |
|--------------------------------|---------:|-------:|---------:|
| Uniform (33/33/33)             |   1.0986 |    —   |    33.3% |
| League base rate               |   1.0762 | 0.6514 |    43.3% |
| **Maher Poisson (this work)**  | **0.9550** | **0.5644** | **52.1%** |
| Dixon-Coles (this work)        |   0.9566 | 0.5657 |    52.1% |
| Pinnacle closing odds          |   0.9160 | 0.5415 |    ~53%  |

The Maher Poisson model captures roughly **76% of the available log-loss signal** between the naive base-rate baseline and Pinnacle's market-leading closing odds — using only goal data, no injury or lineup information.

---

## Methodology

### Data

- **Source:** [football-data.co.uk](https://www.football-data.co.uk) (free CSV downloads, no scraping required)
- **Scope:** Primeira Liga, seasons 2021-22 through 2025-26 (~1,500 matches)
- **Fields used:** match dates, team names, full-time scores, closing odds from a dozen bookmakers

### Models

**1. Maher Poisson (`src/model.py`).** Fit as a Poisson GLM with team and opponent as categorical predictors plus a home indicator. Each team learns an attack rating and a defense rating; expected goals are `exp(attack_team + defense_opponent + home_advantage)`. Fit via maximum likelihood with `statsmodels`.

**2. Dixon-Coles (`src/dixon_coles.py`).** Adds the Dixon & Coles (1997) low-score correction τ and optional time-decay weighting. Likelihood implemented from scratch and optimized with `scipy.optimize.minimize` (SLSQP).

### Evaluation

Train/test split: model trained on matches before the 2024-25 season, evaluated on 2024-25 matches. Three metrics computed:

- **Log-loss** — proper scoring rule, penalizes confident wrong predictions
- **Brier score** — squared error against one-hot truth
- **Accuracy** — top-1 prediction matches outcome

Three baselines: uniform probabilities, league base rate (from training data only), and Pinnacle's closing odds (the sharpest publicly available probability estimate).

---

## What I learned (including the negative results)

**The Maher Poisson model produced excellent ratings.** Without being told who the strong teams are, the model correctly identified Porto, Benfica, and Sporting CP as the top three, separated them from Braga, and grouped relegation-threatened teams (Tondela, Belenenses, Paços Ferreira) at the bottom. The estimated home advantage of ~24% extra goals matches the literature on Portuguese football.

**The diagnostic calibration plot identified a clear, specific weakness:** the model never predicted draw probabilities above ~32%, even though draws happen ~25% of the time. The probability mass was being spread across narrow home/away wins instead of concentrating on draw outcomes. This was the canonical Dixon-Coles signature.

**Dixon-Coles τ correction did *not* improve log-loss.** Contrary to what the diagnostic plot suggested, fitting the full Dixon-Coles model showed `ρ ≈ +0.04` for Primeira Liga — the *opposite* sign from Dixon and Coles' 1997 English-football finding. This indicates that in Primeira Liga, low-score draws are slightly *less* common than independence predicts, with mass concentrating in 1-0/0-1 results instead. A plausible explanation is the structural dominance of the Big 3, who play half of all fixtures and produce narrow favorite-wins rather than stalemates against weaker opposition.

**Time-decay weighting traded log-loss for accuracy.** Sweeping `xi` from 0 to 0.007 showed log-loss monotonically *increased* (0.957 → 0.960) while accuracy monotonically *increased* (52.1% → 54.6%). Time-decayed models pick the modal outcome better but lose probabilistic calibration — they become overconfident in their top choice and get crushed by log-loss when wrong. This is the bias-variance tradeoff in action and a useful illustration that "accuracy" and "log-loss" measure different things.

**The architectural ceiling.** Maher Poisson, Dixon-Coles, and Dixon-Coles + time decay all log-loss around 0.955-0.960. The 0.04 gap to Pinnacle is not a model-architecture problem; it's the value of information the model doesn't have — injuries, lineup confirmations, market signal from sharp bettors. Further gains would require additional data, not additional math.

---

## Project structure

```
primeira-liga-model/
├── app.py                       # Streamlit web app
├── src/
│   ├── model.py                 # Maher Poisson (statsmodels GLM)
│   ├── dixon_coles.py           # Dixon-Coles model (from-scratch MLE)
│   ├── fetch_data.py            # Download CSVs from football-data.co.uk
│   ├── odds.py                  # Odds <-> probabilities + helpers
│   └── save_model.py            # Fit + pickle the model for the app
├── notebooks/
│   ├── 01_explore.ipynb         # Initial exploratory analysis
│   ├── 02_ratings.ipynb         # Team ratings and attack-vs-defense plot
│   └── 03_evaluation.ipynb      # Train/test split, metrics, calibration
├── data/                        # Cached parquet + pickled model (gitignored except model)
└── requirements.txt
```

---

## Getting started

```bash
git clone https://github.com/EduTheMan9/primeira-liga-model.git
cd primeira-liga-model

python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate    # macOS/Linux

pip install -r requirements.txt

# Fetch data and fit the model
python src/fetch_data.py
python src/save_model.py

# Launch the app
streamlit run app.py
```

---

## Limitations and disclaimers

This is a learning project, not a betting tool.

- The model uses **only historical goals and home/away status**. It knows nothing about injuries, suspensions, lineup changes, travel fatigue, midweek European fixtures, or motivation (e.g. a team safe mid-table playing a relegation-threatened side).
- **Newly-promoted teams cannot be predicted** — they have no training history. About 60 matches from the 2024-25 season were excluded from evaluation for this reason.
- **Expected-value calculations in the web app use the same data the model trained on for the most recent historical matchups.** This is a portfolio demo, not a profitable betting strategy. Beating Pinnacle's closing line consistently is genuinely hard, and this model does not do that.
- **Past performance is not predictive of future results.** Even a positive-EV bet loses individually; long-run profitability requires bankroll management and many independent bets.

---

## References

- Maher, M. J. (1982). *Modelling association football scores.* Statistica Neerlandica, 36(3), 109-118.
- Dixon, M. J., & Coles, S. G. (1997). *Modelling association football scores and inefficiencies in the football betting market.* Journal of the Royal Statistical Society: Series C (Applied Statistics), 46(2), 265-280.

---

## About

Built as an independent project by Eduardo, 3rd-year Industrial and Management Engineering student.

Tools: Python, pandas, statsmodels, scipy, matplotlib, Streamlit.
