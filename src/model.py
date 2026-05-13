"""Maher Poisson model for predicting Primeira Liga match outcomes.

Each team has an attack rating and a defense rating. Expected goals are:
    home_goals ~ Poisson(exp(attack_home + defense_away + home_adv))
    away_goals ~ Poisson(exp(attack_away + defense_home))

We fit it as a single Poisson GLM where each match contributes two rows
(one for each side), with categorical team variables and a home indicator.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson


def prepare_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape matches into a 'long' DataFrame: 2 rows per match (home and away)."""
    home = pd.DataFrame({
        "goals":    df["FTHG"].values,
        "team":     df["HomeTeam"].values,
        "opponent": df["AwayTeam"].values,
        "is_home":  1,
    })
    away = pd.DataFrame({
        "goals":    df["FTAG"].values,
        "team":     df["AwayTeam"].values,
        "opponent": df["HomeTeam"].values,
        "is_home":  0,
    })
    return pd.concat([home, away], ignore_index=True)


def fit_model(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit a Poisson GLM and return the fitted model."""
    long_df = prepare_long_format(df)
    model = smf.glm(
        formula="goals ~ team + opponent + is_home",
        data=long_df,
        family=sm.families.Poisson(),
    ).fit()
    return model

def fit_model_until(df: pd.DataFrame, cutoff_date: pd.Timestamp):
    """Fit the model using only matches before cutoff_date.
    
    Useful for evaluation: train on the past, predict the future.
    """
    train_df = df[df["Date"] < cutoff_date]
    if len(train_df) < 100:
        raise ValueError(f"Only {len(train_df)} matches before {cutoff_date}; need more")
    return fit_model(train_df)


def predict_match(model, home_team: str, away_team: str, max_goals: int = 8) -> dict:
    """Predict a single match. Returns probabilities + the expected score grid."""
    # Predict expected goals for each side by feeding the model a 2-row "fake match"
    fixture = pd.DataFrame({
        "team":     [home_team, away_team],
        "opponent": [away_team, home_team],
        "is_home":  [1, 0],
    })
    expected = model.predict(fixture).values
    lambda_home, lambda_away = expected[0], expected[1]

    # Joint score probability assuming independence:
    # P(home=i, away=j) = Poisson(i; λh) * Poisson(j; λa)
    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    score_matrix = np.outer(home_probs, away_probs)

    # Aggregate by result
    p_home = np.tril(score_matrix, k=-1).sum()  # home > away
    p_draw = np.trace(score_matrix)             # home == away
    p_away = np.triu(score_matrix, k=1).sum()   # away > home

    return {
        "home_team":     home_team,
        "away_team":     away_team,
        "lambda_home":   lambda_home,
        "lambda_away":   lambda_away,
        "p_home_win":    p_home,
        "p_draw":        p_draw,
        "p_away_win":    p_away,
        "score_matrix":  score_matrix,
    }


def main():
    """Smoke test: fit the model and predict a marquee Primeira Liga matchup."""
    DATA_DIR = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(DATA_DIR / "matches.parquet")

    print(f"Fitting Maher Poisson model on {len(df)} matches...")
    model = fit_model(df)
    print(f"Model fit. Pseudo R²: {1 - model.deviance / model.null_deviance:.3f}")
    print(f"Home advantage coefficient: {model.params['is_home']:.3f} "
          f"(=> {(np.exp(model.params['is_home']) - 1) * 100:.1f}% more goals at home)")

    # Predict a classic matchup
    pred = predict_match(model, "Benfica", "Porto")
    print(f"\n=== {pred['home_team']} (home) vs {pred['away_team']} ===")
    print(f"Expected goals: {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}")
    print(f"P(home win): {pred['p_home_win']:.1%}")
    print(f"P(draw):     {pred['p_draw']:.1%}")
    print(f"P(away win): {pred['p_away_win']:.1%}")

    # Most likely scorelines (top 5)
    matrix = pred["score_matrix"]
    flat_idx = np.argsort(matrix, axis=None)[::-1][:5]
    print("\nMost likely scorelines:")
    for idx in flat_idx:
        h, a = np.unravel_index(idx, matrix.shape)
        print(f"  {h}-{a}: {matrix[h, a]:.1%}")

def print_prediction(pred: dict) -> None:
    """Human-readable summary of a match prediction."""
    print(f"=== {pred['home_team']} (home) vs {pred['away_team']} ===")
    print(f"Expected goals: {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}")
    print(f"P(home win): {pred['p_home_win']:.1%}")
    print(f"P(draw):     {pred['p_draw']:.1%}")
    print(f"P(away win): {pred['p_away_win']:.1%}")

    matrix = pred["score_matrix"]
    flat_idx = np.argsort(matrix, axis=None)[::-1][:5]
    print("\nMost likely scorelines:")
    for idx in flat_idx:
        h, a = np.unravel_index(idx, matrix.shape)
        print(f"  {h}-{a}: {matrix[h, a]:.1%}")


if __name__ == "__main__":
    main()