"""Dixon-Coles model: Poisson + low-score correction + time decay.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and 
Inefficiencies in the Football Betting Market", Applied Statistics 46(2).

Unlike the basic Maher Poisson model (which we could fit with statsmodels),
Dixon-Coles has a low-score correction that breaks the GLM structure.
We have to write the likelihood ourselves and optimize with scipy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


# ----- the τ correction -----

def tau(home_goals: int, away_goals: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor.
    
    Returns 1 for any score where home_goals >= 2 or away_goals >= 2.
    Adjusts the four corner cases (0,0), (0,1), (1,0), (1,1).
    """
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_h * lambda_a * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lambda_h * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lambda_a * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0


# ----- the model -----

@dataclass
class DixonColesModel:
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    home_adv: float
    rho: float
    xi: float

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Return (λ_home, λ_away) for a matchup."""
        lambda_h = np.exp(self.attack[home_team] + self.defense[away_team] + self.home_adv)
        lambda_a = np.exp(self.attack[away_team] + self.defense[home_team])
        return lambda_h, lambda_a

    def predict_match(self, home_team: str, away_team: str, max_goals: int = 10) -> dict:
        """Generate the joint scoreline distribution and aggregate to H/D/A probabilities."""
        if home_team not in self.attack or away_team not in self.attack:
            raise ValueError(f"Unknown team(s): {home_team}, {away_team}")
        
        lambda_h, lambda_a = self.expected_goals(home_team, away_team)
        
        # Build the (max_goals+1) x (max_goals+1) score matrix
        home_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_h)
        away_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_a)
        score_matrix = np.outer(home_pmf, away_pmf)
        
        # Apply τ correction to the four corner cells
        for h in range(2):
            for a in range(2):
                score_matrix[h, a] *= tau(h, a, lambda_h, lambda_a, self.rho)
        
        # Aggregate by outcome
        p_home = np.tril(score_matrix, k=-1).sum()
        p_draw = np.trace(score_matrix)
        p_away = np.triu(score_matrix, k=1).sum()
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "lambda_home": lambda_h,
            "lambda_away": lambda_a,
            "p_home_win": p_home,
            "p_draw": p_draw,
            "p_away_win": p_away,
            "score_matrix": score_matrix,
        }


# ----- fitting -----

def _unpack_params(params: np.ndarray, teams: list[str]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Parameter vector → (attack[teams], defense[teams], home_adv, rho)."""
    n = len(teams)
    attack = params[:n]
    defense = params[n:2*n]
    home_adv = params[2*n]
    rho = params[2*n + 1]
    return attack, defense, home_adv, rho


def _neg_log_likelihood(
    params: np.ndarray,
    teams: list[str],
    team_idx: dict[str, int],
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Negative log-likelihood for Dixon-Coles. To be minimized."""
    attack, defense, home_adv, rho = _unpack_params(params, teams)
    
    # Vectorized expected goals for every match
    lambda_h = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
    lambda_a = np.exp(attack[away_idx] + defense[home_idx])
    
    # Poisson log-likelihoods (vectorized)
    log_lik = (
        home_goals * np.log(lambda_h) - lambda_h - _logfactorial(home_goals)
        + away_goals * np.log(lambda_a) - lambda_a - _logfactorial(away_goals)
    )
    
    # τ correction: only matters for the four corner cases, do those one at a time
    low_score_mask = (home_goals < 2) & (away_goals < 2)
    if low_score_mask.any():
        idx = np.where(low_score_mask)[0]
        for i in idx:
            t = tau(int(home_goals[i]), int(away_goals[i]), lambda_h[i], lambda_a[i], rho)
            if t <= 0:
                return 1e10  # τ went negative; this region is infeasible
            log_lik[i] += np.log(t)
    
    # Apply time decay weights
    return -np.sum(weights * log_lik)


# Vectorized log(n!) for integer arrays
def _logfactorial(n: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln
    return gammaln(n + 1)


def fit_dixon_coles(
    df: pd.DataFrame,
    xi: float = 0.0,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> DixonColesModel:
    """Fit the Dixon-Coles model.
    
    Args:
        df: DataFrame with Date, HomeTeam, AwayTeam, FTHG, FTAG.
        xi: Time decay rate. 0 = no decay (same as Maher + τ correction).
            Typical values: 0.001 to 0.005 per day.
        cutoff_date: If provided, only fit on matches before this date.
    """
    if cutoff_date is not None:
        df = df[df["Date"] < cutoff_date].copy()
    
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    team_idx = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    
    # Pre-index everything for speed
    home_idx = df["HomeTeam"].map(team_idx).values
    away_idx = df["AwayTeam"].map(team_idx).values
    home_goals = df["FTHG"].values.astype(int)
    away_goals = df["FTAG"].values.astype(int)
    
    # Time decay weights (relative to the most recent match in df)
    most_recent = df["Date"].max()
    days_ago = (most_recent - df["Date"]).dt.days.values
    weights = np.exp(-xi * days_ago)
    
    # Initial parameters: zeros for attack/defense, small positive home_adv, zero rho
    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = 0.2  # home advantage
    x0[2 * n + 1] = 0.0  # rho
    
    # Constraint: average attack = 0 (otherwise the model is unidentified — 
    # you could add a constant to all attacks and subtract it from defenses)
    constraints = {
        "type": "eq",
        "fun": lambda p: p[:n].mean(),
    }
    
    print(f"Fitting Dixon-Coles on {len(df)} matches, {n} teams, xi={xi}...")
    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(teams, team_idx, home_idx, away_idx, home_goals, away_goals, weights),
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 200, "disp": False},
    )
    
    if not result.success:
        print(f"  WARNING: optimization may not have converged: {result.message}")
    print(f"  Done. Final negative log-likelihood: {result.fun:.2f}")
    
    attack_arr, defense_arr, home_adv, rho = _unpack_params(result.x, teams)
    
    return DixonColesModel(
        teams=teams,
        attack=dict(zip(teams, attack_arr)),
        defense=dict(zip(teams, defense_arr)),
        home_adv=home_adv,
        rho=rho,
        xi=xi,
    )


def main():
    DATA_DIR = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(DATA_DIR / "matches.parquet")
    
    # No time decay first — just the τ correction
    model = fit_dixon_coles(df, xi=0.0)
    print(f"\nHome advantage: {model.home_adv:.3f}")
    print(f"Rho (low-score correction): {model.rho:.4f}")
    
    # Predict the same Benfica-Porto we did before
    pred = model.predict_match("Benfica", "Porto")
    print(f"\n=== {pred['home_team']} (home) vs {pred['away_team']} ===")
    print(f"Expected goals: {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}")
    print(f"P(home win): {pred['p_home_win']:.1%}")
    print(f"P(draw):     {pred['p_draw']:.1%}")
    print(f"P(away win): {pred['p_away_win']:.1%}")


if __name__ == "__main__":
    main()