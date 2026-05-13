"""Utilities for working with betting odds and computing expected value."""

from typing import Optional
import pandas as pd
import numpy as np


def implied_probabilities(odds: tuple[float, float, float]) -> np.ndarray:
    """Convert decimal odds for (home, draw, away) into normalized probabilities.
    
    Removes the bookmaker's overround by rescaling so probabilities sum to 1.
    """
    inv = np.array([1 / o for o in odds])
    return inv / inv.sum()


def overround(odds: tuple[float, float, float]) -> float:
    """Bookmaker's margin: how much the implied probs sum to over 1.
    
    Example: 1.07 means a 7% overround.
    """
    return sum(1 / o for o in odds)


def find_latest_h2h(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> Optional[pd.Series]:
    """Return the most recent historical match between two teams (any orientation)."""
    mask = (
        ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team))
        | ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
    )
    matches = df[mask].sort_values("Date", ascending=False)
    if len(matches) == 0:
        return None
    return matches.iloc[0]