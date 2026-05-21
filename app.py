"""Streamlit web app for Primeira Liga match prediction."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# --- Page config ---
st.set_page_config(
    page_title="Primeira Liga Match Predictor",
    page_icon="⚽",
    layout="centered",
)


# --- Load the model once and cache ---
@st.cache_resource
def load_model():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    with open(Path(__file__).parent / "data" / "fitted_model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
teams = sorted(model.teams)


# --- Header ---
st.title("⚽ Primeira Liga Match Predictor")
st.markdown(
    "Statistical match-outcome predictions for the Portuguese Primeira Liga, "
    "based on a Dixon-Coles model fit to 5 seasons of match data."
)


# --- Inputs ---
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home team", teams, index=teams.index("Benfica") if "Benfica" in teams else 0)
with col2:
    away_team = st.selectbox("Away team", teams, index=teams.index("Porto") if "Porto" in teams else 1)


# --- Predict button ---
if home_team == away_team:
    st.warning("Pick two different teams.")
else:
    pred = model.predict_match(home_team, away_team)

    # Probability bar chart
    st.subheader("Match outcome probabilities")
    prob_df = pd.DataFrame({
        "Outcome": [f"{home_team} win", "Draw", f"{away_team} win"],
        "Probability": [pred["p_home_win"], pred["p_draw"], pred["p_away_win"]],
    })
    st.bar_chart(prob_df.set_index("Outcome"), height=300)

    # --- Probabilities as decimal odds (with optional overround) ---
    st.subheader("Match odds")

    overround = st.slider(
        "Bookmaker margin (overround)",
        min_value=1.00,
        max_value=1.10,
        value=1.00,
        step=0.01,
        help=(
            "1.00 = fair odds (probabilities sum to 100%). "
            "Real bookmakers typically use 1.02–1.07."
        ),
    )

    def to_odds(prob: float, margin: float) -> float:
        # Multiplying probability by margin >1 makes implied probs sum to >100%
        # which shortens the displayed odds (bookmaker's edge).
        return 1 / (prob * margin)

    odd_home = to_odds(pred["p_home_win"], overround)
    odd_draw = to_odds(pred["p_draw"], overround)
    odd_away = to_odds(pred["p_away_win"], overround)

    # Big three "betting site" boxes
    c1, c2, c3 = st.columns(3)
    c1.metric(
        label=f"{home_team} win",
        value=f"{odd_home:.2f}",
        delta=f"{pred['p_home_win']:.1%} probability",
        delta_color="off",
    )
    c2.metric(
        label="Draw",
        value=f"{odd_draw:.2f}",
        delta=f"{pred['p_draw']:.1%} probability",
        delta_color="off",
    )
    c3.metric(
        label=f"{away_team} win",
        value=f"{odd_away:.2f}",
        delta=f"{pred['p_away_win']:.1%} probability",
        delta_color="off",
    )

    # Show what €10 returns if the bet wins
    st.caption(
        f"A €10 bet on **{home_team} win** would return **€{10 * odd_home:.2f}**, "
        f"on **draw** would return **€{10 * odd_draw:.2f}**, "
        f"on **{away_team} win** would return **€{10 * odd_away:.2f}**."
    )

    # --- Compare model's odds to Bet365's ---
    st.divider()
    st.subheader("Model vs Bet365")

    # Load the raw matches once (so we can find the most recent meeting)
    @st.cache_data
    def load_matches() -> pd.DataFrame:
        return pd.read_parquet(Path(__file__).parent / "data" / "matches.parquet")

    all_matches = load_matches()

    from src.odds import find_latest_h2h, implied_probabilities, overround

    last_match = find_latest_h2h(all_matches, home_team, away_team)

    if last_match is None:
        st.info(
            f"No historical matches found between {home_team} and {away_team} "
            "in dataset. Bet365 comparison unavailable."
        )
    elif pd.isna(last_match["B365CH"]) or pd.isna(last_match["B365CD"]) or pd.isna(last_match["B365CA"]):
        st.info(
            "Bet365 closing odds missing for the most recent meeting. Comparison unavailable."
        )
    else:
        # If the historical match had teams in reversed orientation, flip the odds
        flipped = last_match["HomeTeam"] != home_team
        if flipped:
            book_home = last_match["B365CA"]
            book_draw = last_match["B365CD"]
            book_away = last_match["B365CH"]
        else:
            book_home = last_match["B365CH"]
            book_draw = last_match["B365CD"]
            book_away = last_match["B365CA"]

        # Convert Bet365 odds to fair probabilities
        book_odds = (book_home, book_draw, book_away)
        book_probs = implied_probabilities(book_odds)
        book_margin = overround(book_odds)

        # Build the comparison table
        comparison_df = pd.DataFrame({
            "Outcome": [f"{home_team} win", "Draw", f"{away_team} win"],
            "Model": [
                f"{1 / pred['p_home_win']:.2f}",
                f"{1 / pred['p_draw']:.2f}",
                f"{1 / pred['p_away_win']:.2f}",
            ],
            "Prob.": [
                f"{pred['p_home_win']:.1%}",
                f"{pred['p_draw']:.1%}",
                f"{pred['p_away_win']:.1%}",
            ],
            "Bet365 odds": [
                f"{book_home:.2f}",
                f"{book_draw:.2f}",
                f"{book_away:.2f}",
            ],
            "Bet365 prob.": [
                f"{book_probs[0]:.1%}",
                f"{book_probs[1]:.1%}",
                f"{book_probs[2]:.1%}",
            ],
        })
        st.table(comparison_df)

        st.caption(
            f"Bet365 odds taken from the most recent {home_team} vs {away_team} match: "
            f"**{last_match['HomeTeam']} {int(last_match['FTHG'])} - "
            f"{int(last_match['FTAG'])} {last_match['AwayTeam']}** "
            f"on {last_match['Date'].date()}. "
            f"Bet365's overround on this match: {(book_margin - 1) * 100:.1f}%."
        )

        # --- Expected value: where would the model bet? ---
        st.subheader("Expected value at Bet365 odds")
        st.markdown(
            "**EV** = (your probability × bookmaker odds) − 1. "
            "Positive means Model thinks the bet is profitable. "
            "Negative means the bookmaker has the edge."
        )

        ev_home = pred["p_home_win"] * book_home - 1
        ev_draw = pred["p_draw"] * book_draw - 1
        ev_away = pred["p_away_win"] * book_away - 1

        ev_df = pd.DataFrame({
            "Outcome": [f"{home_team} win", "Draw", f"{away_team} win"],
            "Bet365 odds": [f"{book_home:.2f}", f"{book_draw:.2f}", f"{book_away:.2f}"],
            "Your prob.": [
                f"{pred['p_home_win']:.1%}",
                f"{pred['p_draw']:.1%}",
                f"{pred['p_away_win']:.1%}",
            ],
            "Expected value": [f"{ev_home:+.1%}", f"{ev_draw:+.1%}", f"{ev_away:+.1%}"],
        })
        st.table(ev_df)

        best_ev = max(ev_home, ev_draw, ev_away)
        if best_ev > 0:
            best_name = [f"{home_team} win", "Draw", f"{away_team} win"][
                [ev_home, ev_draw, ev_away].index(best_ev)
            ]
            st.success(
                f"Model identifies **{best_name}** as the best-value bet "
                f"(EV {best_ev:+.1%} per €1 staked). "
                "Note: this does not constitute betting advice. Model has known limitations."
            )
        else:
            st.info(
                "Model thinks Bet365's odds are fair or favor the bookmaker on every outcome. "
                "No positive-EV bet identified."
            )

    # Expected goals
    st.subheader("Expected scoreline")
    st.markdown(
        f"**{home_team} {pred['lambda_home']:.2f} — {pred['lambda_away']:.2f} {away_team}**  "
        f"(model's expected goals for each side)"
    )

    # Top 5 most likely scorelines
    st.subheader("Most likely scorelines")
    matrix = pred["score_matrix"]
    flat_idx = np.argsort(matrix, axis=None)[::-1][:5]
    rows = []
    for idx in flat_idx:
        h, a = np.unravel_index(idx, matrix.shape)
        rows.append({
            "Scoreline": f"{home_team} {h} - {a} {away_team}",
            "Probability": f"{matrix[h, a]:.1%}",
        })
    st.table(pd.DataFrame(rows))

    # Full scoreline heatmap (only show 0-5 to keep it readable)
    st.subheader("Full scoreline probabilities")
    display_size = 6
    heatmap_df = pd.DataFrame(
        matrix[:display_size, :display_size],
        index=[f"{home_team} {i}" for i in range(display_size)],
        columns=[f"{away_team} {i}" for i in range(display_size)],
    )
    st.dataframe(
        heatmap_df.style.format("{:.1%}").background_gradient(cmap="Blues"),
        use_container_width=True,
    )


# --- Footer ---
st.divider()
st.caption(
    "Model: Dixon-Coles (1997) with τ correction. Trained on Primeira Liga matches "
    "from 2021-22 to 2025-26. Built by Eduardo Tavares, IEM student at Faculdade de Engenharia da Universidade do Porto, with Claude Code assistance."
)
