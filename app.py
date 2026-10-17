import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="NFL Matchup Analytics",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 NFL Matchup Analytics")
st.write(
    "Exploring how player performance and defensive matchups "
    "help predict weekly NFL results."
)

weekly, results, methodology = st.tabs(
    ["Weekly Analysis", "Results Tracker", "Methodology"]
)

with weekly:
    st.header("Weekly Analysis")
    st.info(
        "No projections published yet. "
        "Our first step is choosing the data and building the model."
    )

    st.subheader("Planned analysis")
    st.write(
        "Start with running back rushing yards, using workload, "
        "efficiency, and the opposing defense."
    )

with results:
    st.header("Results Tracker")
    st.write(
        "Pregame predictions and actual results will appear here "
        "as we evaluate the model."
    )

    empty_results = pd.DataFrame(
        columns=[
            "Season",
            "Week",
            "Player",
            "Opponent",
            "Projected Yards",
            "Actual Yards",
        ]
    )
    st.dataframe(empty_results, hide_index=True, use_container_width=True)

with methodology:
    st.header("How the Model Works")
    st.write(
        "The model is under development. We will document its "
        "inputs, calculations, and limitations here."
    )

    st.markdown(
        """
        Our evaluation will:
        - Use only information available before each game.
        - Save predictions before kickoff.
        - Compare forecasts with actual results.
        - Test whether matchup data improves on a player's average.
        """
    )

st.divider()
st.caption(
    "NFL Matchup Analytics | Created by Owen Dunne | "
    "Independent sports analytics project"
)
