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
import hmac
import json
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

MARKETS = {
    "Passing yards": "player_pass_yds",
    "Rushing yards": "player_rush_yds",
    "Receiving yards": "player_reception_yds",
    "Receptions": "player_receptions",
}


def odds_request(path, parameters):
    parameters = {
        **parameters,
        "apiKey": st.secrets["ODDS_API_KEY"],
    }
    url = f"{API_BASE}/{path}?{urlencode(parameters)}"

    try:
        with urlopen(url, timeout=25) as response:
            data = json.load(response)
            remaining = response.headers.get("x-requests-remaining")
            return data, remaining
    except HTTPError as error:
        # Never display the request URL: it contains the API key.
        raise RuntimeError(
            f"Data provider returned HTTP {error.code}. "
            "Check your key, plan, and remaining credits."
        ) from None
    except (URLError, TimeoutError, ValueError):
        raise RuntimeError(
            "Could not load data. Please try again later."
        ) from None


@st.cache_data(ttl=3600, show_spinner=False)
def load_events():
    data, _ = odds_request(f"sports/{SPORT}/events", {})
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def load_props(event_id):
    return odds_request(
        f"sports/{SPORT}/events/{event_id}/odds",
        {
            "bookmakers": "draftkings",
            "markets": ",".join(MARKETS.values()),
            "oddsFormat": "american",
        },
    )


st.divider()
st.header("DraftKings Connection Test")

password = st.text_input("Admin password", type="password")

if password:
    expected = st.secrets.get("ADMIN_PASSWORD", "")

    if not expected:
        st.warning("Add ADMIN_PASSWORD to Streamlit Secrets first.")
    elif not hmac.compare_digest(
        password.encode(), expected.encode()
    ):
        st.error("Incorrect password.")
    else:
        try:
            events = load_events()

            if not events:
                st.info("No upcoming games are available.")
            else:
                event_map = {event["id"]: event for event in events}

                event_id = st.selectbox(
                    "Choose one game",
                    options=list(event_map),
                    format_func=lambda key: (
                        f"{event_map[key]['away_team']} at "
                        f"{event_map[key]['home_team']} — "
                        f"{event_map[key]['commence_time']}"
                    ),
                )

                st.caption(
                    "Imports four stat categories for this game only. "
                    "Identical requests are cached for one hour."
                )

                if st.button("Load this game's DraftKings props"):
                    with st.spinner("Loading props..."):
                        data, remaining = load_props(event_id)

                    rows = []
                    market_labels = {
                        value: label for label, value in MARKETS.items()
                    }

                    for bookmaker in data.get("bookmakers", []):
                        for market in bookmaker.get("markets", []):
                            for outcome in market.get("outcomes", []):
                                rows.append({
                                    "Player": outcome.get("description"),
                                    "Stat": market_labels.get(
                                        market["key"], market["key"]
                                    ),
                                    "Side": outcome.get("name"),
                                    "Line": outcome.get("point"),
                                    "Odds": outcome.get("price"),
                                    "Updated (UTC)": market.get(
                                        "last_update",
                                        bookmaker.get("last_update"),
                                    ),
                                })

                    if rows:
                        st.dataframe(
                            pd.DataFrame(rows),
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.info(
                            "No DraftKings props returned for these "
                            "categories. They may not be posted yet "
                            "or available through this feed."
                        )

                    if remaining is not None:
                        st.caption(
                            f"API credits remaining when fetched: {remaining}"
                        )

        except RuntimeError as error:
            st.error(str(error))
