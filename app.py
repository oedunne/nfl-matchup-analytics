import hmac
import json
from datetime import datetime, timezone
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="NFL Matchup Analytics",
    page_icon="🏈",
    layout="wide",
)

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

MARKETS = {
    "Passing yards": "player_pass_yds",
    "Rushing yards": "player_rush_yds",
    "Receiving yards": "player_reception_yds",
    "Receptions": "player_receptions",
}

PROP_COLUMNS = [
    "Player",
    "Matchup",
    "Kickoff (UTC)",
    "Stat",
    "Side",
    "Line",
    "Odds",
    "Updated (UTC)",
]


def odds_request(path, parameters):
    api_key = st.secrets.get("ODDS_API_KEY", "")

    if not api_key:
        raise RuntimeError("Add ODDS_API_KEY to Streamlit Secrets.")

    parameters = {**parameters, "apiKey": api_key}
    url = f"{API_BASE}/{path}?{urlencode(parameters)}"

    try:
        with urlopen(url, timeout=25) as response:
            data = json.load(response)
            remaining = response.headers.get("x-requests-remaining")
            return data, remaining
    except HTTPError as error:
        raise RuntimeError(
            f"Data provider returned HTTP {error.code}. "
            "Check your key, plan, and remaining credits."
        ) from None
    except (URLError, TimeoutError, OSError, ValueError):
        # Do not expose request URLs, which contain the API key.
        raise RuntimeError(
            "Could not load data. Please try again later."
        ) from None


@st.cache_data(ttl=3600, show_spinner=False)
def load_events():
    data, _ = odds_request(f"sports/{SPORT}/events", {})
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def load_props(event_id):
    data, remaining = odds_request(
        f"sports/{SPORT}/events/{event_id}/odds",
        {
            "bookmakers": "draftkings",
            "markets": ",".join(MARKETS.values()),
            "oddsFormat": "american",
        },
    )
    fetched_at = datetime.now(timezone.utc).isoformat()
    return data, remaining, fetched_at


def props_to_table(data):
    rows = []
    labels = {value: label for label, value in MARKETS.items()}
    matchup = (
        f"{data.get('away_team', '')} at "
        f"{data.get('home_team', '')}"
    )

    for bookmaker in data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            for outcome in market.get("outcomes", []):
                rows.append(
                    {
                        "Player": outcome.get("description"),
                        "Matchup": matchup,
                        "Kickoff (UTC)": data.get("commence_time"),
                        "Stat": labels.get(
                            market["key"], market["key"]
                        ),
                        "Side": outcome.get("name"),
                        "Line": outcome.get("point"),
                        "Odds": outcome.get("price"),
                        "Updated (UTC)": market.get(
                            "last_update",
                            bookmaker.get("last_update"),
                        ),
                    }
                )

    return pd.DataFrame(rows, columns=PROP_COLUMNS)


if "imported_props" not in st.session_state:
    st.session_state.imported_props = pd.DataFrame(
        columns=PROP_COLUMNS
    )

if "props_fetched_at" not in st.session_state:
    st.session_state.props_fetched_at = None


st.title("🏈 NFL Matchup Analytics")
st.write(
    "Weekly player projections informed by performance, "
    "workload, and defensive schemes."
)

weekly, comparisons, results, methodology, admin = st.tabs(
    [
        "Projections & Matchups",
        "Prop Comparisons",
        "Results Tracker",
        "Methodology",
        "Admin / Update Data",
    ]
)


with weekly:
    st.header("Projections & Matchups")
    st.info(
        "No projections published yet. "
        "The model and weekly data imports are under development."
    )

    st.dataframe(
        pd.DataFrame(
            columns=[
                "Player",
                "Position",
                "Opponent",
                "Stat",
                "Season Average",
                "Projection",
                "Matchup Explanation",
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Planned matchup analysis")
    st.markdown(
        """
        - **Receiving:** player production against man and zone
          coverage, combined with the opponent's coverage tendencies.
        - **Rushing:** player efficiency on zone and gap concepts,
          combined with offensive usage and defensive performance.
        - **Workload:** opportunities help translate efficiency
          into projected game totals.
        """
    )


with comparisons:
    st.header("Prop Comparisons")
    st.write(
        "Imported DraftKings lines will be compared with "
        "our projections once the model is ready."
    )

    st.info(
        "No model comparisons yet. Imported lines below "
        "are not recommendations."
    )

    props = st.session_state.imported_props

    if props.empty:
        st.write(
            "Use Admin / Update Data to test importing "
            "props for one game."
        )
    else:
        st.caption(
            "This session's imported game only. "
            f"Fetched: {st.session_state.props_fetched_at}"
        )

        stat_filter = st.selectbox(
            "Stat category",
            ["All"] + sorted(props["Stat"].dropna().unique().tolist()),
        )

        visible_props = props.copy()

        if stat_filter != "All":
            visible_props = visible_props[
                visible_props["Stat"] == stat_filter
            ]

        st.dataframe(
            visible_props,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download imported props as CSV",
            data=props.to_csv(index=False).encode("utf-8"),
            file_name="draftkings_props.csv",
            mime="text/csv",
        )

    st.caption(
        "Line differences alone do not establish a betting "
        "advantage. Odds and forecast uncertainty also matter."
    )


with results:
    st.header("Results Tracker")
    st.info(
        "No predictions saved yet. Permanent prediction "
        "history and results tracking will be added next."
    )

    st.dataframe(
        pd.DataFrame(
            columns=[
                "Season",
                "Week",
                "Player",
                "Opponent",
                "Stat",
                "Prediction Saved At",
                "Projection",
                "Prop Line",
                "Actual Result",
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.write(
        "We will compare model accuracy with a simple "
        "season-average forecast and preserve pregame predictions."
    )


with methodology:
    st.header("Methodology")
    st.write(
        "The model is under development. No validated "
        "projections or success rates are published yet."
    )

    st.subheader("Research question")
    st.write(
        "Does adding defensive scheme information improve "
        "player forecasts compared with using season averages?"
    )

    st.subheader("Planned inputs")
    st.markdown(
        """
        - Defensive coverage frequencies.
        - QB, WR, and TE performance against man and zone.
        - RB performance on zone and gap runs.
        - Defensive performance against zone and gap runs.
        - Player workload and weekly opponents.
        """
    )

    st.subheader("Evaluation")
    st.markdown(
        """
        - Use information available before kickoff.
        - Record the season and data cutoff for each upload.
        - Preserve predictions before games begin.
        - Compare forecasts with actual results and a baseline.
        - Account for sample sizes when interpreting splits.
        """
    )

    st.caption(
        "Injuries and depth-chart adjustments are outside "
        "the initial model's scope."
    )


with admin:
    st.header("Admin / Update Data")
    st.write(
        "Import DraftKings props for one game at a time."
    )

    st.info(
        "Weekly statistics uploads and permanent storage "
        "are not connected yet. Imported props currently "
        "stay in your browser session only and are not "
        "published to other visitors. Download a CSV to keep a copy."
    )

    password = st.text_input(
        "Admin password",
        type="password",
    )

    if password:
        expected = st.secrets.get("ADMIN_PASSWORD", "")

        if not expected:
            st.warning(
                "Add ADMIN_PASSWORD to Streamlit Secrets."
            )
        elif not hmac.compare_digest(
            password.encode(), expected.encode()
        ):
            st.error("Incorrect password.")
        else:
            st.success("Admin access enabled.")

            try:
                events = load_events()

                if not events:
                    st.info("No upcoming games are available.")
                else:
                    event_map = {
                        event["id"]: event for event in events
                    }

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
                        "Requests passing yards, rushing yards, "
                        "receiving yards, and receptions. "
                        "Identical requests are cached for one hour. "
                        "A cache miss uses API credits."
                    )

                    if st.button(
                        "Load this game's DraftKings props",
                        type="primary",
                    ):
                        with st.spinner("Loading props..."):
                            data, remaining, fetched_at = load_props(
                                event_id
                            )

                        imported = props_to_table(data)

                        # Replace the previous game's session data.
                        st.session_state.imported_props = imported
                        st.session_state.props_fetched_at = fetched_at

                        if imported.empty:
                            st.warning(
                                "No DraftKings props returned for "
                                "these categories. They may not be "
                                "posted or available in this feed."
                            )
                        else:
                            st.success(
                                f"Loaded {len(imported)} over/under "
                                "outcomes. View them in Prop Comparisons."
                            )

                        if remaining is not None:
                            st.caption(
                                "API credits remaining when fetched: "
                                f"{remaining}"
                            )

            except RuntimeError as error:
                st.error(str(error))


st.divider()
st.caption(
    "NFL Matchup Analytics — Independent sports analytics "
    "project — Created by Owen Dunne"
)
