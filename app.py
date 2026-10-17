import hmac
import json
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

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


def get_secret(name):
    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


def odds_request(path, parameters):
    api_key = get_secret("ODDS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY is missing from Streamlit Secrets."
        )

    query = urlencode({**parameters, "apiKey": api_key})
    url = f"{API_BASE}/{path}?{query}"

    try:
        with urlopen(url, timeout=25) as response:
            data = json.load(response)
            remaining = response.headers.get(
                "x-requests-remaining"
            )
        return data, remaining

    except HTTPError as error:
        messages = {
            401: "The API key was rejected. Check Streamlit Secrets.",
            403: "The provider denied access. Check your API plan.",
            422: "The provider rejected the requested game or markets.",
            429: "The API quota or request limit has been reached.",
        }
        raise RuntimeError(
            messages.get(
                error.code,
                f"The data provider returned HTTP {error.code}.",
            )
        ) from None

    except (URLError, TimeoutError, OSError):
        raise RuntimeError(
            "Could not connect to the data provider. Try again later."
        ) from None

    except ValueError:
        raise RuntimeError(
            "The provider returned an unreadable response."
        ) from None


@st.cache_data(ttl=3600, show_spinner=False)
def load_events():
    data, _ = odds_request(
        f"sports/{SPORT}/events",
        {},
    )

    if not isinstance(data, list):
        raise RuntimeError("Unexpected game-list response.")

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

    if not isinstance(data, dict):
        raise RuntimeError("Unexpected props response.")

    fetched_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    return data, remaining, fetched_at


def props_to_table(data):
    labels = {
        market_key: label
        for label, market_key in MARKETS.items()
    }

    matchup = (
        f"{data.get('away_team', '')} at "
        f"{data.get('home_team', '')}"
    )

    rows = []

    for bookmaker in data.get("bookmakers", []):
        if bookmaker.get("key") != "draftkings":
            continue

        for market in bookmaker.get("markets", []):
            for outcome in market.get("outcomes", []):
                rows.append(
                    {
                        "Player": outcome.get("description"),
                        "Matchup": matchup,
                        "Kickoff (UTC)": data.get(
                            "commence_time"
                        ),
                        "Stat": labels.get(
                            market.get("key"),
                            market.get("key"),
                        ),
                        "Side": outcome.get("name"),
                        "Line": outcome.get("point"),
                        "Odds": outcome.get("price"),
                        "Updated (UTC)": market.get(
                            "last_update"
                        ) or bookmaker.get("last_update"),
                    }
                )

    return pd.DataFrame(rows, columns=PROP_COLUMNS)


def import_selected_game():
    """Runs before the page renders when the button is clicked."""
    event_id = st.session_state.get("selected_event_id")
    entered = st.session_state.get("admin_password", "")
    expected = get_secret("ADMIN_PASSWORD")

    if not expected or not hmac.compare_digest(
        entered.encode("utf-8"),
        expected.encode("utf-8"),
    ):
        st.session_state.import_notice = (
            "error",
            "Enter the correct admin password first.",
        )
        return

    if not event_id:
        st.session_state.import_notice = (
            "error",
            "Choose a game first.",
        )
        return

    try:
        data, remaining, fetched_at = load_props(event_id)
        table = props_to_table(data)

        st.session_state.imported_props = table
        st.session_state.props_fetched_at = fetched_at
        st.session_state.credits_remaining = remaining

        if table.empty:
            st.session_state.import_notice = (
                "warning",
                "No DraftKings props were returned for these "
                "categories. They may not be posted yet or "
                "available through this feed.",
            )
        else:
            st.session_state.import_notice = (
                "success",
                f"Loaded {len(table)} over/under outcomes. "
                "Open Prop Comparisons to see them.",
            )

    except RuntimeError as error:
        st.session_state.import_notice = (
            "error",
            str(error),
        )


if "imported_props" not in st.session_state:
    st.session_state.imported_props = pd.DataFrame(
        columns=PROP_COLUMNS
    )

if "props_fetched_at" not in st.session_state:
    st.session_state.props_fetched_at = None

if "credits_remaining" not in st.session_state:
    st.session_state.credits_remaining = None

if "import_notice" not in st.session_state:
    st.session_state.import_notice = None


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
        "The model and weekly statistics imports are "
        "under development."
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

    st.subheader("Planned analysis")

    st.markdown(
        """
        - **Receiving:** performance against man and zone
          coverage, combined with opponent coverage tendencies.
        - **Rushing:** performance on zone and gap runs,
          combined with offensive usage and defensive efficiency.
        - **Workload:** opportunities used to translate
          efficiency into projected game totals.
        """
    )


with comparisons:
    st.header("Prop Comparisons")

    st.caption(
        "The model is not connected yet. These are imported "
        "sportsbook lines, not model recommendations."
    )

    props = st.session_state.imported_props

    if props.empty:
        st.info(
            "No props loaded in this session. Go to "
            "Admin / Update Data, enter your password, "
            "choose a game, and load its props."
        )
    else:
        st.write(
            f"**Fetched:** "
            f"{st.session_state.props_fetched_at}"
        )

        categories = sorted(
            props["Stat"].dropna().unique().tolist()
        )

        category = st.selectbox(
            "Stat category",
            ["All"] + categories,
            key="comparison_stat_filter",
        )

        search = st.text_input(
            "Search player",
            placeholder="Enter a player name",
            key="comparison_player_search",
        )

        visible = props.copy()

        if category != "All":
            visible = visible[
                visible["Stat"] == category
            ]

        if search.strip():
            visible = visible[
                visible["Player"]
                .fillna("")
                .str.contains(
                    search.strip(),
                    case=False,
                    regex=False,
                )
            ]

        st.dataframe(
            visible.drop(
                columns=["Kickoff (UTC)", "Updated (UTC)"],
                errors="ignore",
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download this game's props",
            data=props.to_csv(index=False).encode("utf-8"),
            file_name="draftkings_props.csv",
            mime="text/csv",
        )

        st.caption(
            "Each over and under is a separate row. "
            "Only the most recently imported game is shown."
        )

    st.caption(
        "Imported data currently stays in your session only. "
        "Permanent publishing will be added separately."
    )


with results:
    st.header("Results Tracker")

    st.info(
        "No predictions saved yet. Permanent history "
        "and results tracking are not implemented."
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
        "We will preserve pregame forecasts and compare "
        "their accuracy with a season-average baseline."
    )


with methodology:
    st.header("Methodology")

    st.write(
        "The model is under development. No validated "
        "projections or success rates are published yet."
    )

    st.subheader("Research question")

    st.write(
        "Does defensive scheme information improve player "
        "forecasts compared with season averages?"
    )

    st.subheader("Planned inputs")

    st.markdown(
        """
        - Defensive man and zone coverage frequencies.
        - QB, WR, and TE performance against those coverages.
        - RB performance on zone and gap runs.
        - Defensive performance against zone and gap runs.
        - Offensive scheme usage, player workload, and opponents.
        """
    )

    st.subheader("Evaluation")

    st.markdown(
        """
        - Use only information available before kickoff.
        - Record the season and data cutoff for each upload.
        - Save predictions before games begin.
        - Compare predictions with actual results and a baseline.
        - Consider sample sizes when interpreting splits.
        """
    )

    st.caption(
        "Injuries and depth-chart adjustments are outside "
        "the initial model's scope. Projection differences "
        "alone do not establish a betting advantage."
    )


with admin:
    st.header("Admin / Update Data")

    st.info(
        "The DraftKings import is available below. "
        "Weekly statistics uploads and permanent storage "
        "are not connected yet."
    )

    st.text_input(
        "Admin password",
        type="password",
        key="admin_password",
    )

    expected_password = get_secret("ADMIN_PASSWORD")
    entered_password = st.session_state.get(
        "admin_password", ""
    )

    authenticated = bool(expected_password) and (
        hmac.compare_digest(
            entered_password.encode("utf-8"),
            expected_password.encode("utf-8"),
        )
    )

    if not expected_password:
        st.warning(
            "Add ADMIN_PASSWORD to Streamlit Secrets."
        )
    elif entered_password and not authenticated:
        st.error("Incorrect password.")

    if authenticated:
        st.success("Admin access enabled.")
                st.subheader("Weekly Statistics Upload")

        import base64
        import io
        from urllib.request import Request

        DATASETS = {
            "Defensive Man/Zone Coverage": "defensive_coverage",
            "RB vs. Gap Runs": "rb_gap",
            "RB vs. Zone Runs": "rb_zone",
            "Full RB Stats / Receiving": "rb_full",
            "WR vs. Man": "wr_man",
            "WR vs. Zone": "wr_zone",
            "TE vs. Man": "te_man",
            "TE vs. Zone": "te_zone",
            "QB vs. Man": "qb_man",
            "QB vs. Zone": "qb_zone",
            "Defense vs. Zone/Gap Runs": "defensive_run_splits",
            "CBS Stats Allowed — QB": "cbs_allowed_qb",
            "CBS Stats Allowed — RB": "cbs_allowed_rb",
            "CBS Stats Allowed — WR": "cbs_allowed_wr",
            "CBS Stats Allowed — TE": "cbs_allowed_te",
            "Weekly Schedule / Opponents": "weekly_schedule",
        }

        def github_file_request(path, payload=None):
            token = get_secret("GITHUB_TOKEN")
            repo = get_secret("GITHUB_REPO")
            branch = get_secret("GITHUB_BRANCH") or "main"

            if not token or not repo:
                raise RuntimeError(
                    "Add GITHUB_TOKEN and GITHUB_REPO "
                    "to Streamlit Secrets."
                )

            url = (
                f"https://api.github.com/repos/{repo}"
                f"/contents/{path}"
            )

            if payload is None:
                url += "?" + urlencode({"ref": branch})

            request = Request(
                url,
                data=(
                    json.dumps(payload).encode("utf-8")
                    if payload is not None else None
                ),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "Content-Type": "application/json",
                    "User-Agent": "NFL-Matchup-Analytics",
                },
                method="PUT" if payload is not None else "GET",
            )

            try:
                with urlopen(request, timeout=30) as response:
                    return json.load(response)
            except HTTPError as error:
                if error.code == 404 and payload is None:
                    return None

                messages = {
                    401: "GitHub rejected the token. Check its expiration.",
                    403: "Check the token's Contents permission: Read and write.",
                    404: "Check the repository name and token's repository access.",
                    409: "The file changed during saving. Try saving again.",
                    422: "GitHub rejected the save. Check branch permissions.",
                }

                raise RuntimeError(
                    messages.get(
                        error.code,
                        f"GitHub returned HTTP {error.code}.",
                    )
                ) from None
            except (URLError, TimeoutError, OSError, ValueError):
                raise RuntimeError(
                    "Could not complete the GitHub request. "
                    "Check the repository before retrying a save."
                ) from None

        dataset = st.selectbox(
            "Which table are you uploading?",
            list(DATASETS),
            key="weekly_dataset",
        )

        save_path = f"data/{DATASETS[dataset]}.csv"

        st.caption(
            "Saving updates this category only. "
            "Previous versions remain in GitHub history."
        )

        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            key="weekly_stats_file",
        )

        if uploaded_file is not None:
            preview = None

            try:
                if uploaded_file.size > 500_000:
                    st.error("Please use a CSV smaller than 500 KB.")
                else:
                    preview = pd.read_csv(
                        io.BytesIO(uploaded_file.getvalue())
                    )

                    if preview.empty:
                        st.error("This file has no data rows.")
                        preview = None
                    else:
                        st.write(
                            f"**{dataset}:** {len(preview)} rows, "
                            f"{len(preview.columns)} columns"
                        )
                        st.dataframe(
                            preview,
                            hide_index=True,
                            use_container_width=True,
                        )
            except (
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
                UnicodeError,
            ):
                st.error(
                    "Could not read this CSV. Export it as "
                    "UTF-8 CSV with column headings."
                )

            if preview is not None:
                st.caption(
                    "Check that the preview matches the selected "
                    "category before saving."
                )

                if st.button(
                    "Save this table to GitHub",
                    key="save_weekly_table",
                ):
                    try:
                        with st.spinner("Saving table..."):
                            existing = github_file_request(save_path)

                            content = preview.to_csv(
                                index=False
                            ).encode("utf-8")

                            payload = {
                                "message": f"Update {dataset}",
                                "content": base64.b64encode(
                                    content
                                ).decode("ascii"),
                                "branch": (
                                    get_secret("GITHUB_BRANCH")
                                    or "main"
                                ),
                            }

                            if existing is not None:
                                payload["sha"] = existing["sha"]

                            github_file_request(
                                save_path, payload
                            )

                        st.success(
                            f"Saved {dataset} permanently "
                            f"to {save_path}."
                        )
                    except RuntimeError as error:
                        st.error(str(error))

        if st.button(
            "View saved table for this category",
            key="view_saved_table",
        ):
            try:
                saved = github_file_request(save_path)

                if saved is None:
                    st.info(
                        "No saved file found. If you expected one, "
                        "check the repository and token access."
                    )
                else:
                    saved_bytes = base64.b64decode(
                        saved["content"]
                    )
                    saved_table = pd.read_csv(
                        io.BytesIO(saved_bytes)
                    )

                    st.write(f"**Saved table: {dataset}**")
                    st.dataframe(
                        saved_table,
                        hide_index=True,
                        use_container_width=True,
                    )
            except RuntimeError as error:
                st.error(str(error))
            except (ValueError, KeyError, UnicodeError):
                st.error("The saved file could not be read.")

        st.caption(
            "Saved statistics are not connected to projections yet. "
            "Column validation and the model come next."
        )

        st.divider()
        try:
            with st.spinner("Loading available games..."):
                events = load_events()

            if not events:
                st.info("No upcoming games are available.")
            else:
                event_map = {
                    event["id"]: event
                    for event in events
                    if "id" in event
                }

                if event_map:
                    st.selectbox(
                        "Choose one game",
                        options=list(event_map),
                        format_func=lambda event_id: (
                            f"{event_map[event_id]['away_team']} at "
                            f"{event_map[event_id]['home_team']} — "
                            f"{event_map[event_id]['commence_time']}"
                        ),
                        key="selected_event_id",
                    )

                    st.caption(
                        "Loads passing yards, rushing yards, "
                        "receiving yards, and receptions. "
                        "Identical imports are cached for one hour. "
                        "A new request uses API credits."
                    )

                    st.button(
                        "Load this game's DraftKings props",
                        type="primary",
                        on_click=import_selected_game,
                    )

                notice = st.session_state.import_notice

                if notice:
                    notice_type, message = notice

                    if notice_type == "success":
                        st.success(message)
                    elif notice_type == "warning":
                        st.warning(message)
                    else:
                        st.error(message)

                remaining = (
                    st.session_state.credits_remaining
                )

                if remaining is not None:
                    st.caption(
                        "API credits remaining when last fetched: "
                        f"{remaining}"
                    )

        except RuntimeError as error:
            st.error(str(error))


st.divider()
st.caption(
    "NFL Matchup Analytics — Independent sports analytics "
    "project — Created by Owen Dunne"
)
