import base64
import csv
import hashlib
import hmac
import io
import json
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

MODEL_VERSION = '1.0-experimental'
API_BASE = 'https://api.the-odds-api.com/v4'
SPORT = 'americanfootball_nfl'
MARKETS = {'Passing yards': 'player_pass_yds', 'Rushing yards': 'player_rush_yds',
           'Receiving yards': 'player_reception_yds', 'Receptions': 'player_receptions'}
DATASETS = {
    'Defensive Man/Zone Coverage': 'defensive_coverage',
    'RB vs. Gap Runs': 'rb_gap', 'RB vs. Zone Runs': 'rb_zone',
    'Full RB Stats / Receiving': 'rb_full',
    'WR vs. Man': 'wr_man', 'WR vs. Zone': 'wr_zone',
    'TE vs. Man': 'te_man', 'TE vs. Zone': 'te_zone',
    'QB vs. Man': 'qb_man', 'QB vs. Zone': 'qb_zone',
    'Defense vs. Zone/Gap Runs': 'defensive_run_splits',
    'CBS Stats Allowed — QB': 'cbs_allowed_qb',
    'CBS Stats Allowed — RB': 'cbs_allowed_rb',
    'CBS Stats Allowed — WR': 'cbs_allowed_wr',
    'CBS Stats Allowed — TE': 'cbs_allowed_te',
    'Season Receiving / Teams / Games': 'season_receiving',
    'Season Rushing / Teams / Games': 'season_rushing',
    'Season Passing / Teams / Games': 'season_passing',
}
REQUIRED = set(DATASETS.values()) - {'rb_full'}
TEAM_NAMES = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs', 'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams',
    'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
    'SEA': 'Seattle Seahawks', 'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders',
}
ALIASES = {'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU', 'JAC': 'JAX',
           'LA': 'LAR', 'OAK': 'LV', 'SD': 'LAC', 'SL': 'LAR', 'WSH': 'WAS'}


def team_code(value):
    value = str(value).strip()
    if value in TEAM_NAMES:
        return value
    if value in ALIASES:
        return ALIASES[value]
    for code, full in TEAM_NAMES.items():
        if value.casefold() in (full.casefold(), full.split()[-1].casefold()):
            return code
    return None


def player_key(value):
    value = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode()
    value = re.sub(r'^\s*\d+\s*', '', value)
    value = re.sub(r'\b(jr|sr|ii|iii|iv)\b\.?', '', value, flags=re.I)
    return re.sub(r'[^a-z0-9]', '', value.lower())


def number(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '', regex=False)
                         .str.replace('%', '', regex=False).str.strip(), errors='coerce')


def parse_csv(raw, kind):
    rows = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
    if not rows:
        raise ValueError('The CSV is empty.')
    if kind.startswith('cbs_allowed_'):
        header = next((i for i, row in enumerate(rows) if row[:2] == ['Rank', 'Team']), None)
        if header is None:
            raise ValueError('CBS file needs the Rank / Team header row.')
        if kind == 'cbs_allowed_qb':
            cols = ['Rank', 'Team', 'Pass Attempts', 'Completions', 'Pass Yards', 'Pass TD',
                    'INT', 'Rating', 'Rush Attempts', 'Rush Yards', 'YPC', 'Rush TD', 'FL', 'FPTS']
        else:
            cols = ['Rank', 'Team', 'Rush Attempts', 'Rush Yards', 'YPC', 'Rush TD',
                    'Targets', 'Receptions', 'Rec Yards', 'YPR', 'Rec TD', 'FL', 'FPTS']
        if any(len(row) != len(cols) for row in rows[header + 1:] if any(row)):
            raise ValueError('Unexpected CBS column count.')
        d = pd.DataFrame([r for r in rows[header + 1:] if any(r)], columns=cols)
        d['Team'] = d['Team'].str.replace(r'^(QB|RB|WR|TE)\s+vs\s+', '', regex=True)
        d['Team'] = d['Team'].map(team_code)
        if d['Team'].isna().any():
            raise ValueError('Unrecognized team in CBS file.')
        for c in cols[2:]:
            d[c] = number(d[c])
    else:
        d = pd.read_csv(io.BytesIO(raw))
        d.columns = d.columns.str.strip()
        if kind.startswith('season_'):
            needed = {'Name', 'POS', 'GP', 'YDS'}
            needed |= {'REC', 'TGTS'} if kind == 'season_receiving' else {'ATT'}
        elif kind == 'defensive_coverage':
            needed = {'Team', 'Man Rate', 'Zone Rate'}
        elif kind == 'defensive_run_splits':
            needed = {'Team', 'Def Zone YAPG', 'Def Gap YAPG', 'Next Opponent'}
        elif kind.startswith(('wr_', 'te_')):
            needed = {'Player', 'Routes Run', 'Receptions', 'Rec. Yards'}
        elif kind.startswith('qb_'):
            needed = {'Player', 'Plays', 'Pass Yards', 'YPA'}
        else:
            needed = {'Player', 'Rushes', 'Rush Yards', 'Yards Per Carry'}
        missing = needed - set(d.columns)
        if missing:
            raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
        if kind.startswith('season_'):
            d = d[d['Name'].notna() & number(d['GP']).notna()].copy()
            codes = sorted(set(TEAM_NAMES) | set(ALIASES), key=len, reverse=True)
            suffix = '(' + '|'.join(codes) + ')$'
            d['Name'] = d['Name'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
            d['Team'] = d['Name'].str.extract(suffix, expand=False).map(team_code)
            d['Player'] = d['Name'].str.replace(suffix, '', regex=True).str.strip()
            d['Position'] = d['POS'].str.strip()
            for c in needed - {'Name', 'POS'}:
                d[c] = number(d[c])
            if d['Team'].isna().any():
                raise ValueError('Could not separate a player name from its team abbreviation.')
        elif 'Player' in d:
            volume = 'Routes Run' if kind.startswith(('wr_', 'te_')) else ('Plays' if kind.startswith('qb_') else 'Rushes')
            d = d[number(d[volume]).notna()].copy()
            d['Player'] = d['Player'].astype(str).str.replace(r'^\s*\d+\s*', '', regex=True).str.strip()
            for c in d.columns:
                if c != 'Player':
                    d[c] = number(d[c])
        else:
            d['Team'] = d['Team'].map(team_code)
            if d['Team'].isna().any():
                raise ValueError('Unrecognized team abbreviation.')
            if 'Next Opponent' in d:
                d['Opponent'] = d['Next Opponent'].map(team_code)
            for c in d.columns:
                if c not in {'Team', 'Next Opponent', 'Opponent'}:
                    d[c] = number(d[c])
    if 'Player' in d:
        d['Key'] = d['Player'].map(player_key)
        if d['Key'].duplicated().any():
            raise ValueError('Duplicate or ambiguous player names; resolve before importing.')
    elif d['Team'].duplicated().any():
        raise ValueError('Duplicate teams in table.')
    if d.empty:
        raise ValueError('No usable data rows.')
    return d.reset_index(drop=True)


def build_model(tables, cutoff, defense_weight=0.25):
    missing = REQUIRED - tables.keys()
    if missing:
        raise ValueError('Upload these datasets first: ' + ', '.join(sorted(missing)))
    issues, output = [], []
    cov = tables['defensive_coverage'].set_index('Team')
    runs = tables['defensive_run_splits'].set_index('Team')
    splits = {k: v.set_index('Key') for k, v in tables.items() if 'Key' in v}

    def issue(player, reason):
        issues.append({'Player / Team': player, 'Issue': reason})

    def get_split(kind, key):
        t = splits.get(kind)
        return t.loc[key] if t is not None and key in t.index else None

    def defense_factor(pos, opp, col):
        t = tables['cbs_allowed_' + pos.lower()]
        valid = t[col].dropna()
        match = t[t.Team == opp]
        if match.empty or valid.empty or valid.mean() <= 0:
            return None
        val = match.iloc[0][col]
        if pd.isna(val) or val < 0:
            return None
        return max(0.85, min(1.15, 1 + defense_weight * (val / valid.mean() - 1)))

    def coverage_factor(a, b, volume, production, opp, player):
        if a is None or b is None or opp not in cov.index:
            issue(player, 'Missing coverage split or opponent coverage; using season baseline.')
            return 1.0, 'Season baseline; coverage unavailable'
        c = cov.loc[opp]
        man, zone = c['Man Rate'], c['Zone Rate']
        if pd.isna(man) or pd.isna(zone) or min(man, zone) < 0 or man + zone <= 0 or man + zone > 101:
            issue(player, 'Invalid coverage rates; using season baseline.')
            return 1.0, 'Season baseline; invalid coverage'
        va, vb, ya, yb = a[volume], b[volume], a[production], b[production]
        if any(pd.isna(x) for x in (va, vb, ya, yb)) or min(va, vb) <= 0 or min(ya, yb) < 0:
            issue(player, 'Insufficient coverage sample; using season baseline.')
            return 1.0, 'Season baseline; insufficient split sample'
        # Shrink each split to the player's combined split rate.
        pooled = (ya + yb) / (va + vb)
        if pooled <= 0:
            return 1.0, 'No positive split production'
        prior = 50.0
        rate_a = (ya + prior * pooled) / (va + prior)
        rate_b = (yb + prior * pooled) / (vb + prior)
        expected = (man * rate_a + zone * rate_b) / (man + zone)
        reference = (va * rate_a + vb * rate_b) / (va + vb)
        factor = max(0.75, min(1.25, expected / reference))
        return factor, f'Coverage adjustment {factor:.3f}×; man {man:g}%, zone {zone:g}%'

    def add(row, stat, base, opp, factor, explanation, workload):
        pos = row['Position']
        col = {'Passing yards': 'Pass Yards', 'Rushing yards': 'Rush Yards',
               'Receiving yards': 'Rec Yards', 'Receptions': 'Receptions'}[stat]
        df = defense_factor(pos, opp, col)
        if df is None:
            issue(row['Player'], f'Missing CBS {stat} opponent data; projection omitted.')
            return
        output.append({'Player': row['Player'], 'Key': row['Key'], 'Position': pos,
                       'Team': row['Team'], 'Opponent': opp, 'Stat': stat,
                       'Season Average': round(base, 3), 'Projection': round(base * factor * df, 3),
                       'Workload / Game': round(workload, 3), 'Scheme Factor': round(factor, 4),
                       'Defense Factor': round(df, 4), 'Explanation': explanation})

    for kind in ['season_passing', 'season_rushing', 'season_receiving']:
        for _, r in tables[kind].iterrows():
            player, key, pos, team, gp = r['Player'], r['Key'], r['Position'], r['Team'], r['GP']
            wanted = {'season_passing': {'QB'}, 'season_rushing': {'RB', 'QB'},
                      'season_receiving': {'RB', 'WR', 'TE'}}[kind]
            if pos not in wanted:
                continue
            if not pd.notna(gp) or gp <= 0 or gp > cutoff:
                issue(player, f'GP={gp:g} is inconsistent with data through Week {cutoff}; omitted.')
                continue
            if team not in runs.index or pd.isna(runs.loc[team, 'Opponent']):
                issue(player, 'No opponent supplied; omitted (may be a bye).')
                continue
            opp = runs.loc[team, 'Opponent']
            # A next-opponent list can skip a bye: require reciprocal pairing.
            if opp not in runs.index or runs.loc[opp, 'Opponent'] != team:
                issue(player, f'Nonreciprocal opponent {team} → {opp}; verify this week, omitted.')
                continue
            if pd.isna(r['YDS']) or r['YDS'] < 0:
                issue(player, 'Missing/negative season yards; omitted.')
                continue
            if kind == 'season_passing':
                a, b = get_split('qb_man', key), get_split('qb_zone', key)
                f, why = coverage_factor(a, b, 'Plays', 'Pass Yards', opp, player)
                if a is not None and b is not None and a['Pass Yards'] + b['Pass Yards'] > r['YDS'] + 2:
                    issue(player, 'Coverage passing yards exceed season total; omitted.')
                    continue
                add(r, 'Passing yards', r['YDS'] / gp, opp, f, why, r['ATT'] / gp)
            elif kind == 'season_rushing':
                a, b = get_split('rb_gap', key), get_split('rb_zone', key)
                if pos == 'RB' and a is not None and b is not None:
                    if a['Rushes'] + b['Rushes'] > r['ATT'] + 1 or a['Rush Yards'] + b['Rush Yards'] > r['YDS'] + 2:
                        issue(player, 'Gap + zone rushing totals exceed season total; omitted.')
                        continue
                why = 'Season carries/game × season yards/carry; run-scheme defense shown as context'
                add(r, 'Rushing yards', r['YDS'] / gp, opp, 1.0, why, r['ATT'] / gp)
            else:
                a, b = get_split(pos.lower() + '_man', key), get_split(pos.lower() + '_zone', key)
                if pos in {'WR', 'TE'} and a is not None and b is not None:
                    if a['Rec. Yards'] + b['Rec. Yards'] > r['YDS'] + 2 or a['Receptions'] + b['Receptions'] > r['REC']:
                        issue(player, 'Man + zone receiving totals exceed season total; omitted.')
                        continue
                for stat, total, production in [('Receiving yards', 'YDS', 'Rec. Yards'), ('Receptions', 'REC', 'Receptions')]:
                    if pos in {'WR', 'TE'}:
                        f, why = coverage_factor(a, b, 'Routes Run', production, opp, player)
                    else:
                        f, why = 1.0, 'Season receiving baseline; RB coverage splits unavailable'
                    add(r, stat, r[total] / gp, opp, f, why, r['TGTS'] / gp)
    return pd.DataFrame(output), pd.DataFrame(issues).drop_duplicates()


def secret(name):
    try:
        return str(st.secrets.get(name, ''))
    except Exception:
        return ''


def is_admin():
    a, b = st.session_state.get('admin_password', ''), secret('ADMIN_PASSWORD')
    return bool(b) and hmac.compare_digest(a.encode(), b.encode())


def request_json(url, method='GET', payload=None, token=None, missing_ok=False):
    headers = {'User-Agent': 'NFL-Matchup-Analytics', 'Accept': 'application/json'}
    if token:
        headers.update({'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json'})
    if payload is not None:
        headers['Content-Type'] = 'application/json'
    req = Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                  headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as response:
            return json.load(response), dict(response.headers)
    except HTTPError as e:
        if missing_ok and e.code == 404:
            return None, {}
        raise RuntimeError(f'Service returned HTTP {e.code}. Check access, token/key, quota, and branch settings.') from None
    except (URLError, OSError, TimeoutError, ValueError):
        raise RuntimeError('Request failed. For a save, check GitHub before retrying.') from None


def github(path, content=None, create_only=False):
    token, repo = secret('GITHUB_TOKEN'), secret('GITHUB_REPO')
    branch = secret('GITHUB_BRANCH') or 'main'
    if not token or not repo:
        raise RuntimeError('Add GITHUB_TOKEN and GITHUB_REPO to Streamlit Secrets.')
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    existing, _ = request_json(url + '?' + urlencode({'ref': branch}), token=token, missing_ok=True)
    if content is None:
        if existing is None:
            return None
        if not isinstance(existing, dict) or 'content' not in existing:
            raise RuntimeError('Unexpected GitHub file response.')
        return base64.b64decode(existing['content'])
    if not is_admin():
        raise RuntimeError('Admin access required.')
    if create_only and existing is not None:
        raise RuntimeError('This snapshot already exists; it cannot be overwritten.')
    body = {'message': f'Update {path}', 'branch': branch,
            'content': base64.b64encode(content).decode()}
    if existing is not None:
        body['sha'] = existing['sha']
    request_json(url, method='PUT', payload=body, token=token)


@st.cache_data(ttl=3600, show_spinner=False)
def events():
    key = secret('ODDS_API_KEY')
    if not key:
        raise RuntimeError('Add ODDS_API_KEY to Streamlit Secrets.')
    data, _ = request_json(f'{API_BASE}/sports/{SPORT}/events?' + urlencode({'apiKey': key}))
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_props(event_id):
    key = secret('ODDS_API_KEY')
    if not key:
        raise RuntimeError('Add ODDS_API_KEY to Streamlit Secrets.')
    params = {'apiKey': key, 'bookmakers': 'draftkings', 'markets': ','.join(MARKETS.values()), 'oddsFormat': 'american'}
    data, headers = request_json(f'{API_BASE}/sports/{SPORT}/events/{event_id}/odds?' + urlencode(params))
    labels, rows = {v: k for k, v in MARKETS.items()}, []
    for book in data.get('bookmakers', []):
        if book.get('key') != 'draftkings':
            continue
        for market in book.get('markets', []):
            for o in market.get('outcomes', []):
                if o.get('name') not in {'Over', 'Under'} or o.get('point') is None:
                    continue
                rows.append({'Player': o.get('description'), 'Stat': labels.get(market['key'], market['key']),
                             'Side': o['name'], 'Line': o['point'], 'Odds': o.get('price'),
                             'Away': team_code(data.get('away_team')), 'Home': team_code(data.get('home_team')),
                             'Matchup': f"{data.get('away_team')} at {data.get('home_team')}",
                             'Kickoff (UTC)': data.get('commence_time'), 'Event ID': event_id,
                             'Updated (UTC)': market.get('last_update') or book.get('last_update'),
                             'Fetched': datetime.now(timezone.utc).isoformat(), 'Bookmaker': 'DraftKings'})
    return rows, headers.get('x-requests-remaining')


def prop_comparisons(projections, props):
    if projections.empty or props.empty:
        return pd.DataFrame()
    p = props.copy()
    p['Key'] = p['Player'].map(player_key)
    d = p.merge(projections.drop(columns=['Player']), on=['Key', 'Stat'], how='inner', validate='many_to_one')
    d = d[((d.Team == d.Away) & (d.Opponent == d.Home)) | ((d.Team == d.Home) & (d.Opponent == d.Away))].copy()
    d = d[pd.to_datetime(d['Kickoff (UTC)'], utc=True, errors='coerce') > pd.Timestamp.now(tz='UTC')].copy()
    d['Difference'] = d['Projection'] - d['Line']
    d['Difference %'] = d['Difference'].div(d['Line'].replace(0, float('nan'))) * 100
    d['Direction'] = d['Difference'].apply(lambda v: 'Over' if v > 0 else ('Under' if v < 0 else 'Equal'))
    return d


def main():
    st.set_page_config(page_title='NFL Matchup Analytics', page_icon='🏈', layout='wide')
    for key, value in {'tables': {}, 'source_bytes': {}, 'projections': pd.DataFrame(), 'issues': pd.DataFrame(),
                       'props': pd.DataFrame(), 'batch_meta': None, 'model_meta': None}.items():
        if key not in st.session_state:
            st.session_state[key] = value
    s = st.session_state
    st.title('🏈 NFL Matchup Analytics')
    st.caption('Experimental season-average forecasts adjusted for coverage and defensive matchup. No validated betting probabilities.')
    weekly, comparisons, results, methodology, admin = st.tabs([
        'Projections & Matchups', 'Prop Comparisons', 'Results Tracker', 'Methodology', 'Admin / Update Data'])

    with admin:
        st.text_input('Admin password', type='password', key='admin_password')
        if not secret('ADMIN_PASSWORD'):
            st.warning('Add ADMIN_PASSWORD to Streamlit Secrets.')
        elif s.admin_password and not is_admin():
            st.error('Incorrect password.')
        if is_admin():
            st.success('Admin access enabled.')
            st.subheader('Weekly data batch')
            c1, c2, c3 = st.columns(3)
            season = c1.number_input('Season', 2020, 2100, 2026)
            cutoff = c2.number_input('Stats through week', 1, 18, 5)
            week = c3.number_input('Project week', 1, 22, 6)
            label = st.selectbox('CSV category', list(DATASETS))
            kind = DATASETS[label]
            uploads = st.file_uploader('Upload CSVs (one or several)', type=['csv'], accept_multiple_files=True)
            name_map = {
                'te vs man': 'te_man', 'te vs zone': 'te_zone', 'wr vs man': 'wr_man', 'wr vs zone': 'wr_zone',
                'qb vs man': 'qb_man', 'qb vs zone': 'qb_zone', 'defensive passing': 'defensive_coverage',
                'gap schemes': 'rb_gap', 'zone schemes': 'rb_zone', 'full rb stats inc': 'rb_full',
                'defense yapg': 'defensive_run_splits', 'full season stats receiving': 'season_receiving',
                'rb full season': 'season_rushing', 'qb full season': 'season_passing',
                'allowed - qbs': 'cbs_allowed_qb', 'allowed - rbs': 'cbs_allowed_rb',
                'allowed - wrs': 'cbs_allowed_wr', 'allowed - tes': 'cbs_allowed_te'}
            pending = {}
            for f in uploads:
                guessed = next((v for k, v in name_map.items() if k in f.name.lower()), None)
                chosen = guessed or (kind if len(uploads) == 1 else None)
                if chosen is None:
                    st.error(f'{f.name}: cannot identify category. Upload it separately and select its category.')
                    continue
                if chosen in pending:
                    st.error(f'Duplicate category: {chosen}. Keep one file per category.')
                    continue
                try:
                    if f.size > 500_000:
                        raise ValueError('CSV must be under 500 KB.')
                    raw = f.getvalue()
                    table = parse_csv(raw, chosen)
                    pending[chosen] = (raw, table)
                    with st.expander(f'{f.name} → {chosen}: {len(table)} usable rows'):
                        st.dataframe(table.head(20), hide_index=True)
                except (ValueError, KeyError, pd.errors.ParserError) as e:
                    st.error(f'{f.name}: {e}')
            if pending and st.button('Accept previewed files into this batch', type='primary'):
                meta = {'season': int(season), 'cutoff': int(cutoff), 'week': int(week)}
                if s.batch_meta is not None and s.batch_meta != meta:
                    s.tables, s.source_bytes = {}, {}
                for k, (raw, table) in pending.items():
                    s.tables[k], s.source_bytes[k] = table, raw
                s.batch_meta = meta
                s.projections = pd.DataFrame()
                s.model_meta = None
                st.success('Accepted. Files remain in this session until you publish the batch.')
            st.write(f'**Accepted datasets:** {len(s.tables)} / {len(DATASETS)}')
            st.write(', '.join(sorted(s.tables)) or 'None yet')
            confirmed = st.checkbox('All accepted files share the displayed season and cutoff; opponents are for the projection week.')
            if st.button('Validate and calculate projections', disabled=not bool(s.tables)):
                try:
                    meta = {'season': int(season), 'cutoff': int(cutoff), 'week': int(week)}
                    if not confirmed or s.batch_meta != meta or week <= cutoff:
                        raise ValueError('Confirm the batch dates. Projection week must follow the cutoff; reaccept files if dates changed.')
                    p, issues = build_model(s.tables, int(cutoff))
                    s.projections, s.issues = p, issues
                    s.model_meta = {**meta, 'model_version': MODEL_VERSION, 'defense_weight': 0.25,
                                    'calculated_at': datetime.now(timezone.utc).isoformat()}
                    st.success(f'Calculated {len(p)} projections. Review Data checks before publishing.')
                except (ValueError, KeyError) as e:
                    st.error(str(e))
            if st.button('Publish this data batch and projections to GitHub', disabled=s.model_meta is None):
                try:
                    bundle = {'meta': s.model_meta, 'sources': {k: base64.b64encode(v).decode() for k, v in s.source_bytes.items()},
                              'projections': json.loads(s.projections.to_json(orient='records')),
                              'issues': json.loads(s.issues.to_json(orient='records'))}
                    github('data/current_batch.json', json.dumps(bundle, allow_nan=False).encode())
                    st.success('Published permanently. Visitors can load this batch below.')
                except RuntimeError as e:
                    st.error(str(e))
            st.divider()
            st.subheader('DraftKings imports')
            try:
                games = events()
                event_map = {e['id']: e for e in games}
                selected = st.multiselect('Choose games', list(event_map), format_func=lambda k: f"{event_map[k]['away_team']} at {event_map[k]['home_team']}")
                st.caption('Four markets per game. Identical requests cached for one hour; new requests use API credits. Only click when you want to import.')
                if st.button('Import selected games', disabled=not selected):
                    rows, failures = [], []
                    for eid in selected:
                        try:
                            items, remaining = fetch_props(eid)
                            rows.extend(items)
                            st.caption(f'Credits remaining when fetched: {remaining}')
                        except RuntimeError as e:
                            failures.append(str(e))
                    if rows:
                        incoming = pd.DataFrame(rows)
                        old = s.props
                        if not old.empty:
                            old = old[~old['Event ID'].isin(incoming['Event ID'])]
                        s.props = pd.concat([old, incoming], ignore_index=True)
                        st.success(f'Imported {len(rows)} over/under outcomes.')
                    else:
                        st.warning('No props returned. Existing imports retained.')
                    for error in failures:
                        st.error(error)
                if st.button('Publish imported props', disabled=s.props.empty):
                    github('data/current_props.json', s.props.to_json(orient='records').encode())
                    st.success('Props saved permanently.')
            except RuntimeError as e:
                st.error(str(e))

    with weekly:
        if st.button('Load latest published data and props'):
            try:
                raw = github('data/current_batch.json')
                if raw:
                    bundle = json.loads(raw)
                    sources = {k: base64.b64decode(v) for k, v in bundle['sources'].items()}
                    s.tables = {k: parse_csv(v, k) for k, v in sources.items()}
                    s.source_bytes, s.batch_meta = sources, {k: bundle['meta'][k] for k in ['season', 'cutoff', 'week']}
                    s.projections, s.issues = pd.DataFrame(bundle['projections']), pd.DataFrame(bundle['issues'])
                    s.model_meta = bundle['meta']
                else:
                    st.info('No published batch found.')
                raw_props = github('data/current_props.json')
                if raw_props:
                    s.props = pd.DataFrame(json.loads(raw_props))
            except (RuntimeError, ValueError, KeyError) as e:
                st.error(str(e))
        if s.model_meta:
            st.write(f"**{s.model_meta['season']} · Week {s.model_meta['week']} · Stats through Week {s.model_meta['cutoff']}**")
        if s.projections.empty:
            st.info('Load a published batch, or accept all CSVs in Admin and calculate projections.')
        else:
            p = s.projections.copy()
            pos = st.multiselect('Positions', ['QB', 'RB', 'WR', 'TE'], default=['QB', 'RB', 'WR', 'TE'])
            stat = st.selectbox('Projection stat', sorted(p.Stat.unique()))
            p = p[p.Position.isin(pos) & (p.Stat == stat)].copy()
            p['Matchup Change %'] = (p.Projection / p['Season Average'].replace(0, float('nan')) - 1) * 100
            sort = st.selectbox('Rank by', ['Largest matchup boost', 'Largest matchup reduction', 'Highest projection'])
            p = p.sort_values('Projection' if sort == 'Highest projection' else 'Matchup Change %', ascending=sort == 'Largest matchup reduction')
            st.dataframe(p.drop(columns='Key').round(2), hide_index=True, use_container_width=True)
            st.download_button('Download all projections', s.projections.drop(columns='Key').to_csv(index=False), 'projections.csv', 'text/csv')
        with st.expander('Data checks — review before using projections', expanded=not s.issues.empty):
            st.dataframe(s.issues, hide_index=True, use_container_width=True)
            st.caption('Omitted players are not assigned invented teams or workloads. Missing coverage uses a labeled season baseline.')
        with st.expander('Defensive run-scheme context'):
            if 'defensive_run_splits' in s.tables:
                st.dataframe(s.tables['defensive_run_splits'], hide_index=True)
            st.caption('Yards allowed per game combines run volume and efficiency. It is not used as yards per carry or as an efficiency multiplier.')

    with comparisons:
        if is_admin():
            with st.expander('Manual prop entry'):
                st.caption('Use a game from the API event list so the player, opponent, kickoff, and event identifier can be matched.')
                try:
                    manual_games = {e['id']: e for e in events()}
                    if manual_games:
                        with st.form('manual_prop'):
                            manual_event = st.selectbox('Game', list(manual_games), format_func=lambda k: f"{manual_games[k]['away_team']} at {manual_games[k]['home_team']}", key='manual_game')
                            manual_player = st.text_input('Player name')
                            manual_stat = st.selectbox('Stat', list(MARKETS))
                            manual_line = st.number_input('Line', min_value=0.0, value=0.5, step=0.5)
                            over_odds = st.number_input('Over odds (American)', value=-110, step=1)
                            under_odds = st.number_input('Under odds (American)', value=-110, step=1)
                            book = st.text_input('Sportsbook', value='Manual')
                            submitted = st.form_submit_button('Add manual prop')
                        if submitted:
                            event = manual_games[manual_event]
                            if not manual_player.strip() or any(abs(v) < 100 for v in (over_odds, under_odds)):
                                st.error('Enter a player and American odds of +100 or higher, or -100 or lower.')
                            elif pd.to_datetime(event['commence_time'], utc=True) <= pd.Timestamp.now(tz='UTC'):
                                st.error('Choose a game that has not started.')
                            else:
                                now = datetime.now(timezone.utc).isoformat()
                                common = {'Player': manual_player.strip(), 'Stat': manual_stat, 'Line': manual_line,
                                          'Away': team_code(event['away_team']), 'Home': team_code(event['home_team']),
                                          'Matchup': f"{event['away_team']} at {event['home_team']}",
                                          'Kickoff (UTC)': event['commence_time'], 'Event ID': manual_event,
                                          'Updated (UTC)': now, 'Fetched': now, 'Bookmaker': book.strip() or 'Manual'}
                                new_rows = pd.DataFrame([{**common, 'Side': 'Over', 'Odds': over_odds}, {**common, 'Side': 'Under', 'Odds': under_odds}])
                                s.props = pd.concat([s.props, new_rows], ignore_index=True)
                                s.props = s.props.drop_duplicates(['Event ID', 'Player', 'Stat', 'Line', 'Side', 'Bookmaker'], keep='last')
                                st.success('Manual line added. Use Publish imported props in Admin to save it for visitors.')
                except RuntimeError as e:
                    st.error(str(e))
        d = prop_comparisons(s.projections, s.props)
        st.caption('Projection minus line: positive suggests Over, negative suggests Under. Differences are not estimated win probabilities.')
        if not d.empty:
            stat = st.selectbox('Comparison stat', sorted(d.Stat.unique()))
            view = d[d.Stat == stat].copy()
            # One row per player/game/stat/line, with each side’s odds in columns.
            keys = ['Player', 'Matchup', 'Bookmaker', 'Stat', 'Line', 'Projection', 'Difference', 'Difference %', 'Direction']
            view = view.pivot_table(index=keys, columns='Side', values='Odds', aggfunc='first').reset_index()
            rank = st.selectbox('Sort differences', ['Largest absolute difference', 'Largest positive', 'Largest negative'])
            if rank == 'Largest absolute difference':
                view = view.iloc[view.Difference.abs().argsort()[::-1]]
            else:
                view = view.sort_values('Difference', ascending=rank == 'Largest negative')
            search = st.text_input('Search player')
            if search:
                view = view[view.Player.str.contains(search, case=False, regex=False)]
            st.dataframe(view.round(2), hide_index=True, use_container_width=True)
            st.download_button('Download comparisons', view.to_csv(index=False), 'comparisons.csv', 'text/csv')
        else:
            st.info('No matching upcoming props and projections yet. Imports must match both player and opponent; completed games are excluded.')
        with st.expander('Imported sportsbook lines'):
            if not s.props.empty:
                st.caption('Fetch times: ' + ', '.join(sorted(s.props.Fetched.unique())))
                st.dataframe(s.props[['Player', 'Matchup', 'Stat', 'Side', 'Line', 'Odds']], hide_index=True)
        if is_admin() and not d.empty and s.model_meta:
            if st.button('Save pregame prediction snapshot'):
                try:
                    # Recheck kickoff at the moment of saving.
                    future = d[pd.to_datetime(d['Kickoff (UTC)'], utc=True) > pd.Timestamp.now(tz='UTC')]
                    if future.empty:
                        raise RuntimeError('No pregame comparisons remain.')
                    now = datetime.now(timezone.utc)
                    path = f"data/history/{s.model_meta['season']}_w{s.model_meta['week']}_{now.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
                    snapshot = {'meta': s.model_meta, 'saved_at': now.isoformat(),
                                'source_hashes': {k: hashlib.sha256(v).hexdigest() for k, v in s.source_bytes.items()},
                                'comparisons': json.loads(future.to_json(orient='records'))}
                    github(path, json.dumps(snapshot, allow_nan=False).encode(), create_only=True)
                    st.success('Saved immutable pregame snapshot: ' + path)
                except RuntimeError as e:
                    st.error(str(e))

    with results:
        st.write('Load a saved snapshot, then optionally upload actual results to evaluate it.')
        path = st.text_input('Snapshot path from GitHub', placeholder='data/history/2026_w6_....json')
        if st.button('Load snapshot'):
            try:
                if not re.fullmatch(r'data/history/[A-Za-z0-9_-]+\.json', path):
                    raise ValueError('Use the data/history/...json path returned when saving.')
                raw = github(path)
                if raw is None:
                    raise ValueError('Snapshot not found.')
                s.loaded_snapshot = json.loads(raw)
            except (RuntimeError, ValueError) as e:
                st.error(str(e))
        if 'loaded_snapshot' in s:
            snapshot = s.loaded_snapshot
            st.caption('Saved at ' + snapshot['saved_at'])
            history = pd.DataFrame(snapshot['comparisons'])
            history = history.drop_duplicates(['Event ID', 'Key', 'Stat', 'Line', 'Bookmaker'])
            st.dataframe(history[['Player', 'Matchup', 'Stat', 'Projection', 'Line']], hide_index=True)
            st.caption('Actual-results CSV columns: Event ID, Player, Stat, Actual. Actual must be the completed-game total. Evaluation below is a preview.')
            actual_file = st.file_uploader('Actual results CSV', type='csv', key='actual_results')
            if actual_file:
                try:
                    actual = pd.read_csv(actual_file)
                    if not {'Event ID', 'Player', 'Stat', 'Actual'} <= set(actual):
                        raise ValueError('Use the four required columns shown above.')
                    actual['Key'] = actual.Player.map(player_key)
                    actual['Actual'] = number(actual.Actual)
                    if actual.Actual.isna().any() or actual.duplicated(['Event ID', 'Key', 'Stat']).any():
                        raise ValueError('Actual results need numeric values and unique player/event/stat rows.')
                    scored = history.merge(actual[['Event ID', 'Key', 'Stat', 'Actual']], on=['Event ID', 'Key', 'Stat'], validate='many_to_one')
                    scored = scored[pd.to_datetime(scored['Kickoff (UTC)'], utc=True) <= pd.Timestamp.now(tz='UTC')].copy()
                    if scored.empty:
                        raise ValueError('No matching started games found.')
                    scored['Absolute Error'] = (scored.Projection - scored.Actual).abs()
                    scored['Baseline Error'] = (scored['Season Average'] - scored.Actual).abs()
                    scored['Outcome'] = scored.apply(lambda r: 'Push' if r.Actual == r.Line else ('Correct direction' if (r.Projection - r.Line) * (r.Actual - r.Line) > 0 else ('No direction' if r.Projection == r.Line else 'Incorrect direction')), axis=1)
                    # Forecast accuracy is one observation per event/player/stat, regardless of available lines.
                    unique = scored.drop_duplicates(['Event ID', 'Key', 'Stat'])
                    st.dataframe(unique.groupby('Stat').agg(Count=('Actual', 'size'), Model_MAE=('Absolute Error', 'mean'), Baseline_MAE=('Baseline Error', 'mean')).round(2))
                    st.dataframe(scored[['Player', 'Stat', 'Projection', 'Line', 'Actual', 'Absolute Error', 'Outcome']], hide_index=True)
                    st.download_button('Download evaluated results', scored.to_csv(index=False), 'evaluated_results.csv', 'text/csv')
                    if is_admin() and st.button('Save evaluated results to GitHub'):
                        github('data/results/' + uuid.uuid4().hex + '.csv', scored.to_csv(index=False).encode(), create_only=True)
                        st.success('Results saved; original forecasts unchanged.')
                except (ValueError, KeyError, RuntimeError) as e:
                    st.error(str(e))

    with methodology:
        st.markdown('''### Experimental model v1
- **Baseline:** each player's season total divided by games played. Workloads initially use season averages, with no injury or role adjustments.
- **WR/TE coverage:** receptions per route and yards per route, separately for man and zone. Each split is shrunk toward the combined rate using 50 prior routes. Opponent man/zone frequencies are normalized over those two categories. The expected rate is divided by the player's historical split-weighted rate, then applied to the season per-game baseline. Coverage adjustment is capped at ±25%.
- **QB coverage:** the same approach uses passing yards per recorded split play (not per attempt). This is an approximation because split pass-attempt counts are unavailable. A 50-play prior is used.
- **RB rushing and receiving; QB rushing:** season per-game baselines. Gap/zone rushing and defensive yards allowed per game are displayed as context, not forced into an unsupported efficiency adjustment.
- **CBS adjustment:** multiply the opponent's allowed stat relative to the unweighted mean of the 32 defensive per-game values, shrink 75% toward neutral, and cap the resulting adjustment at ±15%. It is applied once per stat. The default 25% weight, priors, and caps are heuristic choices, not fitted or validated parameters.
- **Missing data:** unmatched names, invalid games played, inconsistent split totals, and nonreciprocal opponents are flagged. No fuzzy name matching. Missing coverage may use a labeled baseline; invalid season inputs are omitted.
- **Scope:** forecasts are experimental; opponent strength, game script, injuries, weather, and recent workload changes are not modeled. Man/zone may omit other coverages. Rushing scheme shares faced by a defense do not establish an offense's preferred scheme.
- **History:** timestamped pregame snapshots preserve lines, odds, source hashes, model settings, and projections. Compare accuracy with season averages, by stat. No claimed betting edge or return calculation.
''')
    st.divider()
    st.caption('NFL Matchup Analytics — Independent sports analytics project — Created by Owen Dunne')


if __name__ == '__main__':
    main()
