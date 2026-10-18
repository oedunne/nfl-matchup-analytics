import base64
import csv
import hashlib
import hmac
import io
import json
import math
import html
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

MODEL_VERSION = '2.0-approved-formulas'
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
            needed = {'Player', 'Routes Run', 'Receptions', 'Rec. Yards', 'Targets/Route Run'}
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


def build_model(tables, cutoff, defense_weight=0.5):
    missing = REQUIRED - tables.keys()
    if missing:
        raise ValueError('Upload these datasets first: ' + ', '.join(sorted(missing)))
    issues, output = [], []
    cov = tables['defensive_coverage'].set_index('Team')
    runs = tables['defensive_run_splits'].set_index('Team')
    splits = {k: v.set_index('Key') for k, v in tables.items() if 'Key' in v}

    def issue(player, reason):
        issues.append({'Player / Team': player, 'Issue': reason})

    def get(kind, key):
        t = splits.get(kind)
        return t.loc[key] if t is not None and key in t.index else None

    def valid(v):
        return pd.notna(v) and math.isfinite(float(v))

    def factor(allowed, median):
        if not valid(allowed) or not valid(median) or median <= 0 or allowed < 0:
            return None
        return 1 + defense_weight * (allowed / median - 1)

    def defensive(pos, opp, col):
        t = tables['cbs_allowed_' + pos.lower()]
        match = t[t.Team == opp]
        return None if match.empty else factor(match.iloc[0][col], t[col].median())

    def shares(a, b):
        if not valid(a) or not valid(b) or min(a, b) < 0 or a + b <= 0:
            return None
        return a / (a + b), b / (a + b)

    def mix(opp):
        if opp not in cov.index:
            return None
        c = cov.loc[opp]
        return shares(c['Man Rate'], c['Zone Rate'])

    # Select the same players for every stat, using their primary position stat.
    selected = set()
    for kind, positions in [('season_receiving', {'WR', 'TE'}), ('season_rushing', {'RB'}), ('season_passing', {'QB'})]:
        t = tables[kind].copy()
        t = t[t.Position.isin(positions) & (t.GP > 0) & (t.GP <= cutoff) & t.YDS.notna()].copy()
        t['Rank YPG'] = t.YDS / t.GP
        t = t.sort_values(['Rank YPG', 'Player'], ascending=[False, True])
        for (team, pos), group in t.groupby(['Team', 'Position']):
            selected.update(group.head(2 if pos == 'WR' else 1).Key)

    def add(row, stat, projection, neutral, workload, detail, df=1.0):
        if not all(valid(v) for v in (projection, neutral, workload)):
            issue(row.Player, f'Invalid {stat} inputs; omitted.')
            return
        # Rating measures matchup effect, not player quality or prop value.
        rating = matchup_rating(projection, neutral)
        output.append({'Player': row.Player, 'Key': row.Key, 'Position': row.Position,
                       'Team': row.Team, 'Opponent': runs.loc[row.Team, 'Opponent'], 'Stat': stat,
                       'Season Average': round(row[{'Passing yards': 'YDS', 'Rushing yards': 'YDS', 'Receiving yards': 'YDS', 'Receptions': 'REC'}[stat]] / row.GP, 4),
                       'Projection': round(projection, 4), 'Matchup Rating': rating,
                       'Neutral Projection': round(neutral, 4), 'Workload / Game': round(workload, 4),
                       'Defense Factor': df, 'Calculation': detail})

    for kind in ['season_passing', 'season_rushing', 'season_receiving']:
        for _, r in tables[kind].iterrows():
            player, key, pos, team, gp = r.Player, r.Key, r.Position, r.Team, r.GP
            wanted = {'season_passing': {'QB'}, 'season_rushing': {'RB', 'QB'}, 'season_receiving': {'RB', 'WR', 'TE'}}[kind]
            if pos not in wanted:
                continue
            if not valid(gp) or gp <= 0 or gp > cutoff:
                issue(player, f'GP={gp:g} conflicts with through-Week-{cutoff} label; omitted.')
                continue
            if key not in selected:
                continue
            if team not in runs.index or pd.isna(runs.loc[team, 'Opponent']):
                issue(player, 'No opponent supplied; omitted (may be a bye).')
                continue
            opp = runs.loc[team, 'Opponent']
            if opp not in runs.index or runs.loc[opp, 'Opponent'] != team:
                issue(player, f'Nonreciprocal opponent {team} → {opp}; omitted until verified.')
                continue
            if not valid(r.YDS):
                issue(player, 'Missing season yardage; omitted.')
                continue
            if kind == 'season_passing':
                if not valid(r.ATT) or r.ATT <= 0:
                    continue
                base_ypa, attempts = r.YDS / r.ATT, r.ATT / gp
                a, b, weights = get('qb_man', key), get('qb_zone', key), mix(opp)
                expected_ypa = base_ypa
                detail = 'Season attempts/game × season YPA (coverage unavailable)'
                if a is not None and b is not None and weights:
                    if a['Pass Yards'] + b['Pass Yards'] > r.YDS + 2:
                        issue(player, 'Man + zone passing yards exceed season total; omitted.')
                        continue
                    rates = []
                    for split in (a, b):
                        ypa, yards = split['YPA'], split['Pass Yards']
                        if not valid(ypa) or not valid(yards) or ypa <= 0 or yards <= 0:
                            rates.append(base_ypa)
                            issue(player, 'Unusable/zero split attempt estimate; season YPA used for that split.')
                        else:
                            n = yards / ypa
                            rates.append((yards + 20 * base_ypa) / (n + 20))
                    expected_ypa = sum(w * v for w, v in zip(weights, rates))
                    detail = 'Attempts/game × man/zone YPA; 20-attempt prior; split attempts estimated from yards/YPA'
                else:
                    issue(player, 'Missing QB coverage split; season YPA fallback.')
                df = defensive('QB', opp, 'Pass Yards')
                if df is not None:
                    add(r, 'Passing yards', attempts * expected_ypa * df, attempts * base_ypa, attempts, detail, df)
                else:
                    issue(player, 'Missing valid QB defensive passing benchmark; omitted.')
            elif kind == 'season_rushing':
                if not valid(r.ATT) or r.ATT <= 0:
                    continue
                carries, overall = r.ATT / gp, r.YDS / r.ATT
                if pos == 'QB':
                    df = defensive('QB', opp, 'Rush Yards')
                    if df is not None:
                        add(r, 'Rushing yards', r.YDS / gp * df, r.YDS / gp, carries, 'QB season rushing yards/game × QB-specific defensive multiplier', df)
                    continue
                a, b = get('rb_zone', key), get('rb_gap', key)
                if a is None or b is None or a.Rushes + b.Rushes <= 0:
                    df = defensive('RB', opp, 'Rush Yards')
                    issue(player, 'Missing RB scheme splits; labeled CBS season-baseline fallback.')
                    if df is not None:
                        add(r, 'Rushing yards', r.YDS / gp * df, r.YDS / gp, carries, 'Fallback: season rushing baseline × CBS RB defense', df)
                    continue
                if a.Rushes + b.Rushes > r.ATT + 1 or a['Rush Yards'] + b['Rush Yards'] > r.YDS + 2:
                    issue(player, 'Zone + gap rushing exceeds season totals; omitted.')
                    continue
                own = shares(a.Rushes, b.Rushes)
                if not {'Def Zone %', 'Def Gap %'} <= set(runs.columns):
                    issue(player, 'Missing defensive runs-faced shares; rushing omitted.')
                    continue
                faced = shares(runs.loc[opp, 'Def Zone %'], runs.loc[opp, 'Def Gap %'])
                if faced is None:
                    issue(player, 'Invalid defensive run shares; rushing omitted.')
                    continue
                expected = tuple(0.5 * x + 0.5 * y for x, y in zip(own, faced))
                rates = [(split['Rush Yards'] + 20 * overall) / (split.Rushes + 20) for split in (a, b)]
                dfs = [factor(runs.loc[opp, col], runs[col].median()) for col in ['Def Zone YAPG', 'Def Gap YAPG']]
                if any(v is None for v in dfs):
                    issue(player, 'Missing valid defensive scheme benchmark; rushing omitted.')
                    continue
                classified = min(1.0, (a.Rushes + b.Rushes) / r.ATT)
                ypc = classified * sum(w * rate * adj for w, rate, adj in zip(expected, rates, dfs)) + (1 - classified) * overall
                neutral_ypc = classified * sum(w * rate for w, rate in zip(own, rates)) + (1 - classified) * overall
                detail = f'Carries/game; zone share {expected[0]:.1%}; gap {expected[1]:.1%}; 20-carry priors; unclassified {1-classified:.1%} neutral; no extra CBS rushing multiplier'
                add(r, 'Rushing yards', carries * ypc, carries * neutral_ypc, carries, detail)
            else:
                if not all(valid(r[c]) for c in ['TGTS', 'REC']) or r.TGTS <= 0 or r.REC < 0 or r.REC > r.TGTS:
                    issue(player, 'Invalid receiving targets/receptions; omitted.')
                    continue
                base_targets, base_catch, base_ypt = r.TGTS / gp, r.REC / r.TGTS, r.YDS / r.TGTS
                target_count, catches, yards = base_targets, base_targets * base_catch, base_targets * base_ypt
                neutral_catches, neutral_yards = catches, yards
                detail = 'Season targets/game × season efficiency; RB has no receiving coverage splits'
                if pos in {'WR', 'TE'}:
                    a, b, weights = get(pos.lower() + '_man', key), get(pos.lower() + '_zone', key), mix(opp)
                    if a is not None and b is not None and weights:
                        if a['Rec. Yards'] + b['Rec. Yards'] > r.YDS + 2 or a.Receptions + b.Receptions > r.REC:
                            issue(player, 'Man + zone receiving exceeds season totals; omitted.')
                            continue
                        routes = [a['Routes Run'], b['Routes Run']]
                        targets = [a['Routes Run'] * a['Targets/Route Run'], b['Routes Run'] * b['Targets/Route Run']]
                        if not all(valid(v) and v >= 0 for v in routes + targets) or sum(routes) <= 0 or sum(targets) <= 0:
                            issue(player, 'Unusable split target estimates; using season receiving baseline.')
                        else:
                            historical = shares(*routes)
                            base_tpr = sum(targets) / sum(routes)
                            tprs = [(t + 50 * base_tpr) / (n + 50) for t, n in zip(targets, routes)]
                            catch_rates = [min(1.0, (sp.Receptions + 10 * base_catch) / (t + 10)) for sp, t in zip((a, b), targets)]
                            ypts = [(sp['Rec. Yards'] + 10 * base_ypt) / (t + 10) for sp, t in zip((a, b), targets)]
                            ref_tpr = sum(w * t for w, t in zip(historical, tprs))
                            effective_routes = base_targets / ref_tpr
                            target_count = effective_routes * sum(w * t for w, t in zip(weights, tprs))
                            # Weight production by coverage-specific targets, not just route shares.
                            catches = effective_routes * sum(w * t * c for w, t, c in zip(weights, tprs, catch_rates))
                            yards = effective_routes * sum(w * t * y for w, t, y in zip(weights, tprs, ypts))
                            neutral_catches = effective_routes * sum(w * t * c for w, t, c in zip(historical, tprs, catch_rates))
                            neutral_yards = effective_routes * sum(w * t * y for w, t, y in zip(historical, tprs, ypts))
                            detail = 'Coverage-adjusted targets × target-weighted efficiency; 50-route/10-target priors; split targets estimated from rounded TPRR'
                    else:
                        issue(player, 'Missing receiving coverage split; season receiving baseline used.')
                for stat, production, neutral, col in [('Receiving yards', yards, neutral_yards, 'Rec Yards'), ('Receptions', catches, neutral_catches, 'Receptions')]:
                    df = defensive(pos, opp, col)
                    if df is None:
                        issue(player, f'Missing valid {pos} defensive {stat} benchmark; omitted.')
                    else:
                        add(r, stat, production * df, neutral, target_count, detail, df)
    return pd.DataFrame(output), pd.DataFrame(issues, columns=['Player / Team', 'Issue']).drop_duplicates()


def matchup_rating(projection, neutral):
    if not math.isfinite(neutral) or neutral <= 0 or not math.isfinite(projection):
        return None
    # Smooth symmetric scale: neutral=5.5, stronger increases approach 10,
    # stronger decreases approach 1. This is not a hit probability.
    ratio = max(projection / neutral, 0.000001)
    return round(5.5 + 4.5 * math.tanh(math.log(ratio) / 0.35), 1)


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


def github(path, content=None, create_only=False, expected_hash=None):
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
    if expected_hash is not None:
        current_hash = hashlib.sha256(base64.b64decode(existing['content'])).hexdigest() if existing else 'missing'
        if current_hash != expected_hash:
            raise RuntimeError('Saved picks changed. Refresh and try again.')
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


PROJECTION_COLUMNS = ['Player', 'Position', 'Team', 'Opponent', 'Stat', 'Season Average',
                      'Our Projection', 'DraftKings O/U', 'Matchup / 10', 'Over / Under', 'By How Much']


def joined_projections(projections, props):
    if projections.empty:
        return pd.DataFrame(columns=PROJECTION_COLUMNS)
    out = projections.rename(columns={'Projection': 'Our Projection', 'Matchup Rating': 'Matchup / 10'}).copy()
    if 'Matchup / 10' not in out:
        out['Matchup / 10'] = None
    lines = prop_comparisons(projections, props)
    if not lines.empty:
        lines = lines[lines.Bookmaker.str.casefold() == 'draftkings'].copy()
    if not lines.empty:
        lines = lines.sort_values(['Fetched', 'Updated (UTC)'], ascending=False)
        lines = lines.drop_duplicates(['Key', 'Stat', 'Team', 'Opponent'])
        out = out.merge(lines[['Key', 'Stat', 'Team', 'Opponent', 'Line']], on=['Key', 'Stat', 'Team', 'Opponent'], how='left', validate='one_to_one')
        out = out.rename(columns={'Line': 'DraftKings O/U'})
    else:
        out['DraftKings O/U'] = float('nan')
    delta = out['Our Projection'] - out['DraftKings O/U']
    out['Over / Under'] = delta.apply(lambda v: 'No line' if pd.isna(v) else ('Over' if v > 0 else ('Under' if v < 0 else 'Equal')))
    out['By How Much'] = delta.abs()
    return out[PROJECTION_COLUMNS]


def styled_table(d, key=None):
    if d.empty:
        st.info('No rows match these filters.')
        return
    def direction_color(v):
        return 'background-color: #d1fae5; color: #065f46; font-weight: 700' if v == 'Over' else ('background-color: #ffe4e6; color: #9f1239; font-weight: 700' if v == 'Under' else '')
    style = d.style.format(precision=1, na_rep='—')
    if 'Over / Under' in d:
        style = style.map(direction_color, subset=['Over / Under'])
    if 'Result' in d:
        colors = {'Won': '#166534', 'Lost': '#be123c', 'Pending': '#a16207', 'Push': '#475569', 'Void': '#475569'}
        style = style.map(lambda v: f'color: {colors.get(v,"inherit")}; font-weight:700', subset=['Result'])
    config = {}
    if 'Matchup / 10' in d:
        config['Matchup / 10'] = st.column_config.ProgressColumn('Matchup / 10', min_value=1, max_value=10, format='%.1f')
    st.dataframe(style, hide_index=True, use_container_width=True, column_config=config, key=key)


def select_pick_ui(projections, props):
    if not is_admin():
        st.caption('Enter your Admin password to save a pick.')
        return
    lines = prop_comparisons(projections, props)
    if lines.empty:
        return
    lines = lines.sort_values('Fetched', ascending=False).drop_duplicates(['Event ID', 'Key', 'Stat', 'Line', 'Bookmaker', 'Side'])
    st.subheader('Add a prop to My Picks')
    candidates = lines.drop_duplicates(['Event ID', 'Key', 'Stat', 'Line', 'Bookmaker']).reset_index(drop=True)
    index = st.selectbox('Choose a prop', list(candidates.index), format_func=lambda n: f"{candidates.loc[n,'Player']} · {candidates.loc[n,'Stat']} · {candidates.loc[n,'Line']:g} · {candidates.loc[n,'Bookmaker']}")
    row = candidates.loc[index]
    st.caption(f"Our projection: {row.Projection:.1f} · {row.Matchup} · kickoff {row['Kickoff (UTC)']}")
    side = st.radio('Your prediction', ['Over', 'Under'], index=0 if row.Projection >= row.Line else 1, horizontal=True)
    if st.button('Save selected pick', type='primary'):
        try:
            if pd.to_datetime(row['Kickoff (UTC)'], utc=True) <= pd.Timestamp.now(tz='UTC'):
                raise ValueError('Game has started; choose an upcoming prop.')
            match = lines[(lines['Event ID'] == row['Event ID']) & (lines.Key == row.Key) & (lines.Stat == row.Stat) & (lines.Line == row.Line) & (lines.Bookmaker == row.Bookmaker) & (lines.Side == side)]
            odds = None if match.empty else float(match.iloc[0].Odds)
            identifier = hashlib.sha256(json.dumps([row['Event ID'], row.Key, row.Stat, float(row.Line), row.Bookmaker, side]).encode()).hexdigest()[:24]
            data = github('data/my_picks.json')
            picks = json.loads(data) if data else []
            if any(p['ID'] == identifier for p in picks):
                raise ValueError('That exact pick is already saved.')
            record = {'ID': identifier, 'Player': row.Player, 'Position': row.Position, 'Team': row.Team,
                      'Opponent': row.Opponent, 'Stat': row.Stat, 'Line': float(row.Line), 'Your Pick': side,
                      'Projection When Picked': float(row.Projection), 'Odds': odds, 'Bookmaker': row.Bookmaker,
                      'Event ID': row['Event ID'], 'Kickoff (UTC)': row['Kickoff (UTC)'],
                      'Saved At': datetime.now(timezone.utc).isoformat(), 'Result': 'Pending',
                      'Actual': None, 'Model Version': MODEL_VERSION, 'Result History': []}
            picks.append(record)
            github('data/my_picks.json', json.dumps(picks, allow_nan=False).encode(), expected_hash=hashlib.sha256(data).hexdigest() if data else 'missing')
            st.session_state.my_picks = picks
            st.success('Pick saved permanently. Open My Picks to track it.')
        except (RuntimeError, ValueError) as e:
            st.error(str(e))


def result_from_actual(side, line, actual):
    if actual == line:
        return 'Push'
    return 'Won' if (actual > line) == (side == 'Over') else 'Lost'


def picks_page():
    st.subheader('Your picks. Your record.')
    st.caption('Saved lines and projections stay frozen. Result updates are manual; no automatic grading or payout tracking.')
    if st.button('Load / refresh saved picks'):
        try:
            raw = github('data/my_picks.json')
            st.session_state.my_picks = json.loads(raw) if raw else []
        except (RuntimeError, ValueError) as e:
            st.error(str(e))
    if 'my_picks' not in st.session_state:
        st.info('Load your saved picks above to display your record.')
        return
    picks = st.session_state.my_picks
    won = sum(p['Result'] == 'Won' for p in picks)
    lost = sum(p['Result'] == 'Lost' for p in picks)
    pending = sum(p['Result'] == 'Pending' for p in picks)
    pushed = sum(p['Result'] == 'Push' for p in picks)
    void = sum(p['Result'] == 'Void' for p in picks)
    cols = st.columns(5)
    cols[0].metric('RECORD', f'{won} – {lost}')
    cols[1].metric('HIT RATE', f'{100*won/(won+lost):.1f}%' if won+lost else '—')
    cols[2].metric('PENDING', pending)
    cols[3].metric('PUSHES', pushed)
    cols[4].metric('VOID', void)
    st.caption('Hit rate = wins ÷ (wins + losses). Pushes, voids, and pending picks are excluded. Record is self-reported.')
    if not picks:
        st.info('Save a prop from Prop Comparisons, or load your previously saved picks.')
        return
    d = pd.DataFrame(picks)
    status = st.selectbox('Show results', ['All', 'Pending', 'Won', 'Lost', 'Push', 'Void'])
    visible = d if status == 'All' else d[d.Result == status]
    columns = ['Player', 'Team', 'Opponent', 'Stat', 'Bookmaker', 'Line', 'Your Pick', 'Projection When Picked', 'Result', 'Actual', 'Saved At']
    styled_table(visible[columns])
    st.download_button('Download my picks', d.drop(columns=['Result History'], errors='ignore').to_csv(index=False), 'my_picks.csv', 'text/csv')
    if is_admin():
        with st.expander('Grade or correct a saved pick', expanded=True):
            by_id = {p['ID']: p for p in picks}
            pid = st.selectbox('Pick to update', list(by_id), format_func=lambda k: f"{by_id[k]['Player']} · {by_id[k]['Stat']} {by_id[k]['Your Pick']} {by_id[k]['Line']} · {by_id[k]['Result']}")
            pick = by_id[pid]
            mode = st.radio('How to grade', ['Enter actual stat', 'Mark result manually'], horizontal=True)
            actual = None
            if mode == 'Enter actual stat':
                actual = st.number_input('Actual completed-game stat', value=float(pick['Actual']) if pick.get('Actual') is not None else 0.0, step=1.0)
                grade = result_from_actual(pick['Your Pick'], pick['Line'], actual)
                st.write('Result: **' + grade + '**')
            else:
                grade = st.selectbox('Result', ['Won', 'Lost', 'Push', 'Void', 'Pending'])
            confirmed = st.checkbox('The game is final (or the pick is void).')
            if st.button('Save pick result'):
                try:
                    if grade != 'Pending' and not confirmed:
                        raise ValueError('Confirm the game is final or the pick is void.')
                    if grade in {'Won', 'Lost', 'Push'} and pd.to_datetime(pick['Kickoff (UTC)'], utc=True) > pd.Timestamp.now(tz='UTC'):
                        raise ValueError('This game has not started. Leave the pick pending.')
                    raw = github('data/my_picks.json')
                    latest = json.loads(raw) if raw else []
                    found = next((p for p in latest if p['ID'] == pid), None)
                    if found is None:
                        raise ValueError('Pick not found; refresh the list.')
                    found.setdefault('Result History', []).append({'previous_result': found['Result'], 'previous_actual': found.get('Actual'), 'new_result': grade, 'new_actual': actual, 'at': datetime.now(timezone.utc).isoformat()})
                    found['Result'], found['Actual'] = grade, actual
                    github('data/my_picks.json', json.dumps(latest, allow_nan=False).encode(), expected_hash=hashlib.sha256(raw).hexdigest())
                    st.session_state.my_picks = latest
                    st.rerun()
                except (RuntimeError, ValueError) as e:
                    st.error(str(e))


def main():
    st.set_page_config(page_title='NFL Matchup Analytics', page_icon='🏈', layout='wide')
    for key, value in {'tables': {}, 'source_bytes': {}, 'projections': pd.DataFrame(), 'issues': pd.DataFrame(),
                       'props': pd.DataFrame(), 'batch_meta': None, 'model_meta': None}.items():
        if key not in st.session_state:
            st.session_state[key] = value
    s = st.session_state
    st.markdown('''<style>
    .block-container {padding-top:2rem; max-width:1600px;}
    [data-testid="stMetric"] {background:linear-gradient(120deg,#10243b,#184e64);padding:18px;border-radius:14px;border-bottom:3px solid #34d399;color:#f1f5f9;}
    [data-testid="stMetric"] * {color:#f1f5f9 !important;}
    .stButton>button[kind="primary"] {background:#087f70;border-color:#087f70;border-radius:9px;}
    button[data-baseweb="tab"] {font-weight:650;}
    </style>''', unsafe_allow_html=True)
    st.title('🏈 NFL Matchup Analytics')
    st.caption('Player strengths × defensive matchups · Your projections, your picks, your record')
    weekly, comparisons, picks_tab, results, methodology, admin = st.tabs([
        'Projections & Matchups', 'Prop Comparisons', 'My Picks', 'Model Results', 'Methodology', 'Admin / Update Data'])

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
                    s.model_meta = {**meta, 'model_version': MODEL_VERSION, 'defense_weight': 0.5,
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
                    s.projections, s.issues = build_model(s.tables, int(bundle['meta']['cutoff']))
                    s.model_meta = {**bundle['meta'], 'model_version': MODEL_VERSION, 'defense_weight': 0.5}
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
            st.info('Load your published data, or accept the CSVs in Admin and calculate projections.')
        else:
            p = joined_projections(s.projections, s.props)
            cols = st.columns(3)
            positions = cols[0].multiselect('Positions', ['QB', 'RB', 'WR', 'TE'], default=['QB', 'RB', 'WR', 'TE'])
            stat = cols[1].selectbox('Projection stat', ['All'] + sorted(p.Stat.unique()))
            team = cols[2].selectbox('Team', ['All'] + sorted(p.Team.unique()))
            p = p[p.Position.isin(positions)]
            if stat != 'All':
                p = p[p.Stat == stat]
            if team != 'All':
                p = p[p.Team == team]
                logo_code = {'WAS': 'wsh'}.get(team, team.lower())
                st.image(f'https://a.espncdn.com/i/teamlogos/nfl/500/{logo_code}.png', width=64)
            search = st.text_input('Find a player', key='projection_search')
            if search:
                p = p[p.Player.str.contains(search, case=False, regex=False)]
            order = st.selectbox('Sort table', ['Best matchup', 'Worst matchup', 'Largest prop difference', 'Player name'])
            sortcol = {'Best matchup': 'Matchup / 10', 'Worst matchup': 'Matchup / 10', 'Largest prop difference': 'By How Much', 'Player name': 'Player'}[order]
            p = p.sort_values(sortcol, ascending=order in {'Worst matchup', 'Player name'}, na_position='last')
            st.caption('Up to 2 WRs and 1 QB, TE, RB per team, chosen by primary season yards/game. Props are the latest imported DraftKings lines for matching upcoming games. — means no line imported.')
            styled_table(p)
            st.caption('Matchup rating: 5.5 is neutral; closer to 10 is more favorable for production, closer to 1 is less favorable. It is not a win probability or a rating of the prop price.')
            st.download_button('Download projection table', p.to_csv(index=False), 'projections.csv', 'text/csv')
        with st.expander('Data checks — review before using projections', expanded=not s.issues.empty):
            st.dataframe(s.issues, hide_index=True, use_container_width=True)
            st.caption('Omitted players are not assigned invented teams or workloads. Missing coverage uses a labeled season baseline.')
        with st.expander('Defensive run-scheme context'):
            if 'defensive_run_splits' in s.tables:
                st.dataframe(s.tables['defensive_run_splits'], hide_index=True)
            st.caption('Defensive scheme yards/game feeds the approved median-based adjustments. These values reflect both volume and efficiency.')

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
            styled_table(view.rename(columns={'Direction': 'Over / Under'}).round(2))
            st.download_button('Download comparisons', view.to_csv(index=False), 'comparisons.csv', 'text/csv')
        else:
            st.info('No matching upcoming props and projections yet. Imports must match both player and opponent; completed games are excluded.')
        with st.expander('Imported sportsbook lines'):
            if not s.props.empty:
                st.caption('Fetch times: ' + ', '.join(sorted(s.props.Fetched.unique())))
                st.dataframe(s.props[['Player', 'Matchup', 'Stat', 'Side', 'Line', 'Odds']], hide_index=True)
        select_pick_ui(s.projections, s.props)
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

    with picks_tab:
        picks_page()

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
        st.markdown("""### Model v2 · agreed calculation rules
- **Player selection:** up to two WRs and one QB, TE, RB per team, ranked by primary season yards/game. QB selection carries over to QB rushing; RB selection carries over to RB receiving. This is a production filter, not a verified depth chart.
- **WR/TE targeting:** season targets/game adjusted by opponent-weighted targets/route relative to historical route-weighted targets/route. Each coverage target rate uses a 50-route prior toward the player's combined man/zone target rate.
- **WR/TE production:** each coverage's catch rate and yards/target use a 10-target prior toward season efficiency. Coverage contributions are weighted by expected targets, so targeting is counted once. Split targets are estimated from routes × rounded targets/route; missing splits use labeled season baselines.
- **Defense:** multiplier = 1 + 0.5 × (opponent position-specific stat allowed/game ÷ league median allowed/game − 1). Separate medians for every stat and position; no extra caps. Zero/missing medians are flagged, not divided by.
- **RB rushing:** expected classified zone/gap mix = 50% player's own carry shares + 50% defense's runs-faced shares, normalizing each pair. RB scheme YPC uses a 20-carry prior toward season YPC. Each scheme gets its own median-based defensive yards/game adjustment. Multiply by carries/game. Preserve unclassified carries at overall efficiency with a neutral adjustment. Do not add a second CBS rushing multiplier. Defensive yards/game is used as requested; it combines volume and efficiency.
- **RB receiving:** targets/game × season catch rate or yards/target × matching RB defensive multiplier. No invented RB man/zone receiving splits.
- **QB passing:** attempts/game × opponent-weighted man/zone YPA × QB passing defense multiplier. Smooth with a 20-attempt prior toward season YPA. Split attempts are estimates from passing yards/YPA; unusable estimates fall back to season YPA.
- **QB rushing:** season rushing yards/game × QB-specific defensive rushing multiplier.
- **Matchup rating:** compare the final forecast with the same smoothed model under the player's own scheme mix and a neutral defense. Score = 5.5 + 4.5 × tanh(log(forecast/neutral)/0.35). Neutral is 5.5; 10 is the strongest end and 1 the weakest end of this heuristic scale. It measures production matchup, not player ability or the probability of beating a line. Undefined neutral baselines show no score.
- **My Picks:** save your chosen side before kickoff, freezing the line, projection, odds, and timestamp. Manually mark Won/Lost/Push/Void or enter the actual completed-game stat. Record = wins/losses; hit rate excludes pushes, voids, and pending picks. Result corrections retain an audit trail. Admin password is required to write.
- **Limits:** no injury, weather, role-change, or game-script model. Priors, 50% strengths, and rating scale are experimental settings, not fitted performance claims. Data checks flag cutoff conflicts and ambiguous names. Model Results remains available for evaluating pregame snapshots separately from your personal picks.
""")
    st.divider()
    st.caption('NFL Matchup Analytics — Independent sports analytics project — Created by Owen Dunne')


if __name__ == '__main__':
    main()
