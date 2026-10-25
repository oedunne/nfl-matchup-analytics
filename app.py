import base64
import csv
import hashlib
import hmac
import io
import json
import math
import html
import gzip
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

MODEL_VERSION = '7.0-workload-pressure-box'
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
ALIASES = {'AZ': 'ARI', 'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU', 'JAC': 'JAX',
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
    if kind in EXTRA_KINDS:
        return parse_extra(raw, kind)
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


def build_model(tables, cutoff, defense_weight=0.632):
    missing = REQUIRED - tables.keys()
    if missing:
        raise ValueError('Upload these datasets first: ' + ', '.join(sorted(missing)))
    issues, output = [], []
    workload_map, workload_notes = projected_workloads(tables, cutoff)
    issues.extend(workload_notes)
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
            group = group[group.apply(lambda r: availability(r.Key, r.Team) > 0, axis=1)]
            selected.update(group.head(2 if pos == 'WR' else 1).Key)

    def add(row, stat, projection, neutral, workload, detail, df=1.0, pre_defense=None, scheme_known=False):
        if not all(valid(v) for v in (projection, neutral, workload)):
            issue(row.Player, f'Invalid {stat} inputs; omitted.')
            return
        # Rating measures matchup effect, not player quality or prop value.
        rating = matchup_rating(projection, neutral)
        pre_defense = projection / df if pre_defense is None else pre_defense
        defense_rating = matchup_rating(projection, pre_defense)
        scheme_rating = matchup_rating(pre_defense, neutral) if scheme_known else None
        output.append({'Player': row.Player, 'Key': row.Key, 'Position': row.Position,
                       'Team': row.Team, 'Opponent': runs.loc[row.Team, 'Opponent'], 'Stat': stat,
                       'Season Average': round(row[{'Passing yards': 'YDS', 'Rushing yards': 'YDS', 'Receiving yards': 'YDS', 'Receptions': 'REC'}[stat]] / row.GP, 4),
                       'Projection': round(projection, 4), 'Matchup Rating': rating,
                       'Neutral Projection': round(neutral, 4), 'Workload / Game': round(workload, 4),
                       'Defense Factor': df, 'Defense Rating': defense_rating, 'Scheme Rating': scheme_rating,
                       'Signals': signal_label(defense_rating, scheme_rating), 'Calculation': detail})

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
                base_ypa, attempts = r.YDS / r.ATT, workload_map.get((key, team, 'attempts'), r.ATT / gp)
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
                    add(r, 'Passing yards', attempts * expected_ypa * df, attempts * base_ypa, attempts, detail, df, scheme_known=a is not None and b is not None and weights is not None)
                else:
                    issue(player, 'Missing valid QB defensive passing benchmark; omitted.')
            elif kind == 'season_rushing':
                if not valid(r.ATT) or r.ATT <= 0:
                    continue
                carries, overall = workload_map.get((key, team, 'carries'), r.ATT / gp), r.YDS / r.ATT
                if pos == 'QB':
                    df = defensive('QB', opp, 'Rush Yards')
                    if df is not None:
                        add(r, 'Rushing yards', carries * overall * df, carries * overall, carries, 'QB season rushing yards/game × QB-specific defensive multiplier', df)
                    continue
                a, b = get('rb_zone', key), get('rb_gap', key)
                if a is None or b is None or a.Rushes + b.Rushes <= 0:
                    df = defensive('RB', opp, 'Rush Yards')
                    issue(player, 'Missing RB scheme splits; labeled CBS season-baseline fallback.')
                    if df is not None:
                        add(r, 'Rushing yards', carries * overall * df, carries * overall, carries, 'Fallback: season rushing baseline × CBS RB defense', df)
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
                expected = tuple(0.75 * x + 0.25 * y for x, y in zip(own, faced))
                rates = [(split['Rush Yards'] + 20 * overall) / (split.Rushes + 20) for split in (a, b)]
                dfs = [factor(runs.loc[opp, col], runs[col].median()) for col in ['Def Zone YAPG', 'Def Gap YAPG']]
                if any(v is None for v in dfs):
                    issue(player, 'Missing valid defensive scheme benchmark; rushing omitted.')
                    continue
                classified = min(1.0, (a.Rushes + b.Rushes) / r.ATT)
                ypc = classified * sum(w * rate * adj for w, rate, adj in zip(expected, rates, dfs)) + (1 - classified) * overall
                neutral_ypc = classified * sum(w * rate for w, rate in zip(own, rates)) + (1 - classified) * overall
                pre_defense_ypc = classified * sum(w * rate for w, rate in zip(expected, rates)) + (1 - classified) * overall
                detail = f'Carries/game; zone share {expected[0]:.1%}; gap {expected[1]:.1%}; 20-carry priors; unclassified {1-classified:.1%} neutral; no extra CBS rushing multiplier'
                add(r, 'Rushing yards', carries * ypc, carries * neutral_ypc, carries, detail, pre_defense=carries * pre_defense_ypc, scheme_known=True)
            else:
                if not all(valid(r[c]) for c in ['TGTS', 'REC']) or r.TGTS <= 0 or r.REC < 0 or r.REC > r.TGTS:
                    issue(player, 'Invalid receiving targets/receptions; omitted.')
                    continue
                base_targets, base_catch, base_ypt = workload_map.get((key, team, 'targets'), r.TGTS / gp), r.REC / r.TGTS, r.YDS / r.TGTS
                target_count, catches, yards = base_targets, base_targets * base_catch, base_targets * base_ypt
                neutral_catches, neutral_yards = catches, yards
                has_coverage = False
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
                            has_coverage = True
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
                    if pos == 'RB' and stat == 'Receiving yards':
                        volume_factor = defensive('RB', opp, 'Receptions')
                        allowed = tables['cbs_allowed_rb'].copy()
                        allowed['YPCatch'] = allowed['Rec Yards'] / allowed['Receptions'].replace(0, float('nan'))
                        opponent_row = allowed[allowed.Team == opp]
                        efficiency_factor = None if opponent_row.empty else factor(opponent_row.iloc[0]['YPCatch'], allowed.YPCatch.median())
                        df = None if volume_factor is None or efficiency_factor is None else volume_factor * efficiency_factor
                        detail = 'Projected targets × catch rate × yards/catch; defense catch volume and yards/catch adjusted separately'
                    if df is None:
                        issue(player, f'Missing valid {pos} defensive {stat} benchmark; omitted.')
                    else:
                        add(r, stat, production * df, neutral, target_count, detail, df, scheme_known=has_coverage)
    result = apply_matchup_layers(pd.DataFrame(output), tables, issues)
    return result, pd.DataFrame(issues, columns=['Player / Team', 'Issue']).drop_duplicates()


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
    projections=injury_filtered(projections)
    if projections.empty or props.empty:
        return pd.DataFrame()
    p = props.copy()
    p['Key'] = p['Player'].map(player_key)
    d = p.merge(projections.drop(columns=['Player']), on=['Key', 'Stat'], how='inner', validate='many_to_one')
    d = d[((d.Team == d.Away) & (d.Opponent == d.Home)) | ((d.Team == d.Home) & (d.Opponent == d.Away))].copy()
    d = d[pd.to_datetime(d['Kickoff (UTC)'], utc=True, errors='coerce') > pd.Timestamp.now(tz='UTC')].copy()
    d['Difference'] = d['Projection'] - d['Line']
    d['Difference %'] = d['Difference'].div(d['Line'].replace(0, float('nan'))) * 100
    d['Direction'] = d.apply(lambda r: pick_direction(r.Stat, r.Projection, r.Line), axis=1)
    return d


PROJECTION_COLUMNS = ['Player', 'Position', 'Team', 'Opponent', 'Stat', 'Season Average',
                      'Our Projection', 'DraftKings O/U', 'Matchup / 10', 'Defense / 10', 'Coverage / Scheme / 10', 'Signals', 'Over / Under', 'By How Much']


def joined_projections(projections, props):
    projections=injury_filtered(projections)
    if projections.empty:
        return pd.DataFrame(columns=PROJECTION_COLUMNS)
    out = projections.rename(columns={'Projection': 'Our Projection', 'Matchup Rating': 'Matchup / 10', 'Defense Rating': 'Defense / 10', 'Scheme Rating': 'Coverage / Scheme / 10'}).copy()
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
    for column in ['Defense / 10', 'Coverage / Scheme / 10', 'Signals']:
        if column not in out:
            out[column] = None
    delta = out['Our Projection'] - out['DraftKings O/U']
    out['Over / Under'] = out.apply(lambda r: pick_direction(r.Stat, r['Our Projection'], r['DraftKings O/U']), axis=1)
    out['By How Much'] = delta.abs()
    return out[PROJECTION_COLUMNS]


def styled_table(d, key=None):
    if d.empty:
        st.info('No rows match these filters.')
        return
    def direction_color(v):
        return 'background-color: #d1fae5; color: #065f46; font-weight: 700' if v == 'Over' else ('background-color: #ffe4e6; color: #9f1239; font-weight: 700' if v == 'Under' else '')
    d=d.copy()
    if 'Team' in d:
        d.insert(0,'Team badge',d.Team.map(logo_url))
    style = d.style.format(precision=1, na_rep='—')
    if 'Over / Under' in d:
        style = style.map(direction_color, subset=['Over / Under'])
    if 'Result' in d:
        colors = {'Won': '#166534', 'Lost': '#be123c', 'Pending': '#a16207', 'Push': '#475569', 'Void': '#475569', 'Needs review': '#a16207'}
        style = style.map(lambda v: f'color: {colors.get(v,"inherit")}; font-weight:700', subset=['Result'])
    config = {}
    if 'Team badge' in d:
        config['Team badge']=st.column_config.ImageColumn('',width='small')
    for col in ['Season','Week','Picks','Wins','Losses','Pushes','Pending / Review','Void']:
        if col in d:config[col]=st.column_config.NumberColumn(col,format='%d')
    for rating_column in ['Matchup / 10', 'Defense / 10', 'Coverage / Scheme / 10']:
        if rating_column in d:
            config[rating_column] = st.column_config.ProgressColumn(rating_column, min_value=1, max_value=10, format='%.1f')
    if 'Signals' in d:
        colors = {'Mixed': '#b45309', 'Favorable': '#047857', 'Unfavorable': '#be123c', 'Neutral': '#475569'}
        style = style.map(lambda v: 'color:'+colors.get(v, '#64748b')+';font-weight:700', subset=['Signals'])
    st.dataframe(style, hide_index=True, use_container_width=True, column_config=config, key=key, row_height=46)


def select_pick_ui(projections, props):
    if not is_admin():
        st.caption('Enter your Admin password to save a pick.')
        return
    lines = prop_comparisons(projections, props)
    if lines.empty:
        return
    lines = game_filter_frame(lines,'my_pick_game')
    if lines.empty:return
    lines = lines.sort_values('Fetched', ascending=False).drop_duplicates(['Event ID', 'Key', 'Stat', 'Line', 'Bookmaker', 'Side'])
    st.subheader('Add a prop to My Picks')
    candidates = lines.drop_duplicates(['Event ID', 'Key', 'Stat', 'Line', 'Bookmaker']).reset_index(drop=True)
    index = st.selectbox('Choose a prop', list(candidates.index), format_func=lambda n: f"{candidates.loc[n,'Player']} · {candidates.loc[n,'Stat']} · {candidates.loc[n,'Line']:g} · {candidates.loc[n,'Bookmaker']}")
    row = candidates.loc[index]
    fixture_banner(row.Team,row.Opponent,'Choose your side below')
    c1,c2,c3=st.columns(3)
    c1.metric('Our projection',f'{row.Projection:.1f}')
    c2.metric('Prop line',f'{row.Line:g}')
    c3.metric('Difference',f'{row.Projection-row.Line:+.1f}')
    side = st.radio('Your prediction', ['Over', 'Under'], index=0 if row.Projection >= row.Line else 1, horizontal=True)
    if st.button('Save selected pick', type='primary'):
        try:
            if st.session_state.get('injury_load_error'):
                raise ValueError('Refresh injury notes successfully before saving a new pick.')
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
                      'Actual': None, 'Model Version': MODEL_VERSION, 'Result History': [],
                      'Season': st.session_state.model_meta['season'], 'Week': st.session_state.model_meta['week'],
                      'Day': local_day(row['Kickoff (UTC)']), 'Manual Grade': False,
                      'Injury Notes When Picked':matching_injury(row.Player,row.Team,st.session_state.model_meta['season'],st.session_state.model_meta['week']),
                      'Game Injury Notes When Picked':game_injury_context(row.Team,row.Opponent,st.session_state.model_meta['season'],st.session_state.model_meta['week'])}
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
    st.subheader('🎯 My Picks')
    select_pick_ui(st.session_state.projections,st.session_state.props)
    matchup_review_ui()
    manual_prop_entry_ui()
    st.divider()
    st.caption('Saved lines and projections stay frozen. Use All Model Picks to import results for all three records, or grade a personal pick manually.')
    if st.button('Load / refresh saved picks'):
        try:
            raw = github('data/my_picks.json')
            st.session_state.my_picks = json.loads(raw) if raw else []
        except (RuntimeError, ValueError) as e:
            st.error(str(e))
    if 'my_picks' not in st.session_state:
        st.info('Load your saved picks above to display your record.')
        return
    picks = game_filter_picks(st.session_state.my_picks,'my_saved_game')
    official=official_records(picks)
    st.caption(record_scope_label()+'. Pilot picks remain visible in the table.')
    won = sum(p['Result'] == 'Won' for p in official)
    lost = sum(p['Result'] == 'Lost' for p in official)
    pending = sum(p['Result'] in {'Pending', 'Needs review'} for p in official)
    pushed = sum(p['Result'] == 'Push' for p in official)
    void = sum(p['Result'] == 'Void' for p in official)
    cols = st.columns(5)
    cols[0].metric('RECORD', f'{won} – {lost}')
    cols[1].metric('HIT RATE', f'{100*won/(won+lost):.1f}%' if won+lost else '—')
    cols[2].metric('PENDING', pending)
    cols[3].metric('PUSHES', pushed)
    cols[4].metric('VOID', void)
    st.caption('Hit rate = wins ÷ (wins + losses). Pushes, voids, and pending picks are excluded. Automatic and manually reviewed grades; no payout tracking.')
    if not picks:
        st.info('Choose a game and save your first prop above, or load your saved picks.')
        return
    d = pd.DataFrame(picks)
    d['Injury Alert']=[(n['Status']+': '+n.get('Notes','')) if (n:=injury_for_pick(p)) and n['Status']!='Healthy / cleared' else '' for p in picks]
    d['Record Scope']=['Official' if official_pick(p) else ('Pilot' if record_week(p)==(2026,6) else 'Not in cumulative') for p in picks]
    status = st.selectbox('Show results', ['All', 'Pending', 'Needs review', 'Won', 'Lost', 'Push', 'Void'])
    visible = d if status == 'All' else d[d.Result == status]
    columns = ['Player', 'Team', 'Opponent', 'Stat', 'Bookmaker', 'Line', 'Your Pick', 'Projection When Picked', 'Original Model Projection', 'Manual Projection Override', 'Pick Source', 'Result', 'Actual', 'Injury Alert', 'Record Scope', 'Saved At']
    styled_table(visible.reindex(columns=columns))
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
                    found['Result'], found['Actual'] = grade, (None if grade in {'Pending', 'Void'} else actual)
                    found['Manual Grade'] = grade != 'Pending'
                    github('data/my_picks.json', json.dumps(latest, allow_nan=False).encode(), expected_hash=hashlib.sha256(raw).hexdigest())
                    st.session_state.my_picks = latest
                    st.rerun()
                except (RuntimeError, ValueError) as e:
                    st.error(str(e))


TRACKER_PATH = 'data/model_tracker.json.gz'
LOCAL_ZONE = ZoneInfo('America/Los_Angeles')
STAT_FIELDS = {'Passing yards': 'passing_yards', 'Rushing yards': 'rushing_yards', 'Receiving yards': 'receiving_yards', 'Receptions': 'receptions'}


def utc_now():
    return datetime.now(timezone.utc)


def local_day(kickoff):
    value = pd.to_datetime(kickoff, utc=True, errors='coerce')
    return None if pd.isna(value) else value.tz_convert(LOCAL_ZONE).date().isoformat()


def started(pick, now=None):
    value = pd.to_datetime(pick.get('Kickoff (UTC)'), utc=True, errors='coerce')
    return pd.isna(value) or value <= pd.Timestamp(now or utc_now())


def empty_tracker():
    return {'schema': 1, 'picks': {}, 'days': {}, 'updated_at': None}


def load_tracker():
    raw = github(TRACKER_PATH)
    data = json.loads(gzip.decompress(raw)) if raw else empty_tracker()
    if not isinstance(data.get('picks'), dict) or not isinstance(data.get('days'), dict):
        raise ValueError('Unexpected saved tracker format.')
    return data, hashlib.sha256(raw).hexdigest() if raw else 'missing'


def save_tracker(data, original_hash):
    data['updated_at'] = utc_now().isoformat()
    github(TRACKER_PATH, gzip.compress(json.dumps(data, allow_nan=False).encode(), mtime=0), expected_hash=original_hash)
    st.session_state.model_tracker = data


def sync_top20(data, day, now=None):
    now=now or utc_now()
    entry=data['days'].setdefault(day,{'active':[],'excluded':{},'changes':[]})
    entry.setdefault('excluded_players',{})
    picks=data['picks']
    player=lambda p:p.get('Key') or player_key(p['Player'])
    locked=[pid for pid in entry['active'] if pid in picks and started(picks[pid],now)]
    blocked=set(entry['excluded_players'])|{player(picks[pid]) for pid in entry['excluded'] if pid in picks}
    eligible=[p for pid,p in picks.items() if p.get('Day')==day and not p.get('Excluded') and not p.get('Injury Excluded')
              and pid not in entry['excluded'] and player(p) not in blocked and not started(p,now)
              and pick_direction(p['Stat'],p['Projection When Picked'],p['Line']) in {'Over','Under'}]
    eligible.sort(key=lambda p:(-abs(p['Projection When Picked']-p['Line']),p['Player'],p['Stat'],p['ID']))
    selected,used=locked[:],{player(picks[pid]) for pid in locked}
    teams,passing={},{}
    def cluster(p):
        # Directional passing-production overlap, including RB receiving.
        return (p['Team'],p['Your Pick']) if p['Stat'] in {'Passing yards','Receiving yards','Receptions'} else None
    def count(p):
        teams[p['Team']]=teams.get(p['Team'],0)+1
        c=cluster(p)
        if c:passing[c]=passing.get(c,0)+1
    for pid in locked:count(picks[pid])
    for p in eligible:
        if len(selected)>=20:break
        c=cluster(p)
        if player(p) in used or teams.get(p['Team'],0)>=3 or (c and passing.get(c,0)>=2):
            continue
        selected.append(p['ID']);used.add(player(p));count(p)
    if selected!=entry['active']:
        entry['changes'].append({'at':now.isoformat(),'previous':entry['active'][:],'new':selected[:],'selection_policy':UI_VERSION})
        entry['active']=selected
    return entry


def register_model_slate(projections, props, meta, now=None):
    now = now or utc_now()
    if st.session_state.get('injury_load_error'):
        raise ValueError('Load injury notes successfully before registering new picks.')
    candidates = prop_comparisons(projections, props)
    if candidates.empty:
        raise ValueError('No matching upcoming props. Calculate projections and import games first.')
    candidates = candidates[candidates.Bookmaker.str.casefold() == 'draftkings']
    candidates = candidates.sort_values(['Fetched', 'Updated (UTC)'], ascending=False).drop_duplicates(['Event ID', 'Key', 'Stat'])
    data, version = load_tracker()
    added = 0
    for _, r in candidates.iterrows():
        difference = float(r.Projection - r.Line)
        if pick_direction(r.Stat, r.Projection, r.Line) == 'No pick' or started(r.to_dict(), now):
            continue
        pid = hashlib.sha256(json.dumps([r['Event ID'], r.Key, r.Stat, 'DraftKings']).encode()).hexdigest()[:24]
        if pid in data['picks']:
            continue
        side = 'Over' if difference > 0 else 'Under'
        odds_rows = props[(props['Event ID'] == r['Event ID']) & (props.Player.map(player_key) == r.Key)
                          & (props.Stat == r.Stat) & (props.Line == r.Line) & (props.Side == side)]
        odds = None if odds_rows.empty or pd.isna(odds_rows.iloc[0].Odds) else float(odds_rows.iloc[0].Odds)
        data['picks'][pid] = {
            'ID': pid, 'Player': r.Player, 'Key': r.Key, 'Position': r.Position, 'Team': r.Team,
            'Opponent': r.Opponent, 'Away': r.Away, 'Home': r.Home, 'Game': r.Matchup,
            'Stat': r.Stat, 'Line': float(r.Line), 'Your Pick': side,
            'Projection When Picked': float(r.Projection), 'Season Average': float(r['Season Average']),
            'Odds': odds, 'Bookmaker': 'DraftKings', 'Event ID': r['Event ID'], 'Kickoff (UTC)': r['Kickoff (UTC)'],
            'Day': local_day(r['Kickoff (UTC)']), 'Season': int(meta['season']), 'Week': int(meta['week']),
            'Model Version': MODEL_VERSION, 'Saved At': now.isoformat(), 'Result': 'Pending', 'Actual': None,
            'Excluded': False, 'Exclusion Reason': '', 'Exclusion History': [], 'Result History': [],
            'Result Note': '', 'Manual Grade': False, 'Injury Excluded':False, 'UI Version':UI_VERSION, 'Game Injury Notes When Picked':game_injury_context(r.Team,r.Opponent,meta['season'],meta['week']), 'Injury Notes When Picked':matching_injury(r.Player,r.Team,meta['season'],meta['week']), 'No Pick Range When Saved': no_pick_settings().get(r.Stat, 0.0)}
        added += 1
    for day in sorted({p['Day'] for p in data['picks'].values() if p.get('Day')}):
        sync_top20(data, day, now)
    save_tracker(data, version)
    return added


def exclude_model_pick(pid, reason, top_day=None, restore=False, now=None):
    now = now or utc_now()
    data, version = load_tracker()
    p = data['picks'].get(pid)
    if p is None:
        raise ValueError('Pick not found. Refresh the tracker.')
    if started(p, now):
        raise ValueError('Exclusions and restores lock at kickoff.')
    if not restore and not reason.strip():
        raise ValueError('Choose an exclusion reason.')
    if top_day is None:
        p['Excluded'] = not restore
        p['Exclusion Reason'] = '' if restore else reason
        p['Exclusion History'].append({'at': now.isoformat(), 'excluded': not restore, 'reason': reason})
    else:
        day = data['days'].setdefault(top_day, {'active': [], 'excluded': {}, 'changes': []})
        key = p.get('Key') or player_key(p['Player'])
        day.setdefault('excluded_players', {})
        if restore:
            day['excluded_players'].pop(key, None)
            for old_id in list(day['excluded']):
                old = data['picks'].get(old_id, {})
                if (old.get('Key') or player_key(old.get('Player', ''))) == key:
                    day['excluded'].pop(old_id, None)
        else:
            day['excluded_players'][key] = {'reason': reason, 'at': now.isoformat()}
            day['excluded'][pid] = {'reason': reason, 'at': now.isoformat()}
        day['changes'].append({'at': now.isoformat(), 'pick': pid, 'excluded': not restore, 'reason': reason})
    sync_top20(data, p['Day'], now)
    save_tracker(data, version)


def record_metrics(picks, key_prefix):
    counts = {status: sum(p.get('Result') == status for p in picks) for status in ['Won', 'Lost', 'Push', 'Void', 'Pending', 'Needs review']}
    won, lost = counts['Won'], counts['Lost']
    cols = st.columns(5)
    cols[0].metric('RECORD', f'{won} – {lost}')
    cols[1].metric('HIT RATE', f'{100*won/(won+lost):.1f}%' if won+lost else '—')
    cols[2].metric('PENDING / REVIEW', counts['Pending'] + counts['Needs review'])
    cols[3].metric('PUSHES', counts['Push'])
    cols[4].metric('VOID', counts['Void'])
    st.caption(f'{len(picks)} counted selections · Hit rate excludes pending, review, pushes, voids, and crossed-off picks.')


def pick_display(picks):
    return pd.DataFrame([{'Game': p.get('Game', f"{p['Team']} vs {p['Opponent']}"), 'Player': p['Player'],
                          'Stat': p['Stat'], 'Pick': p['Your Pick'], 'Line': p['Line'],
                          'Projection': p['Projection When Picked'], 'Difference': round(abs(p['Projection When Picked']-p['Line']), 2),
                          'Result': p['Result'], 'Actual': p.get('Actual'), 'Note': p.get('Result Note', '')} for p in picks])


def tracker_controls():
    st.subheader('Save forecasts and update results')
    c1, c2 = st.columns(2)
    if c1.button('Load / refresh model records'):
        try:
            load_analysis_settings()
            data, _ = load_tracker()
            st.session_state.model_tracker = data
        except (ValueError, RuntimeError) as e:
            st.error(str(e))
    if is_admin() and c2.button('Save current model slate + refresh Top 20'):
        try:
            if not st.session_state.model_meta:
                raise ValueError('Calculate or load a model batch first.')
            added = register_model_slate(st.session_state.projections, st.session_state.props, st.session_state.model_meta)
            st.success(f'{added} new forecasts saved. Existing saved lines remain fixed.')
        except (ValueError, RuntimeError) as e:
            st.error(str(e))


def model_records_page():
    st.subheader('All Model Picks')
    st.caption('One official forecast per player, game, and stat. Only imported DraftKings markets with matching projections are registered. Exclusions do not change My Picks.')
    tracker_controls()
    if 'model_tracker' not in st.session_state:
        st.info('Load saved records, or save the current model slate to start tracking.')
        return
    data = st.session_state.model_tracker
    days = sorted({p['Day'] for p in data['picks'].values()}, reverse=True)
    day = st.selectbox('Model record date', ['All dates'] + days)
    picks = [p for p in data['picks'].values() if day == 'All dates' or p['Day'] == day]
    picks = game_filter_picks(picks,'all_model_game')
    active = [p for p in picks if not p.get('Excluded') and not p.get('Injury Excluded')]
    st.caption(record_scope_label()+'. Displayed historical picks include the pilot.')
    record_metrics(official_records(active), 'model')
    styled_table(pick_display(active))
    if active:
        st.download_button('Download all model picks', pd.DataFrame(active).drop(columns=['Result History', 'Exclusion History'], errors='ignore').to_csv(index=False), 'all_model_picks.csv', 'text/csv')
    if is_admin():
        with st.expander('Cross off picks for this week', expanded=True):
            st.caption('✕ removes this pick from the record and Top 20 eligibility before kickoff. Injury exclusions are managed in Injuries. No saved forecast is deleted.')
            candidates = [p for p in active if not started(p)]
            reason = st.selectbox('Reason', ['Injury / availability', 'Role uncertainty', 'Line too low / too volatile', 'Data concern', 'Other'], key='model_exclude_reason')
            search = st.text_input('Find a pick to cross off', key='model_exclude_search')
            if search:
                candidates = [p for p in candidates if search.casefold() in (p['Player']+' '+p['Game']+' '+p['Stat']).casefold()]
            max_page = max(1, math.ceil(len(candidates)/20))
            if st.session_state.get('model_exclude_page', 1) > max_page:
                st.session_state.model_exclude_page = max_page
            page = st.number_input('Page (20 picks per page)', min_value=1, max_value=max_page, value=1, key='model_exclude_page')
            for p in candidates[(page-1)*20:page*20]:
                c1, c2 = st.columns([12, 1])
                c1.write(f"**{p['Player']}** · {p['Your Pick']} {p['Line']:g} {p['Stat']} · {p['Game']}")
                if c2.button('✕', key='model_x_'+p['ID'], help='Exclude from model record and Top 20'):
                    try:
                        exclude_model_pick(p['ID'], reason)
                        st.rerun()
                    except (ValueError, RuntimeError) as e:
                        st.error(str(e))
            if not candidates:
                st.info('No matching pregame picks to exclude.')
        with st.expander('Excluded picks / restore before kickoff'):
            excluded = [p for p in picks if p.get('Excluded') or p.get('Injury Excluded')]
            for p in excluded:
                c1, c2 = st.columns([10, 2])
                c1.write(f"{p['Player']} · {p['Stat']} · {p.get('Exclusion Reason') or ('Injury exclusion' if p.get('Injury Excluded') else '')}")
                if c2.button('Restore', key='model_restore_'+p['ID'], disabled=started(p) or bool(p.get('Injury Excluded'))):
                    try:
                        exclude_model_pick(p['ID'], '', restore=True)
                        st.rerun()
                    except (ValueError, RuntimeError) as e:
                        st.error(str(e))
        results_update_ui()


def best20_page():
    st.subheader('🏆 20 Best Picks of the Day')
    st.caption('One prop per player: the largest qualifying absolute projection-minus-line difference. Crossing off a pick blocks that player for this day. Raw yards and receptions use different units; this is not a calibrated betting-edge score.')
    if 'model_tracker' not in st.session_state:
        st.info('Load or save the model slate in All Model Picks first.')
        return
    data = st.session_state.model_tracker
    today = utc_now().astimezone(LOCAL_ZONE).date().isoformat()
    days = sorted(set(data['days']) | {today})
    day = st.selectbox('Top 20 day (California time)', days, index=days.index(today))
    entry = data['days'].get(day, {'active': [], 'excluded': {}})
    active = [data['picks'][pid] for pid in entry['active'] if pid in data['picks'] and not data['picks'][pid].get('Excluded') and not data['picks'][pid].get('Injury Excluded') and pid not in entry['excluded']]
    all_top = []
    for saved_day in data['days'].values():
        all_top.extend(data['picks'][pid] for pid in saved_day['active'] if pid in data['picks'] and not data['picks'][pid].get('Excluded') and not data['picks'][pid].get('Injury Excluded') and pid not in saved_day['excluded'])
    overall = list({p['ID']: p for p in all_top}.values())
    st.markdown('**Top 20 · Overall record**')
    st.caption(record_scope_label())
    record_metrics(official_records(overall), 'top20_overall')
    with st.expander('Record for the selected day'):
        record_metrics(active, 'top20_day')
    if len(active) < 20:
        st.info(f'{len(active)} eligible selections saved for this day. The list fills to 20 only when enough matching pregame picks are available.')
    if is_admin():
        if st.button('Refresh Top 20 from saved slate'):
            try:
                fresh, version = load_tracker()
                sync_top20(fresh, day)
                save_tracker(fresh, version)
                st.rerun()
            except (ValueError, RuntimeError) as e:
                st.error(str(e))
        reason = st.selectbox('Reason when crossing off a Top 20 pick', ['Injury / availability', 'Role uncertainty', 'Line too low / too volatile', 'Data concern', 'Other'], key='top_exclusion_reason')
    else:
        reason = ''
    st.caption('Top 20 limits: one prop per player, up to three per team, and up to two same-team passing-related picks in each direction. Historical started selections keep their original policy.')
    ranks = {p['ID']: n+1 for n, p in enumerate(sorted(active, key=lambda p: -abs(p['Projection When Picked']-p['Line'])))}
    games = {}
    for p in active:
        games.setdefault(p['Game'], []).append(p)
    for game, group in sorted(games.items(), key=lambda pair: min(ranks[p['ID']] for p in pair[1])):
        fixture_banner(group[0]['Team'],group[0]['Opponent'],game)
        for p in sorted(group, key=lambda p:ranks[p['ID']]):
            with st.container(border=True):
                c1,c2,c3,c4 = st.columns([6,2,2,1])
                c1.markdown(f"**#{ranks[p['ID']]} · {html.escape(p['Player'])}**  \n{p['Your Pick']} {p['Stat']} **{p['Line']:g}**")
                c2.metric('Projection', f"{p['Projection When Picked']:.1f}")
                c3.metric('Difference', f"{abs(p['Projection When Picked']-p['Line']):.1f}")
                c1.caption('Result: ' + p['Result'])
                if is_admin() and c4.button('✕', key='top_x_'+p['ID'], disabled=started(p), help='Exclude this player from today’s Top 20; refill with another player'):
                    try:
                        exclude_model_pick(p['ID'], reason, top_day=day)
                        st.rerun()
                    except (ValueError, RuntimeError) as e:
                        st.error(str(e))
    if active:
        st.download_button('Download this Top 20', pick_display(active).to_csv(index=False), f'top20_{day}.csv', 'text/csv')
    with st.expander('Top 20 exclusions / restore'):
        for pid, details in entry['excluded'].items():
            p = data['picks'].get(pid)
            if not p:
                continue
            st.write(f"{p['Player']} · {p['Stat']} · {details['reason']}")
            if is_admin() and st.button('Restore eligibility', key='top_restore_'+pid, disabled=started(p)):
                try:
                    exclude_model_pick(pid, '', top_day=day, restore=True)
                    st.rerun()
                except (ValueError, RuntimeError) as e:
                    st.error(str(e))
    st.caption('Started selections keep their slots. Top 20 exclusions block that player for the day and leave All Model Picks unchanged. Older started selections retain their original rules.')


@st.cache_data(ttl=900, show_spinner=False)
def download_csv(url):
    try:
        with urlopen(Request(url, headers={'User-Agent': 'NFL-Matchup-Analytics'}), timeout=45) as response:
            return pd.read_csv(io.BytesIO(response.read()), low_memory=False)
    except (HTTPError, URLError, OSError, ValueError):
        raise RuntimeError('Could not load public results data. The season may not be available yet. Use the CSV fallback or try later.') from None


@st.cache_data(ttl=300, show_spinner=False)
def completed_games(day):
    # Confirm final status separately; do not grade partial weekly statistics.
    url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?' + urlencode({'dates': day.replace('-', ''), 'limit': 100})
    data, _ = request_json(url)
    finals = set()
    for event in data.get('events', []):
        if not event.get('status', {}).get('type', {}).get('completed'):
            continue
        for comp in event.get('competitions', []):
            teams = {x.get('homeAway'): team_code(x.get('team', {}).get('abbreviation')) for x in comp.get('competitors', [])}
            date = local_day(event.get('date'))
            if teams.get('home') and teams.get('away') and date:
                finals.add((date, teams['away'], teams['home']))
    return finals


def normalize_weekly_stats(table, season, week):
    t = table.copy()
    # Accommodate current and earlier nflverse field names without guessing data.
    rename = {}
    for new, choices in {'Player': ['player_display_name', 'player_name', 'display_name'],
                         'Team': ['team', 'recent_team', 'team_abbr'], 'Opponent': ['opponent_team', 'opponent']}.items():
        found = next((c for c in choices if c in t), None)
        if found is None:
            raise ValueError(f'Weekly results file has no recognized {new} column.')
        rename[found] = new
    t = t.rename(columns=rename)
    if not {'season','week'} <= set(t):
        raise ValueError('Weekly results require season and week fields.')
    t = t[(number(t.season)==int(season)) & (number(t.week)==int(week))].copy()
    if 'season_type' in t:
        t = t[t.season_type.astype(str).str.upper().isin(['REG', 'REGULAR'])]
    t['Team'], t['Opponent'] = t.Team.map(team_code), t.Opponent.map(team_code)
    t['Key'] = t.Player.map(player_key)
    for stat in STAT_FIELDS.values():
        if stat in t:
            t[stat] = number(t[stat])
    return t


def find_actual(pick, stats):
    key = pick.get('Key') or player_key(pick['Player'])
    found = stats[(stats.Key == key) & (stats.Team == pick['Team']) & (stats.Opponent == pick['Opponent'])]
    if len(found) != 1:
        return None, 'Player missing/ambiguous in weekly stats; check participation and name. No automatic zero.'
    field = STAT_FIELDS.get(pick['Stat'])
    if field not in found or pd.isna(found.iloc[0][field]):
        return None, 'Stat unavailable; not treated as zero.'
    row = found.iloc[0]
    actual = float(row[field])
    opportunities = [pd.to_numeric(row.get(c), errors='coerce') for c in ['attempts', 'carries', 'targets']]
    if actual == 0 and not any(pd.notna(v) and v > 0 for v in opportunities):
        return None, 'Zero stat with no recorded offensive opportunity; verify participation / void rules.'
    return actual, ''


def apply_auto_results(picks, stats, finals, now=None):
    now = now or utc_now()
    changed, graded = 0, 0
    for p in picks:
        if p.get('Manual Grade') or p.get('Result') == 'Void':
            continue
        if not started(p, now):
            continue
        day = p.get('Day') or local_day(p['Kickoff (UTC)'])
        # Personal picks may not store home/away. Match either orientation.
        final = (day, p.get('Away', p['Team']), p.get('Home', p['Opponent'])) in finals or (day, p['Opponent'], p['Team']) in finals or (day, p['Team'], p['Opponent']) in finals
        if not final:
            p['Result Note'] = 'Waiting for confirmed final game status.'
            continue
        actual, note = find_actual(p, stats)
        if actual is None:
            if p.get('Actual') is None:
                p['Result'] = 'Needs review'
            p['Result Note'] = note
            continue
        result = result_from_actual(p['Your Pick'], p['Line'], actual)
        if p.get('Actual') != actual or p.get('Result') != result:
            p.setdefault('Result History', []).append({'at': now.isoformat(), 'previous_result': p.get('Result'), 'previous_actual': p.get('Actual'), 'new_result': result, 'new_actual': actual, 'source': 'nflverse weekly stats; final confirmed separately'})
            p['Actual'], p['Result'] = actual, result
            changed += 1
        p['Result Note'] = 'Auto-graded; later stat corrections may update this result.'
        p['Graded At'] = now.isoformat()
        graded += 1
    return changed, graded


def result_update(season, week, uploaded_stats=None):
    stats_raw = uploaded_stats if uploaded_stats is not None else download_csv(f'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{int(season)}.csv')
    stats = normalize_weekly_stats(stats_raw, season, week)
    if stats.empty:
        raise ValueError('No player results found for this season/week yet. Records were not graded.')
    data, version = load_tracker()
    model = [p for p in data['picks'].values() if p['Season']==int(season) and p['Week']==int(week)]
    raw_personal = github('data/my_picks.json')
    personal = json.loads(raw_personal) if raw_personal else []
    # New personal picks carry season/week. Older picks require exact game date matching below.
    model_events = {p['Event ID'] for p in model}
    selected_personal = [p for p in personal if (p.get('Season')==int(season) and p.get('Week')==int(week)) or p['Event ID'] in model_events]
    days = sorted({p.get('Day') or local_day(p['Kickoff (UTC)']) for p in model + selected_personal})
    finals = set()
    for day in days:
        if day:
            finals |= completed_games(day)
    c1,n1 = apply_auto_results(model, stats, finals)
    c2,n2 = apply_auto_results(selected_personal, stats, finals)
    save_tracker(data, version)
    if raw_personal is not None:
        github('data/my_picks.json', json.dumps(personal, allow_nan=False).encode(), expected_hash=hashlib.sha256(raw_personal).hexdigest())
        st.session_state.my_picks = personal
    return c1+c2,n1,n2


def results_update_ui():
    with st.expander('Update completed-game results', expanded=True):
        st.caption('One import grades All Model Picks, Top 20, and My Picks using frozen lines. Requires both weekly player stats and confirmed final game status. Not live grading.')
        c1,c2=st.columns(2)
        season=c1.number_input('Result season',2020,2100,2026,key='result_season')
        week=c2.number_input('Result week',1,18,6,key='result_week')
        if st.button('Import results and grade all three records',type='primary'):
            try:
                with st.spinner('Loading weekly stats and checking final games...'):
                    changed,n1,n2=result_update(season,week)
                st.success(f'{n1} model and {n2} personal selections matched. {changed} result changes saved. Top 20 uses the same model results. Refresh My Picks if needed.')
            except (RuntimeError,ValueError,KeyError) as e:
                st.error(str(e))
        fallback=st.file_uploader('Fallback: nflverse-format weekly results CSV',type='csv',key='weekly_actual_csv')
        if fallback and st.button('Grade from uploaded weekly CSV'):
            try:
                changed,n1,n2=result_update(season,week,pd.read_csv(fallback))
                st.success(f'Graded {n1} model and {n2} personal selections; {changed} result changes saved.')
            except (RuntimeError,ValueError,KeyError) as e:
                st.error(str(e))
        st.caption('If final-status lookup is unavailable, automatic grading stops; use manual review below. DNPs/voids require review. Excluded selections retain results internally but do not count in their excluded record.')
    with st.expander('Manual result correction / void'):
        data=st.session_state.get('model_tracker',empty_tracker())
        ids=list(data['picks'])
        if not ids:
            return
        pid=st.selectbox('Model pick to grade',ids,format_func=lambda k:f"{data['picks'][k]['Player']} · {data['picks'][k]['Stat']} · {data['picks'][k]['Your Pick']} {data['picks'][k]['Line']}")
        grade=st.selectbox('Manual model result',['Won','Lost','Push','Void','Pending'])
        confirmation=st.checkbox('I verified the completed game or void condition.',key='manual_model_confirm')
        if st.button('Save manual model result'):
            try:
                fresh,version=load_tracker()
                p=fresh['picks'][pid]
                if not confirmation:
                    raise ValueError('Confirm the result before saving.')
                if grade in {'Won','Lost','Push'} and not started(p):
                    raise ValueError('This game has not started.')
                p.setdefault('Result History',[]).append({'at':utc_now().isoformat(),'previous_result':p['Result'],'new_result':grade,'source':'manual'})
                p['Result']=grade
                p['Manual Grade']=grade!='Pending'
                p['Actual']=None
                p['Result Note']='Manually reviewed' if grade!='Pending' else 'Automatic grading re-enabled'
                save_tracker(fresh,version)
                st.rerun()
            except (RuntimeError,ValueError,KeyError) as e:
                st.error(str(e))


DEFAULT_NO_PICK = {'Passing yards': 10.0, 'Receiving yards': 5.0, 'Rushing yards': 5.0, 'Receptions': 0.5}


def no_pick_settings():
    try:
        return {**DEFAULT_NO_PICK, **st.session_state.get('no_pick_ranges', {})}
    except Exception:
        return DEFAULT_NO_PICK.copy()


def pick_direction(stat, projection, line, ranges=None):
    if pd.isna(projection) or pd.isna(line):
        return 'No line'
    difference = float(projection) - float(line)
    threshold = (ranges or no_pick_settings()).get(stat, 0.0)
    if abs(difference) <= threshold + 1e-9:
        return 'No pick'
    return 'Over' if difference > 0 else 'Under'


def signal_label(defense, scheme):
    def direction(rating):
        if rating is None or pd.isna(rating):
            return None
        return 1 if rating > 5.8 else (-1 if rating < 5.2 else 0)
    d, s = direction(defense), direction(scheme)
    if s is None:
        return 'Defense only' if d is not None else 'Unavailable'
    if d is None:
        return 'Scheme only'
    if d*s < 0:
        return 'Mixed'
    if d > 0 or s > 0:
        return 'Favorable'
    if d < 0 or s < 0:
        return 'Unfavorable'
    return 'Neutral'


def settings_ui():
    with st.expander('Official model pick thresholds'):
        st.caption('Rows inside or exactly on the ± range remain visible as No pick. They do not enter new official model records or Top 20. Personal picks remain your choice. Existing saved forecasts are not regraded or silently removed when settings change.')
        current = no_pick_settings()
        with st.form('threshold_settings'):
            values = {}
            columns = st.columns(2)
            for n, (stat, default) in enumerate(DEFAULT_NO_PICK.items()):
                values[stat] = columns[n % 2].number_input('± range: ' + stat, min_value=0.0, value=float(current[stat]), step=0.5)
            save = st.form_submit_button('Save no-pick ranges')
        if save:
            try:
                github('data/analysis_settings.json', json.dumps({'no_pick_ranges': values}).encode())
                st.session_state.no_pick_ranges = values
                st.success('Ranges saved. New forecasts use these ranges; previously saved records stay unchanged.')
            except RuntimeError as e:
                st.error(str(e))


def load_analysis_settings():
    raw = github('data/analysis_settings.json')
    if raw:
        saved = json.loads(raw).get('no_pick_ranges', {})
        if any(k not in DEFAULT_NO_PICK or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for k, v in saved.items()):
            raise ValueError('Saved no-pick ranges are invalid.')
        st.session_state.no_pick_ranges = {**DEFAULT_NO_PICK, **saved}


def summarize_records(picks, group):
    if not picks:
        return pd.DataFrame()
    rows=[]
    frame=pd.DataFrame(picks)
    for label, part in frame.groupby(group, dropna=False):
        statuses=part.Result.value_counts()
        w,l=int(statuses.get('Won',0)),int(statuses.get('Lost',0))
        settled=part[part.Result.isin(['Won','Lost','Push']) & pd.to_numeric(part.Actual,errors='coerce').notna()].copy()
        if not settled.empty:
            settled=settled.drop_duplicates(['Event ID','Key','Stat']) if 'Key' in settled else settled
            error=(pd.to_numeric(settled['Projection When Picked'])-pd.to_numeric(settled.Actual)).abs().mean()
            baseline=(pd.to_numeric(settled['Season Average'])-pd.to_numeric(settled.Actual)).abs().mean() if 'Season Average' in settled else float('nan')
        else:
            error,baseline=float('nan'),float('nan')
        rows.append({group:label,'Picks':len(part),'Wins':w,'Losses':l,'Pushes':int(statuses.get('Push',0)),
                     'Pending / Review':int(statuses.get('Pending',0)+statuses.get('Needs review',0)),
                     'Void':int(statuses.get('Void',0)),'Hit Rate %':100*w/(w+l) if w+l else float('nan'),
                     'Model MAE':error,'Season Baseline MAE':baseline})
    return pd.DataFrame(rows)


def records_dashboard():
    st.subheader('📊 Records hub')
    st.caption('Official cumulative totals start in 2026 Week 7. Week 6 remains a visible pilot baseline. Excluded picks do not count; forecasts and past results are never deleted.')
    if st.button('Refresh all three records',key='dashboard_refresh'):
        try:
            load_record_policy()
            data,_=load_tracker();st.session_state.model_tracker=data
            raw=github('data/my_picks.json');st.session_state.my_picks=json.loads(raw) if raw else []
        except (RuntimeError,ValueError) as error:
            st.error(str(error))
    if 'model_tracker' not in st.session_state:
        st.info('Refresh records above to load saved results.')
        return
    data=st.session_state.model_tracker
    model=[p for p in data['picks'].values() if not p.get('Excluded') and not p.get('Injury Excluded')]
    mine=st.session_state.get('my_picks',[])
    top=[]
    for entry in data['days'].values():
        top.extend(data['picks'][pid] for pid in entry['active'] if pid in data['picks'] and not data['picks'][pid].get('Excluded') and not data['picks'][pid].get('Injury Excluded') and pid not in entry['excluded'])
    top=list({p['ID']:p for p in top}.values())
    groups=[('All Model Picks',model),('My Picks',mine),('Top 20',top)]
    st.markdown('### Official cumulative records')
    st.caption(record_scope_label())
    for title,picks in groups:
        st.markdown('#### '+title)
        if title=='My Picks' and 'my_picks' not in st.session_state:
            st.info('Refresh records to load personal picks.')
        else:
            record_metrics(official_records(picks),'official_'+title)
    st.markdown('### Week-by-week history')
    weekly=weekly_records_table(groups)
    if not weekly.empty:
        weeks=sorted(set(zip(weekly.Season,weekly.Week)))
        selected=st.selectbox('History week',['All weeks']+weeks,format_func=lambda w:w if w=='All weeks' else f'{w[0]} · Week {w[1]}'+(' · Pilot' if w==(2026,6) else ''))
        visible=weekly if selected=='All weeks' else weekly[(weekly.Season==selected[0])&(weekly.Week==selected[1])]
        styled_table(visible)
        st.download_button('Download weekly history',weekly.to_csv(index=False),'weekly_records.csv','text/csv')
    else:
        st.info('Weekly history appears after picks are saved.')
    st.markdown('### Model record breakdowns')
    scope=st.selectbox('Breakdown scope',['Official cumulative','Week 6 pilot','All saved weeks'])
    rows=official_records(model) if scope=='Official cumulative' else ([p for p in model if record_week(p)==(2026,6)] if scope=='Week 6 pilot' else model)
    for col,title in [('Stat','By stat category'),('Position','By position'),('Your Pick','Overs vs Unders')]:
        st.markdown('**'+title+'**')
        report=summarize_records(rows,col)
        if col!='Stat':report=report.drop(columns=['Model MAE','Season Baseline MAE'],errors='ignore')
        styled_table(report)
    st.caption('MAE compares stat-total forecast errors only within a stat category. Pilot results remain available for analysis without affecting official totals.')



WORKING_PATH = 'data/working_batch.json.gz'
SAVED_PROPS_PATH = 'data/saved_props.json.gz'


def read_working_batch():
    raw = github(WORKING_PATH)
    if raw is not None:
        return json.loads(gzip.decompress(raw)), hashlib.sha256(raw).hexdigest()
    legacy = github('data/current_batch.json')
    if legacy:
        old = json.loads(legacy)
        return {'meta': {k:old['meta'][k] for k in ['season','cutoff','week']},
                'sources':old['sources'], 'confirmed':True}, 'missing'
    return {'meta':None,'sources':{},'confirmed':False}, 'missing'


def save_csv_batch(meta, sources, confirmed=False):
    existing, version = read_working_batch()
    if existing.get('sources') and existing.get('meta') != meta:
        raise ValueError('A different week is saved. Use Clear weekly CSVs before starting a new week.')
    bundle = {'meta':meta, 'sources':{k:base64.b64encode(v).decode() for k,v in sources.items()},
              'confirmed':bool(confirmed),'saved_at':utc_now().isoformat()}
    github(WORKING_PATH, gzip.compress(json.dumps(bundle).encode(), mtime=0), expected_hash=version)
    st.session_state.csv_dirty = False
    st.session_state.csv_confirmed = bool(confirmed)


def read_saved_props():
    raw = github(SAVED_PROPS_PATH)
    if raw is not None:
        return pd.DataFrame(json.loads(gzip.decompress(raw))), hashlib.sha256(raw).hexdigest()
    legacy = github('data/current_props.json')
    return (pd.DataFrame(json.loads(legacy)) if legacy else pd.DataFrame()), 'missing'


def merge_props(existing, incoming):
    if incoming.empty:
        return existing.copy()
    old = existing.copy()
    if not old.empty:
        # Replace the fetched game's sportsbook snapshot; keep other games/books.
        pairs = set(zip(incoming['Event ID'], incoming['Bookmaker']))
        old = old[[pair not in pairs for pair in zip(old['Event ID'], old['Bookmaker'])]]
    return pd.concat([old,incoming], ignore_index=True)


def save_props_snapshot(incoming):
    saved, version = read_saved_props()
    merged = merge_props(saved, incoming)
    github(SAVED_PROPS_PATH, gzip.compress(merged.to_json(orient='records').encode(), mtime=0), expected_hash=version)
    st.session_state.props = merged
    st.session_state.props_dirty = False
    return merged


def restore_saved_state(force=False):
    s = st.session_state
    bundle,_ = read_working_batch()
    if force or not s.source_bytes:
        sources={k:base64.b64decode(v) for k,v in bundle.get('sources',{}).items()}
        tables={k:parse_csv(raw,k) for k,raw in sources.items()}
        s.source_bytes,s.tables = sources,tables
        s.batch_meta=bundle.get('meta')
        s.csv_confirmed=bool(bundle.get('confirmed'))
        s.projections,s.issues=pd.DataFrame(),pd.DataFrame()
        s.model_meta=None
        if s.batch_meta and s.csv_confirmed and REQUIRED <= tables.keys():
            load_injuries()
            ensure_recent_workloads(int(s.batch_meta['season']), int(s.batch_meta['cutoff']), tables)
            s.projections,s.issues=build_model(tables,int(s.batch_meta['cutoff']))
            s.model_meta={**s.batch_meta,'model_version':MODEL_VERSION,'defense_weight':0.632}
    props,_=read_saved_props()
    if force or s.props.empty:
        s.props=props
    # Restoring data does not contact The Odds API or create new forecasts.
    load_record_policy()
    try:
        load_injuries()
        s.injury_load_error=None
    except (RuntimeError,ValueError) as error:
        s.injury_load_error=str(error)
        raise
    try:
        load_analysis_settings()
        s.settings_checked=True
    except (RuntimeError,ValueError):
        pass


def import_games_safely(selected, force_fresh=False):
    s=st.session_state
    imported,reused=0,0
    errors=[]
    for eid in selected:
        known=s.props[s.props['Event ID'].eq(eid) & s.props.Bookmaker.eq('DraftKings')] if not s.props.empty else pd.DataFrame()
        if not known.empty and not force_fresh:
            reused+=1
            continue
        try:
            rows,remaining=fetch_props(eid)
            if not rows:
                errors.append(f'{eid}: no props returned; any saved lines were retained.')
                continue
            incoming=pd.DataFrame(rows)
            # Keep fetched lines in memory even if the external save fails.
            s.props=merge_props(s.props,incoming)
            s.props_dirty=True
            try:
                save_props_snapshot(s.props)
            except (RuntimeError,ValueError) as error:
                errors.append('Props fetched, but automatic saving failed: '+str(error)+'. Use Save current props now before closing the page.')
            imported+=1
            s.last_credits_remaining=remaining
        except RuntimeError as error:
            errors.append(str(error))
    return imported,reused,errors


def clear_weekly_csvs():
    _,version=read_working_batch()
    github(WORKING_PATH,gzip.compress(json.dumps({'meta':None,'sources':{},'confirmed':False,'cleared_at':utc_now().isoformat()}).encode(),mtime=0),expected_hash=version)
    s=st.session_state
    s.tables,s.source_bytes={},{}
    s.batch_meta,s.model_meta=None,None
    s.projections,s.issues=pd.DataFrame(),pd.DataFrame()
    s.csv_confirmed,s.csv_dirty=False,False


def clear_saved_props():
    _,version=read_saved_props()
    github(SAVED_PROPS_PATH,gzip.compress(b'[]',mtime=0),expected_hash=version)
    st.session_state.props=pd.DataFrame()
    st.session_state.props_dirty=False


def persistence_controls():
    s=st.session_state
    st.caption('Accepted CSVs and imported props save automatically to GitHub and restore after refresh. Stored files have no weekly expiration. Clearing current data never deletes saved picks or records.')
    if s.get('csv_dirty'):
        st.error('Your newest CSV changes have not saved permanently yet.')
    if s.get('props_dirty'):
        st.error('Your latest props are only in this session. Save them before refreshing.')
    c1,c2=st.columns(2)
    if c1.button('Save accepted CSVs now',disabled=not bool(s.source_bytes)):
        try:
            save_csv_batch(s.batch_meta,s.source_bytes,s.get('csv_confirmed',False))
            st.success('All accepted CSVs saved.')
        except (RuntimeError,ValueError) as e:
            st.error(str(e))
    if c2.button('Save current props now',disabled=s.props.empty):
        try:
            save_props_snapshot(s.props)
            st.success('All imported props saved.')
        except (RuntimeError,ValueError) as e:
            st.error(str(e))
    with st.expander('Start a new week / clear current data'):
        clear_confirm=st.checkbox('I want to clear the selected current data. My saved picks and records will stay.')
        c1,c2=st.columns(2)
        if c1.button('Clear weekly CSVs',disabled=not clear_confirm):
            try:
                clear_weekly_csvs()
                st.rerun()
            except (RuntimeError,ValueError) as e:
                st.error(str(e))
        if c2.button('Clear imported props',disabled=not clear_confirm):
            try:
                clear_saved_props()
                st.rerun()
            except (RuntimeError,ValueError) as e:
                st.error(str(e))


UI_VERSION = '6.0-week7-interface'
RECORD_POLICY_PATH = 'data/record_policy.json'
INJURIES_PATH = 'data/injury_notes.json'
DEFAULT_RECORD_POLICY = {'official_start': {'season':2026,'week':7}, 'pilot_weeks':['2026:6']}
TEAM_COLORS = {'ARI':'#97233f','ATL':'#a71930','BAL':'#6336a8','BUF':'#1769d2','CAR':'#0085ca','CHI':'#d55e18',
               'CIN':'#fb4f14','CLE':'#c85b18','DAL':'#3277b3','DEN':'#fb4f14','DET':'#0076b6','GB':'#19866a',
               'HOU':'#c32945','IND':'#3467bf','JAX':'#007f87','KC':'#df2747','LAC':'#0080c6','LAR':'#3869db',
               'LV':'#626d7c','MIA':'#008e97','MIN':'#7650b2','NE':'#c62b49','NO':'#a88d54','NYG':'#3467ba',
               'NYJ':'#147d64','PHI':'#118577','PIT':'#b78a00','SEA':'#4d8f31','SF':'#b53137','TB':'#d52c32',
               'TEN':'#3487bd','WAS':'#9b3144'}


def logo_url(team):
    code={'WAS':'wsh'}.get(team,str(team).lower())
    return f'https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png'


def fixture_key(team,opponent):
    return ' / '.join(sorted([str(team),str(opponent)]))


def fixture_title(key):
    return ' vs '.join(TEAM_NAMES.get(t,t).split()[-1] for t in key.split(' / '))


def fixture_banner(team,opponent,caption=''):
    a,b=html.escape(str(team)),html.escape(str(opponent))
    title=html.escape(TEAM_NAMES.get(team,a)+' vs '+TEAM_NAMES.get(opponent,b))
    color=TEAM_COLORS.get(team,'#087f70')
    st.markdown(f'''<div class="fixture-banner" style="border-left:5px solid {color}">
    <img src="{logo_url(team)}" alt="{a}"/><div><div class="fixture-title">{title}</div>
    <div class="fixture-caption">{html.escape(caption)}</div></div><img src="{logo_url(opponent)}" alt="{b}"/></div>''',unsafe_allow_html=True)


def game_filter_frame(frame,key,banner=True):
    if frame.empty or not {'Team','Opponent'}<=set(frame):
        return frame
    options=sorted({fixture_key(t,o) for t,o in zip(frame.Team,frame.Opponent)})
    chosen=st.selectbox('Game',['All games']+options,format_func=lambda x:x if x=='All games' else fixture_title(x),key=key)
    if chosen=='All games':
        return frame
    selected=frame[[fixture_key(t,o)==chosen for t,o in zip(frame.Team,frame.Opponent)]]
    if banner and not selected.empty:
        r=selected.iloc[0]
        fixture_banner(r.Team,r.Opponent)
    return selected


def game_filter_picks(picks,key):
    if not picks:
        return picks
    frame=game_filter_frame(pd.DataFrame(picks),key)
    ids=set(frame.ID)
    return [p for p in picks if p['ID'] in ids]


def load_record_policy():
    raw=github(RECORD_POLICY_PATH)
    policy=json.loads(raw) if raw else json.loads(json.dumps(DEFAULT_RECORD_POLICY))
    if not isinstance(policy.get('official_start'),dict) or not {'season','week'}<=set(policy['official_start']):
        raise ValueError('Saved record policy is invalid.')
    st.session_state.record_policy=policy
    return policy


def record_week(pick):
    try:
        return int(pick['Season']),int(pick['Week'])
    except (KeyError,TypeError,ValueError):
        return None


def official_pick(pick,policy=None):
    policy=policy or st.session_state.get('record_policy',DEFAULT_RECORD_POLICY)
    week=record_week(pick)
    if week is None:
        return False
    start=policy['official_start']
    return week>=(int(start['season']),int(start['week'])) and f'{week[0]}:{week[1]}' not in policy.get('pilot_weeks',[])


def official_records(picks):
    return [p for p in picks if official_pick(p)]


def record_scope_label():
    p=st.session_state.get('record_policy',DEFAULT_RECORD_POLICY)['official_start']
    return f"Official cumulative record · {p['season']} Week {p['week']} onward"


def record_policy_ui():
    with st.expander('Pilot week and official cumulative records'):
        policy=st.session_state.get('record_policy',DEFAULT_RECORD_POLICY)
        st.write('**Week 6 (2026) is the pilot. It remains in weekly history and is excluded from official cumulative totals.**')
        st.caption(record_scope_label()+'. Picks without season/week metadata stay in history but need review before they can count officially.')
        with st.form('record_policy_form'):
            c1,c2=st.columns(2)
            season=c1.number_input('Official record start season',2026,2100,int(policy['official_start']['season']))
            week=c2.number_input('Official record start week',1,22,int(policy['official_start']['week']))
            save=st.form_submit_button('Save official record starting week')
        if save:
            try:
                target=(int(season),int(week));old=(int(policy['official_start']['season']),int(policy['official_start']['week']))
                if target!=old:
                    data,_=load_tracker()
                    personal_raw=github('data/my_picks.json')
                    all_rows=list(data['picks'].values())+(json.loads(personal_raw) if personal_raw else [])
                    for p in all_rows:
                        w=record_week(p)
                        if w and w>=(2026,7) and started(p) and ((w>=target)!=(w>=old)):
                            raise ValueError('This would change whether started official-week picks count. Choose record rules before games begin.')
                changed={**policy,'official_start':{'season':int(season),'week':int(week)}}
                github(RECORD_POLICY_PATH,json.dumps(changed).encode())
                st.session_state.record_policy=changed
                st.success('Record policy saved. No forecasts or results were deleted.')
            except (RuntimeError,ValueError) as error:
                st.error(str(error))


def load_injuries():
    raw=github(INJURIES_PATH)
    data=json.loads(raw) if raw else {'entries':{},'history':[]}
    if not isinstance(data.get('entries'),dict):
        raise ValueError('Saved injury notes are invalid.')
    st.session_state.injury_data=data
    st.session_state.injury_load_error=None
    return data,hashlib.sha256(raw).hexdigest() if raw else 'missing'


def matching_injury(player,team,season,week):
    entries=st.session_state.get('injury_data',{}).get('entries',{})
    key=player_key(player)
    return next((r for r in entries.values() if r['Key']==key and r['Team']==team and r['Season']==int(season) and r['Week']==int(week)),None)


def game_injury_context(team,opponent,season,week):
    return [dict(note) for note in st.session_state.get('injury_data',{}).get('entries',{}).values()
            if note['Team'] in {team,opponent} and note['Season']==int(season) and note['Week']==int(week)]


def injury_blocks(player,team,meta=None):
    meta=meta or st.session_state.get('model_meta') or st.session_state.get('batch_meta')
    if not meta:
        return False
    note=matching_injury(player,team,meta['season'],meta['week'])
    return bool(note and note.get('Exclude Props'))


def injury_filtered(projections,meta=None):
    if projections.empty:
        return projections
    return projections[[not injury_blocks(r.Player,r.Team,meta) for _,r in projections.iterrows()]].copy()


def injury_for_pick(pick):
    w=record_week(pick)
    return matching_injury(pick['Player'],pick['Team'],*w) if w else None


def sync_injury_exclusions():
    # A saved pick keeps its original line/projection; only pregame eligibility changes.
    data,version=load_tracker()
    changed=False
    for p in data['picks'].values():
        if started(p):
            continue
        note=injury_for_pick(p)
        block=bool(note and note.get('Exclude Props'))
        current=bool(p.get('Injury Excluded'))
        if block!=current:
            p['Injury Excluded']=block
            p.setdefault('Exclusion History',[]).append({'at':utc_now().isoformat(),'source':'injury','excluded':block,'status':note.get('Status') if note else 'cleared'})
            changed=True
    if changed:
        for day in list(data['days']):
            sync_top20(data,day)
        save_tracker(data,version)
    else:
        st.session_state.model_tracker=data


def save_injury(entry):
    entry=dict(entry)
    if entry['Status']=='Out':
        entry['Exclude Props']=True
        entry['Expected workload percentage']=0
    percent=float(entry.get('Expected workload percentage',100))
    if not 0 <= percent <= 100:raise ValueError('Expected workload percentage must be between 0 and 100.')
    data,version=load_injuries()
    identifier=hashlib.sha256(json.dumps([entry['Season'],entry['Week'],entry['Team'],entry['Key']]).encode()).hexdigest()[:24]
    previous=data['entries'].get(identifier)
    entry={**entry,'ID':identifier,'Updated At':utc_now().isoformat()}
    data['history'].append({'at':utc_now().isoformat(),'previous':previous,'new':entry})
    data['entries'][identifier]=entry
    github(INJURIES_PATH,json.dumps(data).encode(),expected_hash=version)
    st.session_state.injury_data=data
    try:
        sync_injury_exclusions()
        st.session_state.injury_sync_error=None
    except (RuntimeError,ValueError) as error:
        st.session_state.injury_sync_error='Note saved, but model-record exclusions still need syncing: '+str(error)


def injuries_page():
    st.subheader('🩺 Weekly injury room')
    st.caption('Expected workload percentage reduces projected opportunities. Removed carries and targets are redistributed separately within the position group using recent usage. Recalculate projections after saving changes.')
    s=st.session_state
    if st.button('Load / refresh injury notes'):
        try:
            load_injuries()
        except (RuntimeError,ValueError) as error:
            st.error(str(error))
    meta=s.model_meta or s.batch_meta or {'season':2026,'week':7}
    c1,c2=st.columns(2)
    season=c1.number_input('Injury season',2026,2100,int(meta['season']),key='injury_season')
    week=c2.number_input('Injury week',1,22,int(meta['week']),key='injury_week')
    fixtures=set()
    if not s.props.empty:
        upcoming=s.props[pd.to_datetime(s.props['Kickoff (UTC)'],utc=True,errors='coerce')>pd.Timestamp.now(tz='UTC')]
        fixtures.update(fixture_key(a,b) for a,b in zip(upcoming.Away,upcoming.Home))
    if int(season)==int(meta['season']) and int(week)==int(meta['week']) and 'defensive_run_splits' in s.tables:
        fixtures.update(fixture_key(r.Team,r.Opponent) for _,r in s.tables['defensive_run_splits'].iterrows() if pd.notna(r.Opponent))
    chosen=st.selectbox('Injury game',['Choose teams manually']+sorted(fixtures),format_func=lambda v: fixture_title(v) if v!='Choose teams manually' else v)
    if chosen=='Choose teams manually':
        c1,c2=st.columns(2)
        a=c1.selectbox('First team',list(TEAM_NAMES),format_func=lambda t:TEAM_NAMES[t],key='injury_team_a')
        b=c2.selectbox('Opponent team',[t for t in TEAM_NAMES if t!=a],format_func=lambda t:TEAM_NAMES[t],key='injury_team_b')
    else:
        a,b=chosen.split(' / ')
    fixture_banner(a,b,f'{season} · Week {week}')
    entries=[r for r in s.get('injury_data',{}).get('entries',{}).values() if r['Season']==season and r['Week']==week and r['Team'] in {a,b}]
    if entries:
        display=pd.DataFrame(entries)[['Player','Team','Position','Status','Exclude Props','Notes']]
        st.dataframe(display,hide_index=True,use_container_width=True)
    else:
        st.info('No injury notes saved for these teams in this week.')
    if not is_admin():
        st.caption('Enter your Admin password to add or change injuries.')
        return
    if s.get('injury_sync_error'):
        st.warning(s.injury_sync_error)
    if st.button('Sync injury exclusions to saved model picks'):
        try:
            sync_injury_exclusions()
            s.injury_sync_error=None
            st.success('Pregame model exclusions synchronized.')
        except (RuntimeError,ValueError) as error:
            st.error(str(error))
    team=st.selectbox('Injured player’s team',[a,b],format_func=lambda t:TEAM_NAMES[t])
    roster=[]
    for kind in ['season_passing','season_rushing','season_receiving']:
        t=s.tables.get(kind)
        if t is not None:
            roster.extend(t[t.Team==team].Player.tolist())
    roster=sorted(set(roster))
    selected=st.selectbox('Player',['Type a player name']+roster,key='injury_roster')
    player=st.text_input('Player name',value='' if selected=='Type a player name' else selected,key='injury_name_'+team+'_'+player_key(selected))
    positions=['QB','RB','WR','TE','OL','CB','S','LB','DL','K','Other']
    note=matching_injury(player,team,season,week) if player.strip() else None
    inferred=None
    for kind in ['season_passing','season_rushing','season_receiving']:
        roster_table=s.tables.get(kind)
        if roster_table is not None:
            matches=roster_table[(roster_table.Team==team)&(roster_table.Player.map(player_key)==player_key(player))]
            if not matches.empty:
                inferred=matches.iloc[0].Position
                break
    default_position=note['Position'] if note else (inferred if inferred in positions else 'Other')
    position=st.selectbox('Position',positions,index=positions.index(default_position) if default_position in positions else positions.index('Other'),key='injury_position_'+team+'_'+player_key(player))
    statuses=['Out','Limited / snap count','Questionable','Missed practice','Returning','Healthy / cleared']
    status=st.selectbox('Status',statuses,index=statuses.index(note['Status']) if note else 0,key='injury_status_'+team+'_'+player_key(player))
    default=note.get('Exclude Props') if note and note['Status']==status else status in {'Out','Limited / snap count'}
    excluded=st.checkbox('Exclude this player’s props this week',value=True if status=='Out' else bool(default),disabled=status=='Out',key='injury_exclude_'+team+'_'+player_key(player)+'_'+status)
    workload_percent=st.number_input('Expected workload percentage',0.0,100.0,0.0 if status=='Out' else float((note or {}).get('Expected workload percentage',50 if status=='Limited / snap count' else 100)),disabled=status=='Out',key='injury_workload_'+team+'_'+player_key(player)+'_'+status)
    notes=st.text_area('Notes / reason',value=note.get('Notes','') if note else '',key='injury_notes_'+team+'_'+player_key(player))
    st.caption('Out always excludes props. Limited defaults to excluded. Questionable and missed practice do not automatically mean Out. Imported lines stay saved; saved personal picks receive an alert.')
    if st.button('Save injury entry',type='primary'):
        try:
            if not player.strip():
                raise ValueError('Enter a player name.')
            save_injury({'Season':int(season),'Week':int(week),'Team':team,'Opponent':b if team==a else a,
                         'Player':player.strip(),'Key':player_key(player),'Position':position,'Status':status,
                         'Exclude Props':bool(excluded),'Expected workload percentage':float(workload_percent),'Notes':notes.strip()})
            st.success('Injury saved for this week only. Pregame exclusions applied. Recalculate projections to update workloads.')
            st.rerun()
        except (RuntimeError,ValueError) as error:
            st.error(str(error))


def weekly_records_table(groups):
    rows=[]
    for title,picks in groups:
        buckets={}
        for p in picks:
            w=record_week(p)
            if w:
                buckets.setdefault(w,[]).append(p)
        for (season,week),part in sorted(buckets.items()):
            counts={r:sum(p['Result']==r for p in part) for r in ['Won','Lost','Push','Void','Pending','Needs review']}
            w,l=counts['Won'],counts['Lost']
            rows.append({'Record':title,'Season':season,'Week':week,'Label':'Official' if official_pick(part[0]) else ('Pilot' if (season,week)==(2026,6) else 'Not in cumulative'),
                         'Wins':w,'Losses':l,'Pushes':counts['Push'],'Pending / Review':counts['Pending']+counts['Needs review'],'Void':counts['Void'],
                         'Hit Rate %':100*w/(w+l) if w+l else float('nan'),
                         'Model Versions':', '.join(sorted({p.get('Model Version','Unknown') for p in part}))})
    return pd.DataFrame(rows)


def manual_prop_entry_ui():
    s = st.session_state
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
                            s.props_dirty=True
                            save_props_snapshot(s.props)
                            st.success('Manual line added and saved permanently.')
                            st.rerun()
            except RuntimeError as e:
                st.error(str(e))

def app_style():
    st.markdown('''<style>
    .block-container {padding:1.5rem 2rem 4rem;max-width:1650px;}
    [data-testid="stAppViewContainer"] {background:linear-gradient(180deg,rgba(14,165,233,.065),transparent 600px);}
    .hero-panel {background:linear-gradient(115deg,#082f49 0%,#164e63 60%,#087f70 100%);color:white;border-radius:20px;padding:28px 32px;margin-bottom:24px;box-shadow:0 10px 26px #082f491f;}
    .hero-eyebrow {color:#99f6e4;font-size:13px;font-weight:800;letter-spacing:2px;text-transform:uppercase;}
    .hero-title {color:#fff;font-size:clamp(28px,4vw,42px);font-weight:850;line-height:1.2;margin:8px 0;}
    .hero-caption {color:#d5f3f4;font-size:17px;line-height:1.6;}
    .fixture-banner {display:flex;align-items:center;gap:18px;background:linear-gradient(105deg,#f0f9ff,#f0fdfa);border-radius:14px;padding:16px 22px;margin:15px 0;color:#0c354b;box-shadow:0 3px 14px #082f490c;}
    .fixture-banner img {width:64px;height:64px;object-fit:contain;}
    .fixture-banner div:nth-child(2){flex:1;}
    .fixture-title {font-size:23px;font-weight:800;line-height:1.25;}
    .fixture-caption {font-size:14px;color:#486b7c;margin-top:6px;}
    [data-testid="stMetric"] {background:linear-gradient(120deg,#102d48,#155e75);padding:18px;border-radius:14px;border-bottom:4px solid #2dd4bf;margin:5px 0 14px;}
    [data-testid="stMetric"] * {color:#f0fdfa !important;}
    [data-testid="stMetricValue"] {font-size:32px;font-weight:800;}
    h2,h3 {letter-spacing:-.4px;}
    [data-testid="stMarkdownContainer"] p,[data-testid="stWidgetLabel"] p {font-size:16px;line-height:1.55;}
    button[data-baseweb="tab"] {font-size:16px;font-weight:700;padding:12px 16px;}
    [data-baseweb="tab-list"] {gap:6px;flex-wrap:wrap;}
    button[data-baseweb="tab"][aria-selected="true"] {color:#087f70;background:#ccfbf1;border-radius:10px 10px 0 0;}
    .stButton>button[kind="primary"] {background:#087f70;border-color:#087f70;border-radius:10px;font-weight:750;min-height:44px;}
    [data-testid="stExpander"] {border-radius:13px;}
    [data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;border:1px solid #cbd5e144;}
    @media(max-width:700px){.block-container{padding:1rem .8rem;}.fixture-banner{padding:12px;gap:10px;}.fixture-title{font-size:18px;}.fixture-banner img{width:42px;height:42px;}.hero-panel{padding:20px;}}
    </style>''',unsafe_allow_html=True)
    st.markdown('''<div class="hero-panel"><div class="hero-eyebrow">NFL · Matchup research</div>
    <div class="hero-title">NFL Matchup Analytics</div>
    <div class="hero-caption">Explore the matchup. Choose your side. Track what happens.</div></div>''',unsafe_allow_html=True)


# v7 inputs are optional for migration; missing layers are reported, never invented.
EXTRA_DATASETS = {
    'QB under pressure': 'pressure_qb', 'Offensive pressure allowed': 'pressure_offense',
    'Defensive pressure generated': 'pressure_defense', 'Receiver air yards / YAC': 'receiver_ngs',
    'Defensive box alignment': 'box_defense', 'RB vs fewer than 7': 'box_light',
    'RB vs exactly 7': 'box_seven', 'RB vs 8+': 'box_heavy',
    'Favorable WR–CB matchups': 'matchup_best', 'Unfavorable WR–CB matchups': 'matchup_worst',
    'Weekly workload logs (optional fallback)': 'weekly_workload',
}
EXTRA_KINDS = set(EXTRA_DATASETS.values())
DATASETS.update(EXTRA_DATASETS)


def parse_extra(raw, kind):
    rows = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
    if kind == 'weekly_workload':
        d = pd.read_csv(io.BytesIO(raw))
        required = {'season','week','player_display_name','team','attempts','carries','targets'}
        if not required <= set(d):raise ValueError('Weekly logs require: '+', '.join(sorted(required)))
        return d
    if kind in {'pressure_offense','pressure_defense'}:
        records=[]
        # Supports the supplied stacked rank / team / rate layout and ordinary tables.
        for i,row in enumerate(rows):
            if len(row)>1 and row[1].strip().endswith('%') and i:
                code=team_code(rows[i-1][1])
                if code:
                    records.append({'Team':code,'Pressure Rate':float(row[1].strip('%'))/100})
        if not records:
            h=next((i for i,r in enumerate(rows) if 'Team' in r and '2026' in r),None)
            if h is None:raise ValueError('Cannot find Team and season pressure columns.')
            names=rows[h];ti,vi=names.index('Team'),names.index('2026')
            for row in rows[h+1:]:
                if len(row)>max(ti,vi) and team_code(row[ti]):
                    records.append({'Team':team_code(row[ti]),'Pressure Rate':float(row[vi].strip('%'))/100})
        d=pd.DataFrame(records)
        if d.empty or d.Team.duplicated().any() or not d['Pressure Rate'].between(0,1).all():
            raise ValueError('Need unique teams and valid pressure percentages.')
        return d
    anchor = 'PLAYER NAME' if kind=='receiver_ngs' else ('Team' if kind=='box_defense' else 'Player')
    h=next((i for i,r in enumerate(rows) if anchor in r),None)
    if h is None:raise ValueError('Cannot find '+anchor+' header.')
    names=[]
    for i,name in enumerate(rows[h]):
        name=name.strip() or 'Unused '+str(i)
        if name in names:name+=' Opponent'
        names.append(name)
    d=pd.DataFrame([r[:len(names)]+['']*max(0,len(names)-len(r)) for r in rows[h+1:] if any(r)],columns=names)
    # User requested the malformed NGS error column be ignored completely.
    d=d.drop(columns=[c for c in d if '#ERROR' in c],errors='ignore')
    if kind=='receiver_ngs':d=d.rename(columns={'PLAYER NAME':'Player','TEAM':'Team','POS':'Position'})
    if kind=='box_defense':
        required={'Team','Light Box Rate','Heavy Box Rate'}
    elif kind.startswith('box_'):required={'Player','Rushes','Rush Yards','Yards Per Carry'}
    elif kind=='pressure_qb':required={'Player','Plays','ADoT','YPA','Pass Yards','Comp %'}
    elif kind.startswith('matchup_'):required={'Player','Team','Team Opponent','Projected Matchup','Weekly Matchup Score'}
    else:required={'Player','Team','TAY','REC','TAR','YAC/R','xYAC/R'}
    if not required <= set(d):raise ValueError('Missing columns: '+', '.join(sorted(required-set(d))))
    numeric_anchor = 'Light Box Rate' if kind=='box_defense' else ('Rushes' if kind.startswith('box_') else ('Plays' if kind=='pressure_qb' else ('Weekly Matchup Score' if kind.startswith('matchup_') else 'TAR')))
    d=d[number(d[numeric_anchor]).notna()].copy()
    textcols={'Player','Team','Team Opponent','Position','Projected Matchup'}
    for c in d:
        if c not in textcols:d[c]=number(d[c])
    for c in ['Team','Team Opponent']:
        if c in d:
            d[c]=d[c].map(team_code)
            if d[c].isna().any():raise ValueError('Unrecognized team in '+c)
    if 'Player' in d:
        d['Player']=d.Player.str.replace(r'^\s*\d+\s*','',regex=True).str.strip()
        d['Key']=d.Player.map(player_key)
        if not kind.startswith('matchup_') and d.Key.duplicated().any():raise ValueError('Duplicate player rows.')
    if kind=='box_defense':
        if d.Team.duplicated().any():raise ValueError('Duplicate teams.')
        if not (d['Light Box Rate'].between(0,100)&d['Heavy Box Rate'].between(0,100)&((d['Light Box Rate']+d['Heavy Box Rate'])<=100)).all():raise ValueError('Invalid box percentages.')
    if d.empty:raise ValueError('No usable rows.')
    return d.reset_index(drop=True)


@st.cache_data(ttl=3600,show_spinner=False)
def recent_workload_feed(season):
    stats=download_csv(f'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv')
    stats.attrs['participation_note']='Snap-count supplement unavailable; recent workload uses recorded statistical games.'
    try:
        snaps=download_csv(f'https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv')
        if not {'season','week','team','player','offense_snaps','game_type'}<=set(snaps):return stats
        snaps=snaps[(number(snaps.season)==season)&snaps.game_type.eq('REG')&(number(snaps.offense_snaps)>0)].copy()
        # Fill a zero-opportunity game only after that team-week has published stats.
        teamweeks={(team_code(r.team),int(r.week)) for _,r in stats.iterrows()}
        seen={(player_key(r.player_display_name),team_code(r.team),int(r.week)) for _,r in stats.iterrows()}
        additions=[]
        for _,r in snaps.iterrows():
            identity=(player_key(r.player),team_code(r.team),int(r.week))
            if (identity[1],identity[2]) not in teamweeks or identity in seen:continue
            additions.append({'player_display_name':r.player,'team':r.team,'season':season,'week':int(r.week),'season_type':'REG','attempts':0,'carries':0,'targets':0})
            seen.add(identity)
        if additions:stats=pd.concat([stats,pd.DataFrame(additions)],ignore_index=True)
        stats.attrs['participation_note']=''
    except (RuntimeError,ValueError,KeyError,TypeError):pass
    return stats


def ensure_recent_workloads(season,cutoff,tables):
    if 'weekly_workload' in tables:
        st.session_state.workload_feed=tables['weekly_workload']
        st.session_state.workload_feed_error=None
        return
    try:
        st.session_state.workload_feed=recent_workload_feed(season)
        st.session_state.workload_feed_error=None
    except (RuntimeError,ValueError,OSError) as error:
        st.session_state.workload_feed=pd.DataFrame()
        st.session_state.workload_feed_error=str(error)


def availability(key,team):
    s=st.session_state;meta=s.get('batch_meta') or s.get('model_meta') or {}
    for note in s.get('injury_data',{}).get('entries',{}).values():
        if note.get('Key')==key and note.get('Team')==team and note.get('Season')==meta.get('season') and note.get('Week')==meta.get('week'):
            if note.get('Status')=='Out':return 0.0
            return max(0,min(1,float(note.get('Expected workload percentage',100))/100))
    return 1.0


def projected_workloads(tables,cutoff):
    result={};notes=[];roster={};recent={}
    s=st.session_state;meta=s.get('batch_meta') or s.get('model_meta') or {}
    logs=tables.get('weekly_workload',s.get('workload_feed',pd.DataFrame())).copy()
    fields={'attempts','carries','targets'}
    required={'season','week','player_display_name','team'}|fields
    if logs.attrs.get('participation_note'):
        notes.append({'Player / Team':'Workload','Issue':logs.attrs['participation_note']})
    if not logs.empty and required<=set(logs):
        logs=logs[(number(logs.season)==int(meta.get('season',2026)))&(number(logs.week)<=cutoff)&(number(logs.week)>0)].copy()
        if 'season_type' in logs:logs=logs[logs.season_type.eq('REG')]
        logs['Key']=logs.player_display_name.map(player_key);logs['Team']=logs.team.map(team_code)
        # A duplicated player-week is ambiguous; never count the game twice.
        logs=logs.drop_duplicates(['Key','week'],keep=False)
        for (key,team),g in logs.groupby(['Key','Team']):
            g=g.sort_values('week').tail(3)
            for field in fields:
                values=number(g[field])
                if values.notna().all() and (values>=0).all():recent[key,team,field]=float(values.mean())
    else:notes.append({'Player / Team':'Workload','Issue':'Weekly logs unavailable or invalid: season-only workload fallback. Upload weekly workload logs or retry public feed.'})
    for kind,field,col in [('season_passing','attempts','ATT'),('season_rushing','carries','ATT'),('season_receiving','targets','TGTS')]:
        for _,r in tables[kind].iterrows():
            if pd.isna(r.GP) or r.GP<=0 or r.GP>cutoff or pd.isna(r[col]):continue
            baseline=float(r[col]/r.GP);last=recent.get((r.Key,r.Team,field))
            amount=baseline if last is None else .5*baseline+.5*last
            result[r.Key,r.Team,field]=amount*availability(r.Key,r.Team)
            roster[r.Key,r.Team,field]=(r.Position,amount,last if last is not None else baseline)
            if last is None:notes.append({'Player / Team':r.Player,'Issue':'No matching recent '+field+' logs; season workload used.'})
    # Released opportunities go only to healthy same-position teammates, weighted by recent use.
    for team in TEAM_NAMES:
        for position in ['RB','WR','TE']:
            for field in ['carries','targets']:
                group=[(k,v) for k,v in roster.items() if k[1]==team and k[2]==field and v[0]==position]
                for donor,donor_data in group:
                    released=donor_data[1]*(1-availability(donor[0],team))
                    if not released:continue
                    eligible=[(k,v) for k,v in group if k!=donor and availability(k[0],team)>0]
                    total=sum(v[2]*availability(k[0],team) for k,v in eligible)
                    if total>0:
                        for k,v in eligible:result[k]+=released*v[2]*availability(k[0],team)/total
                    else:notes.append({'Player / Team':team+' '+position,'Issue':'No eligible teammate usage to allocate released '+field+'; left unassigned.'})
    return result,notes


def apply_matchup_layers(result,tables,issues):
    if result.empty:return result
    def warn(who,msg):issues.append({'Player / Team':who,'Issue':msg})
    def lookup(kind,key,column='Key'):
        d=tables.get(kind)
        if d is None:return None
        m=d[d[column]==key]
        return None if m.empty else m.iloc[0]
    pressure={}
    for _,r in tables['season_passing'].iterrows():
        offense=lookup('pressure_offense',r.Team,'Team')
        # Opponent comes from the verified current-week source.
        run=lookup('defensive_run_splits',r.Team,'Team')
        defense=None if run is None else lookup('pressure_defense',run.Opponent,'Team')
        qb=lookup('pressure_qb',r.Key)
        if offense is None or defense is None or qb is None or not r.ATT>0:continue
        usual=float(offense['Pressure Rate']);expected=.5*(usual+float(defense['Pressure Rate']))
        baseline=r.YDS/r.ATT
        if not baseline>0 or not pd.notna(qb.YPA) or usual>=.95:continue
        # Approximation: team pressured-dropback share proxies this QB's pressured-attempt share.
        # Compare to the usual baseline rather than applying pressure a second time.
        reliability=max(0,float(qb.Plays))/(max(0,float(qb.Plays))+20)
        sensitivity=(float(qb.YPA)/baseline-1)/max(.05,1-usual)
        adjustment=max(-.127,min(.07,(expected-usual)*sensitivity*reliability))
        pressure[r.Team]={'usual':usual,'expected':expected,'delta':expected-usual,'adjustment':adjustment,'adot':qb.ADoT}
    for i,r in result.iterrows():
        original=float(r.Projection);value=original
        result.loc[i,'Pressure Adjustment']=0.0
        result.loc[i,'Box Weight']=0.0
        if r.Position=='RB' and r.Stat=='Rushing yards':
            defense=lookup('box_defense',r.Opponent,'Team')
            splits=[lookup(k,r.Key) for k in ['box_light','box_seven','box_heavy']]
            season=lookup('season_rushing',r.Key)
            if defense is not None and season is not None and season.ATT>0 and all(x is not None for x in splits):
                overall=season.YDS/season.ATT
                light=float(defense['Light Box Rate'])/100;heavy=float(defense['Heavy Box Rate'])/100
                weights=[light,1-light-heavy,heavy]
                rates=[(x['Rush Yards']+20*overall)/(x.Rushes+20) for x in splits]
                if all(math.isfinite(float(x)) and x>=0 for x in rates):
                    box_ypc=sum(w*v for w,v in zip(weights,rates))
                    value=.68*value+.32*float(r['Workload / Game'])*box_ypc
                    result.loc[i,'Box Weight']=.32
                    result.loc[i,'Calculation']+='; 68% scheme / 32% box; box splits use 20-carry priors'
            else:warn(r.Player,'Box split/alignment unavailable; full scheme estimate retained.')
        context=pressure.get(r.Team)
        if r.Stat=='Passing yards':
            if context:
                value*=1+context['adjustment']
                result.loc[i,'Pressure Adjustment']=context['adjustment']
                result.loc[i,'Calculation']+=f"; pressure {context['usual']:.1%} → {context['expected']:.1%}; adjustment {context['adjustment']:+.1%}; team dropback-rate proxy"
            else:warn(r.Player,'Pressure inputs unavailable; no pressure adjustment.')
        elif r.Position in {'WR','TE'} and r.Stat in {'Receiving yards','Receptions'}:
            ngs=lookup('receiver_ngs',r.Key)
            peers=tables.get('receiver_ngs')
            if context and ngs is not None and ngs.Team==r.Team and pd.notna(context['adot']) and peers is not None:
                # Role-fit heuristic, not observed receiver production on pressured plays.
                # Weighted team target depth is the usual reference; no QB-total scaling.
                teammates=peers[peers.Team==r.Team]
                usual_depth=(teammates.TAY*teammates.TAR).sum()/max(1,teammates.TAR.sum())
                shift=context['delta']*(float(context['adot'])-usual_depth)
                depth_fit=max(-1,min(1,(usual_depth-float(ngs.TAY))/8))
                opportunities=max(-.06,min(.06,-shift*depth_fit*.15))
                adjustment=opportunities
                if r.Stat=='Receiving yards':
                    # Reward creating YAC beyond expectation only when modeled short opportunities increase.
                    excess=float(ngs['YAC/R'])-float(ngs['xYAC/R'])
                    adjustment+=abs(opportunities)*max(-.5,min(.5,excess/5))
                adjustment=max(-.08,min(.08,adjustment))
                value*=1+adjustment
                result.loc[i,'Pressure Adjustment']=adjustment
                result.loc[i,'Calculation']+='; individual target-depth/YAC fit heuristic; no QB-total scaling'
        result.loc[i,'Projection']=round(max(0,value),4)
        result.loc[i,'Matchup Rating']=matchup_rating(max(0,value),r['Neutral Projection'])
    return result


def matchup_candidates():
    s=st.session_state;meta=s.get('model_meta') or {}
    lines=prop_comparisons(s.projections,s.props)
    records=[]
    for kind,desired in [('matchup_best','Over'),('matchup_worst','Under')]:
        matchups=s.tables.get(kind,pd.DataFrame())
        for _,m in matchups.iterrows():
            matches=lines[(lines.Key==m.Key)&(lines.Team==m.Team)&(lines.Opponent==m['Team Opponent'])&(lines.Stat=='Receiving yards')&(lines.Bookmaker=='DraftKings')] if not lines.empty else pd.DataFrame()
            identifier=hashlib.sha256(json.dumps([meta.get('season'),meta.get('week'),m.Key,m.Team,desired]).encode()).hexdigest()[:24]
            record={'Review ID':identifier,'Player':m.Player,'Team':m.Team,'Opponent':m['Team Opponent'],'Cornerback':m['Projected Matchup'],'Matchup Score':m['Weekly Matchup Score'],'Suggested Side':desired,'Source':kind,'Row':None,'Status':'No matching model projection / DraftKings prop'}
            if not matches.empty:
                row=matches.sort_values('Fetched',ascending=False).iloc[0]
                record['Row']=row
                if started({'Kickoff (UTC)':row['Kickoff (UTC)']}):record['Status']='Game started — locked'
                elif injury_blocks(row.Player,row.Team):record['Status']='Injury exclusion — blocked'
                else:
                    direction=pick_direction(row.Stat,float(row.Projection),float(row.Line))
                    record['Status']='Agrees' if direction==desired else 'Review required'
            records.append(record)
    # Contradictory lists must be reviewed; never automatically save opposing picks.
    byplayer={}
    for item in records:byplayer.setdefault((item['Team'],player_key(item['Player'])),set()).add(item['Suggested Side'])
    for item in records:
        if len(byplayer[item['Team'],player_key(item['Player'])])>1 and item['Row'] is not None and item['Status'] in {'Agrees','Review required'}:
            item['Status']='Conflicting lists — review required'
    return records


def save_matchup_pick(item,override=None):
    s=st.session_state;row=item['Row'];meta=s.model_meta
    if row is None or started({'Kickoff (UTC)':row['Kickoff (UTC)']}):raise ValueError('Need an upcoming matched prop.')
    if s.get('injury_load_error') or injury_blocks(row.Player,row.Team):raise ValueError('Injury eligibility blocks this pick.')
    projection=float(row.Projection) if override is None else float(override)
    side=item['Suggested Side']
    direction=pick_direction(row.Stat,projection,float(row.Line))
    if direction!=side:raise ValueError('Adjusted projection must clear the configured no-pick range on the chosen side.')
    raw=github('data/my_picks.json');picks=json.loads(raw) if raw else []
    if any(p.get('Event ID')==row['Event ID'] and player_key(p['Player'])==row.Key and p['Stat']==row.Stat and p.get('Your Pick')==side for p in picks):return False
    lines=s.props.copy()
    lines['Key']=lines.Player.map(player_key)
    matching=lines[(lines['Event ID']==row['Event ID'])&(lines.Key==row.Key)&(lines.Stat==row.Stat)&(lines.Line==row.Line)&(lines.Side==side)&(lines.Bookmaker==row.Bookmaker)]
    odds=None if matching.empty else float(matching.iloc[0].Odds)
    pid=hashlib.sha256(json.dumps([row['Event ID'],row.Key,row.Stat,float(row.Line),row.Bookmaker,side]).encode()).hexdigest()[:24]
    record={'ID':pid,'Player':row.Player,'Position':row.Position,'Team':row.Team,'Opponent':row.Opponent,'Stat':row.Stat,'Line':float(row.Line),'Your Pick':side,'Projection When Picked':projection,'Original Model Projection':float(row.Projection),'Manual Projection Override':override is not None,'Pick Source':item['Source'],'Matchup Score':float(item['Matchup Score']),'Cornerback':item['Cornerback'],'Odds':odds,'Bookmaker':row.Bookmaker,'Event ID':row['Event ID'],'Kickoff (UTC)':row['Kickoff (UTC)'],'Saved At':utc_now().isoformat(),'Result':'Pending','Actual':None,'Model Version':MODEL_VERSION,'Result History':[],'Season':meta['season'],'Week':meta['week'],'Day':local_day(row['Kickoff (UTC)']),'Manual Grade':False}
    picks.append(record)
    github('data/my_picks.json',json.dumps(picks,allow_nan=False).encode(),expected_hash=hashlib.sha256(raw).hexdigest() if raw else 'missing')
    s.my_picks=picks
    return True


def matchup_review_ui():
    s=st.session_state
    if not any(k in s.tables for k in ['matchup_best','matchup_worst']):return
    st.subheader('WR–CB matchup review')
    st.caption('All uploaded matchup rows are included. Receiving yards only. Agreements outside the no-pick range are added to My Picks automatically when this section loads with Admin access. Overrides affect personal picks only; original model forecasts stay unchanged.')
    if s.projections.empty or s.props.empty:
        st.info('Calculate projections and load saved DraftKings props to match these lists.');return
    if not is_admin():st.info('Admin access is required to add or review these picks.');return
    try:
        raw=github('data/matchup_decisions.json')
        decisions=json.loads(raw) if raw else {}
        items=matchup_candidates();added=0
        for item in items:
            if item['Status']=='Agrees' and item['Review ID'] not in decisions:
                added+=int(save_matchup_pick(item))
        if added:st.success(f'Added {added} matchup-supported picks to My Picks.')
        display=[]
        for item in items:
            row=item['Row']
            display.append({k:v for k,v in item.items() if k not in {'Row','Review ID'}}|{'Model Projection':None if row is None else row.Projection,'Line':None if row is None else row.Line,'Decision':decisions.get(item['Review ID'],{}).get('Action','')})
        st.dataframe(pd.DataFrame(display),hide_index=True,use_container_width=True)
        pending=[x for x in items if x['Status'] in {'Review required','Conflicting lists — review required'} and x['Review ID'] not in decisions]
        if not pending:return
        choice=st.selectbox('Matchup to review',range(len(pending)),format_func=lambda i:pending[i]['Player']+' · '+pending[i]['Suggested Side'])
        item=pending[choice];row=item['Row']
        st.write(f"Model **{row.Projection:.1f}** · Line **{row.Line:g}** · Suggested **{item['Suggested Side']}**")
        value=st.number_input('Your adjusted receiving-yard projection',min_value=0.0,value=float(row.Projection),step=1.0,key='match_override_'+item['Review ID'])
        a,b=st.columns(2)
        keep=a.button('Leave model projection as is',key='match_keep_'+item['Review ID'])
        override=b.button('Save adjusted projection and personal pick',key='match_save_'+item['Review ID'])
        if keep or override:
            if override:save_matchup_pick(item,value)
            decisions[item['Review ID']]={'Action':'Keep original' if keep else 'Personal override','Original Projection':float(row.Projection),'Adjusted Projection':None if keep else value,'At':utc_now().isoformat()}
            github('data/matchup_decisions.json',json.dumps(decisions).encode(),expected_hash=hashlib.sha256(raw).hexdigest() if raw else 'missing')
            st.rerun()
    except (RuntimeError,ValueError,KeyError) as error:st.error('Matchup review: '+str(error))



# Separate browser file reader: sends selected bytes through the component channel,
# rather than relying on Streamlit's native file-upload widget or upload endpoint.
CSV_READER_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><style>
body{font:16px system-ui,sans-serif;margin:0;color:#15354b;background:transparent}
#zone{border:2px dashed #0d9488;border-radius:14px;background:#f0fdfa;padding:20px;text-align:center}
#zone.active{background:#ccfbf1;border-style:solid}button{font:600 16px system-ui;background:#087f70;color:white;padding:12px 20px;border:0;border-radius:8px;cursor:pointer}
p{margin:10px 0}#status{font-size:14px;white-space:pre-wrap;margin-top:10px}button:disabled{opacity:.5}
</style></head><body><section id="zone"><strong>Alternative CSV file upload</strong>
<p>Drag downloaded CSV files into this box</p><button id="choose" type="button">Choose CSV files</button>
<input id="files" type="file" accept=".csv,text/csv" multiple hidden>
<div id="status" role="status" aria-live="polite">Files stay in your app. No copying or pasting needed.</div></section>
<script>
const input=document.getElementById('files'),zone=document.getElementById('zone'),status=document.getElementById('status'),button=document.getElementById('choose');
function send(type,extra={}){window.parent.postMessage({isStreamlitMessage:true,type,...extra},'*')}
function height(){send('streamlit:setFrameHeight',{height:Math.max(235,document.body.scrollHeight+12)})}
function read(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=()=>reject(new Error('Could not read '+file.name));reader.readAsDataURL(file)})}
async function accept(files){
 const list=Array.from(files||[]);if(!list.length)return;
 try{
  if(list.length>40)throw new Error('Select no more than 40 files at once.');
  for(const file of list){if(!file.name.toLowerCase().endsWith('.csv'))throw new Error(file.name+': select a .csv file.');if(file.size>500000)throw new Error(file.name+': CSV must be under 500 KB.');if(!file.size)throw new Error(file.name+': file is empty.');}
  button.disabled=true;status.textContent='Reading '+list.length+' file(s)…';height();
  const output=[];for(const file of list)output.push({name:file.name,size:file.size,data:await read(file)});
  send('streamlit:setComponentValue',{value:{nonce:String(Date.now())+'-'+Math.random().toString(36).slice(2),files:output},dataType:'json'});
  status.textContent='Read '+output.length+' file(s):\n'+output.map(f=>f.name).join('\n')+'\nReview the previews below, then click Accept previewed files into this batch.';
 }catch(error){status.textContent=error.message;}finally{button.disabled=false;input.value='';height();}
}
button.addEventListener('click',()=>input.click());input.addEventListener('change',()=>accept(input.files));
for(const event of ['dragenter','dragover'])zone.addEventListener(event,e=>{e.preventDefault();e.stopPropagation();if(e.dataTransfer)e.dataTransfer.dropEffect='copy';zone.classList.add('active')});
zone.addEventListener('dragleave',()=>zone.classList.remove('active'));
zone.addEventListener('drop',e=>{e.preventDefault();e.stopPropagation();zone.classList.remove('active');accept(e.dataTransfer.files)});
window.addEventListener('message',e=>{if(e.source===window.parent&&e.data&&e.data.type==='streamlit:render'){button.disabled=Boolean(e.data.disabled);height();}});
send('streamlit:componentReady',{apiVersion:1});height();
</script></body></html>'''


@st.cache_resource(show_spinner=False)
def csv_reader_component(source):
    import tempfile
    from pathlib import Path
    import streamlit.components.v1 as components
    directory=Path(tempfile.mkdtemp(prefix='nfl-csv-reader-'))
    (directory/'index.html').write_text(source,encoding='utf-8')
    return components.declare_component('nfl_csv_reader',path=str(directory))


def decode_browser_csvs(payload):
    if not isinstance(payload,dict) or not isinstance(payload.get('files'),list):
        raise ValueError('Invalid file selection. Try choosing the CSVs again.')
    if not 1<=len(payload['files'])<=40:raise ValueError('Choose between 1 and 40 CSV files.')
    files=[]
    for item in payload['files']:
        if not isinstance(item,dict):raise ValueError('Invalid selected file.')
        name=str(item.get('name','')).replace('\\','/').rsplit('/',1)[-1]
        if not name.lower().endswith('.csv'):raise ValueError('Only .csv files are supported.')
        encoded=item.get('data','')
        if not isinstance(encoded,str) or len(encoded)>666668:raise ValueError(name+': CSV must be under 500 KB.')
        try:raw=base64.b64decode(encoded,validate=True)
        except (ValueError,TypeError):raise ValueError(name+': file could not be decoded. Select it again.') from None
        if not raw or len(raw)>500000:raise ValueError(name+': select a nonempty CSV under 500 KB.')
        if item.get('size')!=len(raw):raise ValueError(name+': incomplete file transfer. Select it again.')
        file=io.BytesIO(raw);file.name=name;file.size=len(raw);files.append(file)
    return files


def alternate_csv_uploads():
    s=st.session_state
    st.caption('If Browse files does nothing, use the independent file-upload box below. It accepts your downloaded CSV files directly.')
    if st.button('Reset file upload boxes',key='reset_csv_readers'):
        s.csv_reader_generation=s.get('csv_reader_generation',0)+1
        s.pop('browser_csv_files',None);s.pop('browser_csv_nonce',None)
        st.rerun()
    component=csv_reader_component(CSV_READER_HTML)
    payload=component(key='csv_reader_'+str(s.get('csv_reader_generation',0)),default=None)
    if payload and payload.get('nonce')!=s.get('browser_csv_nonce'):
        try:
            s.browser_csv_files=decode_browser_csvs(payload)
            s.browser_csv_nonce=payload.get('nonce')
        except (ValueError,KeyError,TypeError) as error:
            st.error(str(error));return []
    return s.get('browser_csv_files',[])


def main():
    st.set_page_config(page_title='NFL Matchup Analytics', page_icon='🏈', layout='wide')
    for key, value in {'tables': {}, 'source_bytes': {}, 'projections': pd.DataFrame(), 'issues': pd.DataFrame(),
                       'props': pd.DataFrame(), 'batch_meta': None, 'model_meta': None}.items():
        if key not in st.session_state:
            st.session_state[key] = value
    s = st.session_state
    if not s.get('auto_restore_attempted'):
        s.auto_restore_attempted = True
        if secret('GITHUB_TOKEN') and secret('GITHUB_REPO'):
            try:
                restore_saved_state()
            except (RuntimeError, ValueError, KeyError, OSError) as error:
                s.restore_error = str(error)
    if s.get('injury_load_error'):
        st.warning('Injury notes did not load. Refresh them in Injuries before saving new selections.')
    if s.get('restore_error'):
        st.warning('Saved data could not fully restore: '+s.restore_error+'. Use Reload saved data and props to retry.')
    app_style()
    weekly, all_model_tab, picks_tab, best_tab, dashboard_tab, explorer, injuries_tab, methodology, admin = st.tabs([
        'Projections & Matchups', 'All Model Picks', 'My Picks', '20 Best Picks of the Day', 'Records Dashboard', 'Pick Explorer', 'Injuries', 'Methodology', 'Admin / Update Data'])

    with admin:
        st.text_input('Admin password', type='password', key='admin_password')
        if not secret('ADMIN_PASSWORD'):
            st.warning('Add ADMIN_PASSWORD to Streamlit Secrets.')
        elif s.admin_password and not is_admin():
            st.error('Incorrect password.')
        if is_admin():
            st.success('Admin access enabled.')
            if not s.get('settings_checked'):
                try:
                    load_analysis_settings()
                except (RuntimeError, ValueError) as error:
                    st.warning('Could not load saved thresholds: ' + str(error))
                s.settings_checked = True
            settings_ui()
            record_policy_ui()
            persistence_controls()
            st.subheader('Weekly data batch')
            st.caption('Upload the original season files plus the new pressure, box and matchup files. Public weekly logs supply recent workloads without Odds API credits.')
            c1, c2, c3 = st.columns(3)
            season = c1.number_input('Season', 2020, 2100, int((s.batch_meta or {}).get('season',2026)))
            cutoff = c2.number_input('Stats through week', 1, 18, int((s.batch_meta or {}).get('cutoff',6)))
            week = c3.number_input('Project week', 1, 22, int((s.batch_meta or {}).get('week',7)))
            label = st.selectbox('CSV category', list(DATASETS))
            kind = DATASETS[label]
            uploads = list(st.file_uploader('Upload CSVs (one or several)', type=['csv'], accept_multiple_files=True, key='native_csv_'+str(s.get('csv_reader_generation',0))) or [])
            uploads += alternate_csv_uploads()
            name_map = {
                'qbs under pressure':'pressure_qb', 'official offensive pressure':'pressure_offense',
                'defensive pressure':'pressure_defense','average targeted air':'receiver_ngs',
                'defenses alignment':'box_defense','fewer than 7':'box_light','7 in box':'box_seven',
                '8+ in box':'box_heavy','top 20 matchups':'matchup_best','bottom 20 matchups':'matchup_worst',
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
                try:
                    if s.source_bytes and s.batch_meta != meta:
                        raise ValueError('Clear weekly CSVs explicitly before starting a different week.')
                    for k, (raw, table) in pending.items():
                        s.tables[k], s.source_bytes[k] = table, raw
                    s.batch_meta = meta
                    s.projections,s.issues = pd.DataFrame(),pd.DataFrame()
                    s.model_meta = None
                    s.csv_dirty,s.csv_confirmed = True,False
                    save_csv_batch(meta,s.source_bytes,confirmed=False)
                    st.success('Accepted and saved permanently. These CSVs will restore after refreshing.')
                except (RuntimeError,ValueError) as error:
                    st.error(str(error))
            st.write(f'**Accepted datasets:** {len(s.tables)} / {len(DATASETS)}')
            st.write(', '.join(sorted(s.tables)) or 'None yet')
            confirmed = st.checkbox('All accepted files share the displayed season and cutoff; opponents are for the projection week.',value=s.get('csv_confirmed',False))
            if st.button('Validate and calculate projections', disabled=not bool(s.tables)):
                try:
                    meta = {'season': int(season), 'cutoff': int(cutoff), 'week': int(week)}
                    if not confirmed or s.batch_meta != meta or week <= cutoff:
                        raise ValueError('Confirm the batch dates. Projection week must follow the cutoff; reaccept files if dates changed.')
                    s.batch_meta = meta
                    ensure_recent_workloads(int(season), int(cutoff), s.tables)
                    p, issues = build_model(s.tables, int(cutoff))
                    s.projections, s.issues = p, issues
                    s.model_meta = {**meta, 'model_version': MODEL_VERSION, 'defense_weight': 0.632,
                                    'calculated_at': datetime.now(timezone.utc).isoformat()}
                    st.success(f'Calculated {len(p)} projections. Review Data checks before publishing.')
                    s.register_slate = True
                    s.csv_dirty = True
                    save_csv_batch(meta,s.source_bytes,confirmed=True)
                except (RuntimeError,ValueError, KeyError) as e:
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
                force_fresh = st.checkbox('Fetch fresh lines for games already saved (may use credits)',value=False)
                st.caption('Saved game props are reused by default, even after a refresh. Only unsaved games call The Odds API unless you choose fresh lines. Every successful fetch saves automatically.')
                if st.button('Import selected games', disabled=not selected):
                    imported,reused,errors=import_games_safely(selected,force_fresh)
                    st.success(f'{imported} games fetched; {reused} games reused from saved lines.')
                    if imported or reused:
                        s.register_slate=True
                    if s.get('last_credits_remaining') is not None:
                        st.caption('Credits remaining when last fetched: '+str(s.last_credits_remaining))
                    for error in errors:
                        st.error(error)
                if not s.props.empty:
                    games_saved=s.props[['Event ID','Matchup','Fetched']].drop_duplicates('Event ID',keep='last')
                    st.dataframe(games_saved[['Matchup','Fetched']],hide_index=True,use_container_width=True)
            except RuntimeError as e:
                st.error(str(e))

    with injuries_tab:
        injuries_page()

    with weekly:
        if st.button('Reload saved data and props (no Odds API credits)'):
            try:
                restore_saved_state(force=True)
                s.restore_error = None
                st.success('Saved CSVs and props restored without calling The Odds API.')
            except (RuntimeError,ValueError,KeyError,OSError) as e:
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
            p=game_filter_frame(p,'projection_game')
            search = st.text_input('Find a player', key='projection_search')
            if search:
                p = p[p.Player.str.contains(search, case=False, regex=False)]
            order = st.selectbox('Sort table', ['Best matchup', 'Worst matchup', 'Largest prop difference', 'Player name'])
            sortcol = {'Best matchup': 'Matchup / 10', 'Worst matchup': 'Matchup / 10', 'Largest prop difference': 'By How Much', 'Player name': 'Player'}[order]
            p = p.sort_values(sortcol, ascending=order in {'Worst matchup', 'Player name'}, na_position='last')
            st.caption('Up to 2 WRs and 1 QB, TE, RB per team, chosen by primary season yards/game. Props are the latest imported DraftKings lines for matching upcoming games. — means no line imported.')
            styled_table(p)
            st.caption('All ratings: 5.5 is neutral; higher favors production. Defense and coverage/scheme are separate effects. Mixed means they pull in opposite directions. — means unavailable. No pick means the difference is inside the configured range; the projection is still shown.')
            st.download_button('Download projection table', p.to_csv(index=False), 'projections.csv', 'text/csv')
        with st.expander('Data checks — review before using projections', expanded=not s.issues.empty):
            st.dataframe(s.issues, hide_index=True, use_container_width=True)
            st.caption('Omitted players are not assigned invented teams or workloads. Missing coverage uses a labeled season baseline.')
        with st.expander('Defensive run-scheme context'):
            if 'defensive_run_splits' in s.tables:
                st.dataframe(s.tables['defensive_run_splits'], hide_index=True)
            st.caption('Defensive scheme yards/game feeds the approved median-based adjustments. These values reflect both volume and efficiency.')

    if s.get('register_slate') and is_admin() and s.model_meta and not s.props.empty and not s.projections.empty:
        try:
            count = register_model_slate(s.projections, s.props, s.model_meta)
            st.success(f'Model record saved: {count} new picks. Daily Top 20 lists refreshed.')
        except (RuntimeError, ValueError) as error:
            st.warning('Model record not saved yet: ' + str(error) + ' Use Save current model slate in All Model Picks to retry.')
        finally:
            s.register_slate = False

    with picks_tab:
        picks_page()

    with all_model_tab:
        model_records_page()
    with best_tab:
        best20_page()

    with dashboard_tab:
        records_dashboard()

    with explorer:
        st.subheader("Pick Explorer")
        st.caption(
            "Browse saved past and current picks. Week 6 pilot picks "
            "remain available here. These filters do not change any records."
        )

        if st.button("Load / refresh explorer", key="explorer_refresh"):
            try:
                tracker, _ = load_tracker()
                raw = github("data/my_picks.json")
                st.session_state.explorer_data = {
                    "tracker": tracker,
                    "personal": json.loads(raw) if raw else [],
                }
            except (RuntimeError, ValueError) as error:
                st.error(str(error))

        saved = st.session_state.get("explorer_data")
        if saved is None:
            st.info("Click Load / refresh explorer to load your saved picks.")
        else:
            source = st.selectbox(
                "Which picks?",
                ["All Model Picks", "My Picks", "Top 20"],
                key="explorer_source",
            )
            tracker = saved["tracker"]
            if source == "My Picks":
                rows = saved["personal"]
            elif source == "Top 20":
                ids = {
                    pid
                    for day in tracker["days"].values()
                    for pid in day["active"]
                    if pid not in day.get("excluded", {})
                }
                rows = [
                    pick for pid, pick in tracker["picks"].items()
                    if pid in ids
                ]
            else:
                rows = list(tracker["picks"].values())

            if not rows:
                st.info("No saved picks found for this source.")
            else:
                table = pd.DataFrame(rows).copy()
                for column in ["Season", "Week", "Line", "Actual",
                               "Projection When Picked", "Season Average"]:
                    if column not in table:
                        table[column] = float("nan")
                    table[column] = pd.to_numeric(table[column], errors="coerce")
                for column in ["Player", "Position", "Team", "Opponent", "Stat"]:
                    if column not in table:
                        table[column] = "Unknown"
                    table[column] = table[column].fillna("Unknown")
                for column, default in [("Result", "Pending"),
                                        ("Your Pick", "Unknown"),
                                        ("Result Note", "")]:
                    if column not in table:
                        table[column] = default
                    table[column] = table[column].fillna(default)
                table["Excluded"] = (
                    table.get("Excluded", pd.Series(False, index=table.index))
                    .fillna(False).astype(bool)
                    | table.get("Injury Excluded", pd.Series(False, index=table.index))
                    .fillna(False).astype(bool)
                )

                view = table.copy()
                columns = st.columns(3)
                filters = [
                    ("Season", "Season"), ("Week", "Week"),
                    ("Position", "Position"), ("Team", "Team"),
                    ("Stat", "Stat"), ("Result", "Result"),
                    ("Your Pick", "Over / Under"),
                ]
                for index, (column, label) in enumerate(filters):
                    values = sorted(table[column].dropna().unique().tolist())
                    if column in {"Season", "Week"}:
                        values = [int(value) for value in values]
                    selection = columns[index % 3].selectbox(
                        label, ["All"] + values,
                        key="explorer_filter_" + column,
                    )
                    if selection != "All":
                        view = view[view[column] == selection]

                search = st.text_input("Search player", key="explorer_search")
                if search.strip():
                    view = view[view["Player"].str.contains(
                        search.strip(), case=False, regex=False
                    )]
                include_excluded = st.checkbox(
                    "Include crossed-off / injury-excluded model picks",
                    value=False, key="explorer_excluded",
                )
                if source != "My Picks" and not include_excluded:
                    view = view[~view["Excluded"]]

                view["Projection Error"] = (
                    view["Projection When Picked"] - view["Actual"]
                )
                order = st.selectbox(
                    "Sort by",
                    ["Largest projection miss", "Player", "Team", "Week"],
                    key="explorer_sort",
                )
                if order == "Largest projection miss":
                    view = view.assign(
                        _error=view["Projection Error"].abs()
                    ).sort_values("_error", ascending=False, na_position="last")
                else:
                    view = view.sort_values(order, na_position="last")

                st.caption(
                    "Record below applies only to these filtered picks, "
                    "including any pilot weeks you selected. "
                    "Positive projection error means we projected too high."
                )
                record_metrics(view.to_dict("records"), "explorer")
                display = view.rename(columns={
                    "Your Pick": "Pick",
                    "Projection When Picked": "Our Projection",
                    "Result Note": "Notes",
                })
                display_columns = [
                    "Season", "Week", "Player", "Position", "Team", "Opponent",
                    "Stat", "Pick", "Line", "Season Average", "Our Projection",
                    "Actual", "Projection Error", "Result", "Excluded", "Notes",
                ]
                styled_table(display[display_columns])
                st.download_button(
                    "Download filtered picks",
                    display[display_columns].to_csv(index=False),
                    file_name="filtered_picks.csv", mime="text/csv",
                    key="explorer_download",
                )

    with methodology:
        st.markdown("""### Projection formulas · Model v7
- **Workload:** 50% season opportunities/game + 50% opportunities over the last three available player games, through the selected cutoff. Byes do not count as zero. Offensive snap counts supplement zero-opportunity appearances when the team-week statistics are published. If snap counts are unavailable, recent usage relies on recorded statistical appearances and is flagged. Uses carries, targets and pass attempts separately. Public nflverse weekly logs are cached; upload a weekly-log fallback if unavailable. Missing or unmatched logs use a labeled season baseline. A player with fewer than three recorded games uses the available games.
- **Injuries:** Expected workload percentage applies before redistribution. Out means 0%. Released carries and targets are allocated separately to available same-position teammates using recent usage, or season usage if recent data is unavailable. Remaining limited players receive a reduced share. No eligible recipients means opportunities remain unassigned. Recalculate after edits. Defensive injuries and offensive-line injuries remain contextual notes; no invented efficiency adjustment.
- **Player selection:** up to two available WRs and one available QB, TE and RB per team, ranked by primary season yards/game. Out players are replaced in the display by the next eligible player. Redistribution includes the full uploaded position group, including undisplayed players.
- **Coverage:** WR/TE projected targets are adjusted by coverage-specific targeting and catch/yardage efficiency. Target rates use 50-route priors; efficiencies use 10-target priors. Rounded split target rates are estimates. QB passing uses man/zone YPA with 20-attempt priors. Missing splits fall back to season efficiency.
- **Defense:** multiplier = 1 + 0.632 × (opponent position/stat allowed ÷ league median − 1). Applies symmetrically. Missing benchmarks are flagged. No third-down adjustment.
- **RB scheme:** expected zone/gap mix = 75% individual RB carry mix + 25% opponent runs-faced mix. Individual usage is a proxy for offensive play selection. Apply the RB's scheme-specific YPC, with 20-carry priors, and each scheme's defensive yards-allowed adjustment. Unclassified runs remain neutral. No additional CBS rushing multiplier when scheme data exists.
- **RB boxes:** match fewer-than-seven, exactly-seven and eight-plus box splits to defensive light, residual seven and heavy rates. This mapping assumes the source definitions match those categories; the defensive frequencies are a tendency proxy, not a formation-conditioned forecast. Player split YPC uses 20-carry priors. Final expected YPC = 68% scheme estimate + 32% box estimate. Missing box inputs retain the scheme estimate and produce a data-check notice.
- **RB receiving:** projected targets × season catch rate gives catches before defense. Receptions allowed adjusts catch volume; yards allowed per reception separately adjusts receiving efficiency. Receiving yards = adjusted catches × adjusted player yards/catch. Total receiving yards allowed is not applied as an additional boost.
- **QB pressure:** expected pressure rate = equal blend of offensive pressure allowed and opponent pressure generated. Compare it with offensive pressure allowed as the usual baseline. Team pressured-dropback rate proxies the QB's pressured-attempt share. Adjustment = pressure-rate change × (pressured YPA / season YPA − 1) / (1 − usual pressure rate), tempered by plays/(plays+20), capped at −12.7%/+7%. This is an experimental approximation, not a directly measured clean-pocket model. EPA is contextual and is not converted into yards.
- **Receiver pressure fit:** never scale receiver projections automatically to QB passing totals. Estimate target-depth shift from the change in expected pressure and the QB's pressured ADoT relative to target-weighted team receiver depth. Individual depth fit shifts receiving opportunities up to ±6%; YAC per reception above/below expectation can modify receiving yards, with the combined receiver adjustment capped at ±8%. These modest coefficients are role-fit assumptions, not observed receiver-under-pressure splits. The supplied error column is ignored; YAC difference is calculated from the two valid columns. Missing receiver data means no receiver-pressure adjustment.
- **Ratings:** overall compares final projection with its neutral baseline using 5.5 + 4.5 × tanh(log(projection/neutral)/0.35). Defense and coverage/scheme ratings describe those components before the additional pressure/box layers. Ratings are matchup indicators, not hit probabilities.
- **WR–CB lists:** use every uploaded row, restricted to the current week's verified opponent. Receiving-yard agreements beyond the no-pick range automatically enter My Picks under Admin access. Disagreements are shown for review. Manual overrides freeze a personal projection and preserve the original model forecast; they never rewrite All Model or Top 20 forecasts. Missing lines, injury exclusions and started games do not create automatic picks. Multiple CB rows for one receiver do not duplicate the personal pick. Conflicting lists require review.
- **Records:** 2026 Week 6 remains Pilot; official totals begin Week 7. Existing saved projections, lines, versions and results remain frozen. New calculations use Model v7. Weekly and cumulative records remain separate.
- **Persistence:** accepted CSVs and imported props save to GitHub. Saved props are reused unless a fresh fetch is explicitly requested. Calculation, public workload downloads and matchup review do not consume Odds API credits. Public workload logs can update independently of uploaded snapshots; saved official forecasts do not change.
- **No-pick ranges:** default ±10 passing yards, ±5 receiving/rushing yards and ±0.5 receptions. Admin can configure thresholds. Automatic matchup picks must clear these ranges; manual review can leave the model unchanged.
- **Top 20:** rank absolute projection-line differences, one prop per player, maximum three per team and two passing-related picks in the same direction per team. Started slots freeze. Exclusions refill with the next eligible pick; fewer than 20 is possible if candidates are exhausted.
- **Results:** one-click public weekly-stat import with separate final-game confirmation; ambiguous or missing player records require review. Personal picks use their own saved side and line. W/L hit rate excludes pushes, voids and pending picks.
- **Limits:** no weather or third-down model. Coverage, pressure and defensive factors can be correlated; all weights and caps are experimental settings to evaluate prospectively. Missing data and date conflicts appear in Data checks.

""")
    st.divider()
    st.caption('NFL Matchup Analytics — Independent sports analytics project — Created by Owen Dunne')


if __name__ == '__main__':
    main()
