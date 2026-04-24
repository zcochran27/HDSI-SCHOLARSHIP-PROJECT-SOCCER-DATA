import ast
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Categories that drive pass/duel filtering. type_primary is scalar (one-hot
# via OneHotEncoder); type_secondary is list-valued (multi-hot via
# MultiLabelBinarizer).
# ---------------------------------------------------------------------------
PRIMARY_CATEGORIES = ["interception", "duel"]
SECONDARY_CATEGORIES = ["pass", "defensive_duel", "aerial_duel"]

WINDOW = 20.0  # outcome look-ahead in seconds

# Speed-of-play defaults. Windows define per-event lookback rate features;
# clustering happens on those features plus short/long ratios.
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
SOP_WINDOWS = (30, 60)
SOP_K = 4
SOP_RATIO_EPS = 1e-3

SOP_FEATURE_COLS = [
    "n_events_30s", "delta_x_30s", "dist_30s",
    "n_events_60s", "delta_x_60s", "dist_60s",
    "n_events_ratio_30v60", "dist_ratio_30v60", "delta_x_diff_30v60",
    "sop_cluster", "sop_cluster_name",
]

BASE_COLS = [
    "id", "matchId", "matchPeriod", "matchTimestamp", "time",
    "type_primary", "type_secondary",
    "start_x", "start_y",
    "team_id",
    "opponentTeam_id", "opponentTeam_name",
    "player_id", "player_name",
    "outcome",
] + SOP_FEATURE_COLS

PASS_COLS = [
    "end_x", "end_y", "pass_accurate", "pass_angle", "pass_height", "pass_length",
    "pass_recipient_id", "pass_recipient_name",
]

GROUND_DUEL_COLS = [
    "groundDuel_opponent_id", "groundDuel_opponent_name", "groundDuel_keptPossession",
    "groundDuel_progressedWithBall", "groundDuel_stoppedProgress",
    "groundDuel_recoveredPossession", "groundDuel_takeOn", "groundDuel_side",
]

AERIAL_DUEL_COLS = [
    "aerialDuel_opponent_id", "aerialDuel_opponent_name", "aerialDuel_firstTouch"
]

INTERCEPTION_SEC_TAGS = [
    "back_pass", "carry", "counterpressing_recovery", "cross", "cross_blocked",
    "deep_completed_cross", "deep_completion", "forward_pass", "hand_pass",
    "head_pass", "key_pass", "lateral_pass", "long_pass",
    "pass_to_final_third", "pass_to_penalty_area", "progressive_pass",
    "progressive_run", "recovery", "short_or_medium_pass", "smart_pass",
    "through_pass", "touch_in_box",
]

DUEL_SEC_TAGS = [
    "carry", "counterpressing_recovery", "dribbled_past_attempt",
    "foul_suffered", "progressive_run", "sliding_tackle",
]

FILTERS = [
    {"name": "intercepted_pass", "primary": "interception", "secondary": ["pass"],
     "extra_cols": PASS_COLS, "sec_classes": INTERCEPTION_SEC_TAGS,
     "flag_cols": [], "encode_cols": []},
    {"name": "defensive_duels", "primary": None,
     "secondary": ["defensive_duel", "aerial_duel"],
     "extra_cols": GROUND_DUEL_COLS + AERIAL_DUEL_COLS,
     "sec_classes": DUEL_SEC_TAGS,
     "flag_cols": ["defensive_duel", "aerial_duel"],
     "encode_cols": []},
]


def _to_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
        return list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else [parsed]
    return []


def encode_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach boolean primary_* and secondary_* columns via sklearn encoders."""
    out = df.copy()

    primary_enc = OneHotEncoder(
        categories=[PRIMARY_CATEGORIES],
        handle_unknown="ignore",
        sparse_output=False,
    )
    primary_matrix = primary_enc.fit_transform(out[["type_primary"]])
    for idx, cat in enumerate(PRIMARY_CATEGORIES):
        out[f"primary_{cat}"] = primary_matrix[:, idx].astype(bool)

    secondary_lists = out["type_secondary"].apply(_to_list)
    mlb = MultiLabelBinarizer(classes=SECONDARY_CATEGORIES)
    secondary_matrix = mlb.fit_transform(secondary_lists)
    for idx, cat in enumerate(SECONDARY_CATEGORIES):
        out[f"secondary_{cat}"] = secondary_matrix[:, idx].astype(bool)

    return out


def expand_secondary_tags(df: pd.DataFrame, classes: list) -> pd.DataFrame:
    """MultiLabelBinarizer over `type_secondary` restricted to `classes`,
    producing a boolean `sec_<label>` column per class."""
    out = df.copy()
    if not classes or out.empty or "type_secondary" not in out.columns:
        return out

    lists = out["type_secondary"].apply(_to_list)
    mlb = MultiLabelBinarizer(classes=classes)
    matrix = mlb.fit_transform(lists)
    for idx, cat in enumerate(classes):
        out[f"sec_{cat}"] = matrix[:, idx].astype(bool)
    return out


def encode_scalar_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """One-hot encode scalar categorical columns in place, dropping the originals.
    Emits boolean `<col>_<category>` columns; NaN rows get all-False across the group."""
    out = df.copy()
    for col in cols:
        if col not in out.columns or out.empty:
            continue
        values = out[[col]].astype(object).where(out[[col]].notna(), other=np.nan)
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        matrix = enc.fit_transform(values.fillna("__nan__"))
        categories = enc.categories_[0]
        for idx, cat in enumerate(categories):
            if cat == "__nan__":
                continue
            out[f"{col}_{cat}"] = matrix[:, idx].astype(bool)
        # out = out.drop(columns=[col])
    return out


def apply_filter(df: pd.DataFrame, filter_def: dict) -> pd.DataFrame:
    """Return rows matching a single filter, projected to base + filter-specific
    cols, MLB-expanded on type_secondary, and scalar-encoded on encode_cols."""
    mask = pd.Series(True, index=df.index)

    if filter_def.get("primary"):
        mask &= df[f"primary_{filter_def['primary']}"]

    secondary = filter_def.get("secondary")
    if secondary:
        secondary = [secondary] if isinstance(secondary, str) else list(secondary)
        sec_mask = np.logical_or.reduce(
            [df[f"secondary_{s}"].to_numpy() for s in secondary]
        )
        mask &= pd.Series(sec_mask, index=df.index)

    cols = BASE_COLS + list(filter_def.get("extra_cols", []))
    cols = [c for c in cols if c in df.columns]
    result = df.loc[mask, cols].copy()

    for flag in filter_def.get("flag_cols", []):
        src = f"secondary_{flag}"
        if src in df.columns:
            result[f"is_{flag}"] = df.loc[mask, src].to_numpy()

    result = expand_secondary_tags(result, classes=filter_def.get("sec_classes", []))
    result = encode_scalar_cols(result, cols=filter_def.get("encode_cols", []))
    return result.drop(columns=["type_primary", "type_secondary"], errors="ignore")


def filter_dataset(
    input_df: pd.DataFrame,
    filters: list = FILTERS,
) -> dict:
    """Encode primary/secondary type columns, then return one DataFrame per filter
    keyed by filter name, each projected to BASE_COLS + that filter's extra_cols."""
    df = encode_filter_columns(input_df)
    return {f["name"]: apply_filter(df, f) for f in filters}


def add_outcome(df):
    shot_goals = df.loc[
        (df['type_primary'] == 'shot') & (df['shot_isGoal'] == True),
        ['matchId', 'matchPeriod', 'time', 'team_id']
    ].rename(columns={'team_id': 'scoring_team'})

    own_goals = df.loc[
        df['type_primary'] == 'own_goal',
        ['matchId', 'matchPeriod', 'time', 'opponentTeam_id']
    ].rename(columns={'opponentTeam_id': 'scoring_team'})

    goals = (
        pd.concat([shot_goals, own_goals], ignore_index=True)
          .sort_values(['matchId', 'matchPeriod', 'time'])
          .reset_index(drop=True)
    )

    outcome = np.zeros(len(df), dtype=np.int8)
    if len(goals):
        goals_by_mp = {key: g for key, g in goals.groupby(['matchId', 'matchPeriod'], sort=False)}
        for (mid, period), grp in df.groupby(['matchId', 'matchPeriod'], sort=False):
            g = goals_by_mp.get((mid, period))
            if g is None or len(g) == 0:
                continue
            g_times = g['time'].to_numpy()
            g_teams = g['scoring_team'].to_numpy()
            ev_times = grp['time'].to_numpy()
            ev_teams = grp['team_id'].to_numpy()
            ev_idx   = grp.index.to_numpy()
            left  = np.searchsorted(g_times, ev_times,          side='left')
            right = np.searchsorted(g_times, ev_times + WINDOW, side='right')
            has_goal  = left < right
            first_idx = np.clip(left, 0, len(g_teams) - 1)
            first_team = g_teams[first_idx]
            outcome[ev_idx] = np.where(
                has_goal & (first_team == ev_teams), 1,
                np.where(has_goal, -1, 0),
            )
    df['outcome'] = outcome
    return df


def add_sop_window_features(df, window, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH):
    """Per-event features over the prior `window` seconds in the same
    (matchId, matchPeriod). Prior-event coords are flipped to (L-x, W-y) when
    their team_id differs from the current event's team_id, so distances and
    x-progression are oriented from the current team's POV. Emits per-second
    rates: n_events_<W>s, delta_x_<W>s, dist_<W>s."""
    df = df.sort_values(['matchId', 'matchPeriod', 'time']).reset_index(drop=True)

    n_events = np.zeros(len(df), dtype=np.int32)
    delta_x = np.full(len(df), np.nan, dtype=np.float64)
    dist = np.full(len(df), np.nan, dtype=np.float64)

    for _, grp in df.groupby(['matchId', 'matchPeriod'], sort=False):
        times = grp['time'].to_numpy()
        xs = grp['start_x'].to_numpy(dtype=np.float64)
        ys = grp['start_y'].to_numpy(dtype=np.float64)
        teams = grp['team_id'].to_numpy()
        idx = grp.index.to_numpy()

        left = np.searchsorted(times, times - window, side='left')

        for i in range(len(grp)):
            lo, hi = left[i], i
            if lo >= hi:
                continue
            cur_team = teams[i]
            same = teams[lo:hi] == cur_team
            wx = np.where(same, xs[lo:hi], pitch_length - xs[lo:hi])
            wy = np.where(same, ys[lo:hi], pitch_width - ys[lo:hi])
            n_events[idx[i]] = hi - lo
            delta_x[idx[i]] = wx[-1] - wx[0]
            if hi - lo >= 2:
                dist[idx[i]] = np.hypot(np.diff(wx), np.diff(wy)).sum()
            else:
                dist[idx[i]] = 0.0

    df[f'n_events_{window}s'] = n_events / window
    df[f'delta_x_{window}s'] = delta_x / window
    df[f'dist_{window}s'] = dist / window
    return df


def add_sop_ratios(df, short_window, long_window, eps=SOP_RATIO_EPS):
    """Short-vs-long window ratios on already-computed rate columns. >1 means
    the recent short window is faster than the longer baseline. delta_x is
    signed so a ratio is ill-defined — use a difference instead."""
    df[f'n_events_ratio_{short_window}v{long_window}'] = (
        df[f'n_events_{short_window}s'] / (df[f'n_events_{long_window}s'] + eps)
    )
    df[f'dist_ratio_{short_window}v{long_window}'] = (
        df[f'dist_{short_window}s'] / (df[f'dist_{long_window}s'] + eps)
    )
    df[f'delta_x_diff_{short_window}v{long_window}'] = (
        df[f'delta_x_{short_window}s'] - df[f'delta_x_{long_window}s']
    )
    return df


def _default_sop_cluster_names(profile: pd.DataFrame) -> dict:
    """Name clusters by their centroid profile so labels track meaning, not
    KMeans' arbitrary init. Lowest n_events_ratio = decelerating; highest =
    accelerating; remaining 'sustained' clusters split by sign of delta_x
    (positive = forward, negative = backward) when available."""
    ratio_col = next(
        (c for c in profile.columns if c.startswith('n_events_ratio_')), None
    )
    delta_col = next(
        (c for c in ('delta_x_30s',) if c in profile.columns), None
    )
    if ratio_col is None:
        return {k: f'cluster_{k}' for k in profile.index}

    by_ratio = profile.sort_values(ratio_col)
    names = {by_ratio.index[0]: 'decelerating',
             by_ratio.index[-1]: 'accelerating'}

    middle = [k for k in by_ratio.index if k not in names]
    if len(middle) == 1:
        names[middle[0]] = 'sustained'
    elif len(middle) >= 2 and delta_col is not None:
        middle_sorted = profile.loc[middle].sort_values(delta_col)
        names[middle_sorted.index[0]] = 'sustained_backward'
        names[middle_sorted.index[-1]] = 'sustained_forward'
        for k in middle_sorted.index[1:-1]:
            names[k] = f'sustained_{k}'
    else:
        for k in middle:
            names[k] = f'sustained_{k}'
    return names


def add_sop_clusters(df, sop_cols, k=SOP_K, random_state=0,
                     name_fn=_default_sop_cluster_names):
    """Standardize sop_cols, fit KMeans(k), and attach sop_cluster +
    sop_cluster_name. Rows with NaN in any sop_col get NaN labels."""
    X = df[sop_cols].dropna()
    if X.empty:
        df['sop_cluster'] = np.nan
        df['sop_cluster_name'] = np.nan
        return df

    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(Xs)
    df['sop_cluster'] = pd.Series(km.labels_, index=X.index).reindex(df.index)

    profile = df.groupby('sop_cluster')[sop_cols].mean()
    df['sop_cluster_name'] = df['sop_cluster'].map(name_fn(profile))
    return df


def add_speed_of_play(df, windows=SOP_WINDOWS, k=SOP_K, random_state=0,
                      pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH):
    """End-to-end speed-of-play pipeline: window rate features for each window,
    short-vs-long ratios between the first two windows, and KMeans clustering
    with profile-based names. Adds n_events_<W>s / delta_x_<W>s / dist_<W>s
    per window, ratio/diff cols between windows[0] and windows[1], and
    sop_cluster + sop_cluster_name."""
    for w in windows:
        df = add_sop_window_features(df, window=w,
                                     pitch_length=pitch_length,
                                     pitch_width=pitch_width)

    short_w, long_w = windows[0], windows[1]
    df = add_sop_ratios(df, short_window=short_w, long_window=long_w)

    sop_cols = []
    for w in windows:
        sop_cols += [f'n_events_{w}s', f'delta_x_{w}s', f'dist_{w}s']
    sop_cols += [
        f'n_events_ratio_{short_w}v{long_w}',
        f'dist_ratio_{short_w}v{long_w}',
        f'delta_x_diff_{short_w}v{long_w}',
    ]
    df = add_sop_clusters(df, sop_cols=sop_cols, k=k, random_state=random_state)
    return df


def preprocess(
    input_df: pd.DataFrame,
    filters: list = FILTERS,
) -> dict:
    """Full preprocessing pipeline: add `time`, compute outcome and speed-of-
    play features on the full event log, then filter down to per-filter
    DataFrames (base cols + the columns relevant to each filter). Outcome and
    SOP features are computed before filtering so lookback/lookahead can see
    every event in each (matchId, matchPeriod)."""
    df = input_df.copy()
    df['time'] = pd.to_timedelta(df['matchTimestamp']).dt.total_seconds()
    df = df.sort_values(['matchId', 'matchPeriod', 'time']).reset_index(drop=True)
    df = add_outcome(df)
    df = add_speed_of_play(df)
    return filter_dataset(df, filters=filters)
