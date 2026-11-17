"""Shared analysis code: cleaning, lifecycle table, stages, churn, KPIs.
Used by run_analysis.py (report + charts) and app.py (Streamlit dashboard)."""
import re
import unicodedata

import numpy as np
import pandas as pd

TREND_WINDOW = 3     # compare today's rank with the rank 3 playlist days ago
TREND_STEP = 5       # a move of 5+ places counts as "improving" or "deteriorating"
NEW_DAYS = 7         # New Entry = first 7 days on playlist
PEAK_RANK = 10       # Top 10 = peak zone
STAGES = ["New Entry", "Growth", "Peak", "Mature", "Decline"]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_clean(path: str):
    """Return (clean dataframe, list of validation notes)."""
    notes = []
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    notes.append(f"Rows loaded: {len(df):,}; days: {df['date'].nunique()}")

    # Days with more than 50 rows (a second snapshot was stored): keep the first 50 positions
    counts = df.groupby("date").size()
    bad = counts[counts != 50]
    for d, n in bad.items():
        notes.append(f"{d.date()} had {n} rows; kept the first snapshot (50 rows)")
    df = df.drop_duplicates(["date", "position"], keep="first")
    df = df.drop_duplicates(["date", "song", "artist"], keep="first")
    assert (df.groupby("date").size() == 50).all(), "Each day must have 50 rows"

    full = pd.date_range(df["date"].min(), df["date"].max())
    missing = sorted(set(full) - set(df["date"]))
    notes.append("Dates missing from the data: " + ", ".join(str(d.date()) for d in missing))

    df["song_clean"] = df["song"].map(_norm)
    df["artist_clean"] = df["artist"].map(_norm)
    df["song_id"] = df["song_clean"] + " | " + df["artist_clean"]
    notes.append(f"Unique songs (song + artist): {df['song_id'].nunique()}")
    df = df.sort_values(["song_id", "date"]).reset_index(drop=True)
    return df, notes, missing


def add_stages(df: pd.DataFrame, first_date) -> pd.DataFrame:
    """Add days_since_entry, rank trend and lifecycle stage to each song-day."""
    df = df.sort_values(["song_id", "date"]).copy()
    g = df.groupby("song_id")
    df["entry_date"] = g["date"].transform("min")
    df["left_censored"] = df["entry_date"] == first_date   # already on playlist when data starts
    df["days_since_entry"] = (df["date"] - df["entry_date"]).dt.days
    df["rank_change"] = g["position"].diff(TREND_WINDOW)    # negative = rank improved
    stage = np.select(
        [
            (~df["left_censored"]) & (df["days_since_entry"] < NEW_DAYS),
            df["rank_change"] >= TREND_STEP,
            (df["rank_change"] <= -TREND_STEP) & (df["position"] > PEAK_RANK),
            df["position"] <= PEAK_RANK,
        ],
        ["New Entry", "Decline", "Growth", "Peak"],
        default="Mature",
    )
    df["stage"] = stage
    return df


def lifecycle_table(df: pd.DataFrame, first_date, last_date) -> pd.DataFrame:
    rows = []
    for sid, s in df.groupby("song_id"):
        s = s.sort_values("date")
        entry, exit_ = s["date"].min(), s["date"].max()
        peak_pos = s["position"].min()
        peak_date = s.loc[s["position"] == peak_pos, "date"].min()
        daily_move = s["position"].diff().abs()
        rows.append(
            dict(
                song_id=sid,
                song=s["song"].mode().iat[0],
                artist=s["artist"].mode().iat[0],
                entry_date=entry,
                exit_date=exit_,
                days_on_playlist=len(s),
                span_days=(exit_ - entry).days + 1,
                peak_position=peak_pos,
                time_to_peak=(peak_date - entry).days,
                first_position=s["position"].iat[0],
                avg_position=s["position"].mean(),
                avg_popularity=s["popularity"].mean(),
                peak_popularity=s["popularity"].max(),
                days_to_peak_popularity=(s.loc[s["popularity"].idxmax(), "date"] - entry).days,
                rank_volatility=daily_move.mean() if len(s) > 1 else np.nan,
                is_explicit=bool(s["is_explicit"].mode().iat[0]),
                album_type=s["album_type"].mode().iat[0],
                total_tracks=int(s["total_tracks"].mode().iat[0]),
                duration_min=s["duration_ms"].median() / 60000,
                left_censored=entry == first_date,
                still_active=exit_ == last_date,
            )
        )
    lc = pd.DataFrame(rows)
    # Retention Stability Index: 1 = never moves, 0 = swings across the whole chart each day
    lc["retention_stability"] = (1 - lc["rank_volatility"] / 49).clip(0, 1)
    lc["completed"] = (~lc["left_censored"]) & (~lc["still_active"])
    # most common stage a song spent time in
    dom = df.groupby("song_id")["stage"].agg(lambda x: x.value_counts().index[0])
    lc["dominant_stage"] = lc["song_id"].map(dom)
    return lc


def daily_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Entries / exits per day. Only compares a day with the day right before it."""
    days = sorted(df["date"].unique())
    sets = {d: set(g["song_id"]) for d, g in df.groupby("date")}
    out = []
    for prev, cur in zip(days[:-1], days[1:]):
        if (cur - prev).days != 1:
            continue
        entries = len(sets[cur] - sets[prev])
        exits = len(sets[prev] - sets[cur])
        out.append(dict(date=pd.Timestamp(cur), entries=entries, exits=exits))
    flow = pd.DataFrame(out)
    flow["churn_rate"] = flow["entries"] / 50          # share of the 50 slots that changed
    flow["month"] = flow["date"].dt.to_period("M").astype(str)
    # average move in rank for songs that stayed
    pos = df.pivot(index="date", columns="song_id", values="position")
    moves = []
    for d in flow["date"]:
        p = d - pd.Timedelta(days=1)
        both = pos.loc[[p, d]].dropna(axis=1)
        moves.append((both.loc[d] - both.loc[p]).abs().mean())
    flow["avg_rank_move"] = moves
    return flow


def kpis(lc: pd.DataFrame, flow: pd.DataFrame) -> dict:
    exp = lc[lc["is_explicit"]]["days_on_playlist"].mean()
    cln = lc[~lc["is_explicit"]]["days_on_playlist"].mean()
    sgl = lc[lc["album_type"] == "single"]["days_on_playlist"].mean()
    alb = lc[lc["album_type"] == "album"]["days_on_playlist"].mean()
    new = lc[~lc["left_censored"]]
    return {
        "avg_days_on_playlist": lc["days_on_playlist"].mean(),
        "median_days_on_playlist": lc["days_on_playlist"].median(),
        "entry_to_peak_days": new["time_to_peak"].mean(),
        "playlist_churn_rate": flow["churn_rate"].mean(),
        "retention_stability_index": lc["retention_stability"].mean(),
        "explicit_lifecycle_score": exp / cln,
        "single_vs_album_ratio": sgl / alb,
    }
