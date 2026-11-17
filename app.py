"""Streamlit dashboard: Spain Top 50 lifecycle & rotation analysis.
Run:  streamlit run app.py   (needs data.csv in the same folder)"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import lifecycle as L

st.set_page_config(page_title="Spain Top 50 Lifecycle", layout="wide")


def find_data():
    """Look for the CSV next to app.py (data.csv or the original Atlantic_Spain file)."""
    here = Path(__file__).resolve().parent
    for name in ["data.csv", "Atlantic_Spain 1.csv", "Atlantic_Spain_1.csv"]:
        if (here / name).exists():
            return str(here / name)
    found = sorted(here.glob("*.csv"))
    if found:
        return str(found[0])
    st.error(f"No CSV file found. Put your dataset in this folder: {here}")
    st.stop()


@st.cache_data
def load():
    df, notes, missing = L.load_clean(find_data())
    first, last = df["date"].min(), df["date"].max()
    df = L.add_stages(df, first)
    lc = L.lifecycle_table(df, first, last)
    return df, lc, notes, first, last


df, lc_all, notes, first, last = load()

# ---------------- sidebar filters ----------------
st.sidebar.header("Filters")
dr = st.sidebar.date_input("Date range", (first.date(), last.date()),
                           min_value=first.date(), max_value=last.date())
if len(dr) != 2:
    st.info("Pick both a start and an end date.")
    st.stop()
stages = st.sidebar.multiselect("Lifecycle stage", L.STAGES, default=L.STAGES)
expl = st.sidebar.radio("Explicit content", ["All", "Explicit only", "Clean only"])
atype = st.sidebar.multiselect("Album type", ["single", "album"], default=["single", "album"])

f = df[(df["date"] >= pd.Timestamp(dr[0])) & (df["date"] <= pd.Timestamp(dr[1]))]
f = f[f["stage"].isin(stages) & f["album_type"].isin(atype)]
if expl == "Explicit only":
    f = f[f["is_explicit"]]
elif expl == "Clean only":
    f = f[~f["is_explicit"]]
if f.empty:
    st.warning("No data for these filters.")
    st.stop()

# lifecycle table for the filtered songs (full history of each song, so lifetimes stay correct)
lc = lc_all[lc_all["song_id"].isin(f["song_id"].unique())]
# flow uses the date range and the content filters (not the stage filter, which would break entry/exit counts)
base = df[(df["date"] >= pd.Timestamp(dr[0])) & (df["date"] <= pd.Timestamp(dr[1])) & df["album_type"].isin(atype)]
if expl == "Explicit only":
    base = base[base["is_explicit"]]
elif expl == "Clean only":
    base = base[~base["is_explicit"]]

st.title("Spain Top 50: Content Maturity, Lifecycle & Rotation")
st.caption("Atlantic Recording Corporation | Daily Top 50 snapshots, "
           f"{first.date()} to {last.date()}")

# ---------------- KPI cards ----------------
flow_all = L.daily_flow(df)
flow = flow_all[(flow_all["date"] >= pd.Timestamp(dr[0])) & (flow_all["date"] <= pd.Timestamp(dr[1]))]
k = L.kpis(lc, flow if len(flow) else flow_all) if len(lc) > 3 else {}
c = st.columns(6)
if k:
    c[0].metric("Avg days on playlist", f"{k['avg_days_on_playlist']:.1f}", help="Mean days a song appears on the Top 50.")
    c[1].metric("Median days", f"{k['median_days_on_playlist']:.0f}", help="Half of songs leave sooner than this.")
    c[2].metric("Entry-to-peak (days)", f"{k['entry_to_peak_days']:.1f}", help="Songs that entered after the data start.")
    c[3].metric("Daily churn rate", f"{k['playlist_churn_rate']*100:.1f}%", help="Share of the 50 slots that are new each day.")
    c[4].metric("Retention stability", f"{k['retention_stability_index']:.3f}", help="1 = rank never moves.")
    c[5].metric("Explicit score", f"{k['explicit_lifecycle_score']:.2f}", help="Explicit avg days / clean avg days.")
    st.caption(f"Single vs album longevity ratio: **{k['single_vs_album_ratio']:.2f}** "
               "(avg days of singles divided by avg days of album tracks). "
               "KPI values follow the song set and date range chosen on the left.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Song timeline", "Entry vs exit", "Stage distribution", "Content maturity", "Churn analytics"])

with tab1:
    songs = lc.sort_values("days_on_playlist", ascending=False)
    pick = st.multiselect("Choose songs", songs["song_id"].tolist(),
                          default=songs["song_id"].head(3).tolist(),
                          format_func=lambda s: songs.set_index("song_id").loc[s, "song"] + " - "
                          + songs.set_index("song_id").loc[s, "artist"])
    t = df[df["song_id"].isin(pick)]
    fig = px.line(t, x="date", y="position", color="song", hover_data=["popularity", "stage"])
    fig.update_yaxes(autorange="reversed", title="Position (1 = top)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(lc[lc["song_id"].isin(pick)][["song", "artist", "entry_date", "exit_date", "days_on_playlist",
                                               "peak_position", "time_to_peak"]], hide_index=True)

with tab2:
    fl = flow.melt(id_vars="date", value_vars=["entries", "exits"], var_name="type", value_name="songs")
    fl["songs"] = fl.groupby("type")["songs"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    fig = px.line(fl, x="date", y="songs", color="type", title="Entries and exits per day (7-day average)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Entries always equal exits because the playlist is fixed at 50 songs.")

with tab3:
    a, b = st.columns(2)
    sd = f["stage"].value_counts(normalize=True).reindex(L.STAGES).fillna(0).reset_index()
    sd.columns = ["stage", "share"]
    a.plotly_chart(px.bar(sd, x="stage", y="share", title="Share of song-days by stage"), use_container_width=True)
    ps = f.groupby("stage")["popularity"].mean().reindex(L.STAGES).reset_index()
    b.plotly_chart(px.bar(ps, x="stage", y="popularity", title="Average popularity by stage"), use_container_width=True)
    st.markdown(f"**Stage rules:** New Entry = first {L.NEW_DAYS} days after entering; Decline = rank worse by "
                f"{L.TREND_STEP}+ places vs {L.TREND_WINDOW} days ago; Growth = better by {L.TREND_STEP}+ places "
                f"(outside Top {L.PEAK_RANK}); Peak = Top {L.PEAK_RANK}; Mature = everything else (stable mid-rank).")

with tab4:
    a, b = st.columns(2)
    lc2 = lc.assign(content=lc["is_explicit"].map({True: "Explicit", False: "Clean"}))
    a.plotly_chart(px.box(lc2, x="content", y="days_on_playlist", title="Explicit vs clean: days on playlist"),
                   use_container_width=True)
    b.plotly_chart(px.box(lc2, x="album_type", y="days_on_playlist", title="Single vs album: days on playlist"),
                   use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(px.scatter(lc2, x="duration_min", y="days_on_playlist", color="content", opacity=.6,
                              title="Duration (min) vs days on playlist"), use_container_width=True)
    b.plotly_chart(px.scatter(lc2, x="total_tracks", y="retention_stability", color="album_type", opacity=.6,
                              title="Album size vs retention stability"), use_container_width=True)

with tab5:
    m = flow.groupby("month").agg(churn=("churn_rate", "mean"), volatility=("entries", "std")).reset_index()
    a, b = st.columns(2)
    a.plotly_chart(px.bar(m, x="month", y="churn", title="Average daily churn by month"), use_container_width=True)
    b.plotly_chart(px.bar(m, x="month", y="volatility", title="Volatility (std of daily entries) by month"),
                   use_container_width=True)
    st.write("Biggest turnover days:")
    st.dataframe(flow.sort_values("entries", ascending=False).head(10)[["date", "entries", "exits"]], hide_index=True)

with st.expander("Data validation notes"):
    for n in notes:
        st.write("- " + n)
