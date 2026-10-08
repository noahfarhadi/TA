"""
ta_app.py

Streamlit page for technical analysis: RSI, Connors RSI, MACD and Williams %R.
Users upload a price file, choose the asset and the indicator settings, and see
charts, the latest readings and a downloadable table. All calculations are in
ta_indicators.py. Nothing is stored; uploads live in the memory of the session.

Run locally from an Anaconda prompt:
    streamlit run ta_app.py
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import file_io as fio
import ta_indicators as ti

# ---------------------------------------------------------------------------
# Configuration: every default below can be changed here or in the sidebar
# ---------------------------------------------------------------------------
CONFIG = {
    "page_title": "Technical analysis: RSI, Connors RSI, MACD, Williams %R",
    "intro": "Upload a price file, pick an asset and read the indicators.",
    "disclaimer": ("The indicators are calculations on the uploaded prices. They describe past "
                   "price movements, they do not predict future prices, and this page is not "
                   "investment advice."),
    "file_types": ["csv", "txt", "tsv"],
    "date_candidates": ["date", "datetime", "timestamp", "time"],
    "close_candidates": ["adj close", "adj_close", "adjusted_close", "close", "price", "last"],
    "high_candidates": ["high"],
    "low_candidates": ["low"],
    "asset_candidates": ["ticker", "asset", "symbol", "security"],
    "table_rows": 15,
    "bar_up_color": "#2a9d8f",
    "bar_down_color": "#e76f51",
    "panel_height": 220,
    "price_panel_height": 330,
}

FIT = ({"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters
       else {"use_container_width": True})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_table(data: bytes, separator, decimal: str) -> pd.DataFrame:
    """Read an uploaded delimited text file (delimiter and decimal mark are detected by default)."""
    return fio.read_table(data, separator, decimal)


guess = fio.guess_column


def pick(label, columns, guessed, key, optional=False, help_text=None):
    """Selectbox that starts on the guessed column; returns None for '(none)'."""
    options = (["(none)"] if optional else []) + list(columns)
    index = options.index(guessed) if guessed in options else 0
    choice = st.selectbox(label, options, index=index, key=key, help=help_text)
    return None if choice == "(none)" else choice


def build_figure(view_prices, view_ind, panels, p):
    """Price panel on top, one panel per selected indicator below."""
    heights = [CONFIG["price_panel_height"]] + [CONFIG["panel_height"]] * len(panels)
    fig = make_subplots(rows=1 + len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[h / sum(heights) for h in heights])
    fig.add_trace(go.Scatter(x=view_prices.index, y=view_prices["close"], mode="lines", name="Close"),
                  row=1, col=1)
    for i, name in enumerate(panels, start=2):
        if name == "RSI":
            fig.add_trace(go.Scatter(x=view_ind.index, y=view_ind["RSI"], mode="lines", name=f"RSI({p.rsi_length})"),
                          row=i, col=1)
            levels, bounds = [p.rsi_upper, p.rsi_lower], (0, 100)
        elif name == "Connors RSI":
            fig.add_trace(go.Scatter(x=view_ind.index, y=view_ind["Connors RSI"], mode="lines",
                                     name=f"Connors RSI({p.crsi_rsi_length},{p.crsi_streak_length},{p.crsi_rank_length})"),
                          row=i, col=1)
            levels, bounds = [p.crsi_upper, p.crsi_lower], (0, 100)
        elif name == "Williams %R":
            fig.add_trace(go.Scatter(x=view_ind.index, y=view_ind["Williams %R"], mode="lines",
                                     name=f"Williams %R({p.willr_length})"), row=i, col=1)
            levels, bounds = [p.willr_upper, p.willr_lower], (-100, 0)
        else:  # MACD
            hist = view_ind["MACD histogram"]
            colors = np.where(hist >= 0, CONFIG["bar_up_color"], CONFIG["bar_down_color"])
            fig.add_trace(go.Bar(x=hist.index, y=hist, marker_color=colors, name="MACD histogram"), row=i, col=1)
            fig.add_trace(go.Scatter(x=view_ind.index, y=view_ind["MACD"], mode="lines",
                                     name=f"MACD({p.macd_fast},{p.macd_slow})"), row=i, col=1)
            fig.add_trace(go.Scatter(x=view_ind.index, y=view_ind["MACD signal"], mode="lines",
                                     name=f"Signal({p.macd_signal})"), row=i, col=1)
            levels, bounds = [0], None
        for level in levels:
            fig.add_hline(y=level, line_dash="dot", line_color="gray", row=i, col=1)
        if bounds:
            fig.update_yaxes(range=[bounds[0] - 5, bounds[1] + 5], row=i, col=1)
    fig.update_layout(height=sum(heights) + 80, hovermode="x unified", showlegend=True,
                      legend=dict(orientation="h", y=-0.08), margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title=CONFIG["page_title"], layout="wide")
st.title(CONFIG["page_title"])
st.caption(CONFIG["intro"])

with st.sidebar:
    st.header("File settings")
    sep_label = st.selectbox("Delimiter", list(fio.SEPARATORS), index=0,
                             help="Auto-detect works for each file separately.")
    separator = fio.SEPARATORS[sep_label]
    decimal = st.selectbox("Decimal mark", fio.DECIMALS, index=0)
    dayfirst = st.checkbox("Dates are day-first (DD/MM/YYYY)", value=False)

upload = st.file_uploader("Price file (one row per date, or one row per date and asset)",
                          type=CONFIG["file_types"])
with st.expander("What the file needs to contain"):
    st.markdown(
        "- A date column and a close column. High and low columns are needed for Williams %R only.\n"
        "- For a file with one row per date and asset, choose the asset column and then the asset.\n"
        "- For a file with one close column per asset, choose that asset's column as Close.\n"
        "- Use the same price basis for close, high and low (all adjusted or all unadjusted).")
if upload is None:
    st.info("Upload a price file to start.")
    st.stop()

raw = load_table(upload.getvalue(), separator, decimal)
cols = list(raw.columns)
with st.expander("Column mapping", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        date_col = pick("Date", cols, guess(cols, CONFIG["date_candidates"]), "date")
        # If no close-like name is found, start on the first column after the date column
        close_guess = guess(cols, CONFIG["close_candidates"]) or (cols[1] if len(cols) > 1 else cols[0])
        close_col = pick("Close", cols, close_guess, "close")
    with c2:
        high_col = pick("High", cols, guess(cols, CONFIG["high_candidates"]), "high", optional=True)
        low_col = pick("Low", cols, guess(cols, CONFIG["low_candidates"]), "low", optional=True)
    with c3:
        asset_col = pick("Asset column", cols, guess(cols, CONFIG["asset_candidates"]), "asset", optional=True)
        asset_value = None
        if asset_col is not None:
            values = sorted(raw[asset_col].astype(str).str.strip().unique())
            asset_value = st.selectbox("Asset", values)
try:
    use_hl = high_col is not None and low_col is not None
    prices, info = ti.prepare_ohlc(raw, date_col, close_col, high_col if use_hl else None,
                                   low_col if use_hl else None, asset_col, asset_value, dayfirst)
except Exception as exc:
    st.error(f"The file could not be read with this mapping ({exc}).")
    st.stop()
if len(prices) < 5:
    st.error("Fewer than five usable rows. Check the date and close columns.")
    st.stop()
if info["rows_used"] < info["rows_read"]:
    st.caption(f"{info['rows_read'] - info['rows_used']} of {info['rows_read']} rows were ignored "
               "(missing date, missing or non-positive close).")
if info["inconsistent_ohlc"]:
    st.warning(f"{info['inconsistent_ohlc']} rows have a high below the low, or a close outside the "
               "high-low range. Check the price basis of the three columns.")

# Sidebar: indicator settings
with st.sidebar:
    st.header("Indicators")
    with st.expander("RSI", expanded=False):
        show_rsi = st.checkbox("Show RSI", value=True)
        rsi_length = st.number_input("Length", min_value=2, value=14, step=1, key="rsi_len")
        rsi_upper = st.number_input("Upper level", value=70.0, key="rsi_up")
        rsi_lower = st.number_input("Lower level", value=30.0, key="rsi_lo")
    with st.expander("Connors RSI", expanded=False):
        show_crsi = st.checkbox("Show Connors RSI", value=True)
        crsi_rsi = st.number_input("RSI length of the close", min_value=2, value=3, step=1)
        crsi_streak = st.number_input("RSI length of the streak", min_value=2, value=2, step=1)
        crsi_rank = st.number_input("Percent-rank length", min_value=2, value=100, step=1)
        crsi_upper = st.number_input("Upper level", value=90.0, key="crsi_up")
        crsi_lower = st.number_input("Lower level", value=10.0, key="crsi_lo")
    with st.expander("MACD", expanded=False):
        show_macd = st.checkbox("Show MACD", value=True)
        macd_fast = st.number_input("Fast EMA", min_value=2, value=12, step=1)
        macd_slow = st.number_input("Slow EMA", min_value=3, value=26, step=1)
        macd_signal = st.number_input("Signal EMA", min_value=2, value=9, step=1)
    with st.expander("Williams %R", expanded=False):
        has_hl = {"high", "low"}.issubset(prices.columns)
        show_willr = st.checkbox("Show Williams %R", value=has_hl, disabled=not has_hl,
                                 help="Needs High and Low columns." if not has_hl else None)
        willr_length = st.number_input("Length", min_value=2, value=14, step=1, key="wr_len")
        willr_upper = st.number_input("Upper level", value=-20.0, key="wr_up")
        willr_lower = st.number_input("Lower level", value=-80.0, key="wr_lo")

if macd_fast >= macd_slow:
    st.error("The fast EMA length must be smaller than the slow EMA length.")
    st.stop()

params = ti.TaParams(
    rsi_length=int(rsi_length), rsi_upper=float(rsi_upper), rsi_lower=float(rsi_lower),
    crsi_rsi_length=int(crsi_rsi), crsi_streak_length=int(crsi_streak), crsi_rank_length=int(crsi_rank),
    crsi_upper=float(crsi_upper), crsi_lower=float(crsi_lower),
    macd_fast=int(macd_fast), macd_slow=int(macd_slow), macd_signal=int(macd_signal),
    willr_length=int(willr_length), willr_upper=float(willr_upper), willr_lower=float(willr_lower))

# Indicators are always computed on the full history so the warm-up period is not cut off
indicators = ti.compute_indicators(prices, params)
panels = [name for name, flag in [("RSI", show_rsi), ("Connors RSI", show_crsi),
                                  ("MACD", show_macd), ("Williams %R", show_willr and has_hl)] if flag]

# Rows needed before the first value appears
needed = {"RSI": params.rsi_length + 1,
          "Connors RSI": max(params.crsi_rank_length, params.crsi_rsi_length, params.crsi_streak_length) + 2,
          "MACD": params.macd_slow + params.macd_signal - 1,
          "Williams %R": params.willr_length}
for name in panels:
    if len(prices) < needed[name]:
        st.warning(f"{name} needs at least {needed[name]} rows; the file has {len(prices)}.")

# Date range shown (display only)
d1, d2 = st.columns(2)
start = d1.date_input("Show from", value=prices.index.min().date(),
                      min_value=prices.index.min().date(), max_value=prices.index.max().date())
end = d2.date_input("Show to", value=prices.index.max().date(),
                    min_value=prices.index.min().date(), max_value=prices.index.max().date())
view_prices = prices.loc[pd.Timestamp(start):pd.Timestamp(end)]
view_ind = indicators.loc[view_prices.index]
if view_prices.empty:
    st.error("The selected date range contains no rows.")
    st.stop()

if panels:
    st.plotly_chart(build_figure(view_prices, view_ind, panels, params), **FIT)
else:
    st.info("Switch on at least one indicator in the sidebar.")

# Latest readings
snapshot = ti.latest_snapshot(indicators, params)
snapshot = snapshot.loc[snapshot["Indicator"].isin(panels)]
if not snapshot.empty:
    st.subheader("Latest readings")
    shown = snapshot.copy()
    shown["Date"] = pd.to_datetime(shown["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(shown.style.format({"Value": "{:.2f}"}, na_rep="n/a"), hide_index=True)

# Table and download
st.subheader("Data")
export = prices.join(indicators).rename(columns={"close": "Close", "high": "High", "low": "Low"})
st.dataframe(export.tail(CONFIG["table_rows"]).style.format("{:.4f}", na_rep=""))
st.download_button("Download prices and indicators (CSV)", export.to_csv().encode("utf-8"),
                   file_name="indicators.csv", mime="text/csv")

st.divider()
st.caption(CONFIG["disclaimer"])


# ---------------------------------------------------------------------------
# Version log
# ---------------------------------------------------------------------------
# v1.0 (2026-10-08)  New file. Added: CONFIG block, FIT, load_table (cached call
#   to file_io.read_table), guess (= file_io.guess_column), pick, build_figure,
#   the upload and column-mapping steps, the sidebar settings for RSI, Connors
#   RSI, MACD and Williams %R, the chart, the latest-readings table and the
#   CSV download.
