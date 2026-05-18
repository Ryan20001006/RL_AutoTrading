"""
New_Strategy — Data Pipeline
=====================================================================
Step 1  WRDS/CRSP  → S&P 500 constituent history + quarterly top-30 by market cap
Step 2  yfinance   → daily OHLCV for every ticker that ever appears in the top-30
Step 3  stockstats → MACD, RSI, CCI, ADX (same indicators as the paper)

Output (all saved to New_Strategy/data/):
  sp500_history.csv      — raw CRSP constituent records
  top30_by_quarter.csv   — 30 tickers × each quarter date
  combined_data.csv      — final daily dataset ready for the RL environment

Run from RL/ directory:
  python3.11 New_Strategy/data_pipeline.py
"""

import os, warnings, time
import numpy as np
import pandas as pd
import yfinance as yf
import wrds
from stockstats import StockDataFrame
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SP500_HIST_PATH  = os.path.join(DATA_DIR, "sp500_history.csv")
TOP30_PATH       = os.path.join(DATA_DIR, "top30_by_quarter.csv")
PRICES_PATH      = os.path.join(DATA_DIR, "universe_prices.csv")
COMBINED_PATH    = os.path.join(DATA_DIR, "combined_data.csv")

# ── Study window ──────────────────────────────────────────────────────────────
STUDY_START = "2009-01-01"
STUDY_END   = "2020-05-08"
N_TOP       = 30

# ── Fixed data split (mirrors original paper period) ─────────────────────────
TRAIN_START = "2009-01-01"
TRAIN_END   = "2015-10-15"
VAL_START   = "2015-10-16"
VAL_END     = "2015-12-31"
TRADE_START = "2016-01-01"
TRADE_END   = "2020-05-08"

# ── Quarterly rebalancing dates (calendar quarter starts) ─────────────────────
# We'll snap each one to the nearest actual trading day after fetching CRSP dates.
QUARTER_STARTS = pd.date_range(STUDY_START, STUDY_END, freq="QS").tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — WRDS: constituent history & market-cap ranking
# ══════════════════════════════════════════════════════════════════════════════

def fetch_sp500_history(db: wrds.Connection) -> pd.DataFrame:
    """Pull S&P 500 member records from crsp.dsp500list."""
    print("  Querying crsp.dsp500list …")
    sp500 = db.raw_sql("""
        SELECT permno,
               start,
               COALESCE(ending, CURRENT_DATE) AS ending
        FROM crsp.dsp500list
        ORDER BY permno, start
    """, date_cols=["start", "ending"])
    print(f"  → {len(sp500):,} membership records, "
          f"{sp500.permno.nunique():,} unique permnos")
    return sp500


def members_on_date(sp500: pd.DataFrame, date: pd.Timestamp) -> set:
    """Return the set of permnos that were S&P 500 members on *date*."""
    mask = (sp500["start"] <= date) & (sp500["ending"] >= date)
    return set(sp500.loc[mask, "permno"])


def fetch_market_caps(db: wrds.Connection,
                      permnos: list,
                      dates: list) -> pd.DataFrame:
    """
    Pull (permno, permco, date, mcap) from crsp.dsf for the given permnos.
    For each quarter-start date, queries the first 5 calendar days so that
    holidays (e.g. Jan 1) are automatically handled — we use the first
    actual trading day of each quarter in build_top30_table().
    """
    permno_str = ",".join(str(p) for p in permnos)

    # Expand each quarter date to a window of +5 calendar days
    expanded = set()
    for d in dates:
        for delta in range(6):
            expanded.add(d + pd.Timedelta(days=delta))
    date_str = ",".join(f"'{d.strftime('%Y-%m-%d')}'" for d in sorted(expanded))

    print(f"  Querying market caps for {len(permnos):,} permnos "
          f"({len(dates)} quarters × up to 6 days each) …")
    mcap = db.raw_sql(f"""
        SELECT a.permno, a.permco, a.date,
               ABS(a.prc) * a.shrout * 1000 AS mcap
        FROM crsp.dsf a
        WHERE a.date IN ({date_str})
          AND a.permno IN ({permno_str})
          AND a.prc   IS NOT NULL
          AND a.shrout IS NOT NULL
          AND a.shrout > 0
    """, date_cols=["date"])
    return mcap


def fetch_ticker_map(db: wrds.Connection, permnos: list) -> pd.DataFrame:
    """
    Map permno → most-recent ticker + company name via crsp.stocknames.
    Returns a DataFrame indexed by permno.
    """
    permno_str = ",".join(str(p) for p in permnos)
    names = db.raw_sql(f"""
        SELECT DISTINCT ON (permno)
               permno, ticker, comnam
        FROM crsp.stocknames
        WHERE permno IN ({permno_str})
        ORDER BY permno, nameenddt DESC NULLS FIRST
    """)
    return names.set_index("permno")


def build_top30_table(sp500: pd.DataFrame,
                      mcap_df: pd.DataFrame,
                      ticker_map: pd.DataFrame,
                      quarter_dates: list) -> pd.DataFrame:
    """
    For each quarter date, select the top-N_TOP S&P 500 stocks by market cap,
    deduplicating by permco (permanent company ID) so dual-share-class firms
    (e.g. GOOGL/GOOG, BRK.A/BRK.B) appear only once.
    """
    records = []
    for qdate in quarter_dates:
        members = members_on_date(sp500, qdate)

        # Use the first actual trading day within 5 calendar days of quarter start
        # (handles Jan 1 holiday, weekends, etc.)
        window = mcap_df[
            (mcap_df["date"] >= qdate) &
            (mcap_df["date"] <= qdate + pd.Timedelta(days=5)) &
            (mcap_df["permno"].isin(members))
        ].dropna(subset=["mcap"])

        if len(window) == 0:
            print(f"  WARNING: no market cap data near {qdate.date()}, skipping")
            continue

        # Pick the earliest trading day in the window (first day of the quarter)
        actual_date = window["date"].min()
        day = window[window["date"] == actual_date].copy()

        # Deduplicate: keep highest market-cap share class per company
        day = (day.sort_values("mcap", ascending=False)
                  .drop_duplicates(subset="permco", keep="first"))

        top30 = day.nlargest(N_TOP, "mcap").copy()
        top30["rank"]         = range(1, len(top30) + 1)
        top30["quarter_date"] = qdate

        # Attach ticker
        top30["ticker"] = top30["permno"].map(
            lambda p: ticker_map.at[p, "ticker"] if p in ticker_map.index else None
        )
        top30["comnam"] = top30["permno"].map(
            lambda p: ticker_map.at[p, "comnam"] if p in ticker_map.index else None
        )
        records.append(top30[["quarter_date", "rank", "permno",
                               "permco", "ticker", "comnam", "mcap"]])

    result = pd.concat(records, ignore_index=True)
    print(f"  → Top-{N_TOP} table: {len(result):,} rows across "
          f"{result.quarter_date.nunique()} quarters")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — yfinance: download OHLCV for the full universe
# ══════════════════════════════════════════════════════════════════════════════

# Manual overrides for tickers that CRSP records differently from yfinance
TICKER_OVERRIDE = {
    "GOOGL": "GOOGL",
    "GOOG":  "GOOG",
    "BRK.B": "BRK-B",
    "BRK/B": "BRK-B",
    "BRK":   "BRK-B",   # CRSP stores as BRK, yfinance needs BRK-B
    "BF.B":  "BF-B",
}

def yf_ticker(crsp_ticker: str) -> str:
    """Convert CRSP ticker to yfinance-compatible symbol."""
    if crsp_ticker in TICKER_OVERRIDE:
        return TICKER_OVERRIDE[crsp_ticker]
    # CRSP uses spaces for share-class suffixes; yfinance uses '-'
    return crsp_ticker.replace(" ", "-").strip()


def download_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted daily OHLCV from yfinance for *tickers*.
    Returns long-format DataFrame: [date, tic, open, high, low, close, volume, adjcp]
    """
    yf_tickers = [yf_ticker(t) for t in tickers]
    print(f"  Downloading yfinance data for {len(yf_tickers)} tickers …")

    raw = yf.download(
        yf_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    frames = []
    for tic, yf_tic in zip(tickers, yf_tickers):
        try:
            if len(yf_tickers) == 1:
                df = raw.copy()
            else:
                df = raw[yf_tic].copy()
            df = df.dropna(subset=["Close"])
            df["tic"]    = tic          # use original CRSP ticker as label
            df["adjcp"]  = df["Close"]
            df["open"]   = df["Open"]
            df["high"]   = df["High"]
            df["low"]    = df["Low"]
            df["volume"] = df["Volume"]
            df = df.reset_index().rename(columns={"Date": "date"})
            df["date"] = pd.to_datetime(df["date"])
            frames.append(df[["date", "tic", "open", "high",
                               "low", "adjcp", "volume"]])
        except Exception as e:
            print(f"    WARNING: {tic} ({yf_tic}) failed → {e}")

    if not frames:
        raise RuntimeError("No price data downloaded — check tickers.")

    prices = pd.concat(frames, ignore_index=True)
    print(f"  → {len(prices):,} rows, {prices.tic.nunique()} tickers, "
          f"{prices.date.min().date()} – {prices.date.max().date()}")
    return prices


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Technical indicators (MACD, RSI, CCI, ADX)
# ══════════════════════════════════════════════════════════════════════════════

def add_technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MACD, RSI, CCI, ADX for each ticker using stockstats.
    Expects columns: date, tic, open, high, low, adjcp (=close), volume.
    """
    frames = []
    failed = []
    for tic, grp in prices.groupby("tic"):
        grp = grp.sort_values("date").copy()
        grp = grp.rename(columns={"adjcp": "close"})

        try:
            sdf = StockDataFrame.retype(grp.copy())
            grp["macd"] = sdf["macd"].values
            grp["rsi"]  = sdf["rsi_30"].values
            grp["cci"]  = sdf["cci_30"].values
            grp["adx"]  = sdf["dx_30"].values
        except Exception as e:
            failed.append(f"{tic}: {e}")
            grp["macd"] = np.nan
            grp["rsi"]  = np.nan
            grp["cci"]  = np.nan
            grp["adx"]  = np.nan

        grp = grp.rename(columns={"close": "adjcp"})
        frames.append(grp)

    if failed:
        print(f"  WARNING — indicators failed for {len(failed)} tickers:")
        for msg in failed[:5]:   # show first 5 errors
            print(f"    {msg}")

    result = pd.concat(frames, ignore_index=True)
    before = len(result)
    result = result.dropna(subset=["macd", "rsi", "cci", "adx"])
    print(f"  Dropped {before - len(result):,} NaN rows → {len(result):,} remain")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Assemble combined_data.csv
# ══════════════════════════════════════════════════════════════════════════════

def build_combined(prices_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Keep ALL price+indicator data for the full universe (all 50 tickers,
    full 2009-2020 period).  No filtering by active top-30 status —
    that filtering happens at training time in ensemble.py when we
    select the current quarter's 30 tickers.

    Output columns:
      datadate  tic  adjcp  macd  rsi  cci  adx
    """
    merged = prices_ind.copy()
    merged["datadate"] = merged["date"].dt.strftime("%Y%m%d").astype(int)
    merged = merged.sort_values(["datadate", "tic"])

    print(f"  Combined dataset: {merged.datadate.min()} – {merged.datadate.max()}, "
          f"{merged.datadate.nunique()} trading days, "
          f"{merged.tic.nunique()} unique tickers")
    return merged.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    t_total = time.time()

    # ── STEP 1: WRDS ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1 — WRDS: S&P 500 constituents + market-cap ranking")
    print("="*60)

    if os.path.exists(TOP30_PATH):
        print(f"  Loading cached top-30 table from {TOP30_PATH}")
        top30 = pd.read_csv(TOP30_PATH, parse_dates=["quarter_date"])
    else:
        db = wrds.Connection()

        # 1a. Constituent history
        if os.path.exists(SP500_HIST_PATH):
            print(f"  Loading cached constituent history from {SP500_HIST_PATH}")
            sp500 = pd.read_csv(SP500_HIST_PATH,
                                parse_dates=["start", "ending"])
        else:
            sp500 = fetch_sp500_history(db)
            sp500.to_csv(SP500_HIST_PATH, index=False)
            print(f"  Saved → {SP500_HIST_PATH}")

        # 1b. All permnos ever in S&P 500
        all_permnos = sp500["permno"].unique().tolist()

        # 1c. Market caps on each quarter date
        #     Query in yearly batches to avoid very large SQL IN clauses
        mcap_frames = []
        year_batches = {}
        for qd in QUARTER_STARTS:
            yr = qd.year
            year_batches.setdefault(yr, []).append(qd)

        for yr, dates in sorted(year_batches.items()):
            print(f"  Year {yr}: {len(dates)} quarter dates")
            batch = fetch_market_caps(db, all_permnos, dates)
            mcap_frames.append(batch)
            time.sleep(0.5)   # be polite to WRDS

        mcap_df = pd.concat(mcap_frames, ignore_index=True)

        # 1d. Ticker map
        universe_permnos = mcap_df["permno"].unique().tolist()
        ticker_map = fetch_ticker_map(db, universe_permnos)

        db.close()

        # 1e. Build top-30 per quarter
        top30 = build_top30_table(sp500, mcap_df, ticker_map, QUARTER_STARTS)
        top30.to_csv(TOP30_PATH, index=False)
        print(f"  Saved → {TOP30_PATH}")

    # ── Show universe summary ──────────────────────────────────────────────────
    all_tickers = top30["ticker"].dropna().unique().tolist()
    print(f"\n  Total unique tickers ever in top-{N_TOP}: {len(all_tickers)}")
    print(f"  Sample: {sorted(all_tickers)[:10]} …")

    # ── STEP 2: yfinance prices ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2 — yfinance: OHLCV download")
    print("="*60)

    if os.path.exists(PRICES_PATH):
        print(f"  Loading cached prices from {PRICES_PATH}")
        prices = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    else:
        prices = download_prices(all_tickers, STUDY_START, STUDY_END)
        prices.to_csv(PRICES_PATH, index=False)
        print(f"  Saved → {PRICES_PATH}")

    # ── STEP 3: technical indicators ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3 — Technical indicators (MACD, RSI, CCI, ADX)")
    print("="*60)
    prices_ind = add_technical_indicators(prices)
    print(f"  After dropping NaN indicator rows: {len(prices_ind):,} rows")

    # ── STEP 4: assemble combined_data.csv ───────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 4 — Assembling combined_data.csv")
    print("="*60)
    combined = build_combined(prices_ind)
    combined.to_csv(COMBINED_PATH, index=False)
    print(f"  Saved → {COMBINED_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (time.time() - t_total) / 60
    print("\n" + "="*60)
    print("  DONE")
    print("="*60)
    print(f"  Runtime: {elapsed:.1f} min")
    print(f"\n  Files written to {DATA_DIR}/")
    print(f"    sp500_history.csv      — CRSP constituent records")
    print(f"    top30_by_quarter.csv   — {N_TOP} tickers × {top30.quarter_date.nunique()} quarters")
    print(f"    universe_prices.csv    — raw OHLCV")
    print(f"    combined_data.csv      — {len(combined):,} rows  "
          f"({combined.tic.nunique()} tickers × full history, ready for RL)")

    # ── Quick universe snapshot ────────────────────────────────────────────────
    print("\n  Top-30 snapshot for most recent quarter:")
    latest = top30[top30.quarter_date == top30.quarter_date.max()]
    for _, row in latest.iterrows():
        print(f"    {row['rank']:>2}.  {row['ticker']:<8}  "
              f"${row['mcap']/1e9:.0f}B  {row['comnam']}")
