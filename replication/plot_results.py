"""
Standalone plotting script — run after ensemble_drl_replication.py completes.
Loads saved portfolio CSVs and generates the comparison figure with:
  - Our replication ensemble
  - Paper's reported final return (70.4%) as a reference line
  - DJIA (^DJI)
  - S&P 500 (^GSPC)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OOS_START   = "2016-01-04"
OOS_END     = "2020-05-08"
PAPER_FINAL = 0.704   # paper Table 2 ensemble cumulative return

# ── 1. Load replication portfolio ────────────────────────────────────────────
files = sorted(
    [f for f in os.listdir(RESULTS_DIR) if f.startswith("account_value_trade_ensemble_")],
    key=lambda x: int(x.split("_")[-1].replace(".csv", ""))
)
portfolio = pd.concat(
    [pd.read_csv(os.path.join(RESULTS_DIR, f), index_col=0)["account_value"] for f in files],
    ignore_index=True
)
print(f"Portfolio loaded: {len(portfolio)} daily observations from {len(files)} quarters")

# ── 2. Download benchmarks & build shared date index ─────────────────────────
print("Downloading DJIA and S&P 500 …")
dji_raw  = yf.download("^DJI",  start=OOS_START, end=OOS_END, auto_adjust=True, progress=False)
sp5_raw  = yf.download("^GSPC", start=OOS_START, end=OOS_END, auto_adjust=True, progress=False)

dji_close = dji_raw["Close"].squeeze().dropna()
sp5_close = sp5_raw["Close"].squeeze().dropna()

# Use DJIA trading dates as the shared x-axis
trade_dates = pd.to_datetime(dji_close.index)
n = min(len(portfolio), len(trade_dates))

# Cumulative returns (all start at 0)
cum_rep  = (portfolio.iloc[:n].values / portfolio.iloc[0]) - 1
cum_dji  = (dji_close.iloc[:n].values / dji_close.iloc[0]) - 1
cum_sp5  = sp5_close.reindex(trade_dates[:n]).ffill()
cum_sp5  = (cum_sp5.values / cum_sp5.iloc[0]) - 1
dates    = trade_dates[:n]

print(f"  Replication final : {cum_rep[-1]*100:.1f}%")
print(f"  DJIA final        : {cum_dji[-1]*100:.1f}%")
print(f"  S&P 500 final     : {cum_sp5[-1]*100:.1f}%")
print(f"  Paper reported    : {PAPER_FINAL*100:.1f}%")

# ── 3. Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(dates, cum_rep,  color="firebrick",   linewidth=2.2, label="Ensemble DRL — Our Replication")
ax.plot(dates, cum_dji,  color="steelblue",   linewidth=1.6, linestyle="--", label="DJIA (^DJI)")
ax.plot(dates, cum_sp5,  color="darkorange",  linewidth=1.6, linestyle="--", label="S&P 500 (^GSPC)")
ax.axhline(PAPER_FINAL,  color="green",       linewidth=1.4, linestyle=":",
           label=f"Paper ensemble — reported final return (+{PAPER_FINAL*100:.1f}%)")

ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

# Right-edge value labels
for label, val, col in [
    ("Replication",         float(cum_rep[-1]),  "firebrick"),
    ("DJIA",                float(cum_dji[-1]),  "steelblue"),
    ("S&P 500",             float(cum_sp5[-1]),  "darkorange"),
    ("Paper\n(reported)",   PAPER_FINAL,         "green"),
]:
    ax.annotate(f"{val*100:.1f}%",
                xy=(dates[-1], val),
                xytext=(6, 0), textcoords="offset points",
                color=col, fontsize=9, va="center", fontweight="bold")

ax.set_title("Cumulative Return 2016–2020: Ensemble DRL vs Benchmarks", fontsize=14, pad=12)
ax.set_ylabel("Cumulative Return")
ax.set_xlabel("")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.legend(loc="upper left", framealpha=0.92, fontsize=10)
ax.grid(alpha=0.25)
ax.set_xlim(dates[0], dates[-1] + pd.Timedelta(days=40))   # right margin for labels

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "comparison_with_benchmarks.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nFigure saved → {out}")
