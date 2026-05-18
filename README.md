# Deep Reinforcement Learning for Automated Stock Trading

Replication and extension of **Yang et al. (2020) — "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy"** (ICAIF 2020).

---

## Overview

This repository contains:

| Folder | Description |
|--------|-------------|
| `replication/` | Faithful replication of the paper using modern libraries (stable-baselines3 + gymnasium) |
| `New_Strategy/` | Extended strategy replacing DJI fixed-30 with **S&P 500 rolling top-30 by market cap** via WRDS |

---

## Paper Summary

The paper proposes an **ensemble of three RL agents** for automated stock trading:

- **A2C** (Advantage Actor-Critic)
- **PPO** (Proximal Policy Optimization)
- **DDPG** (Deep Deterministic Policy Gradient)

At each quarterly rebalancing, the agent with the highest **validation Sharpe ratio** is selected to trade the next quarter. A **turbulence index** (Mahalanobis distance) triggers full liquidation during market stress.

**State vector** (181-dim for 30 stocks): `balance + prices + shares + MACD + RSI + CCI + ADX`

**Reward**: change in portfolio value × 1e-4

**Original results** (2016–2020, DJI 30 stocks): Cumulative Return +70.4%, Sharpe 1.30, Max Drawdown −9.7%

---

## Replication (`replication/`)

### Files

| File | Description |
|------|-------------|
| `ensemble_drl_replication.py` | Main replication script — trains and trades 2016–2020 on DJI 30 stocks |
| `extended_ensemble.py` | Extended study — trains 2009–2020, trades 2021–2026 on 29 DJI stocks |
| `learning_curve_analysis.py` | Finds optimal timesteps for each agent via validation Sharpe curves |
| `plot_results.py` | Standalone plotting script for comparison figures |

### Key Migration Decisions (stable-baselines → stable-baselines3)

- `gymnasium` API: `step()` returns 5 values; `reset()` returns `(obs, info)`
- PPO hyperparameters to match original PPO2: `n_steps=128, batch_size=16, learning_rate=7e-4, n_epochs=4, ent_coef=0.005`
- Sharpe ratio uses `rf = 0` (matching the paper, not 0.02)
- `DummyVecEnv` auto-resets on `done=True` — evaluation must use raw env directly

### Replication Results (2016–2020)

| Metric | Paper | Our Replication |
|--------|-------|-----------------|
| Cumulative Return | +70.4% | +71.9% |
| Sharpe Ratio | 1.30 | 1.55 |
| Max Drawdown | −9.7% | −7.3% |

### How to Run

```bash
cd RL/

# 1. Full replication (2016-2020)
python3.11 replication/ensemble_drl_replication.py

# 2. Extended study (2021-2026)
python3.11 replication/extended_ensemble.py

# 3. Learning curve analysis (find optimal timesteps)
python3.11 replication/learning_curve_analysis.py
```

---

## New Strategy (`New_Strategy/`)

### Motivation

The original paper has two limitations:
1. **Fixed stock universe** — DJI 30 stocks are fixed regardless of market evolution
2. **Share-based actions** — buying/selling in fixed share quantities disadvantages high-priced stocks

### Design

**Single change**: Replace DJI fixed-30 with **S&P 500 rolling top-30 by market cap**, updated every quarter using point-in-time WRDS/CRSP data (avoiding survivorship bias).

All other methodology is identical to the paper (same RL agents, same ensemble logic, same reward function).

**Data source**: WRDS CRSP (`crsp.dsp500list` for constituent history, `crsp.dsf` for market cap)

**Period**: Train 2009-01-01 → 2015-10-15 | Val 2015-10-16 → 2015-12-31 | Trade 2016-01-01 → 2020-05-08

### Rolling Universe

| Event | Handling |
|-------|---------|
| Stock leaves top-30 | Positions liquidated at quarter end |
| New stock enters top-30 | Starts with zero position |
| Dual share-class (GOOG/GOOGL, BRK) | Deduplicated by `permco`, keep highest market-cap class |

### Files

| File | Description |
|------|-------------|
| `data_pipeline.py` | Downloads S&P 500 constituent history from WRDS + OHLCV from yfinance |
| `ensemble.py` | Rolling ensemble strategy with quarterly universe rebalancing |
| `check_wrds_access.py` | Diagnostic tool to verify WRDS table access |

### How to Run

```bash
# Step 1: Download data (requires WRDS account)
python3.11 New_Strategy/data_pipeline.py

# Step 2: Run ensemble strategy
python3.11 New_Strategy/ensemble.py
```

**Note**: WRDS access is required for Step 1. The pipeline caches intermediate results — subsequent runs are fast.

---

## Installation

```bash
pip install stable-baselines3 gymnasium torch
pip install yfinance stockstats pandas numpy matplotlib
pip install wrds   # only needed for New_Strategy data pipeline
```

Python 3.11 recommended.

---

## Repository Structure

```
RL/
├── replication/
│   ├── ensemble_drl_replication.py   # Paper replication (2016-2020)
│   ├── extended_ensemble.py          # Extended study (2021-2026)
│   ├── learning_curve_analysis.py    # Optimal timestep analysis
│   ├── plot_results.py               # Standalone plot script
│   └── results/                      # Figures
│       ├── comparison_with_benchmarks.png
│       ├── authors_vs_replication.png
│       └── ensemble_replication_results.png
│
└── New_Strategy/
    ├── data_pipeline.py              # WRDS data download
    ├── ensemble.py                   # Rolling S&P 500 top-30 strategy
    └── results/                      # Figures (generated after running)
```

---

## Reference

```bibtex
@inproceedings{yang2020deep,
  title     = {Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy},
  author    = {Yang, Hongyang and Liu, Xiao-Yang and Zhong, Shan and Walid, Anwar},
  booktitle = {Proceedings of the First ACM International Conference on AI in Finance (ICAIF)},
  year      = {2020}
}
```
