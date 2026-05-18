"""
Extended replication of:
  "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy"
  Yang et al., ICAIF 2020

EXTENSION:
  - Combined dataset: 2009-01-01 → 2026-05-12  (29 DJIA stocks; WBA dropped — delisted 2024)
  - In-sample training:  2009-01-01 → 2020-12-31
  - Out-of-sample trade: 2021-01-01 → 2026-05-12  (quarterly rolling, same ensemble logic)

COMPARISON produced at the end:
  Paper original (2016-2020)  vs  Our replication (2016-2020)  vs  Extension (2021-2026)

Run with:
  python3.11 replication/extended_ensemble.py
from the RL/ directory.
"""

# ── 0. Imports ───────────────────────────────────────────────────────────────
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
warnings.filterwarnings("ignore")

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stockstats import StockDataFrame as Sdf

# ── 1. Constants ─────────────────────────────────────────────────────────────
# WBA delisted June 2024 → use 29 stocks throughout for consistency
TICKERS = ['AAPL','AXP','BA','CAT','CSCO','CVX','DD','DIS','GS','HD',
           'IBM','INTC','JNJ','JPM','KO','MCD','MMM','MRK','MSFT','NKE',
           'PFE','PG','RTX','TRV','UNH','V','VZ','WMT','XOM']

STOCK_DIM        = len(TICKERS)       # 29
INITIAL_BALANCE  = 1_000_000
HMAX_NORMALIZE   = 100
TRANSACTION_FEE  = 0.001
REWARD_SCALING   = 1e-4
STATE_DIM        = 1 + STOCK_DIM * 6  # 175  (balance + price + shares + 4 indicators, each ×29)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_extended")
os.makedirs(RESULTS_DIR, exist_ok=True)

OLD_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020",
    "done_data.csv"
)

# ── 2. Data pipeline ─────────────────────────────────────────────────────────

def load_old_data():
    """Load the pre-processed 2009-2020 data, drop WBA, keep the 29 tickers."""
    df = pd.read_csv(OLD_DATA_PATH, index_col=0)
    df = df[df.tic.isin(TICKERS)].copy()
    df = df[["datadate", "tic", "adjcp", "open", "high", "low", "volume",
             "macd", "rsi", "cci", "adx", "turbulence"]]
    return df


def download_new_data(start="2019-12-01", end="2026-05-13"):
    """
    Download daily OHLCV from yfinance for the 29 tickers.
    The overlap with the old data (2019-12-01 → 2020-08-17) lets the turbulence
    recalculation use a continuous price history.
    """
    print("  Downloading from yfinance …")
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)

    rows = []
    for tic in TICKERS:
        sub = pd.DataFrame({
            "date":   raw.index,
            "tic":    tic,
            "adjcp":  raw["Close"][tic].values,
            "open":   raw["Open"][tic].values,
            "high":   raw["High"][tic].values,
            "low":    raw["Low"][tic].values,
            "volume": raw["Volume"][tic].values,
        })
        sub = sub.dropna(subset=["adjcp"])
        rows.append(sub)

    df = pd.concat(rows, ignore_index=True)
    df["datadate"] = df["date"].dt.strftime("%Y%m%d").astype(int)
    return df[["datadate", "tic", "adjcp", "open", "high", "low", "volume"]]


def add_technical_indicators(df):
    """Compute MACD, RSI-30, CCI-30, ADX-30 per ticker using stockstats."""
    records = []
    for tic in df.tic.unique():
        sub = df[df.tic == tic].copy().reset_index(drop=True)
        sub = sub.rename(columns={"adjcp": "close"})
        sdf = Sdf.retype(sub.copy())
        sub["macd"] = sdf["macd"].values
        sub["rsi"]  = sdf["rsi_30"].values
        sub["cci"]  = sdf["cci_30"].values
        sub["adx"]  = sdf["dx_30"].values
        sub = sub.rename(columns={"close": "adjcp"})
        records.append(sub)

    out = pd.concat(records, ignore_index=True)
    out.fillna(method="bfill", inplace=True)
    return out[["datadate", "tic", "adjcp", "open", "high", "low", "volume",
                "macd", "rsi", "cci", "adx"]]


def compute_turbulence(df):
    """
    Mahalanobis-distance turbulence index (Equation 3 in the paper).
    turbulence_t = (y_t - mu)^T Sigma^{-1} (y_t - mu)
    where y_t is the vector of daily returns at time t,
    mu and Sigma are computed over all historical data up to t.
    The first 252 trading days are set to 0.
    """
    pivot = df.pivot(index="datadate", columns="tic", values="adjcp")
    pivot = pivot[TICKERS]                # ensure consistent column order
    returns = pivot.pct_change().dropna()
    unique_dates = returns.index.tolist()
    start = 252
    turb = [0.0] * (start + 1)           # +1 for the dropped first return row

    for i in range(start, len(unique_dates)):
        current = returns.iloc[i].values          # shape (D,)
        history = returns.iloc[:i].values
        mu      = history.mean(axis=0)            # shape (D,)
        cov     = np.cov(history.T)               # shape (D, D)
        try:
            diff = current - mu                   # shape (D,) — 1D keeps result scalar
            val  = float(diff @ np.linalg.inv(cov) @ diff)
            turb.append(max(val, 0.0))
        except np.linalg.LinAlgError:
            turb.append(0.0)

    # align back to the original date index (which includes the first row dropped by pct_change)
    all_dates = pivot.index.tolist()
    turb_padded = [0.0] + turb           # one extra 0 for the first date before pct_change
    turb_series = pd.DataFrame({"datadate": all_dates, "turbulence": turb_padded[:len(all_dates)]})
    return turb_series


def build_combined_dataset(cache_path=os.path.join(os.path.dirname(__file__), "combined_data.csv")):
    """
    Build and cache the full 2009-2026 dataset.
    Recomputes turbulence on the entire combined price history so the index
    is continuous and uses consistent historical covariance.
    """
    if os.path.exists(cache_path):
        print(f"  Loading cached combined dataset from {cache_path}")
        return pd.read_csv(cache_path, index_col=0)

    print("Building combined dataset …")

    # 2009-2020: use pre-processed data (indicators already computed)
    old = load_old_data()
    old_end = old.datadate.max()
    print(f"  Old data: {old.datadate.min()} – {old_end}  ({old.datadate.nunique()} days)")

    # 2020-2026: download and compute indicators
    new_raw = download_new_data()
    # Keep only dates strictly after the old dataset ends to avoid duplicates
    new_raw = new_raw[new_raw.datadate > old_end]
    print(f"  New raw: {new_raw.datadate.min()} – {new_raw.datadate.max()}")

    # Compute indicators on new data (need a small lookback, so download from 2019-12-01)
    # Re-download with lookback for indicator warm-up
    lookback_raw = download_new_data(start="2019-06-01")
    new_with_ind = add_technical_indicators(lookback_raw)
    # Trim back to only new dates
    new_with_ind = new_with_ind[new_with_ind.datadate > old_end]
    print(f"  New data with indicators: {new_with_ind.datadate.min()} – {new_with_ind.datadate.max()}  "
          f"({new_with_ind.datadate.nunique()} days)")

    # Stack old (without turbulence column for now) + new
    old_no_turb  = old.drop(columns=["turbulence"])
    combined_raw = pd.concat([old_no_turb, new_with_ind], ignore_index=True)
    combined_raw = combined_raw.sort_values(["datadate", "tic"]).reset_index(drop=True)

    # Recompute turbulence on full history
    print("  Computing turbulence index on full 2009-2026 history (this takes a few minutes) …")
    t0 = time.time()
    turb = compute_turbulence(combined_raw)
    print(f"    done in {(time.time()-t0)/60:.1f} min")

    combined = combined_raw.merge(turb, on="datadate")
    combined = combined.sort_values(["datadate", "tic"]).reset_index(drop=True)

    combined.to_csv(cache_path)
    print(f"  Saved to {cache_path}")
    return combined


# ── 3. Utility ───────────────────────────────────────────────────────────────

def data_split(df, start, end):
    data = df[(df.datadate >= start) & (df.datadate < end)].copy()
    data = data.sort_values(["datadate", "tic"], ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data


# ── 4. Gymnasium environments (29-stock versions) ────────────────────────────

class StockEnvTrain(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, df):
        super().__init__()
        self.df = df; self.day = 0
        self.action_space      = spaces.Box(-1, 1, (STOCK_DIM,), np.float32)
        self.observation_space = spaces.Box(0, np.inf, (STATE_DIM,), np.float32)
        self._reset_internals()

    def _obs(self):
        d = self.df.loc[self.day]
        return np.array([self.balance] + d.adjcp.tolist() + list(self.shares)
                        + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist(),
                        dtype=np.float32)

    def _reset_internals(self):
        self.balance = float(INITIAL_BALANCE)
        self.shares  = np.zeros(STOCK_DIM, np.float32)
        self.day     = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self._reset_internals(); return self._obs(), {}

    def step(self, actions):
        if self.day >= len(self.df.index.unique()) - 1:
            return self._obs(), 0.0, True, False, {}
        prices = self.df.loc[self.day].adjcp.values
        begin  = self.balance + float(np.dot(prices, self.shares))
        actions = actions * HMAX_NORMALIZE
        idx = np.argsort(actions)
        for i in idx[actions[idx] < 0]:
            qty = min(abs(actions[i]), self.shares[i])
            if qty > 0:
                self.balance += prices[i] * qty * (1 - TRANSACTION_FEE)
                self.shares[i] -= qty
        for i in idx[::-1][actions[idx[::-1]] > 0]:
            if prices[i] <= 0: continue
            qty = min(int(self.balance // prices[i]), int(actions[i]))
            if qty > 0:
                self.balance -= prices[i] * qty * (1 + TRANSACTION_FEE)
                self.shares[i] += qty
        self.day += 1
        new_prices = self.df.loc[self.day].adjcp.values
        end = self.balance + float(np.dot(new_prices, self.shares))
        return self._obs(), (end - begin) * REWARD_SCALING, False, False, {}


class StockEnvValidation(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, df, turb_threshold, iteration):
        super().__init__()
        self.df = df; self.turb_threshold = turb_threshold
        self.iteration = iteration; self.day = 0
        self.action_space      = spaces.Box(-1, 1, (STOCK_DIM,), np.float32)
        self.observation_space = spaces.Box(0, np.inf, (STATE_DIM,), np.float32)
        self._reset_internals()

    def _obs(self):
        d = self.df.loc[self.day]
        return np.array([self.balance] + d.adjcp.tolist() + list(self.shares)
                        + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist(),
                        dtype=np.float32)

    def _reset_internals(self):
        self.balance = float(INITIAL_BALANCE)
        self.shares  = np.zeros(STOCK_DIM, np.float32)
        self.day = 0; self.turbulence = 0.0
        self.asset_memory = [INITIAL_BALANCE]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self._reset_internals(); return self._obs(), {}

    def step(self, actions):
        if self.day >= len(self.df.index.unique()) - 1:
            pd.DataFrame(self.asset_memory).to_csv(
                os.path.join(RESULTS_DIR, f"account_value_validation_{self.iteration}.csv"))
            return self._obs(), 0.0, True, False, {}
        if self.turbulence >= self.turb_threshold:
            actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM, np.float32)
        else:
            actions = actions * HMAX_NORMALIZE
        prices = self.df.loc[self.day].adjcp.values
        begin  = self.balance + float(np.dot(prices, self.shares))
        idx = np.argsort(actions)
        for i in idx[actions[idx] < 0]:
            qty = min(abs(actions[i]), self.shares[i])
            if qty > 0:
                self.balance += prices[i] * qty * (1 - TRANSACTION_FEE)
                self.shares[i] -= qty
        for i in idx[::-1][actions[idx[::-1]] > 0]:
            if prices[i] <= 0: continue
            qty = min(int(self.balance // prices[i]), int(actions[i]))
            if qty > 0:
                self.balance -= prices[i] * qty * (1 + TRANSACTION_FEE)
                self.shares[i] += qty
        self.day += 1
        self.turbulence = float(self.df.loc[self.day].turbulence.iloc[0])
        new_prices = self.df.loc[self.day].adjcp.values
        end = self.balance + float(np.dot(new_prices, self.shares))
        self.asset_memory.append(end)
        return self._obs(), (end - begin) * REWARD_SCALING, False, False, {}


class StockEnvTrade(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, df, turb_threshold, initial=True, previous_state=None,
                 model_name="", iteration=""):
        super().__init__()
        self.df = df; self.turb_threshold = turb_threshold
        self.initial = initial; self.previous_state = previous_state or []
        self.model_name = model_name; self.iteration = iteration; self.day = 0
        self.action_space      = spaces.Box(-1, 1, (STOCK_DIM,), np.float32)
        self.observation_space = spaces.Box(0, np.inf, (STATE_DIM,), np.float32)
        self._reset_internals()

    def _obs(self):
        d = self.df.loc[self.day]
        return np.array([self.balance] + d.adjcp.tolist() + list(self.shares)
                        + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist(),
                        dtype=np.float32)

    def _reset_internals(self):
        if self.initial or not self.previous_state:
            self.balance = float(INITIAL_BALANCE)
            self.shares  = np.zeros(STOCK_DIM, np.float32)
        else:
            ps = self.previous_state
            self.balance = float(ps[0])
            self.shares  = np.array(ps[STOCK_DIM + 1: STOCK_DIM * 2 + 1], np.float32)
        self.day = 0; self.turbulence = 0.0
        self.asset_memory = [
            INITIAL_BALANCE if (self.initial or not self.previous_state)
            else self.balance + float(np.dot(self.df.loc[0].adjcp.values, self.shares))
        ]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self._reset_internals(); return self._obs(), {}

    def step(self, actions):
        if self.day >= len(self.df.index.unique()) - 1:
            pd.DataFrame(self.asset_memory, columns=["account_value"]).to_csv(
                os.path.join(RESULTS_DIR, f"account_value_trade_{self.model_name}_{self.iteration}.csv"))
            return self._obs(), 0.0, True, False, {}
        if self.turbulence >= self.turb_threshold:
            actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM, np.float32)
        else:
            actions = actions * HMAX_NORMALIZE
        prices = self.df.loc[self.day].adjcp.values
        begin  = self.balance + float(np.dot(prices, self.shares))
        idx = np.argsort(actions)
        for i in idx[actions[idx] < 0]:
            qty = min(abs(actions[i]), self.shares[i])
            if qty > 0:
                self.balance += prices[i] * qty * (1 - TRANSACTION_FEE)
                self.shares[i] -= qty
        for i in idx[::-1][actions[idx[::-1]] > 0]:
            if prices[i] <= 0: continue
            qty = min(int(self.balance // prices[i]), int(actions[i]))
            if qty > 0:
                self.balance -= prices[i] * qty * (1 + TRANSACTION_FEE)
                self.shares[i] += qty
        self.day += 1
        self.turbulence = float(self.df.loc[self.day].turbulence.iloc[0])
        new_prices = self.df.loc[self.day].adjcp.values
        end = self.balance + float(np.dot(new_prices, self.shares))
        self.asset_memory.append(end)
        return self._obs(), (end - begin) * REWARD_SCALING, False, False, {}

    def render(self):
        d = self.df.loc[self.day]
        return ([self.balance] + d.adjcp.tolist() + list(self.shares)
                + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist())


# ── 5. Agent helpers ─────────────────────────────────────────────────────────

def make_train_env(df):
    return DummyVecEnv([lambda: StockEnvTrain(df)])

def train_a2c(env, timesteps=100_000):
    t0 = time.time()
    m = A2C("MlpPolicy", env, verbose=0)
    m.learn(total_timesteps=timesteps)
    print(f"    A2C  ({(time.time()-t0)/60:.1f} min)")
    return m

def train_ppo(env, timesteps=100_000):
    t0 = time.time()
    # Match original PPO2 hyperparameters: n_steps=128, batch_size=16, lr=7e-4, n_epochs=4
    m = PPO("MlpPolicy", env, ent_coef=0.005,
            n_steps=128, batch_size=16, learning_rate=7e-4, n_epochs=4, verbose=0)
    m.learn(total_timesteps=timesteps)
    print(f"    PPO  ({(time.time()-t0)/60:.1f} min)")
    return m

def train_ddpg(env, timesteps=100_000):
    t0 = time.time()
    n = env.action_space.shape[-1]
    noise = OrnsteinUhlenbeckActionNoise(np.zeros(n), 0.5 * np.ones(n))
    m = DDPG("MlpPolicy", env, action_noise=noise, verbose=0)
    m.learn(total_timesteps=timesteps)
    print(f"    DDPG ({(time.time()-t0)/60:.1f} min)")
    return m

def validate_model(model, val_df, turb_threshold, iteration):
    env = DummyVecEnv([lambda: StockEnvValidation(val_df, turb_threshold, iteration)])
    obs = env.reset()
    for _ in range(len(val_df.index.unique())):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(act)
        if done[0]: break

def get_sharpe(iteration):
    path = os.path.join(RESULTS_DIR, f"account_value_validation_{iteration}.csv")
    if not os.path.exists(path): return 0.0
    vals = pd.read_csv(path, index_col=0).iloc[:, 0]
    ret  = vals.pct_change().dropna()
    return float((4 ** 0.5) * ret.mean() / ret.std()) if ret.std() > 0 else 0.0

def trade_quarter(model, trade_df, last_state, iter_num, turb_threshold, initial, model_name="ensemble"):
    """Deploy a model for one quarter; return the final state vector."""
    env = DummyVecEnv([lambda: StockEnvTrade(
        trade_df, turb_threshold, initial=initial,
        previous_state=last_state, model_name=model_name, iteration=iter_num
    )])
    obs = env.reset()
    n   = len(trade_df.index.unique())
    new_last = last_state
    for step_i in range(n):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(act)
        if step_i == n - 2:
            new_last = env.envs[0].render()
        if done[0]: break
    return new_last


# ── 6. Main strategy loop (Ensemble + all 3 individual agents) ───────────────

def run_all_strategies(df, unique_trade_date, insample_turb,
                       rebalance_window=63, validation_window=63):
    """
    Each quarter:
      1. Train A2C (100k), PPO (100k), DDPG (100k)
      2. Validate each → pick best Sharpe for ensemble
      3. Trade with all four strategies in parallel (each carrying its own state)
    """
    print(f"\n{'='*60}")
    print("  ALL STRATEGIES  (train 2009-2020 | trade 2021-2026)")
    print(f"{'='*60}")

    insample_90th = np.quantile(insample_turb, 0.90)

    # Separate state trackers for each strategy
    last_ensemble = []
    last_a2c      = []
    last_ppo      = []
    last_ddpg     = []

    ppo_sh, a2c_sh, ddpg_sh, model_used = [], [], [], []
    t_start = time.time()

    for i in range(rebalance_window + validation_window,
                   len(unique_trade_date), rebalance_window):

        initial = (i - rebalance_window - validation_window == 0)
        q = len(model_used) + 1
        print(f"\n[Q{q}]  i={i}  initial={initial}")

        # Dynamic turbulence threshold
        anchor  = unique_trade_date[i - rebalance_window - validation_window]
        end_idx = df.index[df.datadate == anchor].tolist()
        end_idx = end_idx[-1] if end_idx else 0
        hist    = df.iloc[max(0, end_idx - validation_window * 30):end_idx + 1].drop_duplicates("datadate")
        turb_threshold = (insample_90th if hist.turbulence.mean() > insample_90th
                          else float(insample_turb.max()))
        print(f"  turb threshold: {turb_threshold:.1f}")

        val_start = unique_trade_date[i - rebalance_window - validation_window]
        val_end   = unique_trade_date[i - rebalance_window]
        trade_end = unique_trade_date[i]

        train_df = data_split(df, 20090000, val_start)
        val_df   = data_split(df, val_start, val_end)
        trade_df = data_split(df, val_end,   trade_end)
        print(f"  train→{val_start} | val {val_start}→{val_end} | trade {val_end}→{trade_end}")

        env_t = make_train_env(train_df)

        # ── Train all three agents ────────────────────────────────────────
        m_a2c  = train_a2c(env_t,  timesteps=100_000)
        validate_model(m_a2c,  val_df, turb_threshold, i)
        s_a2c  = get_sharpe(i);  a2c_sh.append(s_a2c)
        print(f"    A2C  val Sharpe: {s_a2c:.4f}")

        m_ppo  = train_ppo(env_t,  timesteps=10_000)
        validate_model(m_ppo,  val_df, turb_threshold, i)
        s_ppo  = get_sharpe(i);  ppo_sh.append(s_ppo)
        print(f"    PPO  val Sharpe: {s_ppo:.4f}")

        m_ddpg = train_ddpg(env_t, timesteps=5_000)
        validate_model(m_ddpg, val_df, turb_threshold, i)
        s_ddpg = get_sharpe(i);  ddpg_sh.append(s_ddpg)
        print(f"    DDPG val Sharpe: {s_ddpg:.4f}")

        # ── Ensemble selection ────────────────────────────────────────────
        best_s, best_n, best_m = max(
            (s_ppo, "PPO", m_ppo), (s_a2c, "A2C", m_a2c), (s_ddpg, "DDPG", m_ddpg)
        )
        print(f"  → Ensemble picks: {best_n}  (Sharpe {best_s:.4f})")
        model_used.append(best_n)

        # ── Trade — all four strategies carry independent portfolios ──────
        last_ensemble = trade_quarter(best_m, trade_df, last_ensemble, i,
                                      turb_threshold, initial, model_name="ensemble")
        last_a2c      = trade_quarter(m_a2c,  trade_df, last_a2c,      i,
                                      turb_threshold, initial, model_name="a2c")
        last_ppo      = trade_quarter(m_ppo,  trade_df, last_ppo,      i,
                                      turb_threshold, initial, model_name="ppo")
        last_ddpg     = trade_quarter(m_ddpg, trade_df, last_ddpg,     i,
                                      turb_threshold, initial, model_name="ddpg")

    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min")
    return ppo_sh, a2c_sh, ddpg_sh, model_used


# ── 7. Performance helpers ────────────────────────────────────────────────────

def load_portfolio(results_dir, strategy="ensemble"):
    prefix = f"account_value_trade_{strategy}_"
    files  = sorted(
        [f for f in os.listdir(results_dir) if f.startswith(prefix) and f.endswith(".csv")],
        key=lambda x: int(x.split("_")[-1].replace(".csv", ""))
    )
    frames = [pd.read_csv(os.path.join(results_dir, f), index_col=0)["account_value"]
              for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.Series(dtype=float)

def metrics(series, rf=0.0):
    series    = series.dropna()
    daily_ret = series.pct_change().dropna()
    cum       = series.iloc[-1] / series.iloc[0] - 1
    ann_ret   = (1 + cum) ** (252 / max(len(daily_ret), 1)) - 1
    ann_vol   = daily_ret.std() * np.sqrt(252)
    sharpe    = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    mdd       = ((series - series.cummax()) / series.cummax()).min()
    return dict(cum=cum, ann=ann_ret, vol=ann_vol, sharpe=sharpe, mdd=mdd)

def print_metrics(m, label):
    print(f"\n── {label} ──")
    print(f"  Cumulative Return : {m['cum']*100:.1f}%")
    print(f"  Annual Return     : {m['ann']*100:.1f}%")
    print(f"  Annual Volatility : {m['vol']*100:.1f}%")
    print(f"  Sharpe Ratio      : {m['sharpe']:.2f}")
    print(f"  Max Drawdown      : {m['mdd']*100:.1f}%")


# ── 8. Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 8a. Build / load combined dataset ────────────────────────────────────
    df = build_combined_dataset()
    print(f"\nCombined dataset: {df.datadate.min()} – {df.datadate.max()}, "
          f"{df.datadate.nunique()} trading days, {df.tic.nunique()} tickers")

    # ── 8b. In-sample turbulence baseline (2009–2020) ─────────────────────────
    insample      = df[(df.datadate < 20210101) & (df.datadate >= 20090000)].drop_duplicates("datadate")
    insample_turb = insample.turbulence.values
    print(f"In-sample 90th-pct turbulence: {np.quantile(insample_turb, 0.90):.2f}")

    # ── 8c. Out-of-sample date range (2021-2026) ──────────────────────────────
    unique_trade_date = np.sort(
        df[(df.datadate > 20201001) & (df.datadate <= 20260513)].datadate.unique()
    )
    print(f"Out-of-sample: {unique_trade_date[0]} – {unique_trade_date[-1]}, "
          f"{len(unique_trade_date)} dates")

    # ── 8d. Run all strategies ────────────────────────────────────────────────
    ppo_sh, a2c_sh, ddpg_sh, model_used = run_all_strategies(
        df, unique_trade_date, insample_turb, rebalance_window=63, validation_window=63
    )

    print("\n── Ensemble model selection each quarter ──")
    for q, m in enumerate(model_used, 1):
        print(f"  Q{q:>2}: {m}")

    # ── 8e. Load all portfolios ───────────────────────────────────────────────
    p_ensemble = load_portfolio(RESULTS_DIR, "ensemble")
    p_a2c      = load_portfolio(RESULTS_DIR, "a2c")
    p_ppo      = load_portfolio(RESULTS_DIR, "ppo")
    p_ddpg     = load_portfolio(RESULTS_DIR, "ddpg")

    # ── 8f. Benchmarks ────────────────────────────────────────────────────────
    import yfinance as yf
    OOS_START, OOS_END = "2021-01-01", "2026-05-13"
    print("\nDownloading DJIA and S&P 500 …")
    dji_raw = yf.download("^DJI",  start=OOS_START, end=OOS_END, auto_adjust=True, progress=False)
    sp5_raw = yf.download("^GSPC", start=OOS_START, end=OOS_END, auto_adjust=True, progress=False)
    dji_close = dji_raw["Close"].squeeze().dropna()
    sp5_close = sp5_raw["Close"].squeeze().dropna()
    bench_dates = pd.to_datetime(dji_close.index)

    # ── 8g. Performance table ─────────────────────────────────────────────────
    def bench_metrics(close):
        s = close.values; r = pd.Series(s).pct_change().dropna()
        cum = s[-1]/s[0]-1; ann = (1+cum)**(252/len(r))-1; vol = r.std()*np.sqrt(252)
        return dict(cum=cum, ann=ann, vol=vol, sharpe=ann/vol,
                    mdd=float(((pd.Series(s)-pd.Series(s).cummax())/pd.Series(s).cummax()).min()))

    strategies = {
        "Ensemble":  metrics(p_ensemble) if len(p_ensemble) > 10 else {},
        "A2C":       metrics(p_a2c)      if len(p_a2c)      > 10 else {},
        "PPO":       metrics(p_ppo)      if len(p_ppo)      > 10 else {},
        "DDPG":      metrics(p_ddpg)     if len(p_ddpg)     > 10 else {},
        "DJIA":      bench_metrics(dji_close),
        "S&P 500":   bench_metrics(sp5_close),
    }

    for name, m in strategies.items():
        if m: print_metrics(m, f"{name} (2021–2026)")

    print("\n" + "="*78)
    print("  PERFORMANCE TABLE — Extended Out-of-Sample (2021–2026, rf=0)")
    print("="*78)
    cols = ["Ensemble", "A2C", "PPO", "DDPG", "DJIA", "S&P 500"]
    print(f"{'Metric':<22}" + "".join(f"{c:>11}" for c in cols))
    print("-"*78)
    def fmt(m, k, pct):
        if not m: return "—"
        v = m.get(k, float("nan"))
        return f"{v*100:.1f}%" if pct else f"{v:.2f}"
    for name, key, pct in [("Cumulative Return","cum",True),("Annual Return","ann",True),
                            ("Annual Volatility","vol",True),("Sharpe (rf=0)","sharpe",False),
                            ("Max Drawdown","mdd",True)]:
        row = f"{name:<22}" + "".join(f"{fmt(strategies[c],key,pct):>11}" for c in cols)
        print(row)

    # ── 8h. Plots ─────────────────────────────────────────────────────────────
    n = min(len(p_ensemble), len(p_a2c), len(p_ppo), len(p_ddpg), len(bench_dates))
    D = bench_dates[:n]

    def cum_ret(port):
        v = port.iloc[:n].values
        return (v / v[0]) - 1

    def bench_cum(close):
        v = close.reindex(bench_dates[:n]).ffill().values
        return (v / v[0]) - 1

    fig, axes = plt.subplots(2, 1, figsize=(14, 11))

    # Panel 1: Cumulative return — all strategies + benchmarks
    ax = axes[0]
    ax.plot(D, cum_ret(p_ensemble), color="firebrick",    linewidth=2.2, label="Ensemble")
    ax.plot(D, cum_ret(p_a2c),      color="royalblue",    linewidth=1.6, label="A2C (100k)")
    ax.plot(D, cum_ret(p_ppo),      color="mediumorchid", linewidth=1.6, label="PPO (100k)")
    ax.plot(D, cum_ret(p_ddpg),     color="darkorange",   linewidth=1.6, label="DDPG (100k)")
    ax.plot(D, bench_cum(dji_close),color="steelblue",    linewidth=1.4,
            linestyle="--", label="DJIA (^DJI)")
    ax.plot(D, bench_cum(sp5_close),color="gray",         linewidth=1.4,
            linestyle="--", label="S&P 500 (^GSPC)")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

    for label, val, col in [
        ("Ensemble", float(cum_ret(p_ensemble)[-1]), "firebrick"),
        ("A2C",      float(cum_ret(p_a2c)[-1]),      "royalblue"),
        ("PPO",      float(cum_ret(p_ppo)[-1]),      "mediumorchid"),
        ("DDPG",     float(cum_ret(p_ddpg)[-1]),     "darkorange"),
        ("DJIA",     float(bench_cum(dji_close)[-1]),"steelblue"),
        ("S&P 500",  float(bench_cum(sp5_close)[-1]),"gray"),
    ]:
        ax.annotate(f"{val*100:.1f}%", xy=(D[-1], val), xytext=(6, 0),
                    textcoords="offset points", color=col, fontsize=9,
                    va="center", fontweight="bold")

    ax.set_title("Cumulative Return 2021–2026: All Strategies vs Benchmarks", fontsize=13, pad=10)
    ax.set_ylabel("Cumulative Return")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_xlim(D[0], D[-1] + pd.Timedelta(days=50))

    # Panel 2: Quarterly validation Sharpe (model selection)
    x = np.arange(1, len(ppo_sh) + 1)
    w = 0.25
    axes[1].bar(x - w, ppo_sh,  width=w, label="PPO",  color="mediumorchid", alpha=0.85)
    axes[1].bar(x,     a2c_sh,  width=w, label="A2C",  color="royalblue",    alpha=0.85)
    axes[1].bar(x + w, ddpg_sh, width=w, label="DDPG", color="darkorange",   alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_title("Validation Sharpe by Quarter — Ensemble Model Selection (2021–2026)")
    axes[1].set_xlabel("Quarter"); axes[1].set_ylabel("Sharpe Ratio")
    axes[1].set_xticks(x); axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "extended_all_strategies.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved → {out}")
    print("\nDone.")