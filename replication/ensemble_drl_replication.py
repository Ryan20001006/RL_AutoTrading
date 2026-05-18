"""
Replication of:
  "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy"
  Yang et al., ICAIF 2020

This script is a self-contained, modern replication using:
  - stable-baselines3 (PyTorch-based, replaces the original stable-baselines/TF1)
  - gymnasium (replaces old gym API)
  - done_data.csv  (pre-processed Dow Jones 30 data from the original repo)

Run with:
  python3.11 replication/ensemble_drl_replication.py
from the RL/ directory.
"""

# ── 0. Imports ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, os, time
warnings.filterwarnings("ignore")

import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# ── 1. Constants (match paper exactly) ──────────────────────────────────────
STOCK_DIM        = 30
INITIAL_BALANCE  = 1_000_000
HMAX_NORMALIZE   = 100        # max shares per trade
TRANSACTION_FEE  = 0.001      # 0.1 %
REWARD_SCALING   = 1e-4
STATE_DIM        = 1 + STOCK_DIM * 6  # balance + price×30 + shares×30 + macd×30 + rsi×30 + cci×30 + adx×30 = 181

# Output goes into replication/results/ so the original repo is untouched
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Path to the pre-processed data in the original repo
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020",
    "done_data.csv"
)

# ── 2. Data loading & splitting ──────────────────────────────────────────────
print("Loading pre-processed data …")
df = pd.read_csv(DATA_PATH, index_col=0)
print(f"  shape: {df.shape}   columns: {list(df.columns)}")

def data_split(df, start, end):
    """Return rows where start <= datadate < end, indexed by trading-day integer."""
    data = df[(df.datadate >= start) & (df.datadate < end)].copy()
    data = data.sort_values(["datadate", "tic"], ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data

# Rolling-window out-of-sample dates (same as original run_DRL.py)
unique_trade_date = df[(df.datadate > 20151001) & (df.datadate <= 20200708)].datadate.unique()
unique_trade_date = np.sort(unique_trade_date)
print(f"Out-of-sample window: {unique_trade_date[0]} – {unique_trade_date[-1]}, "
      f"{len(unique_trade_date)} dates")

# ── 3. Turbulence threshold (90th percentile of in-sample turbulence) ────────
insample = df[(df.datadate < 20151000) & (df.datadate >= 20090000)].drop_duplicates("datadate")
INSAMPLE_TURBULENCE_THRESHOLD = np.quantile(insample.turbulence.values, 0.90)
print(f"In-sample 90th-pct turbulence threshold: {INSAMPLE_TURBULENCE_THRESHOLD:.2f}")


# ── 4. Gymnasium environments ────────────────────────────────────────────────
#
# All three environments share the same state representation:
#   s = [balance, adj_close×30, shares×30, MACD×30, RSI×30, CCI×30, ADX×30]
# giving a 181-dimensional observation vector, exactly as in the paper (Section 4.1).
#
# Actions are continuous in [-1, 1]^30 and scaled by HMAX_NORMALIZE = 100 inside
# step(), so each action component represents up to ±100 shares.

class StockEnvTrain(gym.Env):
    """Training environment — no turbulence guard, purely maximises portfolio value."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, df):
        super().__init__()
        self.df  = df
        self.day = 0
        self.action_space      = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0,  high=np.inf, shape=(STATE_DIM,), dtype=np.float32)
        self._reset_internals()

    def _obs(self):
        d = self.df.loc[self.day]
        return np.array(
            [self.balance] + d.adjcp.tolist() + list(self.shares)
            + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist(),
            dtype=np.float32,
        )

    def _reset_internals(self):
        self.balance      = float(INITIAL_BALANCE)
        self.shares       = np.zeros(STOCK_DIM, dtype=np.float32)
        self.day          = 0
        self.asset_memory = [INITIAL_BALANCE]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internals()
        return self._obs(), {}

    def step(self, actions):
        if self.day >= len(self.df.index.unique()) - 1:
            return self._obs(), 0.0, True, False, {}

        prices = self.df.loc[self.day].adjcp.values
        begin  = self.balance + float(np.dot(prices, self.shares))

        actions = actions * HMAX_NORMALIZE
        order   = np.argsort(actions)
        for i in order[actions[order] < 0]:          # sells first
            qty = min(abs(actions[i]), self.shares[i])
            if qty > 0:
                self.balance   += prices[i] * qty * (1 - TRANSACTION_FEE)
                self.shares[i] -= qty
        for i in order[::-1][actions[order[::-1]] > 0]:  # then buys
            if prices[i] <= 0: continue
            qty = min(int(self.balance // prices[i]), int(actions[i]))
            if qty > 0:
                self.balance   -= prices[i] * qty * (1 + TRANSACTION_FEE)
                self.shares[i] += qty

        self.day += 1
        new_prices = self.df.loc[self.day].adjcp.values
        end   = self.balance + float(np.dot(new_prices, self.shares))
        self.asset_memory.append(end)
        return self._obs(), (end - begin) * REWARD_SCALING, False, False, {}

    def render(self):
        return self.asset_memory


class StockEnvValidation(gym.Env):
    """
    Validation environment — records daily portfolio values to CSV so
    get_sharpe() can compute the rolling Sharpe ratio for model selection.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, turbulence_threshold, iteration):
        super().__init__()
        self.df                   = df
        self.turbulence_threshold = turbulence_threshold
        self.iteration            = iteration
        self.day                  = 0
        self.action_space      = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0,  high=np.inf, shape=(STATE_DIM,), dtype=np.float32)
        self._reset_internals()

    def _obs(self):
        d = self.df.loc[self.day]
        return np.array(
            [self.balance] + d.adjcp.tolist() + list(self.shares)
            + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist(),
            dtype=np.float32,
        )

    def _reset_internals(self):
        self.balance      = float(INITIAL_BALANCE)
        self.shares       = np.zeros(STOCK_DIM, dtype=np.float32)
        self.day          = 0
        self.turbulence   = 0.0
        self.asset_memory = [INITIAL_BALANCE]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internals()
        return self._obs(), {}

    def step(self, actions):
        if self.day >= len(self.df.index.unique()) - 1:
            pd.DataFrame(self.asset_memory).to_csv(
                os.path.join(RESULTS_DIR, f"account_value_validation_{self.iteration}.csv")
            )
            return self._obs(), 0.0, True, False, {}

        if self.turbulence >= self.turbulence_threshold:
            actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM, dtype=np.float32)
        else:
            actions = actions * HMAX_NORMALIZE

        prices = self.df.loc[self.day].adjcp.values
        begin  = self.balance + float(np.dot(prices, self.shares))

        order = np.argsort(actions)
        for i in order[actions[order] < 0]:
            qty = min(abs(actions[i]), self.shares[i])
            if qty > 0:
                self.balance   += prices[i] * qty * (1 - TRANSACTION_FEE)
                self.shares[i] -= qty
        for i in order[::-1][actions[order[::-1]] > 0]:
            if prices[i] <= 0: continue
            qty = min(int(self.balance // prices[i]), int(actions[i]))
            if qty > 0:
                self.balance   -= prices[i] * qty * (1 + TRANSACTION_FEE)
                self.shares[i] += qty

        self.day += 1
        self.turbulence = float(self.df.loc[self.day].turbulence.iloc[0])
        new_prices = self.df.loc[self.day].adjcp.values
        end = self.balance + float(np.dot(new_prices, self.shares))
        self.asset_memory.append(end)
        return self._obs(), (end - begin) * REWARD_SCALING, False, False, {}


class StockEnvTrade(gym.Env):
    """
    Live-trading environment — carries forward the previous quarter's portfolio
    (balance + share holdings) across rebalance windows.
    The turbulence guard fully liquidates when the index breaches the threshold,
    matching Equation (10) in the paper.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, turbulence_threshold, initial=True,
                 previous_state=None, model_name="", iteration=""):
        super().__init__()
        self.df                   = df
        self.turbulence_threshold = turbulence_threshold
        self.initial              = initial
        self.previous_state       = previous_state or []
        self.model_name           = model_name
        self.iteration            = iteration
        self.action_space      = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0,  high=np.inf, shape=(STATE_DIM,), dtype=np.float32)
        self._reset_internals()

    def _obs(self):
        d = self.df.loc[self.day]
        return np.array(
            [self.balance] + d.adjcp.tolist() + list(self.shares)
            + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist(),
            dtype=np.float32,
        )

    def _reset_internals(self):
        if self.initial or not self.previous_state:
            self.balance = float(INITIAL_BALANCE)
            self.shares  = np.zeros(STOCK_DIM, dtype=np.float32)
        else:
            ps = self.previous_state
            self.balance = float(ps[0])
            self.shares  = np.array(ps[STOCK_DIM + 1: STOCK_DIM * 2 + 1], dtype=np.float32)
        self.day            = 0
        self.turbulence     = 0.0
        self.asset_memory   = [
            INITIAL_BALANCE if (self.initial or not self.previous_state)
            else self.balance + float(np.dot(self.df.loc[0].adjcp.values, self.shares))
        ]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internals()
        return self._obs(), {}

    def step(self, actions):
        if self.day >= len(self.df.index.unique()) - 1:
            pd.DataFrame(self.asset_memory, columns=["account_value"]).to_csv(
                os.path.join(RESULTS_DIR, f"account_value_trade_{self.model_name}_{self.iteration}.csv")
            )
            return self._obs(), 0.0, True, False, {}

        if self.turbulence >= self.turbulence_threshold:
            actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM, dtype=np.float32)
        else:
            actions = actions * HMAX_NORMALIZE

        prices = self.df.loc[self.day].adjcp.values
        begin  = self.balance + float(np.dot(prices, self.shares))

        order = np.argsort(actions)
        for i in order[actions[order] < 0]:
            qty = min(abs(actions[i]), self.shares[i])
            if qty > 0:
                self.balance   += prices[i] * qty * (1 - TRANSACTION_FEE)
                self.shares[i] -= qty
        for i in order[::-1][actions[order[::-1]] > 0]:
            if prices[i] <= 0: continue
            qty = min(int(self.balance // prices[i]), int(actions[i]))
            if qty > 0:
                self.balance   -= prices[i] * qty * (1 + TRANSACTION_FEE)
                self.shares[i] += qty

        self.day += 1
        self.turbulence = float(self.df.loc[self.day].turbulence.iloc[0])
        new_prices = self.df.loc[self.day].adjcp.values
        end = self.balance + float(np.dot(new_prices, self.shares))
        self.asset_memory.append(end)
        return self._obs(), (end - begin) * REWARD_SCALING, False, False, {}

    def render(self):
        """Return full state vector — used to seed the next quarter's environment."""
        d = self.df.loc[self.day]
        return (
            [self.balance] + d.adjcp.tolist() + list(self.shares)
            + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist()
        )


# ── 5. Agent training helpers ────────────────────────────────────────────────

def make_train_env(df):
    return DummyVecEnv([lambda: StockEnvTrain(df)])

def train_a2c(env, timesteps=25_000):
    t0 = time.time()
    model = A2C("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    print(f"    A2C  trained  ({(time.time()-t0)/60:.1f} min)")
    return model

def train_ppo(env, timesteps=5_000):
    t0 = time.time()
    # Match original PPO2 hyperparameters as closely as possible:
    #   n_steps=128  (PPO2 default, vs SB3 default of 2048 — 16x more frequent updates)
    #   batch_size=16 (= n_steps / nminibatches = 128/8)
    #   learning_rate=7e-4  (PPO2 default, vs SB3 default of 3e-4)
    #   n_epochs=4  (PPO2 noptepochs default, vs SB3 default of 10)
    model = PPO("MlpPolicy", env, ent_coef=0.005,
                n_steps=128, batch_size=16,
                learning_rate=7e-4, n_epochs=4, verbose=0)
    model.learn(total_timesteps=timesteps)
    print(f"    PPO  trained  ({(time.time()-t0)/60:.1f} min)")
    return model

def train_ddpg(env, timesteps=10_000):
    t0 = time.time()
    n_act = env.action_space.shape[-1]
    noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_act), sigma=0.5 * np.ones(n_act)
    )
    model = DDPG("MlpPolicy", env, action_noise=noise, verbose=0)
    model.learn(total_timesteps=timesteps)
    print(f"    DDPG trained  ({(time.time()-t0)/60:.1f} min)")
    return model

def validate_model(model, val_df, turb_threshold, iteration):
    env = DummyVecEnv([lambda: StockEnvValidation(val_df, turb_threshold, iteration)])
    obs = env.reset()
    for _ in range(len(val_df.index.unique())):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done[0]:
            break

def get_sharpe(iteration):
    """Annualised Sharpe from the saved validation CSV (quarterly factor = √4)."""
    path = os.path.join(RESULTS_DIR, f"account_value_validation_{iteration}.csv")
    if not os.path.exists(path):
        return 0.0
    vals = pd.read_csv(path, index_col=0).iloc[:, 0]
    ret  = vals.pct_change().dropna()
    return float((4 ** 0.5) * ret.mean() / ret.std()) if ret.std() > 0 else 0.0

def trade_quarter(model, trade_df, last_state, iter_num,
                  rebalance_window, turb_threshold, initial):
    """Deploy the selected model for one quarter; return the final state vector."""
    env = DummyVecEnv([lambda: StockEnvTrade(
        trade_df, turb_threshold, initial=initial,
        previous_state=last_state, model_name="ensemble", iteration=iter_num
    )])
    obs = env.reset()
    n   = len(trade_df.index.unique())
    new_last_state = last_state
    for step_i in range(n):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if step_i == n - 2:
            new_last_state = env.envs[0].render()
        if done[0]:
            break
    return new_last_state


# ── 6. Ensemble rolling-window strategy ─────────────────────────────────────

def run_ensemble(df, unique_trade_date, rebalance_window=63, validation_window=63):
    """
    Implements Algorithm 1 from the paper:
      For each quarter t:
        1. Train A2C, PPO, DDPG on all data up to t - validation_window
        2. Validate each on [t - validation_window, t); pick model with highest Sharpe
        3. Trade with best model on [t, t + rebalance_window)
    The loop step equals rebalance_window (≈ one quarter = 63 trading days).
    """
    print("\n" + "="*60)
    print("  ENSEMBLE STRATEGY — quarterly rolling rebalance")
    print("="*60)

    last_state = []
    ppo_sh, a2c_sh, ddpg_sh, model_used = [], [], [], []
    t_start = time.time()

    for i in range(rebalance_window + validation_window,
                   len(unique_trade_date),
                   rebalance_window):

        initial = (i - rebalance_window - validation_window == 0)
        print(f"\n[Quarter {len(model_used)+1}]  (iteration index i={i}, initial={initial})")

        # Dynamic turbulence threshold: use in-sample 90th pct if recent market
        # turbulence mean exceeds it, otherwise relax to the empirical maximum.
        anchor_date = unique_trade_date[i - rebalance_window - validation_window]
        end_idx     = df.index[df.datadate == anchor_date].tolist()
        end_idx     = end_idx[-1] if end_idx else 0
        start_idx   = max(0, end_idx - validation_window * 30 + 1)
        hist_turb   = df.iloc[start_idx:end_idx + 1].drop_duplicates("datadate").turbulence
        turb_threshold = (INSAMPLE_TURBULENCE_THRESHOLD
                          if hist_turb.mean() > INSAMPLE_TURBULENCE_THRESHOLD
                          else float(insample.turbulence.max()))
        print(f"  turbulence threshold: {turb_threshold:.1f}")

        # Data windows
        val_start   = unique_trade_date[i - rebalance_window - validation_window]
        val_end     = unique_trade_date[i - rebalance_window]
        trade_end   = unique_trade_date[i]

        train_df = data_split(df, 20090000, val_start)
        val_df   = data_split(df, val_start, val_end)
        trade_df = data_split(df, val_end,   trade_end)
        print(f"  train …→{val_start} | val {val_start}→{val_end} | trade {val_end}→{trade_end}")

        env_train = make_train_env(train_df)

        # Train all three agents and record validation Sharpe
        m_a2c  = train_a2c(env_train)
        validate_model(m_a2c,  val_df, turb_threshold, i)
        s_a2c  = get_sharpe(i);  a2c_sh.append(s_a2c)
        print(f"    A2C  val Sharpe: {s_a2c:.4f}")

        m_ppo  = train_ppo(env_train)
        validate_model(m_ppo,  val_df, turb_threshold, i)
        s_ppo  = get_sharpe(i);  ppo_sh.append(s_ppo)
        print(f"    PPO  val Sharpe: {s_ppo:.4f}")

        m_ddpg = train_ddpg(env_train)
        validate_model(m_ddpg, val_df, turb_threshold, i)
        s_ddpg = get_sharpe(i); ddpg_sh.append(s_ddpg)
        print(f"    DDPG val Sharpe: {s_ddpg:.4f}")

        # Select model with highest validation Sharpe
        best_sharpe, best_name, best_model = max(
            (s_ppo, "PPO", m_ppo), (s_a2c, "A2C", m_a2c), (s_ddpg, "DDPG", m_ddpg)
        )
        print(f"  → Selected: {best_name}  (Sharpe {best_sharpe:.4f})")
        model_used.append(best_name)

        # Trade with selected model
        last_state = trade_quarter(
            best_model, trade_df, last_state, i,
            rebalance_window, turb_threshold, initial
        )

    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min")
    return ppo_sh, a2c_sh, ddpg_sh, model_used


# ── 7. Run ───────────────────────────────────────────────────────────────────
ppo_sh, a2c_sh, ddpg_sh, model_used = run_ensemble(
    df, unique_trade_date, rebalance_window=63, validation_window=63
)

print("\n── Model selected each quarter ──")
for q, m in enumerate(model_used, 1):
    print(f"  Q{q:>2}: {m}")


# ── 8. Performance analysis ──────────────────────────────────────────────────

def load_portfolio():
    """Stitch quarterly trade CSVs into one continuous portfolio-value series."""
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.startswith("account_value_trade_ensemble_")],
        key=lambda x: int(x.split("_")[-1].replace(".csv", ""))
    )
    frames = [pd.read_csv(os.path.join(RESULTS_DIR, f), index_col=0)["account_value"]
              for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.Series(dtype=float)

def performance_metrics(series, label="Strategy"):
    series     = series.dropna()
    daily_ret  = series.pct_change().dropna()
    cum_ret    = series.iloc[-1] / series.iloc[0] - 1
    annual_ret = (1 + cum_ret) ** (252 / max(len(daily_ret), 1)) - 1
    annual_vol = daily_ret.std() * np.sqrt(252)
    sharpe     = annual_ret / annual_vol if annual_vol > 0 else np.nan  # rf=0, matching paper convention
    mdd        = ((series - series.cummax()) / series.cummax()).min()

    print(f"\n── {label} ──")
    print(f"  Cumulative Return : {cum_ret*100:.1f}%")
    print(f"  Annual Return     : {annual_ret*100:.1f}%")
    print(f"  Annual Volatility : {annual_vol*100:.1f}%")
    print(f"  Sharpe Ratio      : {sharpe:.2f}")
    print(f"  Max Drawdown      : {mdd*100:.1f}%")
    return dict(cum=cum_ret, ann=annual_ret, vol=annual_vol, sharpe=sharpe, mdd=mdd)

portfolio = load_portfolio()
print(f"\nPortfolio series: {len(portfolio)} daily observations")

if len(portfolio) > 10:
    metrics = performance_metrics(portfolio, "Ensemble Strategy (Replication)")
else:
    print("Portfolio series too short — check results/ CSVs.")
    metrics = {}


# ── 9. Plots ─────────────────────────────────────────────────────────────────

def fetch_benchmark(ticker, start, end):
    """Download a benchmark index and return as cumulative-return series (starts at 0)."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return None
    close = raw["Close"].squeeze().dropna()
    return (close / close.iloc[0]) - 1

if len(portfolio) > 10:
    # ── Date alignment ───────────────────────────────────────────────────────
    # The out-of-sample window is 2016-01-04 → 2020-05-08.
    # We reconstruct a business-day date index from yfinance DJIA data so the
    # x-axis shows real calendar dates rather than an integer trading-day count.
    OOS_START = "2016-01-04"
    OOS_END   = "2020-05-08"

    dji_raw = yf.download("^DJI", start=OOS_START, end=OOS_END,
                          auto_adjust=True, progress=False)
    trade_dates = pd.to_datetime(dji_raw.index)

    # Align portfolio length to available dates (trim or pad to match)
    n = min(len(portfolio), len(trade_dates))
    port_aligned  = portfolio.iloc[:n].values
    dates_aligned = trade_dates[:n]

    cum_replication = (port_aligned / port_aligned[0]) - 1

    # ── Benchmarks (same date range) ─────────────────────────────────────────
    dji_cum  = fetch_benchmark("^DJI",  OOS_START, OOS_END)
    sp5_cum  = fetch_benchmark("^GSPC", OOS_START, OOS_END)

    # Align benchmark lengths to our date axis
    dji_aligned = dji_cum.reindex(trade_dates[:n]).fillna(method="ffill").values
    sp5_aligned = sp5_cum.reindex(trade_dates[:n]).fillna(method="ffill").values

    # ── Paper's reported final cumulative return (endpoint reference) ─────────
    PAPER_FINAL_CUM = 0.704   # Table 2: 70.4%

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ── Panel 1: Cumulative return ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(dates_aligned, cum_replication, color="firebrick",  linewidth=2.0,
            label="Ensemble DRL — Our Replication")
    ax.plot(dates_aligned, dji_aligned,     color="steelblue",  linewidth=1.5,
            linestyle="--", label="Dow Jones Industrial Avg (DJIA)")
    ax.plot(dates_aligned, sp5_aligned,     color="darkorange", linewidth=1.5,
            linestyle="--", label="S&P 500")

    # Paper's ensemble endpoint marker + horizontal dashed reference
    ax.axhline(PAPER_FINAL_CUM, color="green", linewidth=1.2, linestyle=":",
               label=f"Paper ensemble final return ({PAPER_FINAL_CUM*100:.1f}%)")
    ax.scatter([dates_aligned[-1]], [PAPER_FINAL_CUM],
               color="green", zorder=5, s=60)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="-", alpha=0.4)
    ax.set_title("Cumulative Return: Ensemble DRL vs Benchmarks (2016–2020)", fontsize=13)
    ax.set_ylabel("Cumulative Return")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1,4,7,10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)

    # Annotate final values on the right edge
    for label, val, col in [
        ("Replication",    float(cum_replication[-1]), "firebrick"),
        ("DJIA",           float(dji_aligned[-1]),     "steelblue"),
        ("S&P 500",        float(sp5_aligned[-1]),     "darkorange"),
        ("Paper (reported)", PAPER_FINAL_CUM,          "green"),
    ]:
        ax.annotate(f"{val*100:.1f}%", xy=(dates_aligned[-1], val),
                    xytext=(8, 0), textcoords="offset points",
                    color=col, fontsize=9, va="center")

    # ── Panel 2: Quarterly Sharpe per agent ───────────────────────────────────
    x = np.arange(1, len(ppo_sh) + 1)
    w = 0.25
    axes[1].bar(x - w, ppo_sh,  width=w, label="PPO",  color="steelblue",   alpha=0.85)
    axes[1].bar(x,     a2c_sh,  width=w, label="A2C",  color="darkorange",  alpha=0.85)
    axes[1].bar(x + w, ddpg_sh, width=w, label="DDPG", color="forestgreen", alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_title("Validation Sharpe Ratio by Quarter and Agent (Model Selection)")
    axes[1].set_xlabel("Quarter"); axes[1].set_ylabel("Sharpe Ratio")
    axes[1].set_xticks(x); axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_fig = os.path.join(RESULTS_DIR, "ensemble_replication_results.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved → {out_fig}")


# ── 10. Comparison table ─────────────────────────────────────────────────────

print("\n" + "="*60)
print("  COMPARISON: Paper (Table 2) vs This Replication")
print("="*60)
paper = dict(cum=0.704, ann=0.130, vol=0.097, sharpe=1.30, mdd=-0.097)

rows = [
    ("Cumulative Return", f"{paper['cum']*100:.1f}%",
     f"{metrics.get('cum', float('nan'))*100:.1f}%" if metrics else "—"),
    ("Annual Return",     f"{paper['ann']*100:.1f}%",
     f"{metrics.get('ann', float('nan'))*100:.1f}%" if metrics else "—"),
    ("Annual Volatility", f"{paper['vol']*100:.1f}%",
     f"{metrics.get('vol', float('nan'))*100:.1f}%" if metrics else "—"),
    ("Sharpe Ratio",      f"{paper['sharpe']:.2f}",
     f"{metrics.get('sharpe', float('nan')):.2f}" if metrics else "—"),
    ("Max Drawdown",      f"{paper['mdd']*100:.1f}%",
     f"{metrics.get('mdd', float('nan'))*100:.1f}%" if metrics else "—"),
]

print(f"{'Metric':<25} {'Paper':>10} {'Replication':>14}")
print("-" * 52)
for name, paper_val, rep_val in rows:
    print(f"{name:<25} {paper_val:>10} {rep_val:>14}")

print("\nDone.")
