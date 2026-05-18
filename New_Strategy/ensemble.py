"""
New_Strategy — Rolling S&P 500 Top-30 Ensemble DRL
=====================================================================
Replicates the Yang et al. (2020) ensemble methodology with one change:
  DJI fixed 30 stocks  →  S&P 500 rolling top-30 by market cap (WRDS)

Period  :  train 2009-01-01 → 2015-10-15
           val   2015-10-16 → 2015-12-31
           trade 2016-01-01 → 2020-05-08  (rolling quarterly)

Run from RL/ directory:
  python3.11 New_Strategy/ensemble.py
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
warnings.filterwarnings("ignore")

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COMBINED_PATH = os.path.join(DATA_DIR, "combined_data.csv")
TOP30_PATH    = os.path.join(DATA_DIR, "top30_by_quarter.csv")

# ── Constants ─────────────────────────────────────────────────────────────────
STOCK_DIM       = 30
INITIAL_BALANCE = 1_000_000
HMAX_NORMALIZE  = 100
TRANSACTION_FEE = 0.001
REWARD_SCALING  = 1e-4
STATE_DIM       = 1 + STOCK_DIM * 6   # 181

REBALANCE_WINDOW  = 63   # trading days per quarter
VALIDATION_WINDOW = 63

# Date boundaries (integer YYYYMMDD format)
TRAIN_START_INT = 20090101
TRAIN_END_INT   = 20151015
VAL_START_INT   = 20151016
VAL_END_INT     = 20151231
TRADE_START_INT = 20160101
TRADE_END_INT   = 20200508


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df    = pd.read_csv(COMBINED_PATH)
    top30 = pd.read_csv(TOP30_PATH, parse_dates=["quarter_date"])
    return df, top30


def get_quarter_tickers(top30: pd.DataFrame, trade_date_int: int) -> list:
    """
    Return the 30 tickers whose quarter is the most recent one that
    started on or before *trade_date_int*.
    """
    trade_date = pd.to_datetime(str(trade_date_int), format="%Y%m%d")
    past = top30[top30["quarter_date"] <= trade_date]
    if past.empty:
        past = top30  # fallback: use earliest quarter
    latest_q = past["quarter_date"].max()
    tickers = top30.loc[top30["quarter_date"] == latest_q, "ticker"].tolist()
    return tickers


def data_split(df: pd.DataFrame,
               start_int: int,
               end_int: int,
               tickers: list) -> pd.DataFrame:
    """
    Filter df to [start_int, end_int) for the given tickers.
    Drops any trading day that doesn't have exactly STOCK_DIM rows
    (handles rare missing data for delisted/acquired stocks).
    """
    sub = df[
        (df["datadate"] >= start_int) &
        (df["datadate"] <  end_int) &
        (df["tic"].isin(tickers))
    ].copy()
    sub = sub.sort_values(["datadate", "tic"], ignore_index=True)

    # Keep only days with exactly 30 stocks
    counts = sub.groupby("datadate")["tic"].count()
    good_days = counts[counts == STOCK_DIM].index
    sub = sub[sub["datadate"].isin(good_days)]

    sub.index = sub["datadate"].factorize()[0]
    return sub


def compute_turbulence(df: pd.DataFrame) -> pd.Series:
    """
    Mahalanobis-distance turbulence index on daily returns.
    Returns a Series indexed by datadate.
    """
    prices = df.pivot(index="datadate", columns="tic", values="adjcp")
    returns = prices.pct_change().dropna()

    mu  = returns.mean().values
    cov = returns.cov().values

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    turb = {}
    for date, row in returns.iterrows():
        diff = row.values - mu
        turb[date] = float(diff @ inv_cov @ diff)

    return pd.Series(turb)


# ══════════════════════════════════════════════════════════════════════════════
#  Gym environments
# ══════════════════════════════════════════════════════════════════════════════

class StockEnvTrain(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df):
        super().__init__()
        self.df  = df
        self.day = 0
        self.action_space      = spaces.Box(-1, 1, (STOCK_DIM,), np.float32)
        self.observation_space = spaces.Box(0, np.inf, (STATE_DIM,), np.float32)
        self.balance = float(INITIAL_BALANCE)
        self.shares  = np.zeros(STOCK_DIM, np.float32)

    def _obs(self):
        d = self.df.loc[self.day]
        return np.array(
            [self.balance] + d.adjcp.tolist() + list(self.shares)
            + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist(),
            dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = float(INITIAL_BALANCE)
        self.shares  = np.zeros(STOCK_DIM, np.float32)
        self.day     = 0
        return self._obs(), {}

    def step(self, actions):
        if self.day >= len(self.df.index.unique()) - 1:
            return self._obs(), 0.0, True, False, {}
        prices  = self.df.loc[self.day].adjcp.values
        begin   = self.balance + float(np.dot(prices, self.shares))
        actions = actions * HMAX_NORMALIZE
        idx     = np.argsort(actions)
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

    def __init__(self, df):
        super().__init__()
        self.df           = df
        self.day          = 0
        self.action_space      = spaces.Box(-1, 1, (STOCK_DIM,), np.float32)
        self.observation_space = spaces.Box(0, np.inf, (STATE_DIM,), np.float32)
        self.balance      = float(INITIAL_BALANCE)
        self.shares       = np.zeros(STOCK_DIM, np.float32)
        self.asset_memory = [INITIAL_BALANCE]

    def _obs(self):
        d = self.df.loc[self.day]
        return np.array(
            [self.balance] + d.adjcp.tolist() + list(self.shares)
            + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist(),
            dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance      = float(INITIAL_BALANCE)
        self.shares       = np.zeros(STOCK_DIM, np.float32)
        self.day          = 0
        self.asset_memory = [INITIAL_BALANCE]
        return self._obs(), {}

    def step(self, actions):
        if self.day >= len(self.df.index.unique()) - 1:
            return self._obs(), 0.0, True, False, {}
        prices  = self.df.loc[self.day].adjcp.values
        actions = actions * HMAX_NORMALIZE
        idx     = np.argsort(actions)
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
        self.asset_memory.append(end)
        return self._obs(), 0.0, False, False, {}

    def get_sharpe(self):
        vals = pd.Series(self.asset_memory)
        ret  = vals.pct_change().dropna()
        return float((4 ** 0.5) * ret.mean() / ret.std()) if ret.std() > 0 else 0.0


class StockEnvTrade(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, initial_balance, turbulence_threshold):
        super().__init__()
        self.df                    = df
        self.day                   = 0
        self.action_space          = spaces.Box(-1, 1, (STOCK_DIM,), np.float32)
        self.observation_space     = spaces.Box(0, np.inf, (STATE_DIM,), np.float32)
        self.balance               = float(initial_balance)
        self.shares                = np.zeros(STOCK_DIM, np.float32)
        self.asset_memory          = [initial_balance]
        self.turbulence            = 0.0
        self.turbulence_threshold  = turbulence_threshold
        self.dates                 = sorted(df["datadate"].unique())

    def _obs(self):
        d = self.df.loc[self.day]
        return np.array(
            [self.balance] + d.adjcp.tolist() + list(self.shares)
            + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist(),
            dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        return self._obs(), {}

    def step(self, actions):
        if self.day >= len(self.df.index.unique()) - 1:
            return self._obs(), 0.0, True, False, {}

        # Turbulence guard: liquidate fully if turbulence spikes
        current_date = self.dates[self.day]
        prices = self.df.loc[self.day].adjcp.values

        if self.turbulence >= self.turbulence_threshold:
            # Sell everything
            for i in range(STOCK_DIM):
                if self.shares[i] > 0:
                    self.balance += prices[i] * self.shares[i] * (1 - TRANSACTION_FEE)
                    self.shares[i] = 0
        else:
            actions = actions * HMAX_NORMALIZE
            idx     = np.argsort(actions)
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
        portfolio_value = self.balance + float(np.dot(new_prices, self.shares))
        self.asset_memory.append(portfolio_value)
        return self._obs(), 0.0, False, False, {}

    def final_value(self):
        return self.asset_memory[-1]


# ══════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_agent(agent_name: str, train_df: pd.DataFrame,
                timesteps: int = 30_000) -> object:
    env = DummyVecEnv([lambda: StockEnvTrain(train_df)])
    t0  = time.time()

    if agent_name == "A2C":
        model = A2C("MlpPolicy", env, verbose=0)
    elif agent_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=0,
                    ent_coef=0.005, n_steps=128, batch_size=16,
                    learning_rate=7e-4, n_epochs=4)
    elif agent_name == "DDPG":
        n     = env.action_space.shape[-1]
        noise = OrnsteinUhlenbeckActionNoise(np.zeros(n), 0.5 * np.ones(n))
        model = DDPG("MlpPolicy", env, action_noise=noise, verbose=0)

    model.learn(total_timesteps=timesteps)
    elapsed = (time.time() - t0) / 60
    print(f"    {agent_name}  ({elapsed:.1f} min)")
    return model


def validate_agent(model, val_df: pd.DataFrame) -> float:
    env = StockEnvValidation(val_df)
    obs, _ = env.reset()
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
    sharpe = env.get_sharpe()
    print(f"      val Sharpe: {sharpe:.4f}")
    return sharpe


# ══════════════════════════════════════════════════════════════════════════════
#  Main rolling strategy
# ══════════════════════════════════════════════════════════════════════════════

def run_strategy(df: pd.DataFrame, top30: pd.DataFrame):
    """
    Rolling quarterly ensemble strategy.
    Each quarter:
      1. Look up the current top-30 tickers
      2. Train A2C, PPO, DDPG on expanding training window
      3. Pick best model by validation Sharpe
      4. Trade the next quarter; carry portfolio value forward
    """
    all_dates   = sorted(df["datadate"].unique())
    oos_dates   = [d for d in all_dates if d >= TRADE_START_INT]
    n_quarters  = len(range(0, len(oos_dates), REBALANCE_WINDOW))

    print(f"  OOS dates: {oos_dates[0]} – {oos_dates[-1]}  "
          f"({len(oos_dates)} days,  ~{n_quarters} quarters)")

    portfolio_value = float(INITIAL_BALANCE)
    account_values  = []   # list of (datadate, portfolio_value)
    quarter_log     = []

    for q_idx, i in enumerate(range(0, len(oos_dates), REBALANCE_WINDOW)):

        trade_start = oos_dates[i]
        trade_end   = oos_dates[min(i + REBALANCE_WINDOW, len(oos_dates) - 1)]
        is_initial  = (q_idx == 0)

        print(f"\n[Q{q_idx+1}]  trade {trade_start} → {trade_end}"
              f"  (initial={is_initial})")

        # ── 1. Universe for this quarter ──────────────────────────────────────
        tickers = get_quarter_tickers(top30, trade_start)
        if len(tickers) == 0:
            print("  WARNING: no tickers found, skipping quarter")
            continue

        # ── 2. Compute turbulence threshold on training window ─────────────
        train_df_full = data_split(df, TRAIN_START_INT, trade_start, tickers)
        if len(train_df_full) == 0:
            print("  WARNING: empty training set, skipping quarter")
            continue

        turb_series = compute_turbulence(train_df_full)
        turb_thresh = float(np.percentile(turb_series, 90))
        print(f"  turb threshold: {turb_thresh:.1f}")

        # ── 3. Validation window (63 days before trade start) ─────────────
        if is_initial:
            val_df = data_split(df, VAL_START_INT, TRADE_START_INT, tickers)
        else:
            val_start_idx = max(0, i - VALIDATION_WINDOW)
            val_start_date = oos_dates[val_start_idx]
            val_df = data_split(df, val_start_date, trade_start, tickers)

        # ── 4. Training set (up to val start) ─────────────────────────────
        if is_initial:
            train_df = data_split(df, TRAIN_START_INT, VAL_START_INT, tickers)
        else:
            val_start_idx = max(0, i - VALIDATION_WINDOW)
            val_start_date = oos_dates[val_start_idx]
            train_df = data_split(df, TRAIN_START_INT, val_start_date, tickers)

        if len(train_df) == 0 or len(val_df) == 0:
            print("  WARNING: empty train/val, skipping quarter")
            continue

        print(f"  train: {train_df['datadate'].min()} → {train_df['datadate'].max()}"
              f"  ({train_df.index.nunique()} days)")
        print(f"  val  : {val_df['datadate'].min()} → {val_df['datadate'].max()}"
              f"  ({val_df.index.nunique()} days)")

        # ── 5. Train all three agents ─────────────────────────────────────
        models, sharpes = {}, {}
        for agent in ["A2C", "PPO", "DDPG"]:
            ts = {"A2C": 30_000, "PPO": 100_000, "DDPG": 10_000}[agent]
            models[agent]  = train_agent(agent, train_df, timesteps=ts)
            sharpes[agent] = validate_agent(models[agent], val_df)

        best_agent = max(sharpes, key=sharpes.get)
        print(f"  → Ensemble picks: {best_agent}  (Sharpe {sharpes[best_agent]:.4f})")

        # ── 6. Trade quarter ──────────────────────────────────────────────
        trade_end_exclusive = (oos_dates[min(i + REBALANCE_WINDOW, len(oos_dates) - 1)]
                               if i + REBALANCE_WINDOW < len(oos_dates)
                               else TRADE_END_INT + 1)
        trade_df = data_split(df, trade_start, trade_end_exclusive, tickers)

        if len(trade_df) == 0:
            print("  WARNING: empty trade window, skipping")
            continue

        trade_env = StockEnvTrade(trade_df, portfolio_value, turb_thresh)
        obs, _    = trade_env.reset()
        done      = False
        while not done:
            act, _ = models[best_agent].predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = trade_env.step(act)
            done = terminated or truncated

        # Record daily portfolio values for this quarter
        for day_i, val in enumerate(trade_env.asset_memory):
            date_int = trade_df["datadate"].unique()[min(day_i, len(trade_df["datadate"].unique())-1)]
            account_values.append({"datadate": date_int, "account_value": val})

        portfolio_value = trade_env.final_value()
        cum_return = (portfolio_value / INITIAL_BALANCE - 1) * 100

        quarter_log.append({
            "quarter":       q_idx + 1,
            "trade_start":   trade_start,
            "trade_end":     trade_end,
            "best_agent":    best_agent,
            "val_sharpe":    sharpes[best_agent],
            "end_value":     portfolio_value,
            "cum_return_pct": cum_return,
        })
        print(f"  End value: ${portfolio_value:,.0f}  "
              f"(cumulative return: {cum_return:+.1f}%)")

    return pd.DataFrame(account_values), pd.DataFrame(quarter_log)


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics & Plot
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(portfolio: pd.Series) -> dict:
    ret   = portfolio.pct_change().dropna()
    annual_ret = ret.mean() * 252
    annual_vol = ret.std() * (252 ** 0.5)
    sharpe     = annual_ret / annual_vol if annual_vol > 0 else 0.0
    rolling_max = portfolio.cummax()
    drawdown    = (portfolio - rolling_max) / rolling_max
    max_dd      = drawdown.min()
    cum_ret     = portfolio.iloc[-1] / portfolio.iloc[0] - 1
    return {"Cumulative Return": cum_ret, "Sharpe Ratio": sharpe,
            "Annual Return": annual_ret, "Max Drawdown": max_dd}


def plot_results(account_values: pd.DataFrame):
    portfolio = account_values.set_index("datadate")["account_value"]

    # Download benchmarks
    start_str = pd.to_datetime(str(account_values["datadate"].min()),
                               format="%Y%m%d").strftime("%Y-%m-%d")
    end_str   = pd.to_datetime(str(account_values["datadate"].max()),
                               format="%Y%m%d").strftime("%Y-%m-%d")

    print("\nDownloading benchmarks …")
    dji = yf.download("^DJI",  start=start_str, end=end_str,
                      auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    sp5 = yf.download("^GSPC", start=start_str, end=end_str,
                      auto_adjust=True, progress=False)["Close"].squeeze().dropna()

    dates = pd.to_datetime(dji.index)
    n     = min(len(portfolio), len(dates))

    cum_port = portfolio.values[:n] / portfolio.values[0] - 1
    cum_dji  = dji.values[:n] / dji.values[0] - 1
    cum_sp5  = sp5.reindex(dji.index[:n]).ffill().values
    cum_sp5  = cum_sp5 / cum_sp5[0] - 1

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates[:n], cum_port, color="firebrick",  linewidth=2.2,
            label="New Strategy — S&P 500 Top-30 Ensemble")
    ax.plot(dates[:n], cum_dji,  color="steelblue",  linewidth=1.6,
            linestyle="--", label="DJIA (^DJI)")
    ax.plot(dates[:n], cum_sp5,  color="darkorange", linewidth=1.6,
            linestyle="--", label="S&P 500 (^GSPC)")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

    for label, val, col in [
        ("New Strategy", float(cum_port[-1]), "firebrick"),
        ("DJIA",         float(cum_dji[-1]),  "steelblue"),
        ("S&P 500",      float(cum_sp5[-1]),  "darkorange"),
    ]:
        ax.annotate(f"{val*100:.1f}%",
                    xy=(dates[n-1], val), xytext=(6, 0),
                    textcoords="offset points", color=col,
                    fontsize=9, va="center", fontweight="bold")

    ax.set_title("New Strategy: S&P 500 Rolling Top-30 Ensemble DRL\n"
                 "(Train: 2009–2015  |  Trade: 2016–2020)", fontsize=13, pad=12)
    ax.set_ylabel("Cumulative Return")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_xlim(dates[0], dates[n-1] + pd.Timedelta(days=40))

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "new_strategy_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("Loading data …")
    df, top30 = load_data()
    print(f"  combined_data: {df.datadate.min()} – {df.datadate.max()}, "
          f"{df.tic.nunique()} tickers")
    print(f"  top30 table  : {top30.quarter_date.min().date()} – "
          f"{top30.quarter_date.max().date()}, "
          f"{top30.quarter_date.nunique()} quarters")

    print("\n" + "="*60)
    print("  ROLLING ENSEMBLE STRATEGY  (2016–2020)")
    print("="*60)

    account_values, quarter_log = run_strategy(df, top30)

    # ── Save results ──────────────────────────────────────────────────────────
    account_values.to_csv(os.path.join(RESULTS_DIR, "account_values.csv"), index=False)
    quarter_log.to_csv(os.path.join(RESULTS_DIR, "quarter_log.csv"),    index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")

    # ── Performance summary ───────────────────────────────────────────────────
    portfolio = account_values.set_index("datadate")["account_value"]
    metrics   = compute_metrics(portfolio)

    print("\n" + "="*55)
    print("  PERFORMANCE SUMMARY")
    print("="*55)
    print(f"  Cumulative Return : {metrics['Cumulative Return']*100:+.1f}%")
    print(f"  Annual Return     : {metrics['Annual Return']*100:+.1f}%")
    print(f"  Sharpe Ratio      : {metrics['Sharpe Ratio']:.4f}")
    print(f"  Max Drawdown      : {metrics['Max Drawdown']*100:.1f}%")

    print("\n  Quarter-by-quarter log:")
    print(f"  {'Q':<4} {'Agent':<6} {'Val Sharpe':>10} {'End Value':>14} {'Cum Ret':>10}")
    print("  " + "-"*50)
    for _, row in quarter_log.iterrows():
        print(f"  Q{int(row.quarter):<3} {row.best_agent:<6} "
              f"{row.val_sharpe:>10.4f} "
              f"${row.end_value:>13,.0f} "
              f"{row.cum_return_pct:>+9.1f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(account_values)
    print("\nDone.")
