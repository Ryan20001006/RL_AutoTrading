"""
Learning Curve Analysis — find optimal timesteps for A2C, PPO, DDPG
without full trial-and-error.

Method: train each agent N_SEEDS times up to max timesteps, evaluating
every EVAL_INTERVAL steps on the same validation window.
Final plot shows mean ± 1 std band across seeds → stable optimal estimate.

Run from RL/ directory:
  python3.11 replication/learning_curve_analysis.py
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# ── Constants (29-stock setup) ───────────────────────────────────────────────
STOCK_DIM       = 29
INITIAL_BALANCE = 1_000_000
HMAX_NORMALIZE  = 100
TRANSACTION_FEE = 0.001
REWARD_SCALING  = 1e-4
STATE_DIM       = 1 + STOCK_DIM * 6   # 175

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_learning_curve")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "combined_data.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── How many independent seeds to average over ───────────────────────────────
N_SEEDS = 5   # increase for more stability; each seed multiplies runtime

# ── Evaluation interval (steps between each checkpoint evaluation) ───────────
EVAL_INTERVAL = 5_000
MAX_TIMESTEPS = {
    "A2C":  100_000,
    "PPO":  200_000,
    "DDPG": 100_000,
}

# ── Fixed train/val split (use Q1 of the extended study) ────────────────────
TRAIN_END = 20201002
VAL_START = 20201002
VAL_END   = 20210104

# ── Minimal gym environments ─────────────────────────────────────────────────

def data_split(df, start, end):
    data = df[(df.datadate >= start) & (df.datadate < end)].copy()
    data = data.sort_values(["datadate", "tic"], ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data

class StockEnvTrain(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, df):
        super().__init__()
        self.df = df; self.day = 0
        self.action_space      = spaces.Box(-1, 1, (STOCK_DIM,), np.float32)
        self.observation_space = spaces.Box(0, np.inf, (STATE_DIM,), np.float32)
        self.balance = float(INITIAL_BALANCE)
        self.shares  = np.zeros(STOCK_DIM, np.float32)

    def _obs(self):
        d = self.df.loc[self.day]
        return np.array([self.balance] + d.adjcp.tolist() + list(self.shares)
                        + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist(),
                        dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = float(INITIAL_BALANCE)
        self.shares  = np.zeros(STOCK_DIM, np.float32)
        self.day = 0
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
        self.df = df; self.day = 0
        self.action_space      = spaces.Box(-1, 1, (STOCK_DIM,), np.float32)
        self.observation_space = spaces.Box(0, np.inf, (STATE_DIM,), np.float32)
        self.balance = float(INITIAL_BALANCE)
        self.shares  = np.zeros(STOCK_DIM, np.float32)
        self.asset_memory = [INITIAL_BALANCE]

    def _obs(self):
        d = self.df.loc[self.day]
        return np.array([self.balance] + d.adjcp.tolist() + list(self.shares)
                        + d.macd.tolist() + d.rsi.tolist() + d.cci.tolist() + d.adx.tolist(),
                        dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = float(INITIAL_BALANCE)
        self.shares  = np.zeros(STOCK_DIM, np.float32)
        self.day = 0; self.asset_memory = [INITIAL_BALANCE]
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
        return float((4**0.5) * ret.mean() / ret.std()) if ret.std() > 0 else 0.0


# ── Checkpoint callback ───────────────────────────────────────────────────────

class CheckpointEvalCallback(BaseCallback):
    def __init__(self, val_df, eval_interval, verbose=0):
        super().__init__(verbose)
        self.val_df        = val_df
        self.eval_interval = eval_interval
        self.timestep_log  = []
        self.sharpe_log    = []
        self._last_eval    = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_interval:
            self._last_eval = self.num_timesteps
            self.timestep_log.append(self.num_timesteps)
            self.sharpe_log.append(self._evaluate())
        return True

    def _evaluate(self):
        env = StockEnvValidation(self.val_df)
        obs, _ = env.reset()
        done = False
        while not done:
            act, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
        return env.get_sharpe()


# ── Run one seed ──────────────────────────────────────────────────────────────

def run_one_seed(agent_name, train_df, val_df, max_ts, eval_interval, seed):
    env_train = DummyVecEnv([lambda: StockEnvTrain(train_df)])
    callback  = CheckpointEvalCallback(val_df, eval_interval)

    if agent_name == "A2C":
        model = A2C("MlpPolicy", env_train, seed=seed, verbose=0)

    elif agent_name == "PPO":
        model = PPO("MlpPolicy", env_train, seed=seed, verbose=0,
                    ent_coef=0.005, n_steps=128, batch_size=16,
                    learning_rate=7e-4, n_epochs=4)

    elif agent_name == "DDPG":
        n     = env_train.action_space.shape[-1]
        noise = OrnsteinUhlenbeckActionNoise(np.zeros(n), 0.5 * np.ones(n))
        model = DDPG("MlpPolicy", env_train, action_noise=noise, seed=seed, verbose=0)

    model.learn(total_timesteps=max_ts, callback=callback)
    return callback.timestep_log, callback.sharpe_log


# ── Run all seeds for one agent, return mean ± std arrays ────────────────────

def run_learning_curve(agent_name, train_df, val_df, max_ts, eval_interval, n_seeds):
    print(f"\n{'─'*55}")
    print(f"  {agent_name}  —  {max_ts:,} steps  ×  {n_seeds} seeds")
    print(f"{'─'*55}")

    all_sharpes = []
    ts_grid     = None
    t0          = time.time()

    for seed in range(n_seeds):
        print(f"  seed {seed+1}/{n_seeds} …", end=" ", flush=True)
        ts_log, sh_log = run_one_seed(agent_name, train_df, val_df,
                                      max_ts, eval_interval, seed)
        print(f"done  (peak Sharpe {max(sh_log):.3f} @ {ts_log[int(np.argmax(sh_log))]:,})")
        all_sharpes.append(sh_log)
        if ts_grid is None:
            ts_grid = ts_log   # all seeds produce the same timestep grid

    elapsed = (time.time() - t0) / 60
    arr     = np.array(all_sharpes)          # shape: (n_seeds, n_checkpoints)
    mean_sh = arr.mean(axis=0)
    std_sh  = arr.std(axis=0)

    best_idx = int(np.argmax(mean_sh))
    print(f"  → Mean peak: Sharpe {mean_sh[best_idx]:.4f} ± {std_sh[best_idx]:.4f}"
          f"  at {ts_grid[best_idx]:,} steps  [{elapsed:.1f} min total]")

    return ts_grid, mean_sh, std_sh


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading data …")
    df = pd.read_csv(DATA_PATH, index_col=0)
    train_df = data_split(df, 20090000, TRAIN_END)
    val_df   = data_split(df, VAL_START, VAL_END)
    print(f"  Train: {train_df.datadate.min()} – {train_df.datadate.max()}  "
          f"({train_df.index.nunique()} days)")
    print(f"  Val:   {val_df.datadate.min()} – {val_df.datadate.max()}  "
          f"({val_df.index.nunique()} days)")
    print(f"  Seeds: {N_SEEDS}  (total runs: {3 * N_SEEDS})\n")

    results = {}
    for agent in ["A2C", "PPO", "DDPG"]:
        ts, mean_sh, std_sh = run_learning_curve(
            agent, train_df, val_df,
            max_ts=MAX_TIMESTEPS[agent],
            eval_interval=EVAL_INTERVAL,
            n_seeds=N_SEEDS,
        )
        results[agent] = (ts, mean_sh, std_sh)

    # ── Plot ──────────────────────────────────────────────────────────────────
    colours = {"A2C": "royalblue", "PPO": "mediumorchid", "DDPG": "darkorange"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, agent in zip(axes, ["A2C", "PPO", "DDPG"]):
        ts, mean_sh, std_sh = results[agent]

        ax.plot(ts, mean_sh, color=colours[agent], linewidth=2.0, label="Mean Sharpe")
        ax.fill_between(ts,
                        mean_sh - std_sh,
                        mean_sh + std_sh,
                        color=colours[agent], alpha=0.2, label="± 1 std")
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

        best_idx = int(np.argmax(mean_sh))
        ax.axvline(ts[best_idx], color="red", linewidth=1.2,
                   linestyle=":", label=f"Peak @ {ts[best_idx]:,}")
        ax.scatter([ts[best_idx]], [mean_sh[best_idx]], color="red", zorder=5, s=60)
        ax.annotate(f"  Best: {ts[best_idx]:,} steps\n  Sharpe: {mean_sh[best_idx]:.3f}",
                    xy=(ts[best_idx], mean_sh[best_idx]),
                    xytext=(10, -18), textcoords="offset points",
                    fontsize=8, color="red")

        ax.set_title(f"{agent} — Validation Sharpe vs Timesteps", fontsize=11)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Validation Sharpe")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Learning Curve Analysis ({N_SEEDS} seeds, mean ± 1 std)\n"
                 "(Train: 2009–2020  |  Val: 2020-Q4)", fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "learning_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved → {out}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  OPTIMAL TIMESTEPS SUMMARY  (averaged over seeds)")
    print("="*65)
    print(f"{'Agent':<8} {'Mean Sharpe':>12} {'± Std':>8} {'Optimal Steps':>15} {'Paper Steps':>12}")
    print("-"*65)
    paper_ts = {"A2C": 30_000, "PPO": 100_000, "DDPG": 10_000}
    for agent in ["A2C", "PPO", "DDPG"]:
        ts, mean_sh, std_sh = results[agent]
        best_idx = int(np.argmax(mean_sh))
        print(f"{agent:<8} {mean_sh[best_idx]:>12.4f} {std_sh[best_idx]:>8.4f} "
              f"{ts[best_idx]:>15,} {paper_ts[agent]:>12,}")

    print("\nDone.")
