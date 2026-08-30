"""
================================================================================
SKRIP TRAINING RPPO v2 — OPTIMIZED RISK-MANAGED AGENT
================================================================================
Agent menerima 22 indikator teknikal + 4 state = 26 dim.
Fully autonomous: agent memutuskan sendiri kapan BUY, SELL, HOLD.
SL/TP/Trailing Stop di-handle otomatis oleh environment.

Perubahan dari v1:
- Multi-environment training (n_envs=4)
- Total timesteps: 500K (dari 100K)
- Arsitektur lebih ramping (LSTM 128, net [256,128])
- Hyperparameter tuning: gamma, ent_coef, clip_range, etc.
- Evaluation score: Profit Factor × sqrt(trade_count) × (1 - MaxDD/100)
- Model HANYA disimpan jika performa validasi LEBIH BAIK
================================================================================
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from stable_baselines3.common.utils import get_linear_fn

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from indicators import load_dataset
from trading_env import IndicatorTradingEnv


# =============================================================================
# CALLBACKS
# =============================================================================

class TradingMetricsCallback(BaseCallback):
    """Callback untuk memantau metrik trading selama training."""
    def __init__(self, check_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            env = self.training_env.envs[0]
            # Unwrap through Monitor wrapper
            while hasattr(env, 'env'):
                env = env.env
            info = env._get_info()
            if self.verbose > 0:
                print(
                    f"\n[RPPO {self.n_calls:06d}/{self.model._total_timesteps}] "
                    f"Balance: ${info['balance']:8.2f} ({info['roi_pct']:+6.2f}%) | "
                    f"DD: {info['drawdown_pct']:5.2f}% | "
                    f"Trades: {info['trade_count']:4d} (W:{info['winning_trades']} L:{info['losing_trades']}) | "
                    f"Win Rate: {info['win_rate_pct']:5.1f}% | "
                    f"Hold: {info['hold_duration']:3d} | Flat: {info['flat_duration']:3d}"
                )
            self.logger.record("trading/balance", info['balance'])
            self.logger.record("trading/roi_pct", info['roi_pct'])
            self.logger.record("trading/drawdown_pct", info['drawdown_pct'])
            self.logger.record("trading/trade_count", info['trade_count'])
            self.logger.record("trading/win_rate_pct", info['win_rate_pct'])
            self.logger.record("trading/total_pnl", info['total_pnl'])
            self.logger.record("trading/hold_duration", info['hold_duration'])
        return True


class CurriculumCallback(BaseCallback):
    """Callback untuk Progressive Difficulty Scheduling (Curriculum Learning)."""
    def __init__(self, total_timesteps: int, initial_days: int = 252, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.initial_days = initial_days

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        # Apply to all environments
        for i, env_wrapper in enumerate(self.training_env.envs):
            env = env_wrapper
            while hasattr(env, 'env'):
                env = env.env
            total_len = env.total_data_len - 1
            current_max = int(self.initial_days + progress * (total_len - self.initial_days))
            env.curriculum_max_step = min(current_max, total_len)

        if self.n_calls % 10000 == 0 and self.verbose > 0:
            env = self.training_env.envs[0]
            while hasattr(env, 'env'):
                env = env.env
            print(f"[Curriculum] Jendela data aktif: {env.curriculum_max_step} steps.")

        return True


class TradeLogCallback(BaseCallback):
    """Callback untuk export trade log setiap 10 episode."""
    def __init__(self, log_dir: str = "trade_logs", verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_count = 0
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [False])
        if any(dones):
            self.episode_count += 1
            env = self.training_env.envs[0]
            while hasattr(env, 'env'):
                env = env.env
            trade_log = env.get_trade_log_df()
            if len(trade_log) > 0 and self.episode_count % 10 == 0:
                path = os.path.join(self.log_dir, f"trade_log_ep{self.episode_count:04d}.csv")
                trade_log.to_csv(path, index=False)
        return True


class SaveOnlyIfBetterCallback(BaseCallback):
    """
    =========================================================================
    HANYA SIMPAN JIKA LEBIH BAIK (STRICT BEST MODEL SAVER)
    =========================================================================
    Evaluasi model secara berkala pada dataset validasi.
    
    Skor Evaluasi (v2):
        PF = Profit Factor = Gross Profit / Gross Loss
        SCORE = PF × sqrt(trade_count) × (1 - MaxDD/100)
        
        Minimum: trade_count >= 20 untuk dianggap valid.
    =========================================================================
    """
    def __init__(
        self,
        eval_features: pd.DataFrame,
        eval_prices: pd.DataFrame,
        eval_raw_atr: pd.Series,
        initial_balance: float = 1000.0,
        eval_freq: int = 10_000,
        save_dir: str = "models",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_features = eval_features
        self.eval_prices = eval_prices
        self.eval_raw_atr = eval_raw_atr
        self.initial_balance = initial_balance
        self.eval_freq = eval_freq
        self.save_dir = save_dir
        self.best_path = os.path.join(save_dir, "best_model")

        self.best_score = -np.inf
        self.best_metrics = None
        self.eval_history = []

        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Jalankan evaluasi deterministik
        metrics = self._evaluate()

        # Hitung skor v2: Profit Factor × sqrt(trade_count) × (1 - MaxDD/100)
        if metrics['trade_count'] < 20:
            score = -999.0  # Not enough trades
        else:
            pf = metrics['profit_factor']
            tc_bonus = np.sqrt(metrics['trade_count'])
            dd_penalty = max(0.01, 1.0 - metrics['max_dd_pct'] / 100.0)
            score = pf * tc_bonus * dd_penalty

        metrics['score'] = score
        metrics['timestep'] = self.num_timesteps
        self.eval_history.append(metrics)

        if self.verbose > 0:
            print(f"\n{'='*75}")
            print(f"[EVALUASI @ {self.num_timesteps:,} STEPS]")
            print(
                f"  ROI: {metrics['roi_pct']:+.2f}% | "
                f"MaxDD: {metrics['max_dd_pct']:.2f}% | "
                f"Trades: {metrics['trade_count']} | "
                f"Win Rate: {metrics['win_rate_pct']:.1f}% | "
                f"PF: {metrics['profit_factor']:.2f} | "
                f"Total PnL: ${metrics['total_pnl']:+.2f}"
            )
            print(f"  Skor: {score:.4f} (Terbaik saat ini: {self.best_score:.4f})")

        # HANYA SIMPAN JIKA LEBIH BAIK!
        if score > self.best_score:
            self.best_score = score
            self.best_metrics = metrics

            # Simpan model tunggal
            self.model.save(self.best_path)

            # Simpan metrik terbaik ke json
            metrics_file = os.path.join(self.save_dir, "best_metrics.json")
            save_metrics = {k: v for k, v in metrics.items() if k != 'balance_history'}
            with open(metrics_file, "w") as f:
                json.dump(save_metrics, f, indent=2, default=str)

            if self.verbose > 0:
                print(f"  >>> [SUKSES DISIMPAN] Model lebih baik -> {self.best_path}.zip")
            print(f"{'='*75}\n")
        else:
            if self.verbose > 0:
                print(f"  [TIDAK DISIMPAN] Performa belum melampaui skor terbaik ({self.best_score:.4f})")
            print(f"{'='*75}\n")

        return True

    def _evaluate(self) -> dict:
        """Evaluasi deterministik pada validation dataset."""
        env = IndicatorTradingEnv(
            features_df=self.eval_features,
            prices_df=self.eval_prices,
            raw_atr=self.eval_raw_atr,
            initial_balance=self.initial_balance,
            lot_size=0.1,
            spread_cost=0.50,
            max_steps_per_episode=None,
            random_start=False,
        )

        obs, info = env.reset()
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        balance_history = [self.initial_balance]
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, lstm_states = self.model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            episode_start = np.array([terminated or truncated])

            obs, reward, terminated, truncated, info = env.step(int(action))
            balance_history.append(info['balance'])

        bal_arr = np.array(balance_history)
        running_max = np.maximum.accumulate(bal_arr)
        dd_arr = (running_max - bal_arr) / (running_max + 1e-8) * 100.0

        # Profit Factor
        trade_log = env.get_trade_log_df()
        if len(trade_log) > 0:
            gross_profit = trade_log[trade_log['pnl_usd'] > 0]['pnl_usd'].sum()
            gross_loss = abs(trade_log[trade_log['pnl_usd'] < 0]['pnl_usd'].sum())
            profit_factor = gross_profit / (gross_loss + 1e-8)
        else:
            profit_factor = 0.0

        return {
            'roi_pct': round(float(info['roi_pct']), 2),
            'max_dd_pct': round(float(np.max(dd_arr)), 2),
            'final_balance': round(float(info['balance']), 2),
            'trade_count': int(info['trade_count']),
            'winning_trades': int(info['winning_trades']),
            'losing_trades': int(info['losing_trades']),
            'win_rate_pct': round(float(info['win_rate_pct']), 2),
            'total_pnl': round(float(info['total_pnl']), 2),
            'profit_factor': round(profit_factor, 4),
        }


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train(
    csv_path: str = "XAUUSD_H1.csv",
    total_timesteps: int = 500_000,
    initial_balance: float = 1000.0,
    n_envs: int = 4,
    save_dir: str = "models",
    log_dir: str = "logs"
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 75)
    print("1. MEMUAT DATASET XAU/USD H1 — TECHNICAL INDICATORS + MULTI-TIMEFRAME")
    print("=" * 75)
    features_df, prices_df, raw_atr = load_dataset(csv_path)
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])

    # Split: 70% Train (2017-2022)
    train_mask = (prices_df['Date'] >= '2017-01-01') & (prices_df['Date'] <= '2022-12-31')
    train_features = features_df[train_mask].reset_index(drop=True)
    train_prices   = prices_df[train_mask].reset_index(drop=True)
    train_atr      = raw_atr[train_mask].reset_index(drop=True)

    # Validation set (2023-2024)
    val_mask = (prices_df['Date'] >= '2023-01-01') & (prices_df['Date'] <= '2024-09-15')
    val_features = features_df[val_mask].reset_index(drop=True)
    val_prices   = prices_df[val_mask].reset_index(drop=True)
    val_atr      = raw_atr[val_mask].reset_index(drop=True)

    num_indicators = len(features_df.columns)
    print(f"  Data Training   : {len(train_features):,} steps (2017-01-01 s/d 2022-12-31)")
    print(f"  Data Validation : {len(val_features):,} steps (2023-01-01 s/d 2024-09-15)")
    print(f"  Indikator       : {num_indicators} fitur teknikal (Tanpa OHLC)")
    print(f"  State Space     : {num_indicators + 4} dim (indicators + atr_norm + position + unrealized_pnl + hold_norm)")
    print(f"  Action Space    : 3 (HOLD / BUY / SELL)")
    print(f"  Risk Management : SL=2×ATR, TP=3×ATR, Trailing Stop")
    print(f"  Agent Type      : FULLY AUTONOMOUS")
    print(f"  Parallel Envs   : {n_envs}")
    print(f"  Total Timesteps : {total_timesteps:,}")
    print(f"  Penyimpanan     : HANYA SIMPAN JIKA LEBIH BAIK (PF × sqrt(TC) × DD_penalty)\n")

    def make_env(seed: int):
        def _init():
            env = IndicatorTradingEnv(
                features_df=train_features,
                prices_df=train_prices,
                raw_atr=train_atr,
                initial_balance=initial_balance,
                lot_size=0.1,
                spread_cost=0.50,
                max_steps_per_episode=252 * 24,
                random_start=True,
                trade_log_path=None,
            )
            env.reset(seed=seed)
            return Monitor(env)
        return _init

    # Multi-environment (DummyVecEnv for LSTM compatibility)
    vec_env = DummyVecEnv([make_env(seed=i * 42) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Arsitektur RPPO v2 — lebih ramping
    policy_kwargs = dict(
        lstm_hidden_size=128,
        n_lstm_layers=1,
        net_arch=dict(
            pi=[256, 128],
            vf=[256, 128]
        ),
        activation_fn=torch.nn.ReLU,
        enable_critic_lstm=True,
        shared_lstm=False
    )

    lr_schedule = get_linear_fn(start=3e-4, end=1e-5, end_fraction=1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()}")

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=device
    )

    # Callbacks
    metrics_callback = TradingMetricsCallback(check_freq=4096, verbose=1)
    curriculum_callback = CurriculumCallback(total_timesteps=total_timesteps, initial_days=252, verbose=1)
    trade_log_callback = TradeLogCallback(log_dir="trade_logs", verbose=1)
    
    # Callback Simpan Hanya Jika Lebih Baik (setiap 10.000 timesteps)
    better_saver_callback = SaveOnlyIfBetterCallback(
        eval_features=val_features,
        eval_prices=val_prices,
        eval_raw_atr=val_atr,
        initial_balance=initial_balance,
        eval_freq=10_000,
        save_dir=save_dir,
        verbose=1,
    )

    print("=" * 75)
    print(f"2. MEMULAI TRAINING RPPO v2 ({total_timesteps:,} TIMESTEPS)")
    print("=" * 75)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            metrics_callback,
            curriculum_callback,
            trade_log_callback,
            better_saver_callback,
        ],
        progress_bar=True
    )

    print("\nTraining selesai! Menyimpan status VecNormalize dan model final...")
    model.save(os.path.join(save_dir, "rppo_indicator_final"))
    vec_env.save(os.path.join(save_dir, "vec_normalize.pkl"))

    if better_saver_callback.best_metrics:
        print(f"\n{'='*75}")
        print(f"HASIL MODEL TERBAIK TERSIMPAN:")
        print(f"  File           : {save_dir}/best_model.zip")
        print(f"  ROI            : {better_saver_callback.best_metrics['roi_pct']:+.2f}%")
        print(f"  Max Drawdown   : {better_saver_callback.best_metrics['max_dd_pct']:.2f}%")
        print(f"  Win Rate       : {better_saver_callback.best_metrics['win_rate_pct']:.1f}%")
        print(f"  Profit Factor  : {better_saver_callback.best_metrics['profit_factor']:.2f}")
        print(f"  Total PnL      : ${better_saver_callback.best_metrics['total_pnl']:+.2f}")
        print(f"  Skor           : {better_saver_callback.best_score:.4f}")
        print(f"{'='*75}\n")

    return model


if __name__ == "__main__":
    train()
