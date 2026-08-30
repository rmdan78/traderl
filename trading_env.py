"""
================================================================================
INDICATOR-ONLY TRADING ENVIRONMENT — OPTIMIZED v2 (RISK-MANAGED)
================================================================================
Agent menerima 22 indikator teknikal (z-scored) + 4 state dim = 26 dim total.
TIDAK ada OHLC mentah yang diberikan ke agent.

Action Space (Discrete 3):
    0 = HOLD   (tahan posisi / tetap flat)
    1 = BUY    (buka LONG / tutup SHORT lalu buka LONG)
    2 = SELL   (buka SHORT / tutup LONG lalu buka SHORT)

Risk Management (OTOMATIS — agent TIDAK perlu belajar kapan exit):
    - Dynamic Stop Loss  = Entry ± SL_ATR_MULT × ATR (default 2.0)
    - Dynamic Take Profit = Entry ± TP_ATR_MULT × ATR (default 3.0)
    - Trailing Stop: setelah profit > 1×ATR → SL ke breakeven
                     setelah profit > 2×ATR → SL ke entry + 1×ATR
    - Intrabar check menggunakan High/Low (bukan hanya Close)

Reward System (Risk-Adjusted):
    - Close Reward: tanh(PnL / Risk) — risk = jarak entry ke SL
    - Asymmetric: loss × 1.5 (agent belajar fear loss > greed profit)
    - Step PnL: delta unrealized PnL (bukan absolute)
    - Inactivity penalty: hanya jika flat > 48 bars
    - TIDAK ada open reward (menghindari overtrading)
================================================================================
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import csv
import os


class Action:
    HOLD = 0
    BUY = 1
    SELL = 2


class IndicatorTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    # -------------------------------------------------------------------------
    # Risk Management Constants
    # -------------------------------------------------------------------------
    SL_ATR_MULT = 2.0              # Stop Loss = 2 × ATR
    TP_ATR_MULT = 3.0              # Take Profit = 3 × ATR
    TRAIL_BREAKEVEN_ATR = 1.0      # Trailing SL → breakeven setelah 1×ATR profit
    TRAIL_LOCK_ATR = 2.0           # Trailing SL → +1×ATR setelah 2×ATR profit

    # -------------------------------------------------------------------------
    # Reward Constants
    # -------------------------------------------------------------------------
    CLOSE_REWARD_SCALE = 0.1       # Skala reward saat close trade
    CLOSE_REWARD_NORM = 1.0        # Normalisasi: PnL/Risk (sudah risk-adjusted)
    LOSS_MULTIPLIER = 1.5          # Asymmetric: loss 1.5x lebih berat

    PNL_STEP_SCALE = 0.003         # Skala feedback delta PnL per step
    PNL_STEP_NORM = 10.0           # Normalisasi step PnL USD

    MAX_FREE_FLAT_BARS = 48        # 48 bar (2 hari) bebas penalti flat
    INACTIVITY_PENALTY = -0.002    # Penalti ringan jika pasif > 48 bar

    MAX_DRAWDOWN_PCT = 20.0        # Terminate jika drawdown > 20%

    def __init__(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        raw_atr: pd.Series,
        initial_balance: float = 1000.0,
        lot_size: float = 0.1,
        spread_cost: float = 0.50,
        contract_size: float = 100.0,
        max_steps_per_episode: int = None,
        random_start: bool = False,
        trade_log_path: str = None,
    ):
        super(IndicatorTradingEnv, self).__init__()

        # --- Data ---
        self.features_df = features_df.copy()
        self.prices_df = prices_df.copy()

        self.features_data = self.features_df.values.astype(np.float32)
        self.num_indicators = self.features_data.shape[1]

        # Prices (INTERNAL ONLY — agent never sees these)
        self.raw_open  = self.prices_df['Open'].values.astype(np.float64)
        self.raw_high  = self.prices_df['High'].values.astype(np.float64)
        self.raw_low   = self.prices_df['Low'].values.astype(np.float64)
        self.raw_close = self.prices_df['Close'].values.astype(np.float64)

        # ATR for SL/TP (INTERNAL ONLY)
        self.raw_atr = raw_atr.values.astype(np.float64)
        # Normalized ATR for observation (price-relative)
        self.atr_pct = self.raw_atr / (self.raw_close + 1e-8)

        self.total_data_len = len(self.features_data)

        # --- Trading parameters ---
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.spread_cost = spread_cost
        self.contract_size = contract_size  # 100 for XAUUSD, 100000 for EURUSD

        self.max_steps_per_episode = max_steps_per_episode
        self.random_start = random_start
        self.curriculum_max_step = self.total_data_len - 1
        self.episode_end_step = self.total_data_len - 1

        self.trade_log_path = trade_log_path

        # --- Spaces ---
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL

        # Observation: num_indicators + atr_norm + position + unrealized_pnl + hold_duration_norm
        self.num_obs = self.num_indicators + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_obs,),
            dtype=np.float32
        )

        self._init_state()

    def _init_state(self):
        """Initialize/reset all tracking state."""
        self.current_step = 0
        self.episode_step_count = 0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance

        # Position tracking
        self.position = 0          # -1 (SHORT), 0 (FLAT), +1 (LONG)
        self.entry_price = 0.0
        self.hold_duration = 0     # Steps since position opened
        self.flat_duration = 0     # Steps since went flat

        # Risk management state
        self.stop_loss = 0.0       # Current SL price
        self.take_profit = 0.0     # Current TP price
        self.entry_atr = 0.0       # ATR at time of entry (for risk calculation)
        self.prev_unrealized_pnl = 0.0  # For delta PnL reward

        # Statistics
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_reward = 0.0

        # Trade log (list of dicts)
        self.trade_log = []
        self._current_trade = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Export trade log from previous episode (if any)
        if len(self.trade_log) > 0 and self.trade_log_path:
            self._export_trade_log()

        # Reset state
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.hold_duration = 0
        self.flat_duration = 0
        self.episode_step_count = 0

        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.entry_atr = 0.0
        self.prev_unrealized_pnl = 0.0

        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_reward = 0.0

        self.trade_log = []
        self._current_trade = None

        # Determine episode boundaries
        active_total_len = getattr(self, 'curriculum_max_step', self.total_data_len - 1)

        if self.max_steps_per_episode is None:
            episode_len = active_total_len
        else:
            episode_len = min(self.max_steps_per_episode, active_total_len)

        max_start = max(0, active_total_len - episode_len)

        if getattr(self, 'random_start', False) and max_start > 0:
            self.current_step = self.np_random.integers(0, max_start)
        else:
            self.current_step = 0

        self.episode_end_step = min(self.current_step + episode_len, active_total_len)

        return self._get_observation(), self._get_info()

    def _get_observation(self) -> np.ndarray:
        """
        Observation = [22 indikator z-scored] + [atr_norm, position, unrealized_pnl, hold_norm]
        TIDAK ADA OHLC MENTAH!
        """
        indicators = self.features_data[self.current_step]

        # ATR as percentage of price (normalized, not z-scored)
        atr_norm = float(self.atr_pct[self.current_step]) * 100.0  # Scale to ~0.5-3.0 range

        # Unrealized PnL (normalized via tanh)
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            current_price = self.raw_close[self.current_step]
            price_diff = current_price - self.entry_price
            pnl_usd = self.position * price_diff * self.lot_size * self.contract_size
            unrealized_pnl = float(np.tanh(pnl_usd / (self.initial_balance * 0.05)))

        pos_float = float(self.position)

        # Normalized hold duration (tanh scaled, ~0-1 range for 0-50 bars)
        hold_norm = float(np.tanh(self.hold_duration / 25.0))

        obs = np.concatenate([
            indicators,
            [atr_norm, pos_float, unrealized_pnl, hold_norm]
        ]).astype(np.float32)

        return obs

    def _compute_unrealized_pnl_usd(self, price: float) -> float:
        """Calculate unrealized PnL in USD."""
        if self.position == 0 or self.entry_price <= 0:
            return 0.0
        price_diff = price - self.entry_price
        return self.position * price_diff * self.lot_size * self.contract_size

    def _set_sl_tp(self, entry_price: float, direction: int, atr: float):
        """Set Stop Loss and Take Profit based on ATR."""
        self.entry_atr = max(atr, 0.5)  # Minimum ATR floor to avoid tiny SL

        sl_distance = self.SL_ATR_MULT * self.entry_atr
        tp_distance = self.TP_ATR_MULT * self.entry_atr

        if direction == 1:  # LONG
            self.stop_loss = entry_price - sl_distance
            self.take_profit = entry_price + tp_distance
        else:  # SHORT
            self.stop_loss = entry_price + sl_distance
            self.take_profit = entry_price - tp_distance

    def _update_trailing_stop(self, current_price: float):
        """Update trailing stop loss based on profit."""
        if self.position == 0 or self.entry_atr <= 0:
            return

        if self.position == 1:  # LONG
            profit_distance = current_price - self.entry_price
            if profit_distance >= self.TRAIL_LOCK_ATR * self.entry_atr:
                # Lock profit: SL → entry + 1×ATR
                new_sl = self.entry_price + self.entry_atr
                self.stop_loss = max(self.stop_loss, new_sl)
            elif profit_distance >= self.TRAIL_BREAKEVEN_ATR * self.entry_atr:
                # Breakeven: SL → entry price
                self.stop_loss = max(self.stop_loss, self.entry_price)

        else:  # SHORT
            profit_distance = self.entry_price - current_price
            if profit_distance >= self.TRAIL_LOCK_ATR * self.entry_atr:
                # Lock profit: SL → entry - 1×ATR
                new_sl = self.entry_price - self.entry_atr
                self.stop_loss = min(self.stop_loss, new_sl)
            elif profit_distance >= self.TRAIL_BREAKEVEN_ATR * self.entry_atr:
                # Breakeven: SL → entry price
                self.stop_loss = min(self.stop_loss, self.entry_price)

    def _check_sl_tp_hit(self) -> Optional[str]:
        """
        Check if SL or TP was hit during the current bar using High/Low.
        Returns 'SL', 'TP', or None.
        """
        if self.position == 0:
            return None

        high = self.raw_high[self.current_step]
        low = self.raw_low[self.current_step]

        if self.position == 1:  # LONG
            if low <= self.stop_loss:
                return 'SL'
            if high >= self.take_profit:
                return 'TP'
        else:  # SHORT
            if high >= self.stop_loss:
                return 'SL'
            if low <= self.take_profit:
                return 'TP'

        return None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step:
        1. Check SL/TP hit (intrabar)
        2. Update trailing stop
        3. Calculate step PnL
        4. Process action (HOLD / BUY / SELL)
        5. Apply rewards
        6. Advance step
        """
        current_price = self.raw_close[self.current_step]
        current_atr = self.raw_atr[self.current_step]
        reward = 0.0

        # =====================================================================
        # 1. CHECK SL/TP HIT (before processing action)
        # =====================================================================
        sl_tp_hit = self._check_sl_tp_hit()
        if sl_tp_hit is not None:
            # Determine exit price
            if sl_tp_hit == 'SL':
                exit_price = self.stop_loss
            else:  # TP
                exit_price = self.take_profit

            # Close trade at SL/TP price
            close_pnl = self._close_trade(exit_price, reason=sl_tp_hit)
            reward += self._compute_close_reward(close_pnl)

            # Apply balance
            self.balance += close_pnl

            self.position = 0
            self.entry_price = 0.0
            self.hold_duration = 0
            self.flat_duration = 0
            self.stop_loss = 0.0
            self.take_profit = 0.0
            self.prev_unrealized_pnl = 0.0

        # =====================================================================
        # 2. UPDATE TRAILING STOP (if still in position)
        # =====================================================================
        if self.position != 0:
            self._update_trailing_stop(current_price)

        # =====================================================================
        # 3. CALCULATE STEP PnL FROM EXISTING POSITION (delta-based)
        # =====================================================================
        if self.position != 0 and self.entry_price > 0 and self.current_step > 0:
            curr_unrealized = self._compute_unrealized_pnl_usd(current_price)
            delta_pnl = curr_unrealized - self.prev_unrealized_pnl
            self.prev_unrealized_pnl = curr_unrealized

            # Delta PnL feedback (normalized via tanh)
            step_pnl_feedback = float(np.tanh(delta_pnl / self.PNL_STEP_NORM)) * self.PNL_STEP_SCALE
            reward += step_pnl_feedback

        # =====================================================================
        # 4. PROCESS ACTION
        # =====================================================================
        if action == Action.HOLD:
            if self.position != 0:
                self.hold_duration += 1
                self.flat_duration = 0
            else:
                self.flat_duration += 1
                # Inactivity penalty (only if flat > 48 bars)
                if self.flat_duration > self.MAX_FREE_FLAT_BARS:
                    reward += self.INACTIVITY_PENALTY

        elif action == Action.BUY:
            if self.position == 1:
                # Already LONG → same as HOLD
                self.hold_duration += 1
                self.flat_duration = 0
            elif self.position == -1:
                # Flip SHORT → LONG (close SHORT first)
                close_pnl = self._close_trade(current_price, reason='FLIP')
                reward += self._compute_close_reward(close_pnl)
                self.balance += close_pnl

                # Open LONG
                self._open_position(current_price, 1, current_atr)
            else:
                # FLAT → Open LONG
                self._open_position(current_price, 1, current_atr)

        elif action == Action.SELL:
            if self.position == -1:
                # Already SHORT → same as HOLD
                self.hold_duration += 1
                self.flat_duration = 0
            elif self.position == 1:
                # Flip LONG → SHORT (close LONG first)
                close_pnl = self._close_trade(current_price, reason='FLIP')
                reward += self._compute_close_reward(close_pnl)
                self.balance += close_pnl

                # Open SHORT
                self._open_position(current_price, -1, current_atr)
            else:
                # FLAT → Open SHORT
                self._open_position(current_price, -1, current_atr)

        # =====================================================================
        # 5. UPDATE BALANCE & STATS
        # =====================================================================
        # Mark-to-market balance (update with unrealized PnL for tracking)
        if self.position != 0:
            unrealized = self._compute_unrealized_pnl_usd(current_price)
            mtm_balance = self.balance + unrealized
        else:
            mtm_balance = self.balance

        mtm_balance = max(1.0, mtm_balance)
        if mtm_balance > self.peak_balance:
            self.peak_balance = mtm_balance

        self.total_reward += reward

        # =====================================================================
        # 6. ADVANCE STEP
        # =====================================================================
        self.current_step += 1
        self.episode_step_count += 1

        # =====================================================================
        # 7. TERMINATION CONDITIONS
        # =====================================================================
        # Drawdown protection (20%)
        current_dd = (self.peak_balance - mtm_balance) / (self.peak_balance + 1e-8) * 100.0
        terminated = bool(current_dd >= self.MAX_DRAWDOWN_PCT)

        # Balance too low
        if self.balance < (self.initial_balance * 0.3):
            terminated = True

        if getattr(self, 'episode_end_step', None) is not None:
            truncated = bool(self.current_step >= self.episode_end_step)
        else:
            truncated = bool(self.current_step >= self.total_data_len - 1)

        # Force close at end of episode
        if (terminated or truncated) and self.position != 0:
            close_price = self.raw_close[min(self.current_step, self.total_data_len - 1)]
            close_pnl = self._close_trade(close_price, reason='EPISODE_END')
            reward += self._compute_close_reward(close_pnl)
            self.balance += close_pnl
            self.position = 0
            self.entry_price = 0.0
            self.hold_duration = 0
            self.stop_loss = 0.0
            self.take_profit = 0.0

        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    # =========================================================================
    # REWARD COMPUTATION
    # =========================================================================

    def _compute_close_reward(self, trade_pnl: float) -> float:
        """
        Risk-adjusted close reward with asymmetric loss penalty.
        reward = tanh(PnL / Risk) * SCALE
        loss is multiplied by LOSS_MULTIPLIER (1.5x)
        """
        # Risk = SL distance in USD
        risk_usd = self.entry_atr * self.SL_ATR_MULT * self.lot_size * self.contract_size
        risk_usd = max(risk_usd, 1.0)  # Floor

        pnl_over_risk = trade_pnl / risk_usd
        base_reward = float(np.tanh(pnl_over_risk / self.CLOSE_REWARD_NORM)) * self.CLOSE_REWARD_SCALE

        # Asymmetric: amplify loss signal
        if base_reward < 0:
            base_reward *= self.LOSS_MULTIPLIER

        return base_reward

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def _open_position(self, price: float, direction: int, atr: float):
        """Open a new position with SL/TP."""
        self.position = direction
        self.entry_price = price
        self.hold_duration = 0
        self.flat_duration = 0
        self.balance -= self.spread_cost
        self.prev_unrealized_pnl = 0.0

        # Set SL/TP
        self._set_sl_tp(price, direction, atr)

        # Log trade open
        dir_str = "LONG" if direction == 1 else "SHORT"
        self._open_trade(price, dir_str)

    # =========================================================================
    # TRADE LOGGING
    # =========================================================================

    def _open_trade(self, price: float, direction: str):
        """Record opening of a new trade."""
        self._current_trade = {
            'trade_num': self.trade_count + 1,
            'entry_step': self.current_step,
            'entry_price': price,
            'direction': direction,
            'balance_at_entry': self.balance,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_atr': self.entry_atr,
        }

    def _close_trade(self, close_price: float, reason: str = 'MANUAL') -> float:
        """Record closing of a trade, compute PnL, update stats. Returns trade PnL."""
        if self._current_trade is None:
            return 0.0

        entry_price = self._current_trade['entry_price']
        direction = self._current_trade['direction']

        # Calculate trade PnL
        if direction == "LONG":
            trade_pnl = (close_price - entry_price) * self.lot_size * self.contract_size
        else:
            trade_pnl = (entry_price - close_price) * self.lot_size * self.contract_size

        trade_pnl -= self.spread_cost  # Deduct spread on exit

        hold_bars = self.current_step - self._current_trade['entry_step']

        self.trade_count += 1
        self.total_pnl += trade_pnl
        if trade_pnl > 0:
            self.winning_trades += 1
        elif trade_pnl < 0:
            self.losing_trades += 1

        # Log the trade
        self.trade_log.append({
            'trade_num': self.trade_count,
            'direction': direction,
            'entry_step': self._current_trade['entry_step'],
            'exit_step': self.current_step,
            'hold_bars': hold_bars,
            'entry_price': round(entry_price, 2),
            'exit_price': round(close_price, 2),
            'stop_loss': round(self._current_trade.get('stop_loss', 0), 2),
            'take_profit': round(self._current_trade.get('take_profit', 0), 2),
            'pnl_usd': round(trade_pnl, 4),
            'balance_after': round(self.balance + trade_pnl, 2),
            'result': 'WIN' if trade_pnl > 0 else ('LOSS' if trade_pnl < 0 else 'BREAK_EVEN'),
            'exit_reason': reason,
        })

        self._current_trade = None
        return trade_pnl

    def _export_trade_log(self):
        """Export trade log to CSV file."""
        if not self.trade_log_path or len(self.trade_log) == 0:
            return

        os.makedirs(os.path.dirname(self.trade_log_path) if os.path.dirname(self.trade_log_path) else '.', exist_ok=True)

        fieldnames = [
            'trade_num', 'direction', 'entry_step', 'exit_step', 'hold_bars',
            'entry_price', 'exit_price', 'stop_loss', 'take_profit',
            'pnl_usd', 'balance_after', 'result', 'exit_reason'
        ]

        with open(self.trade_log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trade_log)

    def get_trade_log_df(self) -> pd.DataFrame:
        """Return trade log as a DataFrame."""
        if len(self.trade_log) == 0:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)

    # =========================================================================
    # INFO & RENDER
    # =========================================================================

    def _get_info(self) -> Dict[str, Any]:
        roi_pct = ((self.balance - self.initial_balance) / self.initial_balance) * 100.0
        win_rate = (self.winning_trades / max(1, self.trade_count)) * 100.0

        # Use mark-to-market for drawdown
        if self.position != 0 and self.entry_price > 0:
            curr_price = self.raw_close[min(self.current_step, self.total_data_len - 1)]
            unrealized = self._compute_unrealized_pnl_usd(curr_price)
            mtm_balance = self.balance + unrealized
        else:
            mtm_balance = self.balance

        drawdown_pct = max(
            0.0,
            (self.peak_balance - mtm_balance) / (self.peak_balance + 1e-8)
        ) * 100.0

        pos_str = "FLAT"
        if self.position > 0:
            pos_str = "LONG"
        elif self.position < 0:
            pos_str = "SHORT"

        return {
            'step': self.current_step,
            'balance': self.balance,
            'peak_balance': self.peak_balance,
            'drawdown_pct': drawdown_pct,
            'roi_pct': roi_pct,
            'position': pos_str,
            'position_raw': self.position,
            'hold_duration': self.hold_duration,
            'flat_duration': self.flat_duration,
            'trade_count': self.trade_count,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': win_rate,
            'total_pnl': self.total_pnl,
            'total_reward': self.total_reward,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
        }

    def render(self):
        info = self._get_info()
        print(
            f"Step: {self.episode_step_count:04d} | "
            f"Pos: {info['position']:<6} (hold:{info['hold_duration']:3d} flat:{info['flat_duration']:3d}) | "
            f"Balance: ${info['balance']:.2f} ({info['roi_pct']:+.2f}%) | "
            f"DD: {info['drawdown_pct']:.2f}% | "
            f"Trades: {info['trade_count']} (W:{info['winning_trades']} L:{info['losing_trades']}) | "
            f"Win Rate: {info['win_rate_pct']:.1f}% | "
            f"PnL: ${info['total_pnl']:+.2f} | "
            f"SL: {info['stop_loss']:.2f} TP: {info['take_profit']:.2f}"
        )
