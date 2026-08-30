"""
================================================================================
MODUL REAL-TIME FEATURE ENGINE (MT5 -> 26-DIM RL OBSERVATION)
================================================================================
Mengonversi candle mentah OHLCV dari MT5 menjadi vektor observasi 26 dimensi
yang 100% identik dengan environment training RL (RecurrentPPO):
  - 22 Indikator Teknikal (16 H1 + 6 H4/Regime, Rolling Z-Score 252)
  - 4 State Dimensi: [atr_norm, position, unrealized_pnl, hold_duration_norm]
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

from indicators import compute_features, calc_atr


class MT5FeatureEngine:
    """Mesin kalkulasi fitur teknikal dan observasi state untuk agent RL."""

    def __init__(
        self,
        min_candles: int = 500,
        zscore_window: int = 252,
        initial_balance_ref: float = 1000.0,
    ):
        self.min_candles = min_candles
        self.zscore_window = zscore_window
        self.initial_balance_ref = initial_balance_ref

    def build_observation(
        self,
        candles_df: pd.DataFrame,
        current_position: int = 0,        # 1 (BUY), -1 (SELL), 0 (FLAT)
        unrealized_pnl_usd: float = 0.0,   # Floating PnL dari MT5
        hold_bars: int = 0,                # Durasi hold dalam jumlah bar H1
    ) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        """
        Menghitung 26 dimensi observasi dari data candle MT5 terbaru.

        Returns:
            obs (np.ndarray): Vektor 26 dimensi float32
            latest_close (float): Harga Close candle terakhir yang selesai
            latest_atr (float): Nilai ATR raw candle terakhir (untuk SL/TP)
            meta (dict): Metadata ringkasan fitur
        """
        if len(candles_df) < self.zscore_window + 50:
            raise ValueError(
                f"Jumlah candle ({len(candles_df)}) kurang dari kebutuhan minimum "
                f"({self.zscore_window + 50}) untuk normalisasi Z-Score yang stabil."
            )

        # 1. Hitung 22 fitur teknikal (Z-scored) & Raw ATR dari modul indicators
        features_clean, prices_clean, raw_atr_clean = compute_features(
            candles_df, zscore_window=self.zscore_window
        )

        if len(features_clean) == 0:
            raise ValueError("Gagal menghasilkan fitur bersih dari candle MT5.")

        # Ambil baris data terakhir (bar candle yang baru saja closed)
        last_indicators = features_clean.iloc[-1].values.astype(np.float32)
        last_price = float(prices_clean['Close'].iloc[-1])
        last_raw_atr = float(raw_atr_clean.iloc[-1])

        # 2. Hitung 4 state feature tambahan (sesuai trading_env.py)
        # a. atr_norm: persentase ATR terhadap harga
        atr_norm = float(last_raw_atr / (last_price + 1e-8)) * 100.0

        # b. pos_float: status posisi saat ini (-1.0, 0.0, 1.0)
        pos_float = float(current_position)

        # c. unrealized_pnl: normalisasi tanh terhadap 5% initial balance
        norm_factor = self.initial_balance_ref * 0.05
        unrealized_pnl_norm = float(np.tanh(unrealized_pnl_usd / (norm_factor + 1e-8)))

        # d. hold_norm: normalisasi tanh durasi hold terhadap scale 25 bar
        hold_norm = float(np.tanh(hold_bars / 25.0))

        # 3. Gabungkan menjadi 26 dimensi
        obs = np.concatenate([
            last_indicators,
            [atr_norm, pos_float, unrealized_pnl_norm, hold_norm]
        ]).astype(np.float32)

        meta = {
            "num_indicators": len(last_indicators),
            "total_obs_dim": len(obs),
            "latest_time": str(prices_clean['Date'].iloc[-1]),
            "latest_close": last_price,
            "latest_atr": last_raw_atr,
            "atr_norm": atr_norm,
            "position": current_position,
            "unrealized_pnl_usd": unrealized_pnl_usd,
            "unrealized_pnl_norm": unrealized_pnl_norm,
            "hold_bars": hold_bars,
            "hold_norm": hold_norm,
        }

        return obs, last_price, last_raw_atr, meta
