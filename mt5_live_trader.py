"""
================================================================================
METATRADER 5 (MT5) LIVE TRADING AGENT — REINFORCEMENT LEARNING
================================================================================
Menjalankan model AI RL terbaik (RecurrentPPO 26-dim) secara mandiri di MT5:
  - Membaca live H1 candle & status akun secara berkala
  - Menghitung 22 indikator teknikal (Z-scored) + 4 state obs = 26 dimensi
  - Agent memutuskan secara otonom: 0=HOLD, 1=BUY, 2=SELL
  - Manajemen risiko otomatis: Dynamic SL (2x ATR), TP (3x ATR)
  - Intrabar Trailing Stop cepat (Breakeven @ 1x ATR, Lock Profit @ 2x ATR)
  - Emergency Circuit Breaker (Max Drawdown & Daily Loss protection)
  - Mendukung mode DRY_RUN, DEMO, dan REAL
================================================================================
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from mt5_connector import MT5Connector
from mt5_feature_engine import MT5FeatureEngine
from mt5_notifier import MT5Notifier


# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MT5LiveTrader")


class MT5LiveTrader:
    """Sistem Utama Eksekusi Live Trading AI Agent."""

    def __init__(
        self,
        config_path: str = "mt5_config.json",
        env_path: str = ".env",
        dry_run: bool = False,
        symbol_override: Optional[str] = None,
        lot_override: Optional[float] = None,
        mode_override: Optional[str] = None,
    ):
        # 1. Muat Environment Credentials
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info(f"Loaded credentials from {env_path}")

        # 2. Muat Konfigurasi JSON
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Overrides
        self.dry_run = dry_run
        self.symbol = symbol_override or self.config["trading"]["symbol"]
        self.timeframe = self.config["trading"]["timeframe"]
        self.lot_size = lot_override if lot_override is not None else float(self.config["trading"]["lot_size"])
        self.magic_number = int(self.config["trading"]["magic_number"])
        self.order_comment = self.config["trading"]["order_comment"]
        self.max_spread = int(self.config["trading"].get("max_spread_points", 50))
        self.slippage = int(self.config["trading"].get("slippage_points", 20))

        # Risk Management Settings
        self.sl_atr_mult = float(self.config["risk_management"]["sl_atr_multiplier"])
        self.tp_atr_mult = float(self.config["risk_management"]["tp_atr_multiplier"])
        self.trail_be_atr = float(self.config["risk_management"]["trailing_breakeven_atr"])
        self.trail_lock_atr = float(self.config["risk_management"]["trailing_lock_profit_atr"])
        self.max_dd_pct = float(self.config["risk_management"]["max_account_drawdown_pct"])
        self.max_daily_loss = float(self.config["risk_management"]["max_daily_loss_usd"])

        # Agent Settings
        self.model_path = self.config["agent"]["model_path"]
        self.min_warmup = int(self.config["agent"]["min_candles_warmup"])
        self.zscore_win = int(self.config["agent"]["rolling_zscore_window"])
        self.init_bal_ref = float(self.config["agent"]["initial_balance_ref"])

        # Execution Settings
        self.intrabar_check_sec = int(self.config["execution"].get("intrabar_trailing_check_sec", 3))
        self.reconnect_sec = int(self.config["execution"].get("reconnect_interval_sec", 5))

        # Credentials
        account = os.getenv("MT5_ACCOUNT")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        path = os.getenv("MT5_PATH")
        self.mode = (mode_override or os.getenv("MT5_MODE", "DEMO")).upper()

        # Inisialisasi Sub-Komponen
        self.connector = MT5Connector(
            account=account,
            password=password,
            server=server,
            path=path,
            magic_number=self.magic_number,
        )
        self.feature_engine = MT5FeatureEngine(
            min_candles=self.min_warmup,
            zscore_window=self.zscore_win,
            initial_balance_ref=self.init_bal_ref,
        )
        self.notifier = MT5Notifier()

        # State Internal
        self.model = None
        self.is_recurrent = True
        self.lstm_states = None
        self.episode_start = np.ones((1,), dtype=bool)

        self.last_bar_time: Optional[datetime] = None
        self.active_position_ticket: Optional[int] = None
        self.active_position_dir: int = 0         # 1 (BUY), -1 (SELL), 0 (FLAT)
        self.active_position_entry: float = 0.0
        self.active_position_atr: float = 0.0
        self.active_position_sl: float = 0.0
        self.active_position_tp: float = 0.0
        self.active_hold_bars: int = 0
        self.active_trailing_stage: int = 0      # 0=Initial, 1=Breakeven, 2=Locked

        # Tracking Kinerja Harian & DD
        self.start_balance: float = 0.0
        self.peak_equity: float = 0.0
        self.circuit_breaker_active = False

    def load_ai_model(self):
        """Memuat model RL terbaik dari checkpoint."""
        logger.info(f"Memuat model AI RL dari: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file tidak ditemukan di {self.model_path}")

        try:
            self.model = RecurrentPPO.load(self.model_path)
            self.is_recurrent = True
            logger.info("Sukses memuat RecurrentPPO (MlpLstmPolicy) model.")
        except Exception:
            self.model = PPO.load(self.model_path)
            self.is_recurrent = False
            logger.info("Sukses memuat standard PPO model.")

    def initialize(self) -> bool:
        """Koneksi ke MT5 dan persiapan awal."""
        logger.info("=" * 70)
        logger.info("MEMULAI MT5 AI LIVE TRADING AGENT")
        logger.info(f"Mode Operasi  : {'[DRY RUN - SIMULASI]' if self.dry_run else f'[{self.mode} TRADING]'}")
        logger.info(f"Simbol Target : {self.symbol} | Timeframe: {self.timeframe} | Lot: {self.lot_size}")
        logger.info(f"Magic Number  : {self.magic_number} | Model: {self.model_path}")
        logger.info("=" * 70)

        # 1. Sambungkan ke MT5
        if not self.connector.connect():
            logger.error("Gagal terhubung ke terminal MetaTrader 5!")
            return False

        # 2. Verifikasi Akun
        acc = self.connector.get_account_info()
        if not acc:
            logger.error("Gagal membaca informasi akun MT5.")
            return False

        self.start_balance = acc["balance"]
        self.peak_equity = acc["equity"]

        logger.info(f"Akun Login   : #{acc['login']} ({acc['name']}) @ {acc['server']}")
        logger.info(f"Saldo Awal   : ${acc['balance']:.2f} {acc['currency']} | Leverage: 1:{acc['leverage']}")
        logger.info(f"Equity Awal  : ${acc['equity']:.2f} | Floating: ${acc['floating_profit']:+.2f}")

        # 3. Verifikasi Simbol Broker
        sym_info = self.connector.get_symbol_info(self.symbol)
        if not sym_info:
            logger.error(f"Simbol '{self.symbol}' tidak tersedia pada broker MT5 ini.")
            return False

        logger.info(
            f"Simbol Broker: {sym_info['symbol']} | Digits: {sym_info['digits']} | "
            f"Spread: {sym_info['spread']} pts (${sym_info['spread_usd']:.2f}) | "
            f"Min Lot: {sym_info['volume_min']} | Max Lot: {sym_info['volume_max']}"
        )

        # 4. Sinkronisasi Posisi Terbuka Eksisting (jika ada)
        self._sync_existing_positions(sym_info['symbol'])

        # 5. Muat AI Model
        self.load_ai_model()

        # 6. Kirim Notifikasi Startup ke Telegram
        mode_label = "[DRY RUN - SIMULASI]" if self.dry_run else f"[{self.mode} TRADING]"
        self.notifier.notify_startup(
            mode=mode_label,
            symbol=sym_info['symbol'],
            timeframe=self.timeframe,
            lot=self.lot_size,
            balance=acc["balance"],
            equity=acc["equity"],
            server=acc["server"],
            login=acc["login"],
            magic=self.magic_number,
        )

        return True

    def _sync_existing_positions(self, resolved_symbol: str):
        """Mendeteksi apakah ada posisi aktif milik bot sebelumnya."""
        positions = self.connector.get_open_positions(symbol=resolved_symbol, magic=self.magic_number)
        if positions:
            pos = positions[0]
            self.active_position_ticket = pos["ticket"]
            self.active_position_dir = pos["direction"]
            self.active_position_entry = pos["price_open"]
            self.active_position_sl = pos["sl"]
            self.active_position_tp = pos["tp"]
            self.active_hold_bars = 1
            logger.info(
                f"Ditemukan posisi aktif #{pos['ticket']}: {pos['type']} {pos['volume']} lot @ {pos['price_open']} "
                f"(Floating: ${pos['profit']:+.2f})"
            )
        else:
            self.active_position_ticket = None
            self.active_position_dir = 0
            self.active_hold_bars = 0
            logger.info("Tidak ada posisi aktif milik bot saat ini (FLAT).")

    def check_circuit_breaker(self, acc_info: Dict[str, Any]) -> bool:
        """
        Memeriksa apakah batas kerugian maksimum (Circuit Breaker) tercapai.
        """
        equity = acc_info["equity"]
        if equity > self.peak_equity:
            self.peak_equity = equity

        # 1. Hitung Max Account Drawdown %
        current_dd = (self.peak_equity - equity) / (self.peak_equity + 1e-8) * 100.0
        if current_dd >= self.max_dd_pct:
            reason = f"Max Drawdown Terlampaui ({current_dd:.2f}% >= {self.max_dd_pct:.2f}%)"
            logger.critical(f"[CIRCUIT BREAKER] {reason}!")
            self._emergency_stop(reason, f"Peak: ${self.peak_equity:.2f}, Equity: ${equity:.2f}")
            return True

        # 2. Hitung Daily Loss USD
        daily_pnl = equity - self.start_balance
        if daily_pnl < 0 and abs(daily_pnl) >= self.max_daily_loss:
            reason = f"Max Daily Loss Terlampaui (${daily_pnl:.2f} <= -${self.max_daily_loss:.2f})"
            logger.critical(f"[CIRCUIT BREAKER] {reason}!")
            self._emergency_stop(reason, f"Start: ${self.start_balance:.2f}, Equity: ${equity:.2f}")
            return True

        return False

    def _emergency_stop(self, reason: str, details: str):
        """Menutup seluruh posisi aktif dan menghentikan bot."""
        self.circuit_breaker_active = True
        self.notifier.notify_circuit_breaker(reason, details)

        if not self.dry_run and self.active_position_ticket:
            logger.warning(f"Menutup posisi #{self.active_position_ticket} demi keamanan...")
            self.connector.close_position(self.active_position_ticket, self.slippage)

        self.active_position_ticket = None
        self.active_position_dir = 0

    def process_bar_close(self, candles_df: pd.DataFrame, current_unrealized_pnl: float):
        """
        Dipanggil saat satu bar candle H1 selesai:
        1. Ekstraksi fitur 26 dimensi
        2. Evaluasi AI Agent
        3. Eksekusi Action (BUY / SELL / HOLD)
        """
        bar_time = candles_df['Date'].iloc[-1]
        logger.info(f"\n{'='*70}\n[BAR CLOSE EVENT @ {bar_time}]")

        # 1. Bangun Vektor Observasi 26-dim
        try:
            obs, last_close, last_atr, meta = self.feature_engine.build_observation(
                candles_df=candles_df,
                current_position=self.active_position_dir,
                unrealized_pnl_usd=current_unrealized_pnl,
                hold_bars=self.active_hold_bars,
            )
        except Exception as e:
            logger.error(f"Gagal menghitung observasi: {e}")
            return

        # 2. Evaluasi Model AI (RecurrentPPO / PPO)
        if self.is_recurrent:
            action, self.lstm_states = self.model.predict(
                obs, state=self.lstm_states, episode_start=self.episode_start, deterministic=True
            )
            self.episode_start = np.array([False])
        else:
            action, _ = self.model.predict(obs, deterministic=True)

        action_int = int(action)
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_str = action_names.get(action_int, "UNKNOWN")

        logger.info(f"  Harga Close : {last_close:.2f} | Raw ATR(14): {last_atr:.2f}")
        logger.info(f"  Posisi Aktif: {self.active_position_dir} (Hold: {self.active_hold_bars} bars, PnL: ${current_unrealized_pnl:+.2f})")
        logger.info(f"  >> KEPUTUSAN AGENT: [{action_int}] {action_str} <<")

        # 3. Cek Spread Filter
        sym_info = self.connector.get_symbol_info(self.symbol)
        if sym_info and sym_info['spread'] > self.max_spread:
            warn_msg = f"Spread saat ini ({sym_info['spread']} pts) melebihi batas aman ({self.max_spread} pts). Menunda open order."
            logger.warning(warn_msg)
            self.notifier.notify_warning("Spread Melebar", warn_msg)
            return

        # 4. Eksekusi Berdasarkan Action
        self._execute_agent_decision(action_int, last_close, last_atr, sym_info)

    def _execute_agent_decision(
        self, action: int, current_price: float, current_atr: float, sym_info: Optional[Dict[str, Any]]
    ):
        """Mengeksekusi logika BUY / SELL / HOLD dengan Dynamic SL/TP."""
        digits = sym_info['digits'] if sym_info else 2
        sl_dist = self.sl_atr_mult * max(current_atr, 0.5)
        tp_dist = self.tp_atr_mult * max(current_atr, 0.5)

        # ---------------------------------------------------------------------
        # ACTION 0: HOLD
        # ---------------------------------------------------------------------
        if action == 0:
            if self.active_position_dir != 0:
                self.active_hold_bars += 1
                logger.info(f"  Agent memilih HOLD (Melanjutkan posisi {self.active_hold_bars} bar).")
            else:
                self.active_hold_bars = 0
                logger.info("  Agent memilih HOLD (Tetap Flat/Tidak Ada Posisi).")

        # ---------------------------------------------------------------------
        # ACTION 1: BUY
        # ---------------------------------------------------------------------
        elif action == 1:
            if self.active_position_dir == 1:
                # Sudah dalam posisi BUY -> Lanjutkan HOLD
                self.active_hold_bars += 1
                logger.info(f"  Sudah dalam posisi BUY -> HOLD (Bar ke-{self.active_hold_bars}).")
            elif self.active_position_dir == -1:
                # Flip SHORT -> BUY: Tutup SHORT lalu buka BUY
                logger.info("  [FLIP] Menutup posisi SELL eksisting sebelum membuka BUY...")
                if not self.dry_run and self.active_position_ticket:
                    self._close_active_position(reason="FLIP (Sinyal AI Berbalik ke BUY)")
                self._open_new_position("BUY", current_price, sl_dist, tp_dist, current_atr, digits)
            else:
                # FLAT -> Buka BUY baru
                self._open_new_position("BUY", current_price, sl_dist, tp_dist, current_atr, digits)

        # ---------------------------------------------------------------------
        # ACTION 2: SELL
        # ---------------------------------------------------------------------
        elif action == 2:
            if self.active_position_dir == -1:
                # Sudah dalam posisi SELL -> Lanjutkan HOLD
                self.active_hold_bars += 1
                logger.info(f"  Sudah dalam posisi SELL -> HOLD (Bar ke-{self.active_hold_bars}).")
            elif self.active_position_dir == 1:
                # Flip LONG -> SELL: Tutup LONG lalu buka SELL
                logger.info("  [FLIP] Menutup posisi BUY eksisting sebelum membuka SELL...")
                if not self.dry_run and self.active_position_ticket:
                    self._close_active_position(reason="FLIP (Sinyal AI Berbalik ke SELL)")
                self._open_new_position("SELL", current_price, sl_dist, tp_dist, current_atr, digits)
            else:
                # FLAT -> Buka SELL baru
                self._open_new_position("SELL", current_price, sl_dist, tp_dist, current_atr, digits)

    def _close_active_position(self, reason: str = "SIGNAL"):
        """Helper untuk menutup posisi aktif dan mengirim notifikasi."""
        if not self.active_position_ticket:
            return

        ticket = self.active_position_ticket
        success, res = self.connector.close_position(ticket, self.slippage)
        acc = self.connector.get_account_info()

        if success:
            self.notifier.notify_trade_close(
                symbol=self.symbol,
                ticket=ticket,
                profit=0.0,
                reason=reason,
                balance=acc["balance"] if acc else None,
                equity=acc["equity"] if acc else None,
                open_price=self.active_position_entry,
            )
        else:
            err_msg = res.get("error", "Gagal menutup posisi")
            self.notifier.notify_error(f"Gagal Close Posisi #{ticket}", str(err_msg))

        self.active_position_ticket = None
        self.active_position_dir = 0
        self.active_hold_bars = 0
        self.active_trailing_stage = 0

    def _open_new_position(
        self, direction: str, price: float, sl_dist: float, tp_dist: float, atr: float, digits: int
    ):
        """Helper untuk membuka order baru di MT5."""
        if direction == "BUY":
            sl = price - sl_dist
            tp = price + tp_dist
            dir_code = 1
        else:
            sl = price + sl_dist
            tp = price - tp_dist
            dir_code = -1

        logger.info(
            f"  Membuka {direction} {self.lot_size} {self.symbol} @ ~{price:.{digits}f} "
            f"| SL: {sl:.{digits}f} (-{sl_dist:.2f}) | TP: {tp:.{digits}f} (+{tp_dist:.2f})"
        )

        if self.dry_run:
            logger.info("  [DRY RUN] Order disimulasikan (tidak dikirim ke broker).")
            self.active_position_ticket = 999999
            self.active_position_dir = dir_code
            self.active_position_entry = price
            self.active_position_atr = atr
            self.active_position_sl = sl
            self.active_position_tp = tp
            self.active_hold_bars = 1
            self.active_trailing_stage = 0

            self.notifier.notify_trade_open(
                symbol=self.symbol,
                direction=direction,
                lot=self.lot_size,
                entry=price,
                sl=sl,
                tp=tp,
                atr=atr,
                ticket=999999,
                reason="AI Decision (DRY RUN)",
            )
            return

        success, res = self.connector.open_order(
            symbol=self.symbol,
            direction=direction,
            lot=self.lot_size,
            sl_dist=sl_dist,
            tp_dist=tp_dist,
            comment=self.order_comment,
            slippage_points=self.slippage,
        )

        if success and res:
            self.active_position_ticket = res.get("ticket")
            self.active_position_dir = dir_code
            self.active_position_entry = res.get("price", price)
            self.active_position_atr = atr
            self.active_position_sl = res.get("sl", sl)
            self.active_position_tp = res.get("tp", tp)
            self.active_hold_bars = 1
            self.active_trailing_stage = 0

            self.notifier.notify_trade_open(
                symbol=self.symbol,
                direction=direction,
                lot=self.lot_size,
                entry=self.active_position_entry,
                sl=self.active_position_sl,
                tp=self.active_position_tp,
                atr=atr,
                ticket=self.active_position_ticket,
                reason="AI RecurrentPPO Decision",
            )
        else:
            err_detail = res.get("comment", res.get("error", "Order Ditolak"))
            logger.error(f"Gagal mengeksekusi order {direction}: {res}")
            self.notifier.notify_error(
                title=f"Order {direction} Ditolak Broker",
                message=f"Simbol: {self.symbol}, Lot: {self.lot_size}, Error: {err_detail}",
                action="Bot akan mencoba kembali pada candle berikutnya jika kondisi terpenuhi.",
            )

    def update_intrabar_trailing_stop(self):
        """
        Pengecekan cepat (setiap 3 detik) untuk Trailing Stop otomatis di MT5:
          1. Profit >= 1.0x ATR -> Geser SL ke Breakeven (Entry Price)
          2. Profit >= 2.0x ATR -> Kunci Profit: Geser SL ke Entry ± 1.0x ATR
        """
        if self.active_position_dir == 0 or self.active_position_ticket is None:
            return

        # Sinkronisasi status posisi saat ini dari MT5
        resolved_sym = self.connector.resolve_symbol(self.symbol)
        positions = self.connector.get_open_positions(symbol=resolved_sym, magic=self.magic_number)

        if not positions:
            if not self.dry_run:
                # Posisi telah tertutup (hit SL atau TP alami di broker)
                logger.info(f"[POSISI SELESAI] Tiket #{self.active_position_ticket} telah closed.")
                acc = self.connector.get_account_info()
                self.notifier.notify_trade_close(
                    symbol=self.symbol,
                    ticket=self.active_position_ticket,
                    profit=0.0,
                    reason="SL/TP Hit (Otomatis Broker)",
                    balance=acc["balance"] if acc else None,
                    equity=acc["equity"] if acc else None,
                    open_price=self.active_position_entry,
                )
                self.active_position_ticket = None
                self.active_position_dir = 0
                self.active_hold_bars = 0
                self.active_trailing_stage = 0
            return

        pos = positions[0]
        curr_price = pos["price_current"]
        entry_price = pos["price_open"]
        current_sl = pos["sl"]
        atr = max(self.active_position_atr, 0.5)

        new_sl = None
        stage_str = ""

        # Logika Trailing Stop BUY
        if pos["direction"] == 1:
            profit_dist = curr_price - entry_price
            if profit_dist >= self.trail_lock_atr * atr and self.active_trailing_stage < 2:
                # Lock +1x ATR
                target_sl = entry_price + (1.0 * atr)
                if target_sl > current_sl + 1e-4:
                    new_sl = target_sl
                    self.active_trailing_stage = 2
                    stage_str = f"Lock Profit (+1x ATR @ {target_sl:.2f})"
            elif profit_dist >= self.trail_be_atr * atr and self.active_trailing_stage < 1:
                # Breakeven
                target_sl = entry_price
                if target_sl > current_sl + 1e-4:
                    new_sl = target_sl
                    self.active_trailing_stage = 1
                    stage_str = f"Breakeven (SL -> Entry @ {target_sl:.2f})"

        # Logika Trailing Stop SELL
        elif pos["direction"] == -1:
            profit_dist = entry_price - curr_price
            if profit_dist >= self.trail_lock_atr * atr and self.active_trailing_stage < 2:
                # Lock +1x ATR
                target_sl = entry_price - (1.0 * atr)
                if current_sl <= 0 or target_sl < current_sl - 1e-4:
                    new_sl = target_sl
                    self.active_trailing_stage = 2
                    stage_str = f"Lock Profit (+1x ATR @ {target_sl:.2f})"
            elif profit_dist >= self.trail_be_atr * atr and self.active_trailing_stage < 1:
                # Breakeven
                target_sl = entry_price
                if current_sl <= 0 or target_sl < current_sl - 1e-4:
                    new_sl = target_sl
                    self.active_trailing_stage = 1
                    stage_str = f"Breakeven (SL -> Entry @ {target_sl:.2f})"

        if new_sl is not None and not self.dry_run:
            logger.info(f"[TRAILING STOP UPDATE] Posisi #{pos['ticket']} -> {stage_str}")
            success, _ = self.connector.modify_position_sltp(pos["ticket"], sl=new_sl, tp=pos["tp"])
            if success:
                old_sl_val = self.active_position_sl
                self.active_position_sl = new_sl
                self.notifier.notify_trailing_update(
                    symbol=self.symbol,
                    ticket=pos["ticket"],
                    old_sl=old_sl_val,
                    new_sl=new_sl,
                    stage=stage_str,
                    current_price=curr_price,
                )

    def run(self):
        """Loop utama proses trading live."""
        if not self.initialize():
            logger.error("Gagal melakukan inisialisasi awal. Program berhenti.")
            return

        logger.info("\nAI Live Trading Agent siap & aktif! Menunggu bar H1...")
        last_heartbeat = time.time()
        was_disconnected = False

        try:
            while True:
                time.sleep(self.intrabar_check_sec)

                # 1. Cek Koneksi MT5
                if not self.connector.is_alive():
                    if not was_disconnected:
                        logger.warning("Koneksi MT5 terputus. Mencoba menghubungkan kembali...")
                        self.notifier.notify_warning("Koneksi MT5 Terputus", "Koneksi ke terminal terputus. Bot mencoba auto-reconnect...")
                        was_disconnected = True

                    if not self.connector.connect():
                        time.sleep(self.reconnect_sec)
                        continue
                    else:
                        was_disconnected = False
                        self.notifier.notify_warning("Koneksi MT5 Pulih", "Berhasil terhubung kembali ke MetaTrader 5.")

                # 2. Cek Status Akun & Circuit Breaker
                acc_info = self.connector.get_account_info()
                if acc_info:
                    if self.check_circuit_breaker(acc_info):
                        logger.critical("Circuit Breaker aktif! Menghentikan proses live trading.")
                        break

                # 3. Jalankan Fast Intrabar Trailing Stop Monitor
                self.update_intrabar_trailing_stop()

                # 4. Pengecekan Bar Selesai (H1 Candle Close)
                # Ambil 500 candle terbaru untuk evaluasi
                candles_df = self.connector.fetch_candles(
                    symbol=self.symbol, timeframe=self.timeframe, count=self.min_warmup + 10
                )
                if candles_df is None or len(candles_df) == 0:
                    continue

                # Candle terakhir pada index -1 adalah candle yang SEDANG BERJALAN (belum closed)
                # Candle pada index -2 adalah candle yang BARU SAJA CLOSED (definitif)
                closed_candles_df = candles_df.iloc[:-1].copy().reset_index(drop=True)
                latest_closed_time = closed_candles_df['Date'].iloc[-1]

                if self.last_bar_time is None:
                    # Inisialisasi awal
                    self.last_bar_time = latest_closed_time
                    logger.info(f"Candle closed terakhir yang terdeteksi: {self.last_bar_time}")
                    # Jalankan evaluasi pertama kali saat startup
                    floating = acc_info["floating_profit"] if acc_info else 0.0
                    self.process_bar_close(closed_candles_df, floating)

                elif latest_closed_time > self.last_bar_time:
                    # Candle H1 baru telah selesai terbentuk!
                    self.last_bar_time = latest_closed_time
                    floating = acc_info["floating_profit"] if acc_info else 0.0
                    self.process_bar_close(closed_candles_df, floating)

                # 5. Heartbeat & Ringkasan Terminal setiap interval waktu
                now = time.time()
                if now - last_heartbeat >= self.config["execution"].get("heartbeat_interval_sec", 60):
                    last_heartbeat = now
                    if acc_info:
                        pos_str = (
                            f"{'BUY' if self.active_position_dir == 1 else 'SELL'} #{self.active_position_ticket}"
                            if self.active_position_dir != 0 else "FLAT"
                        )
                        logger.info(
                            f"[HEARTBEAT] Bal: ${acc_info['balance']:.2f} | Eq: ${acc_info['equity']:.2f} | "
                            f"Float: ${acc_info['floating_profit']:+.2f} | Posisi: {pos_str} (Hold: {self.active_hold_bars}b) | "
                            f"Last Bar: {self.last_bar_time}"
                        )

        except KeyboardInterrupt:
            logger.info("\nProgram dihentikan oleh pengguna (Ctrl+C).")
            self.notifier.notify_shutdown("Dihentikan secara manual oleh pengguna (Ctrl+C).")
        except Exception as e:
            logger.critical(f"Terjadi error fatal tak terduga: {e}", exc_info=True)
            self.notifier.notify_error("Fatal Bot Exception", str(e), action="Bot berhenti beroperasi.")
        finally:
            self.connector.disconnect()
            logger.info("MT5 Live Trader selesai.")


def main():
    parser = argparse.ArgumentParser(description="MT5 AI RL Live Trading Agent")
    parser.add_argument("--config", default="mt5_config.json", help="Path ke file mt5_config.json")
    parser.add_argument("--env", default=".env", help="Path ke file .env")
    parser.add_argument("--dry-run", action="store_true", help="Jalankan simulasi tanpa mengeksekusi order riil ke MT5")
    parser.add_argument("--symbol", default=None, help="Override simbol trading (e.g. XAUUSD, EURUSD)")
    parser.add_argument("--lot", type=float, default=None, help="Override ukuran lot (e.g. 0.01, 0.1)")
    parser.add_argument("--mode", default=None, help="Override mode (DEMO atau REAL)")

    args = parser.parse_args()

    trader = MT5LiveTrader(
        config_path=args.config,
        env_path=args.env,
        dry_run=args.dry_run,
        symbol_override=args.symbol,
        lot_override=args.lot,
        mode_override=args.mode,
    )
    trader.run()


if __name__ == "__main__":
    main()
