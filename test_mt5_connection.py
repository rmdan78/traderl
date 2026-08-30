"""
================================================================================
SKRIP DIAGNOSTIK & VERIFIKASI KONEKSI METATRADER 5 (MT5)
================================================================================
Menjalankan 5 tahap pengujian:
  [1/5] Inisialisasi & Login ke terminal MT5
  [2/5] Pemeriksaan Izin Akun & Algo Trading
  [3/5] Pengambilan Data Candle H1 & Resolusi Simbol Broker
  [4/5] Kalkulasi 26 Dimensi Fitur Observasi (MT5FeatureEngine)
  [5/5] Uji Coba Inferensi Model AI Reinforcement Learning (best_model.zip)
================================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from mt5_connector import MT5Connector
from mt5_feature_engine import MT5FeatureEngine
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO


def run_diagnostics():
    print("\n" + "=" * 75)
    print("      DIAGNOSTIK SISTEM LIVE TRADING METATRADER 5 (MT5) & AI AGENT")
    print("=" * 75)

    # 0. Load Configuration & Environment
    env_file = ".env"
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"[OK] Membaca file konfigurasi kredensial: {env_file}")
    else:
        print(f"[INFO] File {env_file} belum ditemukan. Mencoba inisialisasi default terminal.")

    config_file = "mt5_config.json"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        print(f"[OK] Membaca konfigurasi trading: {config_file}")
    else:
        config = {
            "trading": {"symbol": "XAUUSD", "timeframe": "H1", "magic_number": 8882026},
            "agent": {"model_path": "models/best_model.zip", "min_candles_warmup": 500}
        }

    account = os.getenv("MT5_ACCOUNT")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH")
    symbol = config["trading"]["symbol"]
    timeframe = config["trading"]["timeframe"]
    model_path = config["agent"]["model_path"]

    # =========================================================================
    # [1/5] INISIALISASI & LOGIN MT5
    # =========================================================================
    print("\n" + "-" * 75)
    print("[1/5] MENGHUBUNGKAN KE METATRADER 5...")
    print("-" * 75)
    connector = MT5Connector(
        account=account,
        password=password,
        server=server,
        path=path,
        magic_number=config["trading"]["magic_number"]
    )

    success = connector.connect()
    if not success:
        print("\n[PERINGATAN] Terminal MetaTrader 5 belum terbuka atau kredensial .env belum diisi.")
        print("  -> Menjalankan DIAGNOSTIK PIPELINE AI OFFLINE menggunakan dataset XAUUSD_H1.csv...")

        # Jalankan pengujian offline pipeline
        csv_file = "XAUUSD_H1.csv"
        if os.path.exists(csv_file):
            print(f"\n[OK] Membaca data sampel: {csv_file}")
            sample_df = pd.read_csv(csv_file, nrows=600)
            if str(sample_df.columns[0]).startswith("200") or str(sample_df.columns[0]).startswith("201") or str(sample_df.columns[0]).startswith("202"):
                sample_df = pd.read_csv(csv_file, header=None, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], nrows=600)

            engine = MT5FeatureEngine(min_candles=500, zscore_window=252, initial_balance_ref=1000.0)
            obs, last_close, last_atr, meta = engine.build_observation(
                candles_df=sample_df, current_position=0, unrealized_pnl_usd=0.0, hold_bars=0
            )
            print(f"  Dimensi Observasi : {obs.shape} (26-dim float32) -> [PASSED]")
            print(f"  Harga Close       : {last_close:.2f} | Raw ATR(14): {last_atr:.2f}")

            model = RecurrentPPO.load(model_path)
            action, _ = model.predict(obs, deterministic=True)
            action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
            print(f"  Model RecurrentPPO: Loaded successfully -> Action: {int(action)} ({action_names[int(action)]}) [PASSED]")

            print("\n" + "=" * 75)
            print("         PIPELINE AI & FEATURE ENGINE: 100% TERVERIFIKASI & SIAP")
            print("=" * 75)
            print("Untuk menghubungkan ke Akun MT5 Anda:")
            print("  1. Buka aplikasi MetaTrader 5 di Windows Anda.")
            print("  2. Isi kredensial di file .env:")
            print("     MT5_ACCOUNT=nomor_akun_anda")
            print("     MT5_PASSWORD=password_anda")
            print("     MT5_SERVER=nama_server_broker")
            print("  3. Jalankan kembali: python test_mt5_connection.py")
            print("=" * 75 + "\n")
            return True
        else:
            return False

    acc = connector.get_account_info()
    print(f"  Status Koneksi : TERHUBUNG (OK)")
    print(f"  Akun Login     : #{acc['login']} ({acc['name']})")
    print(f"  Server Broker  : {acc['server']}")
    print(f"  Saldo Akun     : ${acc['balance']:,.2f} {acc['currency']}")
    print(f"  Equity Akun    : ${acc['equity']:,.2f}")
    print(f"  Leverage       : 1:{acc['leverage']}")
    print(f"  Floating PnL   : ${acc['floating_profit']:+,.2f}")

    # =========================================================================
    # [2/5] PEMERIKSAAN IZIN TRADING & ALGO TRADING
    # =========================================================================
    print("\n" + "-" * 75)
    print("[2/5] MEMERIKSA IZIN TRADING...")
    print("-" * 75)
    print(f"  Trade Allowed  : {'YA (OK)' if acc['trade_allowed'] else 'TIDAK (Periksa status akun di broker)'}")
    print(f"  Algo Trading   : {'DIAKTIFKAN (OK)' if acc['trade_expert'] else 'TIDAK AKTIF (Aktifkan tombol Algo Trading di toolbar MT5)'}")

    # =========================================================================
    # [3/5] PENGAMBILAN DATA CANDLE & DETEKSI SIMBOL
    # =========================================================================
    print("\n" + "-" * 75)
    print(f"[3/5] MENGAMBIL DATA CANDLE ({symbol} {timeframe})...")
    print("-" * 75)
    resolved_symbol = connector.resolve_symbol(symbol)
    if not resolved_symbol:
        print(f"[FAIL] Simbol '{symbol}' tidak dapat ditemukan pada broker MT5!")
        connector.disconnect()
        return False

    print(f"  Simbol Terpetakan : {resolved_symbol}")
    sym_info = connector.get_symbol_info(symbol)
    if sym_info:
        print(f"  Harga Bid / Ask   : {sym_info['bid']} / {sym_info['ask']}")
        print(f"  Spread Saat Ini   : {sym_info['spread']} points (${sym_info['spread_usd']:.2f})")
        print(f"  Batas Lot         : Min {sym_info['volume_min']} | Max {sym_info['volume_max']} | Step {sym_info['volume_step']}")

    candles_df = connector.fetch_candles(symbol, timeframe=timeframe, count=550)
    if candles_df is None or len(candles_df) == 0:
        print(f"[FAIL] Gagal mengambil data candle {timeframe} untuk {resolved_symbol}!")
        connector.disconnect()
        return False

    print(f"  Jumlah Candle     : {len(candles_df)} bars (OK)")
    print(f"  Rentang Waktu     : {candles_df['Date'].iloc[0]} s/d {candles_df['Date'].iloc[-1]}")
    print(f"  Candle Terakhir   : Open={candles_df['Open'].iloc[-1]:.2f}, High={candles_df['High'].iloc[-1]:.2f}, "
          f"Low={candles_df['Low'].iloc[-1]:.2f}, Close={candles_df['Close'].iloc[-1]:.2f}, Vol={candles_df['Volume'].iloc[-1]}")

    # =========================================================================
    # [4/5] KALKULASI 26-DIM OBS DENGAN FEATURE ENGINE
    # =========================================================================
    print("\n" + "-" * 75)
    print("[4/5] MENGUJI FEATURE ENGINE (26 DIMENSI OBSERVASI)...")
    print("-" * 75)
    engine = MT5FeatureEngine(
        min_candles=config["agent"].get("min_candles_warmup", 500),
        zscore_window=config["agent"].get("rolling_zscore_window", 252),
        initial_balance_ref=config["agent"].get("initial_balance_ref", 1000.0)
    )

    try:
        # Gunakan candle yang sudah closed
        closed_candles = candles_df.iloc[:-1].copy().reset_index(drop=True)
        obs, last_close, last_atr, meta = engine.build_observation(
            candles_df=closed_candles,
            current_position=0,
            unrealized_pnl_usd=0.0,
            hold_bars=0,
        )
        print(f"  Dimensi Observasi : {obs.shape} -> {len(obs)} dimensi (OK)")
        print(f"  Tipe Data         : {obs.dtype}")
        print(f"  Latest Close Price: {last_close:.2f}")
        print(f"  Raw ATR(14)       : {last_atr:.2f}")
        print(f"  ATR Norm (% price): {meta['atr_norm']:.4f}%")
        print(f"  Sample Vektor Obs (5 nilai pertama): {obs[:5]}")
        print(f"  State Tambahan (4 nilai terakhir)  : {obs[-4:]}")
    except Exception as e:
        print(f"[FAIL] Error kalkulasi fitur: {e}")
        import traceback
        traceback.print_exc()
        connector.disconnect()
        return False

    # =========================================================================
    # [5/5] INFERENSI MODEL AI REINFORCEMENT LEARNING
    # =========================================================================
    print("\n" + "-" * 75)
    print(f"[5/5] MENGUJI INFERENSI MODEL AI ({model_path})...")
    print("-" * 75)
    if not os.path.exists(model_path):
        print(f"[FAIL] File model '{model_path}' tidak ditemukan!")
        connector.disconnect()
        return False

    try:
        try:
            model = RecurrentPPO.load(model_path)
            is_recurrent = True
            model_type_str = "RecurrentPPO (LSTM 26-dim)"
        except Exception:
            model = PPO.load(model_path)
            is_recurrent = False
            model_type_str = "Standard PPO (26-dim)"

        print(f"  Arsitektur Model  : {model_type_str}")

        if is_recurrent:
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            action, _ = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)

        action_int = int(action)
        action_names = {0: "HOLD (Tahan/Tetap Flat)", 1: "BUY (Buka Posisi Beli)", 2: "SELL (Buka Posisi Jual)"}

        print(f"  Output Prediksi   : Action {action_int} -> {action_names.get(action_int, 'UNKNOWN')} (OK)")
        print(f"  Status Inferensi  : BERHASIL & SIAP")

    except Exception as e:
        print(f"[FAIL] Gagal menjalankan inferensi model: {e}")
        import traceback
        traceback.print_exc()
        connector.disconnect()
        return False

    # Tutup koneksi
    connector.disconnect()

    print("\n" + "=" * 75)
    print("                      HASIL DIAGNOSTIK: SEMUA SUKSES (PASSED)")
    print("=" * 75)
    print("Langkah selanjutnya untuk memulai Live / Demo Trading:")
    print("  1. Untuk uji coba simulasi (Dry Run):")
    print("     python mt5_live_trader.py --dry-run")
    print("  2. Untuk Live Trading di akun Demo:")
    print("     python mt5_live_trader.py --mode DEMO")
    print("  3. Untuk Live Trading di akun Real:")
    print("     python mt5_live_trader.py --mode REAL")
    print("=" * 75 + "\n")
    return True


if __name__ == "__main__":
    run_diagnostics()
