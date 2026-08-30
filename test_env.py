"""
================================================================================
SCRIPT PENGUJIAN ENVIRONMENT v2 — RISK-MANAGED TRADING
================================================================================
Menguji:
1. Kepatuhan Gymnasium (check_env)
2. Simulasi trading deterministik dengan 3 aksi (HOLD, BUY, SELL)
3. Verifikasi SL/TP system
4. Verifikasi reward system
5. Verifikasi trade logging
================================================================================
"""

import numpy as np
from gymnasium.utils.env_checker import check_env
from indicators import load_dataset
from trading_env import IndicatorTradingEnv, Action


def run_compliance_check():
    print("=" * 70)
    print("1. MEMERIKSA KEPATUHAN GYMNASIUM (check_env)...")
    print("=" * 70)

    features, prices, raw_atr = load_dataset("XAUUSD_H1.csv")

    env = IndicatorTradingEnv(
        features_df=features,
        prices_df=prices,
        raw_atr=raw_atr,
        initial_balance=1000.0,
        lot_size=0.1,
        spread_cost=0.50,
        max_steps_per_episode=200,
        random_start=False,
    )

    try:
        check_env(env.unwrapped)
        print(">> SUKSES: IndicatorTradingEnv v2 100% kompatibel dengan standar Gymnasium!\n")
    except Exception as e:
        print(f">> GAGAL pada check_env: {e}\n")


def run_deterministic_trade_simulation():
    print("=" * 70)
    print("2. SIMULASI TRADING DETERMINISTIK (3 AKSI + SL/TP)")
    print("=" * 70)

    features, prices, raw_atr = load_dataset("XAUUSD_H1.csv")

    env = IndicatorTradingEnv(
        features_df=features,
        prices_df=prices,
        raw_atr=raw_atr,
        initial_balance=1000.0,
        lot_size=0.1,
        spread_cost=0.50,
        max_steps_per_episode=500,
        random_start=False,
    )

    obs, info = env.reset()
    print(f"Start Balance    : ${info['balance']:.2f}")
    print(f"Observation Dim  : {obs.shape}")
    print(f"Num Indicators   : {env.num_indicators}")
    print(f"Total Obs Dim    : {env.num_obs}")
    print(f"Action Space     : {env.action_space}\n")

    action_names = {Action.HOLD: "HOLD", Action.BUY: "BUY", Action.SELL: "SELL"}
    test_actions = [
        Action.HOLD,   # Step 1: HOLD saat flat
        Action.BUY,    # Step 2: Buka LONG -> SL/TP set
        Action.HOLD,   # Step 3: Hold LONG
        Action.HOLD,   # Step 4: Hold LONG
        Action.HOLD,   # Step 5: Hold LONG
        Action.SELL,   # Step 6: Flip LONG -> SHORT (close LONG, open SHORT)
        Action.HOLD,   # Step 7: Hold SHORT
        Action.HOLD,   # Step 8: Hold SHORT
        Action.BUY,    # Step 9: Flip SHORT -> LONG
        Action.HOLD,   # Step 10: Hold LONG
    ]

    print(f"{'Step':>4} | {'Aksi':<5} | {'Reward':>8} | {'Posisi':<6} | {'Hold':>4} | {'Balance':>10} | {'Trades':>6} | {'PnL':>8} | {'SL':>10} | {'TP':>10}")
    print("-" * 100)

    for step_num, act in enumerate(test_actions, start=1):
        obs, reward, terminated, truncated, info = env.step(act)
        print(
            f"{step_num:4d} | {action_names[act]:<5} | {reward:+8.5f} | "
            f"{info['position']:<6} | {info['hold_duration']:4d} | "
            f"${info['balance']:9.2f} | {info['trade_count']:6d} | "
            f"${info['total_pnl']:+7.2f} | "
            f"${info['stop_loss']:9.2f} | ${info['take_profit']:9.2f}"
        )
        if terminated:
            print("  >> TERMINATED (drawdown > 20%)")
            break

    print(f"\n--- Statistik Akhir ---")
    print(f"Total Trades  : {info['trade_count']}")
    print(f"Win Rate      : {info['win_rate_pct']:.1f}%")
    print(f"Total PnL     : ${info['total_pnl']:+.2f}")
    print(f"Balance       : ${info['balance']:.2f}")
    print(f"ROI           : {info['roi_pct']:+.2f}%")


def run_reward_verification():
    print("\n" + "=" * 70)
    print("3. VERIFIKASI REWARD & RISK MANAGEMENT SYSTEM")
    print("=" * 70)

    features, prices, raw_atr = load_dataset("XAUUSD_H1.csv")

    env = IndicatorTradingEnv(
        features_df=features,
        prices_df=prices,
        raw_atr=raw_atr,
        initial_balance=1000.0,
        lot_size=0.1,
        spread_cost=0.50,
        max_steps_per_episode=200,
        random_start=False,
    )

    obs, _ = env.reset()

    # 1. Test flat hold (no penalty if < 48 bars)
    _, r_flat, _, _, _ = env.step(Action.HOLD)
    print(f"HOLD saat FLAT (1 bar)  -> reward: {r_flat:+.5f} (bebas penalti wajar)")

    # 2. Test open position (NO open reward in v2)
    _, r_open, _, _, info = env.step(Action.BUY)
    print(f"BUY (open LONG)         -> reward: {r_open:+.5f} (no open reward, step PnL only)")
    print(f"  SL: ${info['stop_loss']:.2f}  TP: ${info['take_profit']:.2f}")

    # Verify SL/TP are set
    assert info['stop_loss'] > 0, "Stop Loss should be set after BUY"
    assert info['take_profit'] > 0, "Take Profit should be set after BUY"
    assert info['take_profit'] > info['stop_loss'], "TP should be above SL for LONG"

    # 3. Test flip
    _, r_flip, _, _, info = env.step(Action.SELL)
    print(f"SELL (flip LONG->SHORT) -> reward: {r_flip:+.5f} (close reward + new position)")
    print(f"  SL: ${info['stop_loss']:.2f}  TP: ${info['take_profit']:.2f}")

    # For SHORT: SL should be above entry, TP below
    assert info['stop_loss'] > info['take_profit'], "SL should be above TP for SHORT"

    print(f"\n>> Risk Management Constants:")
    print(f"  SL ATR Multiplier : {env.SL_ATR_MULT}x")
    print(f"  TP ATR Multiplier : {env.TP_ATR_MULT}x")
    print(f"  Trailing BE ATR   : {env.TRAIL_BREAKEVEN_ATR}x")
    print(f"  Trailing Lock ATR : {env.TRAIL_LOCK_ATR}x")
    print(f"  Max Drawdown      : {env.MAX_DRAWDOWN_PCT}%")
    print(f"  Loss Multiplier   : {env.LOSS_MULTIPLIER}x")
    print(">> RISK MANAGEMENT SYSTEM VERIFIED! [OK]\n")


def run_trade_log_check():
    print("=" * 70)
    print("4. VERIFIKASI TRADE LOG")
    print("=" * 70)

    features, prices, raw_atr = load_dataset("XAUUSD_H1.csv")

    env = IndicatorTradingEnv(
        features_df=features,
        prices_df=prices,
        raw_atr=raw_atr,
        initial_balance=1000.0,
        lot_size=0.1,
        spread_cost=0.50,
        max_steps_per_episode=200,
        random_start=False,
        trade_log_path="test_trade_log.csv",
    )

    obs, _ = env.reset()

    actions = [
        Action.BUY, Action.HOLD, Action.HOLD,
        Action.SELL,  # Flip LONG -> SHORT
        Action.HOLD, Action.HOLD,
        Action.BUY,   # Flip SHORT -> LONG
    ]
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        if term:
            break

    trade_log = env.get_trade_log_df()
    print(f"\nTrade Log ({len(trade_log)} trades):")
    if len(trade_log) > 0:
        print(trade_log.to_string(index=False))
        # Verify new columns
        assert 'exit_reason' in trade_log.columns, "exit_reason column missing"
        assert 'stop_loss' in trade_log.columns, "stop_loss column missing"
        assert 'take_profit' in trade_log.columns, "take_profit column missing"
        print(f"\n>> Trade log berisi {len(trade_log)} trades dengan SL/TP info [OK]")
    else:
        print("  (no trades yet)")


def run_long_simulation():
    """Run a longer simulation to see SL/TP hits."""
    print("\n" + "=" * 70)
    print("5. SIMULASI PANJANG — OBSERVASI SL/TP HITS")
    print("=" * 70)

    features, prices, raw_atr = load_dataset("XAUUSD_H1.csv")

    env = IndicatorTradingEnv(
        features_df=features,
        prices_df=prices,
        raw_atr=raw_atr,
        initial_balance=1000.0,
        lot_size=0.1,
        spread_cost=0.50,
        max_steps_per_episode=2000,
        random_start=False,
    )

    obs, _ = env.reset()

    # Random-ish actions to trigger SL/TP
    np.random.seed(42)
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated) and step < 2000:
        # Simple strategy: alternate buy/sell every ~20 bars
        if step % 20 == 0:
            action = np.random.choice([Action.BUY, Action.SELL])
        else:
            action = Action.HOLD
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

    trade_log = env.get_trade_log_df()
    if len(trade_log) > 0:
        print(f"\nTotal trades: {len(trade_log)}")
        print(f"Exit reasons:")
        print(trade_log['exit_reason'].value_counts().to_string())
        print(f"\nSample trades:")
        print(trade_log.head(10).to_string(index=False))
        print(f"\nWin Rate: {info['win_rate_pct']:.1f}%")
        print(f"Total PnL: ${info['total_pnl']:+.2f}")
        print(f"Balance: ${info['balance']:.2f}")
    else:
        print("  No trades generated")

    if terminated:
        print(f"\n>> Episode terminated (drawdown protection triggered)")


if __name__ == "__main__":
    run_compliance_check()
    run_deterministic_trade_simulation()
    run_reward_verification()
    run_trade_log_check()
    run_long_simulation()
