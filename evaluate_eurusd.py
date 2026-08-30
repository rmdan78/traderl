"""
================================================================================
SKRIP EVALUASI MODEL RPPO — CROSS-INSTRUMENT TEST (EURUSD H1)
================================================================================
Menguji model yang dilatih pada XAUUSD terhadap data EURUSD H1.
Ini adalah uji generalisasi sejati — data dan instrumen BERBEDA.

Perbedaan kunci EURUSD vs XAUUSD:
    - Contract size: 100,000 (forex) vs 100 (gold)
    - Spread cost: ~$1.50 per trade (1.5 pip) vs $0.50
    - Pip value (0.1 lot): $1 per pip
    - Volatility: jauh lebih rendah dari gold

Output: metrik performa, chart, dan trade log CSV
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from indicators import load_dataset
from trading_env import IndicatorTradingEnv, Action


# =============================================================================
# KONFIGURASI EURUSD
# =============================================================================
EURUSD_CONFIG = {
    'csv_path': 'EURUSD_H1.csv',
    'contract_size': 100_000,       # Forex standard: 100,000 units per lot
    'lot_size': 0.1,                # Mini lot
    'spread_cost': 1.50,            # ~1.5 pip spread (konservatif)
    'initial_balance': 1000.0,
}
# Pip value untuk 0.1 lot EURUSD = 0.0001 * 100,000 * 0.1 = $1.00 per pip


def load_model(model_path: str = None):
    """Load model RPPO atau PPO."""
    if model_path is None or not os.path.exists(model_path):
        candidates = [
            "models/best_model.zip",
            "models/rppo_indicator_final.zip",
        ]
        found = False
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Model tidak ditemukan pada {candidates}")

    try:
        model = RecurrentPPO.load(model_path)
        is_recurrent = True
        print(f"  Loaded RecurrentPPO (LSTM) dari: {model_path}")
    except Exception:
        model = PPO.load(model_path)
        is_recurrent = False
        print(f"  Loaded PPO dari: {model_path}")

    return model, is_recurrent


def run_backtest(model, is_recurrent, test_features, test_prices, test_atr,
                 config, label="", log_path=None):
    """Jalankan backtest deterministik pada EURUSD."""
    env = IndicatorTradingEnv(
        features_df=test_features,
        prices_df=test_prices,
        raw_atr=test_atr,
        initial_balance=config['initial_balance'],
        lot_size=config['lot_size'],
        spread_cost=config['spread_cost'],
        contract_size=config['contract_size'],
        max_steps_per_episode=None,
        random_start=False,
        trade_log_path=log_path,
    )

    obs, info = env.reset()
    balance_history = [config['initial_balance']]
    positions = [0]
    prices_list = [test_prices['Close'].iloc[0]]
    rewards_list = []

    action_counts = {0: 0, 1: 0, 2: 0}
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        if is_recurrent:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            episode_start = np.array([terminated or truncated])
        else:
            action, _ = model.predict(obs, deterministic=True)

        action_int = int(action)
        action_counts[action_int] = action_counts.get(action_int, 0) + 1

        obs, reward, terminated, truncated, info = env.step(action_int)

        balance_history.append(info['balance'])
        positions.append(info['position_raw'])
        rewards_list.append(reward)
        if env.current_step < len(test_prices):
            prices_list.append(test_prices['Close'].iloc[env.current_step])

    # Export trade log
    trade_log = env.get_trade_log_df()
    if log_path and len(trade_log) > 0:
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
        trade_log.to_csv(log_path, index=False)
        print(f"  Trade log saved: {log_path} ({len(trade_log)} trades)")

    bal_arr = np.array(balance_history)
    price_arr = np.array(prices_list[:len(bal_arr)])

    # --- Metrik ---
    final_value = bal_arr[-1]
    initial_balance = config['initial_balance']
    cum_return = (final_value - initial_balance) / initial_balance * 100.0

    step_rets = np.diff(bal_arr) / (bal_arr[:-1] + 1e-8)
    ann = np.sqrt(252 * 24)
    sharpe = (np.mean(step_rets) / (np.std(step_rets) + 1e-8)) * ann

    neg_rets = step_rets[step_rets < 0]
    sortino = (np.mean(step_rets) / (np.std(neg_rets) + 1e-8)) * ann if len(neg_rets) > 0 else 0.0

    running_max = np.maximum.accumulate(bal_arr)
    dd_arr = (running_max - bal_arr) / (running_max + 1e-8) * 100.0
    max_dd = np.max(dd_arr)

    hours = len(test_features)
    years = max(hours / (252.0 * 24.0), 0.1)
    cagr = ((final_value / initial_balance) ** (1.0 / years) - 1.0) * 100.0
    calmar = cagr / (max_dd + 1e-8)

    bh_ret = (price_arr[-1] - price_arr[0]) / price_arr[0] * 100.0

    pos_arr = np.array(positions)
    long_pct  = np.mean(pos_arr > 0) * 100.0
    short_pct = np.mean(pos_arr < 0) * 100.0
    flat_pct  = np.mean(pos_arr == 0) * 100.0

    win_rate = info['win_rate_pct']

    # Profit Factor
    profit_factor = 0.0
    avg_hold = 0
    sl_count = 0
    tp_count = 0
    avg_win = 0.0
    avg_loss = 0.0
    if len(trade_log) > 0:
        if 'hold_bars' in trade_log.columns:
            avg_hold = trade_log['hold_bars'].mean()
        winners = trade_log[trade_log['pnl_usd'] > 0]
        losers = trade_log[trade_log['pnl_usd'] < 0]
        gross_profit = winners['pnl_usd'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl_usd'].sum()) if len(losers) > 0 else 0.001
        profit_factor = gross_profit / gross_loss
        avg_win = winners['pnl_usd'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl_usd'].mean() if len(losers) > 0 else 0
        if 'exit_reason' in trade_log.columns:
            sl_count = len(trade_log[trade_log['exit_reason'] == 'SL'])
            tp_count = len(trade_log[trade_log['exit_reason'] == 'TP'])

    metrics = dict(
        label=label,
        initial_balance=initial_balance,
        final_value=final_value,
        cum_return=cum_return,
        cagr=cagr,
        bh_ret=bh_ret,
        sharpe=sharpe,
        sortino=sortino,
        max_dd=max_dd,
        calmar=calmar,
        trade_count=info['trade_count'],
        winning_trades=info['winning_trades'],
        losing_trades=info['losing_trades'],
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        long_pct=long_pct,
        short_pct=short_pct,
        flat_pct=flat_pct,
        avg_hold_bars=avg_hold,
        total_pnl=info['total_pnl'],
        sl_count=sl_count,
        tp_count=tp_count,
        balance_history=balance_history,
        dd_arr=dd_arr,
        positions=positions,
        price_arr=price_arr,
        action_counts=action_counts,
        trade_log=trade_log,
    )
    return metrics


def print_metrics(m):
    """Print metrik performa."""
    print(f"\n  Modal Awal                : ${m['initial_balance']:.2f}")
    print(f"  Balance Akhir             : ${m['final_value']:.2f}")
    print(f"  Cumulative Return         : {m['cum_return']:+.2f}%")
    print(f"  CAGR                      : {m['cagr']:+.2f}%")
    print(f"  Buy & Hold Return         : {m['bh_ret']:+.2f}%")
    print(f"  Sharpe Ratio (Annualized) : {m['sharpe']:.2f}")
    print(f"  Sortino Ratio             : {m['sortino']:.2f}")
    print(f"  Max Drawdown              : {m['max_dd']:.2f}%")
    print(f"  Calmar Ratio              : {m['calmar']:.2f}")
    print(f"  Profit Factor             : {m['profit_factor']:.2f}")
    print(f"  Total Trades              : {m['trade_count']} (W:{m['winning_trades']} L:{m['losing_trades']})")
    print(f"  Win Rate                  : {m['win_rate']:.2f}%")
    print(f"  Avg Win                   : ${m['avg_win']:+.2f}")
    print(f"  Avg Loss                  : ${m['avg_loss']:+.2f}")
    print(f"  Total PnL                 : ${m['total_pnl']:+.2f}")
    print(f"  Avg Hold Duration         : {m['avg_hold_bars']:.1f} bars")
    print(f"  SL Exits / TP Exits       : {m['sl_count']} / {m['tp_count']}")
    print(f"  Distribusi Posisi         : LONG {m['long_pct']:.1f}%  SHORT {m['short_pct']:.1f}%  FLAT {m['flat_pct']:.1f}%")
    print(f"  Distribusi Aksi           : HOLD:{m['action_counts'].get(0,0)}  BUY:{m['action_counts'].get(1,0)}  SELL:{m['action_counts'].get(2,0)}")


def plot_results(metrics, output_path="evaluation_chart_eurusd.png"):
    """Plot 4-panel chart untuk EURUSD."""
    fig = plt.figure(figsize=(16, 11), facecolor='#0f172a')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    palette = {'portfolio': '#3b82f6', 'bh': '#f59e0b', 'dd': '#ef4444',
               'long': '#22c55e', 'flat': '#94a3b8', 'short': '#ef4444',
               'grid': '#1e293b', 'text': '#e2e8f0'}

    def style_ax(ax, title):
        ax.set_facecolor('#1e293b')
        ax.set_title(title, color=palette['text'], fontsize=11, fontweight='bold', pad=8)
        ax.tick_params(colors=palette['text'], labelsize=8)
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, color=palette['grid'], linewidth=0.5, alpha=0.8)

    m = metrics
    idx = range(len(m['balance_history']))

    # Panel A: Balance
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(idx, m['balance_history'], color=palette['portfolio'], lw=1.8,
              label=f"Agent ({m['cum_return']:+.1f}%)")
    ax_a.axhline(m['initial_balance'], color='#475569', ls=':', lw=1)
    ax_a.set_ylabel("Balance (USD)", color=palette['text'], fontsize=9)
    ax_a.legend(loc='upper left', fontsize=8, facecolor='#1e293b', labelcolor=palette['text'])
    style_ax(ax_a, "Panel A: Balance — EURUSD H1")

    # Panel B: Drawdown
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.fill_between(idx, -np.array(m['dd_arr']), 0, color=palette['dd'], alpha=0.5,
                      label=f"Max DD: {m['max_dd']:.2f}%")
    ax_b.set_ylabel("Drawdown (%)", color=palette['text'], fontsize=9)
    ax_b.legend(loc='lower left', fontsize=8, facecolor='#1e293b', labelcolor=palette['text'])
    style_ax(ax_b, "Panel B: Drawdown Profile")

    # Panel C: Strategy vs Buy & Hold
    ax_c = fig.add_subplot(gs[1, 0])
    strat_ret = (np.array(m['balance_history']) / m['initial_balance'] - 1.0) * 100.0
    bh_ret = (m['price_arr'] / m['price_arr'][0] - 1.0) * 100.0
    ax_c.plot(idx, strat_ret, color=palette['portfolio'], lw=1.8,
              label=f"Agent ({m['cum_return']:+.1f}%)")
    ax_c.plot(range(len(bh_ret)), bh_ret, color=palette['bh'], lw=1.2, ls=':',
              label=f"Buy & Hold ({m['bh_ret']:+.1f}%)")
    ax_c.axhline(0, color='#475569', ls='-', lw=0.5)
    ax_c.set_xlabel("Steps (H1)", color=palette['text'], fontsize=9)
    ax_c.set_ylabel("Cumulative Return (%)", color=palette['text'], fontsize=9)
    ax_c.legend(loc='upper left', fontsize=8, facecolor='#1e293b', labelcolor=palette['text'])
    style_ax(ax_c, "Panel C: Strategy vs Buy & Hold")

    # Panel D: Action Distribution
    ax_d = fig.add_subplot(gs[1, 1])
    labels = ['HOLD', 'BUY', 'SELL']
    counts = [m['action_counts'].get(i, 0) for i in range(3)]
    colors = [palette['flat'], palette['long'], palette['short']]
    total = sum(counts) + 1e-8
    bars = ax_d.bar(labels, counts, color=colors, alpha=0.85, edgecolor='#0f172a', linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.005,
                  f"{cnt:,}\n({cnt/total*100:.1f}%)",
                  ha='center', va='bottom', color=palette['text'], fontsize=8, fontweight='bold')
    ax_d.set_ylabel("Number of Steps", color=palette['text'], fontsize=9)
    style_ax(ax_d, "Panel D: Action Distribution")

    fig.suptitle(
        "Cross-Instrument Test — XAUUSD Model on EURUSD H1",
        color=palette['text'], fontsize=13, fontweight='bold', y=0.98
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    print(f"\n>> Grafik disimpan: {output_path}")


def evaluate_eurusd(
    model_path: str = None,
    csv_path: str = None,
    output_chart: str = "evaluation_chart_eurusd.png"
):
    """Evaluasi model XAUUSD pada data EURUSD H1."""
    config = EURUSD_CONFIG.copy()
    if csv_path:
        config['csv_path'] = csv_path

    print("=" * 75)
    print("  CROSS-INSTRUMENT TEST: Model XAUUSD → Data EURUSD H1")
    print("=" * 75)

    print(f"\n  Konfigurasi EURUSD:")
    print(f"    CSV Path      : {config['csv_path']}")
    print(f"    Contract Size  : {config['contract_size']:,}")
    print(f"    Lot Size       : {config['lot_size']}")
    print(f"    Spread Cost    : ${config['spread_cost']:.2f} (~{config['spread_cost']:.1f} pip)")
    print(f"    Initial Balance: ${config['initial_balance']:.2f}")

    # Load data
    print(f"\n  Memuat data EURUSD...")
    features_df, prices_df, raw_atr = load_dataset(config['csv_path'])

    if 'Date' in prices_df.columns:
        prices_df['Date'] = pd.to_datetime(prices_df['Date'])
        print(f"  Periode data: {prices_df['Date'].iloc[0]} — {prices_df['Date'].iloc[-1]}")
    print(f"  Total data points: {len(features_df):,}")
    print(f"  Indikator: {len(features_df.columns)} fitur")

    # Load model
    print(f"\n{'='*75}")
    model, is_recurrent = load_model(model_path)

    # Run full backtest
    print(f"\n{'='*75}")
    print("  [BACKTEST] EURUSD H1 — Full Period")
    print("=" * 75)
    m = run_backtest(
        model, is_recurrent,
        features_df, prices_df, raw_atr,
        config=config,
        label="EURUSD Full",
        log_path="backtest_logs/backtest_eurusd.csv"
    )
    print_metrics(m)

    # Plot
    plot_results(m, output_path=output_chart)

    # Summary
    print(f"\n{'='*75}")
    print("  RINGKASAN CROSS-INSTRUMENT TEST")
    print("=" * 75)
    if m['profit_factor'] > 1.0 and m['cum_return'] > 0:
        print("  ✅ Model BERHASIL generalisasi ke EURUSD!")
        print(f"     Profit Factor: {m['profit_factor']:.2f} | ROI: {m['cum_return']:+.2f}%")
    elif m['profit_factor'] > 0.8:
        print("  ⚠️  Model MARGINAL pada EURUSD (PF antara 0.8-1.0)")
        print(f"     Profit Factor: {m['profit_factor']:.2f} | ROI: {m['cum_return']:+.2f}%")
        print("     Pertimbangkan training ulang dengan data EURUSD.")
    else:
        print("  ❌ Model TIDAK generalisasi ke EURUSD")
        print(f"     Profit Factor: {m['profit_factor']:.2f} | ROI: {m['cum_return']:+.2f}%")
        print("     Model perlu di-train ulang dengan data EURUSD.")
    print("=" * 75)


if __name__ == "__main__":
    evaluate_eurusd()
