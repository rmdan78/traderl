"""
================================================================================
MODUL FEATURE ENGINEERING — TECHNICAL INDICATORS + MULTI-TIMEFRAME
================================================================================
Agent TIDAK menerima OHLCV mentah. Menerima 22 indikator teknikal
yang di-normalize dengan rolling z-score:

 === H1 INDICATORS (16) ===
  1. RSI (14)
  2. MACD Line (12, 26)
  3. MACD Signal (9)
  4. MACD Histogram
  5. Bollinger %B (20, 2)
  6. Bollinger Bandwidth (20, 2)
  7. ATR (14)
  8. Stochastic %K (14, 3)
  9. Stochastic %D (3)
 10. EMA Crossover Ratio (9/21)
 11. ADX (14)
 12. CCI (20)
 13. Williams %R (14)
 14. OBV (On-Balance Volume)
 15. MFI (14)
 16. ROC (10)

 === MULTI-TIMEFRAME & REGIME (6) ===
 17. RSI H4 (14)         — bigger picture momentum
 18. MACD Hist H4         — bigger picture trend
 19. EMA Cross H4 (9/21)  — bigger picture trend direction
 20. Volatility Regime     — ATR / ATR_50 ratio (high/low vol detection)
 21. ROC (5)              — short-term momentum
 22. ROC (20)             — medium-term momentum

Total: 22 fitur indikator + 4 state = 26 obs dim
================================================================================
"""

import numpy as np
import pandas as pd


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling Z-Score normalization."""
    mean = series.rolling(window=window, min_periods=50).mean()
    std = series.rolling(window=window, min_periods=50).std()
    return (series - mean) / (std + 1e-8)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


# =============================================================================
# INDIVIDUAL INDICATOR FUNCTIONS
# =============================================================================

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD Line, Signal Line, Histogram."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger %B and Bandwidth."""
    sma = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (close - lower) / (upper - lower + 1e-8)
    bandwidth = (upper - lower) / (sma + 1e-8)
    return pct_b, bandwidth


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = _true_range(high, low, close)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    return atr


def calc_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                    k_period: int = 14, d_period: int = 3):
    """Stochastic Oscillator %K and %D."""
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    pct_k = ((close - lowest_low) / (highest_high - lowest_low + 1e-8)) * 100.0
    pct_d = pct_k.rolling(window=d_period, min_periods=1).mean()
    return pct_k, pct_d


def calc_ema_crossover(close: pd.Series, fast: int = 9, slow: int = 21) -> pd.Series:
    """EMA Crossover — ratio of (fast - slow) / slow."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    return (ema_fast - ema_slow) / (ema_slow + 1e-8)


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = _true_range(high, low, close)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()

    plus_di = 100.0 * _ema(plus_dm, period) / (atr + 1e-8)
    minus_di = 100.0 * _ema(minus_dm, period) / (atr + 1e-8)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    adx = _ema(dx, period)
    return adx


def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=1).mean()
    mad = tp.rolling(window=period, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad + 1e-8)
    return cci


def calc_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R."""
    highest_high = high.rolling(window=period, min_periods=1).max()
    lowest_low = low.rolling(window=period, min_periods=1).min()
    wr = ((highest_high - close) / (highest_high - lowest_low + 1e-8)) * -100.0
    return wr


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    sign = np.sign(close.diff()).fillna(0)
    obv = (sign * volume).cumsum()
    return obv


def calc_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
             period: int = 14) -> pd.Series:
    """Money Flow Index."""
    tp = (high + low + close) / 3.0
    mf = tp * volume
    tp_diff = tp.diff()

    pos_mf = mf.where(tp_diff > 0, 0.0)
    neg_mf = mf.where(tp_diff < 0, 0.0)

    pos_sum = pos_mf.rolling(window=period, min_periods=1).sum()
    neg_sum = neg_mf.rolling(window=period, min_periods=1).sum()

    mr = pos_sum / (neg_sum + 1e-8)
    mfi = 100.0 - (100.0 / (1.0 + mr))
    return mfi


def calc_roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change."""
    roc = ((close - close.shift(period)) / (close.shift(period) + 1e-8)) * 100.0
    return roc


# =============================================================================
# MULTI-TIMEFRAME HELPERS
# =============================================================================

def _resample_h4(series_dict: dict, agg_map: dict) -> pd.DataFrame:
    """
    Simulasi resampling H1 → H4 tanpa datetime index.
    Mengelompokkan setiap 4 bar H1 dan menerapkan aggregation,
    lalu forward-fill kembali ke resolusi H1.
    """
    n = len(list(series_dict.values())[0])
    # Buat group labels: [0,0,0,0,1,1,1,1,...]
    groups = np.arange(n) // 4

    df_temp = pd.DataFrame(series_dict)
    df_temp['_group'] = groups

    agg_result = df_temp.groupby('_group').agg(agg_map)

    # Map kembali ke H1 resolution
    result = pd.DataFrame(index=range(n))
    for col in agg_result.columns:
        result[col] = agg_result[col].values[groups]

    return result


# =============================================================================
# MAIN FEATURE ENGINEERING
# =============================================================================

def compute_features(
    df: pd.DataFrame,
    zscore_window: int = 252 * 24
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Mengekstrak 22 fitur indikator teknikal dari data OHLCV.
    Agent TIDAK menerima OHLC mentah — hanya indikator.

    Returns:
        features_df: DataFrame berisi 22 indikator (z-score normalized)
        prices_df:   DataFrame berisi OHLCV + Date + Hour (untuk kalkulasi PnL internal)
        raw_atr:     Series ATR mentah (untuk SL/TP di environment)
    """
    df = df.copy()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['Hour'] = df['Date'].dt.hour
        df.set_index('Date', inplace=True)
    else:
        df['Hour'] = 0

    df_resampled = df.reset_index()

    # Extract raw price series
    open_s  = df_resampled['Open'].astype(float)
    high_s  = df_resampled['High'].astype(float)
    low_s   = df_resampled['Low'].astype(float)
    close_s = df_resampled['Close'].astype(float)
    volume_s = df_resampled['Volume'].astype(float) if 'Volume' in df_resampled.columns else pd.Series(np.ones(len(df_resampled)))

    # =========================================================================
    # 16 CORE H1 TECHNICAL INDICATORS
    # =========================================================================
    features = pd.DataFrame(index=df_resampled.index)

    # 1. RSI (14)
    features['rsi'] = calc_rsi(close_s, 14)

    # 2-4. MACD (12, 26, 9) → Line, Signal, Histogram
    macd_line, macd_signal, macd_hist = calc_macd(close_s, 12, 26, 9)
    features['macd_line'] = macd_line
    features['macd_signal'] = macd_signal
    features['macd_hist'] = macd_hist

    # 5-6. Bollinger Bands (20, 2) → %B, Bandwidth
    boll_pctb, boll_bw = calc_bollinger(close_s, 20, 2.0)
    features['bollinger_pctb'] = boll_pctb
    features['bollinger_bw'] = boll_bw

    # 7. ATR (14)
    raw_atr = calc_atr(high_s, low_s, close_s, 14)
    features['atr'] = raw_atr.copy()

    # 8-9. Stochastic (14, 3) → %K, %D
    stoch_k, stoch_d = calc_stochastic(high_s, low_s, close_s, 14, 3)
    features['stochastic_k'] = stoch_k
    features['stochastic_d'] = stoch_d

    # 10. EMA Crossover Ratio (9/21)
    features['ema_cross'] = calc_ema_crossover(close_s, 9, 21)

    # 11. ADX (14)
    features['adx'] = calc_adx(high_s, low_s, close_s, 14)

    # 12. CCI (20)
    features['cci'] = calc_cci(high_s, low_s, close_s, 20)

    # 13. Williams %R (14)
    features['williams_r'] = calc_williams_r(high_s, low_s, close_s, 14)

    # 14. OBV
    features['obv'] = calc_obv(close_s, volume_s)

    # 15. MFI (14)
    features['mfi'] = calc_mfi(high_s, low_s, close_s, volume_s, 14)

    # 16. ROC (10)
    features['roc'] = calc_roc(close_s, 10)

    # =========================================================================
    # 6 MULTI-TIMEFRAME & REGIME FEATURES
    # =========================================================================

    # --- H4 Indicators (simulasi resampling setiap 4 bar H1) ---
    h4_data = _resample_h4(
        {'Open': open_s, 'High': high_s, 'Low': low_s, 'Close': close_s},
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    )

    # 17. RSI H4 (14)
    features['rsi_h4'] = calc_rsi(h4_data['Close'], 14)

    # 18. MACD Histogram H4
    _, _, macd_hist_h4 = calc_macd(h4_data['Close'], 12, 26, 9)
    features['macd_hist_h4'] = macd_hist_h4

    # 19. EMA Cross H4 (9/21)
    features['ema_cross_h4'] = calc_ema_crossover(h4_data['Close'], 9, 21)

    # 20. Volatility Regime: ATR / ATR_50 ratio
    atr_50 = raw_atr.rolling(window=50, min_periods=10).mean()
    features['vol_regime'] = raw_atr / (atr_50 + 1e-8)

    # 21. ROC (5) — short-term momentum
    features['roc_5'] = calc_roc(close_s, 5)

    # 22. ROC (20) — medium-term momentum
    features['roc_20'] = calc_roc(close_s, 20)

    num_indicators = len(features.columns)
    print(f"  [Indicators] Total indikator: {num_indicators}")

    # =========================================================================
    # Rolling Z-Score Normalization
    # =========================================================================
    for col in features.columns:
        features[col] = compute_rolling_zscore(features[col], window=zscore_window)

    # Drop NaN dan Clip
    valid_idx = features.dropna().index
    features_clean = features.loc[valid_idx].reset_index(drop=True)
    prices_clean   = df_resampled.loc[valid_idx].reset_index(drop=True)
    raw_atr_clean  = raw_atr.loc[valid_idx].reset_index(drop=True)

    features_clean = features_clean.clip(lower=-5.0, upper=5.0).astype(np.float32)

    return features_clean, prices_clean, raw_atr_clean


def load_dataset(
    csv_path: str = "XAUUSD_H1.csv"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Memuat file CSV dan menghitung indikator teknikal + normalisasi.

    Returns:
        features_df, prices_df, raw_atr
    """
    sample_df = pd.read_csv(csv_path, nrows=5)
    first_col = str(sample_df.columns[0])

    if first_col.startswith('200') or first_col.startswith('201') or first_col.startswith('202'):
        num_cols = len(sample_df.columns)
        if num_cols == 6:
            col_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        elif num_cols == 5:
            col_names = ['Date', 'Open', 'High', 'Low', 'Close']
        else:
            col_names = [f'col_{i}' for i in range(num_cols)]
        df = pd.read_csv(csv_path, header=None, names=col_names)
    else:
        df = pd.read_csv(csv_path)

    return compute_features(df)


if __name__ == "__main__":
    print("Memproses dataset XAUUSD_H1.csv — Technical Indicators + Multi-Timeframe...")
    features, prices, raw_atr = load_dataset("XAUUSD_H1.csv")
    print(f"Selesai! Dimensi fitur: {features.shape}")
    print(f"Kolom fitur ({len(features.columns)}):")
    for i, col in enumerate(features.columns):
        print(f"  [{i+1:2d}] {col}")
    print(f"\nSample data (5 baris pertama):")
    print(features.head())
    print(f"\nRaw ATR sample: {raw_atr.head().tolist()}")
