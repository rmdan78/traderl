#!/usr/bin/env bash
# ==============================================================================
# RUNNER SCRIPT UNTUK LINUX (DEBIAN / UBUNTU) - HEADLESS VIA XVFB
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WINEPREFIX="${WINEPREFIX:-$HOME/.wine}"
export WINEARCH=win64
export WINEDEBUG=-all
export DISPLAY=:99

# 1. Pastikan Virtual Framebuffer (Xvfb) aktif jika tidak ada monitor fisik
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "[INFO] Menjalankan Xvfb virtual display (:99)..."
    Xvfb :99 -screen 0 1024x768x16 -nolisten tcp &
    sleep 2
fi

cd "$SCRIPT_DIR"

# 2. Jalankan bot trading di dalam WINE environment
echo "[INFO] Memulai MT5 AI Live Trader via WINE..."
DISPLAY=:99 wine python mt5_live_trader.py "$@"
