#!/usr/bin/env bash
# ==============================================================================
# SKRIP OTOMATISASI INSTALASI MT5 AI TRADING AGENT DI DEBIAN 12 (BOOKWORM)
# ==============================================================================
# Skrip ini akan memasang:
# 1. Paket dependensi Linux & WINE 64-bit + 32-bit architecture
# 2. Xvfb (Virtual Framebuffer agar MT5 bisa berjalan headless tanpa GUI desktop)
# 3. MetaTrader 5 Terminal di dalam WINE
# 4. Python 3.10 Windows di dalam WINE
# 5. Semua library AI (PyTorch, Stable-Baselines3, MetaTrader5, dll.)
# ==============================================================================

set -e

echo "============================================================================"
echo "    MEMULAI SETUP MT5 AI RL TRADING AGENT UNTUK DEBIAN 12 (HEADLESS VPS)"
echo "============================================================================"

# Pastikan dijalankan sebagai root atau sudo
if [ "$EUID" -ne 0 ]; then
  echo "[ERROR] Harap jalankan skrip ini dengan sudo atau sebagai root:"
  echo "        sudo bash setup_debian12.sh"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
REAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(eval echo ~$REAL_USER)

echo "[1/6] Mengupdate sistem dan memeriksa alokasi memori..."

# Optimasi untuk VPS RAM 1GB: Buat Swap 2GB otomatis jika belum ada
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
if [ "$TOTAL_MEM_KB" -lt 1800000 ]; then
    echo "  [INFO] Terdeteksi RAM <= 1.5GB. Memeriksa Swap Memory..."
    SWAP_EXISTS=$(swapon --show | wc -l)
    if [ "$SWAP_EXISTS" -le 1 ]; then
        echo "  [OPTIMASI] Membuat 2GB Swap file untuk mencegah Out-of-Memory (OOM)..."
        fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
        chmod 600 /swapfile
        mkswap /swapfile
        swapon /swapfile
        if ! grep -q '/swapfile' /etc/fstab; then
            echo '/swapfile none swap sw 0 0' >> /etc/fstab
        fi
        echo "  [OK] Swap 2GB berhasil diaktifkan!"
    else
        echo "  [OK] Swap sudah tersedia."
    fi
fi

dpkg --add-architecture i386
apt-get update
apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    gnupg2 \
    software-properties-common \
    xvfb \
    xauth \
    cabextract \
    winbind \
    procps \
    unzip

echo "[2/6] Memasang WINE (WineHQ) di Debian 12..."
mkdir -pm755 /etc/apt/keyrings
wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key
wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/debian/dists/bookworm/winehq-bookworm.sources
apt-get update
apt-get install -y --install-recommends winehq-stable || apt-get install -y wine wine64 wine32

echo "[3/6] Menginisialisasi WINE Prefix..."
export WINEPREFIX="$USER_HOME/.wine"
export WINEARCH=win64
export WINEDEBUG=-all
export DISPLAY=:99

# Jalankan Xvfb virtual display
Xvfb :99 -screen 0 1024x768x16 &
XVFB_PID=$!
sleep 2

# Inisialisasi Wine sebagai user biasa
su - $REAL_USER -c "DISPLAY=:99 WINEPREFIX=$WINEPREFIX WINEARCH=win64 WINEDEBUG=-all wineboot -i" || true
sleep 3

echo "[4/6] Mendownload dan memasang Python 3.10 Windows di WINE..."
TEMP_DIR="/tmp/mt5_setup"
mkdir -p $TEMP_DIR
cd $TEMP_DIR

PYTHON_INSTALLER="python-3.10.11-amd64.exe"
if [ ! -f "$PYTHON_INSTALLER" ]; then
    wget -q --show-progress "https://www.python.org/ftp/python/3.10.11/$PYTHON_INSTALLER"
fi

su - $REAL_USER -c "DISPLAY=:99 WINEPREFIX=$WINEPREFIX WINEDEBUG=-all wine $TEMP_DIR/$PYTHON_INSTALLER /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1"
sleep 5

echo "[5/6] Mendownload dan memasang MetaTrader 5 Terminal di WINE..."
MT5_INSTALLER="mt5setup.exe"
if [ ! -f "$MT5_INSTALLER" ]; then
    wget -q --show-progress -O "$MT5_INSTALLER" "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"
fi

echo "Memasang MT5 (proses instalasi background)..."
su - $REAL_USER -c "DISPLAY=:99 WINEPREFIX=$WINEPREFIX WINEDEBUG=-all wine $TEMP_DIR/$MT5_INSTALLER /auto" || true
sleep 10

echo "[6/6] Memasang Library Python AI (MetaTrader5, PyTorch, SB3, dll.)..."
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

su - $REAL_USER -c "DISPLAY=:99 WINEPREFIX=$WINEPREFIX WINEDEBUG=-all wine python -m pip install --upgrade pip setuptools wheel"
su - $REAL_USER -c "DISPLAY=:99 WINEPREFIX=$WINEPREFIX WINEDEBUG=-all wine python -m pip install -r $PROJECT_DIR/requirements.txt"

# Buat shortcut runner script
RUNNER_SCRIPT="$PROJECT_DIR/run_live_linux.sh"
cat << 'EOF' > "$RUNNER_SCRIPT"
#!/usr/bin/env bash
# Runner script untuk MT5 Live Trader di Linux (Headless via Xvfb)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WINEPREFIX="$HOME/.wine"
export WINEARCH=win64
export WINEDEBUG=-all
export DISPLAY=:99

# Pastikan Xvfb berjalan
if ! pgrep -x "Xvfb" > /dev/null; then
    Xvfb :99 -screen 0 1024x768x16 -nolisten tcp &
    sleep 2
fi

cd "$SCRIPT_DIR"
DISPLAY=:99 wine python mt5_live_trader.py "$@"
EOF

chmod +x "$RUNNER_SCRIPT"
chown -R $REAL_USER:$REAL_USER "$PROJECT_DIR" "$USER_HOME/.wine"

# Matikan Xvfb sementara
kill $XVFB_PID 2>/dev/null || true

echo "============================================================================"
echo "          INSTALASI DEBIAN 12 SUKSES & SELESAI!"
echo "============================================================================"
echo "Cara Menjalankan Bot di Debian 12:"
echo ""
echo "1. Uji Coba Diagnostik:"
echo "   ./run_live_linux.sh --dry-run"
echo ""
echo "2. Jalankan Mode Live DEMO:"
echo "   ./run_live_linux.sh --mode DEMO"
echo ""
echo "3. Jalankan Mode Live REAL:"
echo "   ./run_live_linux.sh --mode REAL --lot 0.01"
echo ""
echo "4. Menjalankan 24/7 di Background (Systemd Service):"
echo "   sudo cp mt5_agent.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable --now mt5_agent"
echo "============================================================================"
