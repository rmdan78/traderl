# Panduan Deploy & Menjalankan MT5 AI Trading Agent di Debian 12 (Headless VPS)

Panduan ini menjelaskan cara menjalankan **MT5 AI Reinforcement Learning Live Trading Agent** di VPS berbasis **Debian 12 (Bookworm)** secara *headless* (tanpa perlu monitor/GUI desktop) menggunakan **WINE & Xvfb** atau **Docker**.

---

## 🌟 Opsi 1: Instalasi Otomatis 1-Klik (Rekomendasi)

### Langkah 1: Upload / Clone Project ke VPS Debian 12
Upload seluruh folder project ini ke VPS Anda (misal ke `/root/reinforcement`):
```bash
cd /root/reinforcement
```

### Langkah 2: Berikan Izin Eksekusi & Jalankan Setup
Jalankan skrip instalasi otomatis:
```bash
chmod +x setup_debian12.sh run_live_linux.sh
sudo bash setup_debian12.sh
```
Skrip ini akan secara otomatis:
- Memasang WINE 64-bit & arsitektur 32-bit
- Memasang Xvfb (*Virtual Framebuffer*)
- Memasang Python 3.10 Windows di dalam WINE
- Memasang MetaTrader 5 Terminal di dalam WINE
- Menginstal seluruh dependensi AI (`PyTorch`, `Stable-Baselines3`, `MetaTrader5`, dll.)

### Langkah 3: Konfigurasi `.env`
Edit file `.env` di VPS:
```bash
nano .env
```
Isi kredensial akun MT5 dan bot Telegram Anda:
```env
MT5_ACCOUNT=463887667
MT5_PASSWORD=password_anda
MT5_SERVER=Exness-MT5Trial17
MT5_MODE=DEMO

TELEGRAM_BOT_TOKEN=8916816574:AAG_nWI-YE-GA5y7tWDGjv_nRbu235CgTX4
TELEGRAM_CHAT_ID=5161749982
```

### Langkah 4: Jalankan Bot di Debian 12

- **Uji Coba Simulasi (Dry Run):**
  ```bash
  ./run_live_linux.sh --dry-run
  ```
- **Live Trading Akun DEMO:**
  ```bash
  ./run_live_linux.sh --mode DEMO
  ```
- **Live Trading Akun REAL:**
  ```bash
  ./run_live_linux.sh --mode REAL --lot 0.01
  ```

---

## 🔄 Opsi Menjalankan 24/7 di Background (Systemd Daemon)

Agar bot tetap berjalan saat terminal SSH ditutup dan otomatis menyala kembali jika VPS restart:

1. **Pasang Systemd Service:**
   ```bash
   sudo cp mt5_agent.service /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

2. **Aktifkan & Jalankan Service:**
   ```bash
   sudo systemctl enable --now mt5_agent
   ```

3. **Cek Status Bot:**
   ```bash
   sudo systemctl status mt5_agent
   ```

4. **Melihat Log Real-Time:**
   ```bash
   sudo journalctl -u mt5_agent -f
   ```

5. **Menghentikan Bot:**
   ```bash
   sudo systemctl stop mt5_agent
   ```

---

## 🐳 Opsi 2: Menggunakan Docker Container

Jika Anda lebih suka menggunakan Docker di Debian 12:

1. **Pastikan Docker & Docker Compose terpasang:**
   ```bash
   sudo apt install -y docker.io docker-compose
   ```

2. **Build & Jalankan Container:**
   ```bash
   docker compose up -d --build
   ```

3. **Melihat Log:**
   ```bash
   docker logs -f mt5_ai_live_agent
   ```

4. **Menghentikan Container:**
   ```bash
   docker compose down
   ```
