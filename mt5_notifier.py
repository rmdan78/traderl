"""
================================================================================
MODUL NOTIFIKASI TELEGRAM BOT (REAL-TIME TRADING ALERTS)
================================================================================
Mengirim alert & laporan real-time langsung ke Telegram:
- Status Startup & Shutdown Bot
- Notifikasi Open Trade (BUY/SELL, Volume, Entry, SL, TP, ATR)
- Notifikasi Close Trade (Profit/Loss USD, Persentase, Alasan Keluar)
- Notifikasi Update Trailing Stop (Breakeven / Lock Profit)
- Alert Error & Peringatan (Requote, Koneksi Terputus, Spread Melebar)
- Alert Emergency Circuit Breaker (Max Drawdown / Daily Loss)
- Laporan Heartbeat / Ringkasan Akun
================================================================================
"""

import os
import logging
import requests
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger("MT5Notifier")


class MT5Notifier:
    """Pengirim notifikasi via Telegram Bot API dengan formatting lengkap."""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.bot_token and self.chat_id)

        if self.enabled:
            logger.info("Notifikasi Telegram diaktifkan.")
        else:
            logger.info("Notifikasi Telegram belum aktif (TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID di .env masih kosong).")

    def send_message(self, text: str) -> bool:
        """Kirim pesan teks Markdown ke Telegram."""
        if not self.enabled:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                return True
            else:
                logger.warning(f"Telegram API response error ({resp.status_code}): {resp.text}")
                return False
        except Exception as e:
            logger.warning(f"Gagal mengirim notifikasi Telegram: {e}")
            return False

    def notify_startup(
        self,
        mode: str,
        symbol: str,
        timeframe: str,
        lot: float,
        balance: float,
        equity: float,
        server: str,
        login: int,
        magic: int,
    ):
        """Alert saat bot trading mulai berjalan."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"🚀 *[MT5 AI TRADING AGENT — BOT ONLINE]*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• Waktu: `{now_str}`\n"
            f"• Mode: *{mode}*\n"
            f"• Akun: `#{login}` @ `{server}`\n"
            f"• Simbol: `{symbol}` ({timeframe})\n"
            f"• Ukuran Lot: `{lot}`\n"
            f"• Magic Number: `{magic}`\n"
            f"• Saldo Awal: `${balance:,.2f}`\n"
            f"• Equity Awal: `${equity:,.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 *Status*: Agent RecurrentPPO aktif memantau pasar!"
        )
        self.send_message(msg)

    def notify_shutdown(self, reason: str = "Dihentikan oleh pengguna"):
        """Alert saat bot dihentikan."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"🛑 *[MT5 AI TRADING AGENT — BOT OFFLINE]*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• Waktu: `{now_str}`\n"
            f"• Status: *OFFLINE / STOPPED*\n"
            f"• Keterangan: `{reason}`\n"
            f"━━━━━━━━━━━━━━━━━━━"
        )
        self.send_message(msg)

    def notify_trade_open(
        self,
        symbol: str,
        direction: str,
        lot: float,
        entry: float,
        sl: float,
        tp: float,
        atr: float,
        ticket: Optional[int] = None,
        reason: str = "AI Model Decision",
    ):
        """Alert saat posisi baru dibuka."""
        emoji = "🟢" if direction == "BUY" else "🔴"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ticket_str = f"#{ticket}" if ticket else "Dry-Run"

        sl_dist = abs(entry - sl) if sl > 0 else 0
        tp_dist = abs(tp - entry) if tp > 0 else 0

        msg = (
            f"{emoji} *[MT5 AI AGENT — POSISI DIBUKA]*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• Tiket: `{ticket_str}`\n"
            f"• Waktu: `{now_str}`\n"
            f"• Simbol: `{symbol}`\n"
            f"• Aksi: *{direction}*\n"
            f"• Volume: `{lot} Lot`\n"
            f"• Harga Entry: `{entry:.2f}`\n"
            f"• Stop Loss: `{sl:.2f}` (-{sl_dist:.2f})\n"
            f"• Take Profit: `{tp:.2f}` (+{tp_dist:.2f})\n"
            f"• ATR (14): `{atr:.2f}`\n"
            f"• Sinyal: `{reason}`\n"
            f"━━━━━━━━━━━━━━━━━━━"
        )
        self.send_message(msg)

    def notify_trade_close(
        self,
        symbol: str,
        ticket: int,
        profit: float,
        reason: str = "SIGNAL",
        balance: Optional[float] = None,
        equity: Optional[float] = None,
        open_price: Optional[float] = None,
        close_price: Optional[float] = None,
    ):
        """Alert saat posisi ditutup."""
        is_win = profit >= 0
        emoji = "💰" if is_win else "🛑"
        status_str = "PROFIT (WIN)" if is_win else "LOSS"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        price_info = ""
        if open_price and close_price:
            price_info = f"• Harga: Entry `{open_price:.2f}` ➔ Exit `{close_price:.2f}`\n"

        balance_info = ""
        if balance is not None and equity is not None:
            balance_info = f"• Saldo Saat Ini: `${balance:,.2f}` | Equity: `${equity:,.2f}`\n"

        msg = (
            f"{emoji} *[MT5 AI AGENT — POSISI DITUTUP]*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• Tiket: `#{ticket}`\n"
            f"• Waktu: `{now_str}`\n"
            f"• Simbol: `{symbol}`\n"
            f"• Hasil: *{status_str}*\n"
            f"• Realized PnL: *${profit:+,.2f}*\n"
            f"{price_info}"
            f"• Alasan Keluar: `{reason}`\n"
            f"{balance_info}"
            f"━━━━━━━━━━━━━━━━━━━"
        )
        self.send_message(msg)

    def notify_trailing_update(
        self,
        symbol: str,
        ticket: int,
        old_sl: float,
        new_sl: float,
        stage: str,
        current_price: Optional[float] = None,
    ):
        """Alert saat Trailing Stop menggeser SL."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        price_str = f"• Harga Saat Ini: `{current_price:.2f}`\n" if current_price else ""
        msg = (
            f"⚡ *[MT5 AI AGENT — TRAILING STOP AKTIF]*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• Tiket: `#{ticket}` ({symbol})\n"
            f"• Waktu: `{now_str}`\n"
            f"• Tahap: *{stage}*\n"
            f"{price_str}"
            f"• Stop Loss Baru: `{old_sl:.2f}` ➔ *`{new_sl:.2f}`*\n"
            f"• Status Risiko: {'🔒 BEBAS RISIKO (Breakeven)' if 'Breakeven' in stage else '💰 PROFIT TERKUNCI'}\n"
            f"━━━━━━━━━━━━━━━━━━━"
        )
        self.send_message(msg)

    def notify_error(self, title: str, message: str, action: str = "Bot melanjutkan monitoring"):
        """Alert saat terjadi error eksekusi / koneksi."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"⚠️ *[MT5 AI AGENT — ERROR ALERT]*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• Waktu: `{now_str}`\n"
            f"• Masalah: *{title}*\n"
            f"• Detail: `{message}`\n"
            f"• Tindakan: `{action}`\n"
            f"━━━━━━━━━━━━━━━━━━━"
        )
        self.send_message(msg)

    def notify_warning(self, title: str, message: str):
        """Peringatan umum (misal spread melebar / koneksi lambat)."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"🔔 *[MT5 AI AGENT — PERINGATAN]*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• Waktu: `{now_str}`\n"
            f"• Info: *{title}*\n"
            f"• Detail: `{message}`\n"
            f"━━━━━━━━━━━━━━━━━━━"
        )
        self.send_message(msg)

    def notify_circuit_breaker(self, reason: str, details: str):
        """Alert kritis saat batas Drawdown / Daily Loss tercapai."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"🚨🚨 *[MT5 AI AGENT — CIRCUIT BREAKER TRIGGERED!]* 🚨🚨\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• Waktu: `{now_str}`\n"
            f"• Status: *EMERGENCY STOP DIAKTIFKAN*\n"
            f"• Penyebab: *{reason}*\n"
            f"• Detail Akun: `{details}`\n"
            f"• Tindakan Pengamanan: Seluruh posisi bot ditutup paksa dan trading dihentikan sementara untuk mengamankan sisa modal.\n"
            f"━━━━━━━━━━━━━━━━━━━"
        )
        self.send_message(msg)

    def notify_heartbeat(
        self, balance: float, equity: float, floating_pnl: float, position_str: str, last_bar: str
    ):
        """Laporan status berkala."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"💓 *[MT5 AI AGENT — STATUS UPDATE]*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• Waktu: `{now_str}`\n"
            f"• Saldo: `${balance:,.2f}` | Equity: `${equity:,.2f}`\n"
            f"• Floating PnL: *${floating_pnl:+,.2f}*\n"
            f"• Posisi Aktif: `{position_str}`\n"
            f"• Last Candle Closed: `{last_bar}`\n"
            f"━━━━━━━━━━━━━━━━━━━"
        )
        self.send_message(msg)
