"""
================================================================================
SKRIP PENGUJIAN NOTIFIKASI TELEGRAM BOT
================================================================================
Gunakan skrip ini untuk menguji apakah Bot Token dan Chat ID Telegram Anda
sudah benar dan dapat menerima pesan dari bot trading.
================================================================================
"""

import os
import sys
from dotenv import load_dotenv
from mt5_notifier import MT5Notifier


def test_telegram_alert():
    print("\n" + "=" * 70)
    print("        UJI COBA NOTIFIKASI TELEGRAM BOT MT5")
    print("=" * 70)

    # 1. Muat .env
    load_dotenv(".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    print(f"• TELEGRAM_BOT_TOKEN : {'Terisi (' + token[:6] + '...' + token[-4:] + ')' if token else 'KOSONG (Wajib diisi di .env)'}")
    print(f"• TELEGRAM_CHAT_ID   : {chat_id if chat_id else 'KOSONG (Wajib diisi di .env)'}")
    print("-" * 70)

    if not token or not chat_id:
        print("[FAIL] Token atau Chat ID masih kosong di file .env!")
        print("\nPanduan Membuat Bot Telegram (Gratis & 1 Menit):")
        print("  1. Buka Telegram, cari @BotFather.")
        print("  2. Kirim perintah: /newbot lalu ikuti petunjuk nama bot.")
        print("  3. Copy API Token yang diberikan, masukkan ke .env di TELEGRAM_BOT_TOKEN.")
        print("  4. Cari bot @userinfobot di Telegram, klik Start untuk melihat ID angka Anda.")
        print("  5. Copy Id angka tersebut, masukkan ke .env di TELEGRAM_CHAT_ID.")
        print("  6. Buka bot yang baru Anda buat, lalu klik START / kirim pesan 'halo'.")
        print("  7. Jalankan kembali: python test_telegram.py")
        print("=" * 70 + "\n")
        return False

    notifier = MT5Notifier(bot_token=token, chat_id=chat_id)

    print("\n[MENGIRIM TEST ALERT KE TELEGRAM...]")

    # 1. Test Text Alert
    sent = notifier.send_message(
        "👋 *Halo dari MT5 AI Trading Agent!*\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "✅ *Koneksi Telegram Berhasil Terhubung!*\n"
        "Bot Anda sekarang siap mengirimkan laporan:\n"
        "  • 🟢 Sinyal Open BUY / SELL\n"
        "  • 💰 Laporan Profit / Loss Close Trade\n"
        "  • ⚡ Update Trailing Stop Breakeven\n"
        "  • 🚨 Alert Error & Proteksi Circuit Breaker\n"
        "━━━━━━━━━━━━━━━━━━━"
    )

    if sent:
        print("\n[SUKSES] Pesan tes berhasil terkirim ke Telegram Anda!")
        print("Silakan cek aplikasi Telegram di HP / Laptop Anda.")
        print("=" * 70 + "\n")
        return True
    else:
        print("\n[FAIL] Gagal mengirim pesan ke Telegram.")
        print("Kemungkinan penyebab:")
        print("  1. Anda belum menekan tombol START pada bot Anda di Telegram.")
        print("  2. Token bot salah atau Chat ID salah.")
        print("  3. Koneksi internet terganggu.")
        print("=" * 70 + "\n")
        return False


if __name__ == "__main__":
    test_telegram_alert()
