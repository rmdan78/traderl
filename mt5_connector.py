"""
================================================================================
MODUL KONEKTOR METATRADER 5 (MT5)
================================================================================
Menyediakan antarmuka (interface) terisolasi dan tangguh ke platform MetaTrader 5:
- Inisialisasi & Login akun (Real / Demo)
- Pengambilan data candle (OHLCV) historis & live
- Pengecekan status akun (Balance, Equity, Margin, Leverage, PnL)
- Deteksi simbol broker (mendukung suffix seperti XAUUSDm, GOLD, dll.)
- Eksekusi order (Market BUY/SELL, Close, Trailing SL/TP)
================================================================================
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


logger = logging.getLogger("MT5Connector")


TIMEFRAME_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,    # mt5.TIMEFRAME_H1
    "H4": 16388,    # mt5.TIMEFRAME_H4
    "D1": 16408,    # mt5.TIMEFRAME_D1
    "W1": 32769,    # mt5.TIMEFRAME_W1
    "MN1": 49153,   # mt5.TIMEFRAME_MN1
}


class MT5Connector:
    """Konektor API ke MetaTrader 5."""

    def __init__(
        self,
        account: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
        magic_number: int = 8882026,
    ):
        self.account = int(account) if account and str(account).isdigit() and int(account) > 0 else None
        self.password = password if password else None
        self.server = server if server else None
        self.path = path if path and os.path.exists(path) else None
        self.magic_number = magic_number
        self.connected = False
        self._matched_symbols_cache = {}

        if not MT5_AVAILABLE:
            raise ImportError(
                "Paket 'MetaTrader5' belum terpasang. Jalankan: pip install MetaTrader5"
            )

    def connect(self) -> bool:
        """Inisialisasi dan login ke terminal MT5."""
        has_credentials = bool(self.account and self.account > 0 and self.password and self.server)

        # 1. Inisialisasi Terminal
        init_kwargs = {}
        if self.path:
            init_kwargs["path"] = self.path

        if has_credentials:
            init_kwargs["login"] = self.account
            init_kwargs["password"] = self.password
            init_kwargs["server"] = self.server

        # Coba inisialisasi dengan argumen
        if not mt5.initialize(**init_kwargs):
            # Jika gagal dengan kredensial, coba inisialisasi default terminal (auto-attach)
            if has_credentials:
                logger.warning("Inisialisasi dengan kredensial gagal. Mencoba auto-attach ke terminal MT5 yang aktif...")
            if not mt5.initialize():
                err = mt5.last_error()
                logger.error(f"Gagal inisialisasi MT5: {err}")
                self.connected = False
                return False

        # 2. Login akun jika kredensial lengkap
        if has_credentials:
            authorized = mt5.login(login=self.account, password=self.password, server=self.server)
            if not authorized:
                err = mt5.last_error()
                logger.error(f"Gagal login ke akun MT5 #{self.account} @ {self.server}: {err}")
                # Cek apakah sudah ada akun yang login di terminal
                acc_check = mt5.account_info()
                if acc_check is None:
                    self.connected = False
                    return False
                logger.warning(f"Menggunakan akun terminal aktif #{acc_check.login} @ {acc_check.server}")

        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Tidak dapat mengambil informasi akun MT5.")
            self.connected = False
            return False

        self.connected = True
        logger.info(
            f"Terhubung ke MT5: Akun #{account_info.login} ({account_info.name}) | "
            f"Server: {account_info.server} | Saldo: ${account_info.balance:.2f} | "
            f"Equity: ${account_info.equity:.2f}"
        )
        return True

    def disconnect(self):
        """Tutup koneksi MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Koneksi MT5 telah ditutup.")

    def is_alive(self) -> bool:
        """Cek apakah koneksi MT5 masih aktif."""
        if not self.connected:
            return False
        acc = mt5.account_info()
        return acc is not None

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Mengambil data status akun MT5 saat ini."""
        if not self.is_alive():
            if not self.connect():
                return None

        acc = mt5.account_info()
        if acc is None:
            return None

        return {
            "login": acc.login,
            "name": acc.name,
            "server": acc.server,
            "currency": acc.currency,
            "leverage": acc.leverage,
            "balance": float(acc.balance),
            "equity": float(acc.equity),
            "margin": float(acc.margin),
            "free_margin": float(acc.margin_free),
            "margin_level": float(acc.margin_level) if acc.margin_level else 0.0,
            "floating_profit": float(acc.profit),
            "trade_allowed": bool(acc.trade_allowed),
            "trade_expert": bool(acc.trade_expert),
        }

    def resolve_symbol(self, requested_symbol: str) -> Optional[str]:
        """
        Mendeteksi simbol yang valid di broker (misal XAUUSD vs XAUUSDm vs GOLD).
        """
        if requested_symbol in self._matched_symbols_cache:
            return self._matched_symbols_cache[requested_symbol]

        # Coba langsung
        info = mt5.symbol_info(requested_symbol)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(requested_symbol, True)
            self._matched_symbols_cache[requested_symbol] = requested_symbol
            return requested_symbol

        # Cari kandidat variasi nama broker
        candidates = [
            requested_symbol,
            requested_symbol + "m",
            requested_symbol + ".m",
            requested_symbol + ".a",
            requested_symbol + ".pro",
            requested_symbol + ".raw",
            requested_symbol + "c",
        ]
        if requested_symbol.upper().startswith("XAU"):
            candidates.extend(["GOLD", "GOLDm", "XAUUSD.ecn", "XAUUSD.s"])
        elif requested_symbol.upper().startswith("EURUSD"):
            candidates.extend(["EURUSD.ecn", "EURUSD.s"])

        for cand in candidates:
            info = mt5.symbol_info(cand)
            if info is not None:
                mt5.symbol_select(cand, True)
                logger.info(f"Simbol '{requested_symbol}' dipetakan ke '{cand}' pada broker.")
                self._matched_symbols_cache[requested_symbol] = cand
                return cand

        # Jika masih belum ketemu, cari di seluruh daftar simbol MT5
        all_symbols = mt5.symbols_get()
        if all_symbols:
            for s in all_symbols:
                if requested_symbol.upper() in s.name.upper():
                    mt5.symbol_select(s.name, True)
                    logger.info(f"Simbol '{requested_symbol}' dipetakan ke '{s.name}' pada broker.")
                    self._matched_symbols_cache[requested_symbol] = s.name
                    return s.name

        logger.error(f"Simbol '{requested_symbol}' tidak ditemukan di broker MT5.")
        return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Ambil spesifikasi simbol dari broker."""
        resolved = self.resolve_symbol(symbol)
        if not resolved:
            return None

        info = mt5.symbol_info(resolved)
        if info is None:
            return None

        return {
            "symbol": resolved,
            "digits": info.digits,
            "point": info.point,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread,
            "spread_usd": round(info.spread * info.point, info.digits),
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_contract_size": info.trade_contract_size,
            "filling_mode": info.filling_mode,
        }

    def fetch_candles(
        self, symbol: str, timeframe: str = "H1", count: int = 500
    ) -> Optional[pd.DataFrame]:
        """
        Mengambil sejumlah candle OHLCV dari MT5.

        Returns DataFrame dengan kolom:
        ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        """
        if not self.is_alive():
            if not self.connect():
                return None

        resolved_symbol = self.resolve_symbol(symbol)
        if not resolved_symbol:
            return None

        tf_const = getattr(mt5, f"TIMEFRAME_{timeframe.upper()}", None)
        if tf_const is None:
            tf_const = TIMEFRAME_MAP.get(timeframe.upper(), mt5.TIMEFRAME_H1)

        rates = mt5.copy_rates_from_pos(resolved_symbol, tf_const, 0, count)
        if rates is None or len(rates) == 0:
            logger.error(f"Gagal mengambil candle untuk {resolved_symbol}: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        })
        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        return df[cols].copy()

    def get_open_positions(
        self, symbol: Optional[str] = None, magic: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Mengambil posisi terbuka di MT5 yang sesuai filter simbol dan magic number."""
        if not self.is_alive():
            if not self.connect():
                return []

        resolved_symbol = self.resolve_symbol(symbol) if symbol else None

        if resolved_symbol:
            positions = mt5.positions_get(symbol=resolved_symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        magic_filter = magic if magic is not None else self.magic_number
        result = []
        for pos in positions:
            if magic_filter is not None and pos.magic != magic_filter:
                continue

            # pos.type: 0 = POSITION_TYPE_BUY, 1 = POSITION_TYPE_SELL
            pos_dict = {
                "ticket": pos.ticket,
                "time": datetime.fromtimestamp(pos.time),
                "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                "direction": 1 if pos.type == mt5.POSITION_TYPE_BUY else -1,
                "magic": pos.magic,
                "volume": pos.volume,
                "price_open": pos.price_open,
                "sl": pos.sl,
                "tp": pos.tp,
                "price_current": pos.price_current,
                "profit": pos.profit,
                "symbol": pos.symbol,
                "comment": pos.comment,
            }
            result.append(pos_dict)

        return result

    def _get_filling_type(self, symbol_info) -> int:
        """Menentukan filling mode yang didukung broker (FOK / IOC / RETURN)."""
        filling = symbol_info.filling_mode
        # Cek bit mask
        if filling & 1:  # SYMBOL_FILLING_FOK
            return mt5.ORDER_FILLING_FOK
        elif filling & 2:  # SYMBOL_FILLING_IOC
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN

    def open_order(
        self,
        symbol: str,
        direction: str,  # 'BUY' atau 'SELL'
        lot: float,
        sl_dist: float = 0.0,
        tp_dist: float = 0.0,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "RL_Agent",
        slippage_points: int = 20,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Membuka market order BUY atau SELL di MT5 dengan kalkulasi SL/TP dinamis yang valid.
        """
        if not self.is_alive():
            if not self.connect():
                return False, {"error": "Koneksi MT5 terputus"}

        resolved_symbol = self.resolve_symbol(symbol)
        if not resolved_symbol:
            return False, {"error": f"Simbol {symbol} tidak valid"}

        sym_info = mt5.symbol_info(resolved_symbol)
        if sym_info is None:
            return False, {"error": f"Tidak dapat membaca info {resolved_symbol}"}

        # Normalisasi lot ke batas broker
        lot = max(sym_info.volume_min, min(sym_info.volume_max, lot))
        step = sym_info.volume_step
        lot = round(lot / step) * step
        lot = round(lot, 2)

        digits = sym_info.digits
        point = sym_info.point
        stops_level = getattr(sym_info, "trade_stops_level", 0)
        min_stop_distance = max(stops_level * point, 10 * point)

        order_type_str = direction.upper()
        if order_type_str == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = sym_info.ask

            # Hitung SL & TP berdasarkan jarak dari harga live Ask
            if sl_dist > 0:
                eff_sl_dist = max(sl_dist, min_stop_distance)
                sl = price - eff_sl_dist
            elif sl > 0 and sl >= price:
                sl = price - min_stop_distance

            if tp_dist > 0:
                eff_tp_dist = max(tp_dist, min_stop_distance)
                tp = price + eff_tp_dist
            elif tp > 0 and tp <= price:
                tp = price + min_stop_distance

        elif order_type_str == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = sym_info.bid

            # Untuk SELL: SL HARUS LEBIH TINGGI dari Bid, TP HARUS LEBIH RENDAH dari Bid
            if sl_dist > 0:
                eff_sl_dist = max(sl_dist, min_stop_distance)
                sl = price + eff_sl_dist
            elif sl > 0 and sl <= price:
                sl = price + min_stop_distance

            if tp_dist > 0:
                eff_tp_dist = max(tp_dist, min_stop_distance)
                tp = price - eff_tp_dist
            elif tp > 0 and tp >= price:
                tp = price - min_stop_distance
        else:
            return False, {"error": f"Arah order '{direction}' tidak dikenal"}

        sl = round(float(sl), digits) if sl > 0 else 0.0
        tp = round(float(tp), digits) if tp > 0 else 0.0
        filling = self._get_filling_type(sym_info)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": resolved_symbol,
            "volume": float(lot),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": int(slippage_points),
            "magic": int(self.magic_number),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            logger.error(f"Order send gagal tanpa response: {err}")
            return False, {"error": str(err)}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order ditolak MT5: code={result.retcode}, comment={result.comment}")
            return False, {
                "retcode": result.retcode,
                "comment": result.comment,
                "request": request,
            }

        logger.info(
            f"[ORDER BERHASIL] {order_type_str} {lot} {resolved_symbol} @ {price:.{digits}f} "
            f"| SL: {sl:.{digits}f} | TP: {tp:.{digits}f} | Ticket: {result.order}"
        )
        return True, {
            "ticket": result.order,
            "deal": result.deal,
            "price": result.price,
            "volume": result.volume,
            "sl": sl,
            "tp": tp,
            "retcode": result.retcode,
        }

    def close_position(
        self, ticket: int, slippage_points: int = 20
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Menutup posisi yang ada berdasarkan tiket posisi."""
        if not self.is_alive():
            if not self.connect():
                return False, {"error": "Koneksi MT5 terputus"}

        pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list or len(pos_list) == 0:
            return False, {"error": f"Posisi tiket #{ticket} tidak ditemukan"}

        pos = pos_list[0]
        sym_info = mt5.symbol_info(pos.symbol)
        if sym_info is None:
            return False, {"error": f"Tidak dapat membaca info {pos.symbol}"}

        # Untuk menutup BUY, kirim order SELL; untuk menutup SELL, kirim order BUY
        if pos.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = sym_info.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = sym_info.ask

        filling = self._get_filling_type(sym_info)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": int(ticket),
            "symbol": pos.symbol,
            "volume": float(pos.volume),
            "type": close_type,
            "price": float(price),
            "deviation": int(slippage_points),
            "magic": int(self.magic_number),
            "comment": f"Close #{ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err_msg = result.comment if result else str(mt5.last_error())
            logger.error(f"Gagal menutup posisi #{ticket}: {err_msg}")
            return False, {"error": err_msg}

        logger.info(f"[POSISI DITUTUP] #{ticket} {pos.symbol} @ {price:.{sym_info.digits}f} (Profit: ${pos.profit:+.2f})")
        return True, {"ticket": ticket, "retcode": result.retcode}

    def modify_position_sltp(
        self, ticket: int, sl: float, tp: float
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Mengubah Stop Loss dan Take Profit posisi yang aktif dengan validasi ketat."""
        if not self.is_alive():
            if not self.connect():
                return False, {"error": "Koneksi MT5 terputus"}

        pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list or len(pos_list) == 0:
            return False, {"error": f"Posisi tiket #{ticket} tidak ditemukan"}

        pos = pos_list[0]
        sym_info = mt5.symbol_info(pos.symbol)
        digits = sym_info.digits if sym_info else 2
        point = sym_info.point if sym_info else 0.01
        stops_level = getattr(sym_info, "trade_stops_level", 0) if sym_info else 0
        min_stop_distance = max(stops_level * point, 10 * point)

        sl = round(float(sl), digits) if sl > 0 else 0.0
        tp = round(float(tp), digits) if tp > 0 else 0.0

        # Validasi arah SL / TP terhadap harga live saat ini
        if pos.type == mt5.POSITION_TYPE_BUY:
            current_bid = sym_info.bid if sym_info else pos.price_current
            if sl > 0 and sl >= current_bid - min_stop_distance:
                sl = round(current_bid - min_stop_distance, digits)
        else:  # POSITION_TYPE_SELL
            current_ask = sym_info.ask if sym_info else pos.price_current
            if sl > 0 and sl <= current_ask + min_stop_distance:
                sl = round(current_ask + min_stop_distance, digits)

        # Jika SL dan TP sama persis dengan yang ada, tidak perlu kirim modifikasi
        if abs(pos.sl - sl) < 1e-5 and abs(pos.tp - tp) < 1e-5:
            return True, {"modified": False, "reason": "No change"}

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": pos.symbol,
            "sl": float(sl),
            "tp": float(tp),
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err_msg = result.comment if result else str(mt5.last_error())
            logger.warning(f"Gagal modifikasi SL/TP #{ticket}: {err_msg}")
            return False, {"error": err_msg}

        logger.info(f"[SL/TP DIUPDATE] #{ticket} SL -> {sl:.{digits}f} | TP -> {tp:.{digits}f}")
        return True, {"ticket": ticket, "sl": sl, "tp": tp}
