# -*- coding: utf-8 -*-
"""
دریافت دیتای کندل (OHLCV) از صرافی‌ها با ccxt.
هیچ API Key لازم نیست چون فقط دیتای عمومی قیمت گرفته می‌شه.
اگه یک صرافی جواب نداد، خودکار می‌ره سراغ بعدی.
"""
import ccxt
import pandas as pd


def _get_exchange(name):
    exchange_class = getattr(ccxt, name)
    return exchange_class({"enableRateLimit": True, "timeout": 15000})


def fetch_ohlcv(symbol, timeframe="15m", limit=300, exchange_name="binance"):
    ex = _get_exchange(exchange_name)
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def fetch_ohlcv_with_fallback(symbol, timeframe, limit, exchange_order):
    last_err = None
    for ex_name in exchange_order:
        try:
            return fetch_ohlcv(symbol, timeframe, limit, ex_name), ex_name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"دریافت دیتا برای {symbol} از هیچ‌کدام از صرافی‌ها ممکن نشد: {last_err}")


def is_symbol_allowed(symbol, cfg):
    """
    فیلتر مطلق و همیشگی: این نمادها هیچ‌وقت نباید معامله بشن، فارغ از این‌که از
    کجا توی لیست اومدن (لیست پویا، لیست دستی، هر چیز دیگه). طلا (پشتوانه‌طلا) و
    جفت‌های استیبل‌کوین به استیبل‌کوین رفتار متفاوتی از کریپتوی معمولی دارن
    (نوسان خیلی کم یا رفتار غیرکریپتویی) و منطق این ربات براشون طراحی نشده.
    """
    base = symbol.split("/")[0].upper()
    gold_tokens = getattr(cfg, "EXCLUDE_GOLD_TOKENS", [])
    if base in [g.upper() for g in gold_tokens]:
        return False
    stablecoins = getattr(cfg, "STABLECOIN_BASES", [])
    if base in [s.upper() for s in stablecoins]:
        return False
    return True


def get_top_symbols(exchange_order, quote="USDT", top_n=250, exclude_keywords=None, cfg=None):
    """
    پرحجم‌ترین جفت‌ارزهای اسپات (بر اساس حجم معاملات ۲۴ ساعته) رو برمی‌گردونه.
    اول از صرافی اول لیست تلاش می‌کنه، اگه نشد می‌ره سراغ بعدی.
    """
    exclude_keywords = exclude_keywords or []
    last_err = None
    for ex_name in exchange_order:
        try:
            ex = _get_exchange(ex_name)
            markets = ex.load_markets()
            tickers = ex.fetch_tickers()

            candidates = []
            for sym, market in markets.items():
                if not market.get("active", True):
                    continue
                if market.get("quote") != quote:
                    continue
                if market.get("type") not in (None, "spot"):
                    continue
                if any(kw in sym for kw in exclude_keywords):
                    continue
                if cfg is not None and not is_symbol_allowed(sym, cfg):
                    continue
                t = tickers.get(sym)
                if not t:
                    continue
                vol = t.get("quoteVolume") or 0
                candidates.append((sym, vol))

            candidates.sort(key=lambda x: x[1], reverse=True)
            top = [sym for sym, _ in candidates[:top_n]]
            if top:
                return top, ex_name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"دریافت لیست نمادهای برتر ممکن نشد: {last_err}")
