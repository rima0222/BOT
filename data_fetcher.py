# -*- coding: utf-8 -*-
"""
دریافت دیتای کندل (OHLCV) از صرافی‌ها با ccxt.
هیچ API Key لازم نیست چون فقط دیتای عمومی قیمت گرفته می‌شه.
اگه یک صرافی جواب نداد، خودکار می‌ره سراغ بعدی.
"""
import threading

import ccxt
import pandas as pd

_local = threading.local()
# آمار منبع دیتای زنده (برای نمایش در پنل): {صرافی: تعداد درخواست موفق}
source_stats = {}
_stats_lock = threading.Lock()


def _get_exchange(name):
    """یک نمونه برای هر صرافی در هر رشته (بازار‌ها فقط یک‌بار بارگذاری می‌شن، نه در هر درخواست)."""
    cache = getattr(_local, "ex", None)
    if cache is None:
        cache = {}
        _local.ex = cache
    if name not in cache:
        exchange_class = getattr(ccxt, name)
        cache[name] = exchange_class({"enableRateLimit": True, "timeout": 15000})
    return cache[name]


def _count(ex_name):
    with _stats_lock:
        source_stats[ex_name] = source_stats.get(ex_name, 0) + 1


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
            df = fetch_ohlcv(symbol, timeframe, limit, ex_name)
            if df is None or not len(df):
                raise RuntimeError("کندلی برنگشت")
            _count(ex_name)
            return df, ex_name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"دریافت دیتا برای {symbol} از هیچ‌کدام از صرافی‌ها ممکن نشد: {last_err}")


def drop_forming_candle(df, timeframe, now_ms=None):
    """
    حذف کندلی که هنوز بسته نشده. سیگنال فقط باید روی کندل‌های بسته‌شده ساخته بشه؛
    وگرنه مثلاً «کندل تاییدی» وسط تشکیل شدنش صعودی به نظر میاد ولی نزولی بسته می‌شه
    (یکی از دلایل اصلی تفاوت نتیجه‌ی زنده با بک‌تست و استاپ‌های زودهنگام).
    """
    import time as _time
    import market_data
    if df is None or len(df) == 0:
        return df
    now_ms = now_ms if now_ms is not None else int(_time.time() * 1000)
    open_ms = df["timestamp"].values.astype("datetime64[ms]").astype("int64")
    closes = market_data.close_times_ms(open_ms, timeframe)
    return df[closes <= now_ms].reset_index(drop=True)


def fetch_closed_ohlcv(symbol, timeframe, limit, exchange_order):
    """آخرین `limit` کندلِ بسته‌شده (یکی بیشتر گرفته می‌شه چون آخری معمولاً در حال تشکیله)."""
    df, ex_name = fetch_ohlcv_with_fallback(symbol, timeframe, limit + 1, exchange_order)
    df = drop_forming_candle(df, timeframe)
    return df.tail(limit).reset_index(drop=True), ex_name


def fetch_since(symbol, timeframe, since_ms, exchange_order, max_pages=3, limit=1000):
    """کندل‌ها از since_ms تا الان (برای جبران وقفه‌ها، چند صفحه)؛ شامل کندل در حال تشکیل."""
    last_err = None
    for ex_name in exchange_order:
        try:
            ex = _get_exchange(ex_name)
            rows, cursor = [], int(since_ms)
            for _ in range(max_pages):
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
                if not batch:
                    break
                rows.extend(batch)
                if len(batch) < limit:
                    break
                cursor = batch[-1][0] + 1
            if not rows:
                raise RuntimeError("کندلی برنگشت")
            _count(ex_name)
            return rows, ex_name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"دریافت کندل‌های {symbol} ممکن نشد: {last_err}")


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
