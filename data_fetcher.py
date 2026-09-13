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
