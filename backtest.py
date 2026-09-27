# -*- coding: utf-8 -*-
"""
بک‌تست تکی (یک تنظیم مشخص) — نسخه‌ی ۲، روی موتور سریع.

تغییرات نسبت به نسخه‌ی قبل:
  - دیتا از کش محلی (market_data) میاد؛ فقط بار اول دانلود می‌شه.
  - همه‌ی نمادها هم‌زمان و به ترتیب زمان واقعی شبیه‌سازی می‌شن (سرمایه و سقف
    پوزیشن‌ها مشترک و درست).
  - SL/TP با سایه‌ی کندل فعال می‌شن (مثل سفارش واقعی صرافی)، با فرض بدبینانه.
  - سیگنال‌ها دقیقاً معادل کد زنده‌ان (tests/test_engine.py این رو تایید می‌کنه)،
    ولی ده‌ها برابر سریع‌تر، و برای همه‌ی استراتژی‌ها و حالت‌های ترکیب.
  - نتیجه علاوه بر کل بازه، جدا برای ۷۰٪ اول و ۳۰٪ آخر هم گزارش می‌شه تا ثبات
    عملکرد در زمان دیده بشه.
"""
import logging
import os
import threading
from datetime import datetime

import data_fetcher
import fast_backtest
import market_data
import sim_engine

log = logging.getLogger("backtest")

_cache_lock = threading.Lock()
_cache = {"obj": None}


def build_config(live_config, overrides=None):
    """یک نسخه‌ی مستقل از تنظیمات (برای این‌که تغییرات بک‌تست روی config زنده اثر نذاره)."""
    class Cfg:
        pass
    cfg = Cfg()
    for k in dir(live_config):
        if k.isupper():
            setattr(cfg, k, getattr(live_config, k))
    if overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)
    return cfg


def get_cache(cfg):
    with _cache_lock:
        if _cache["obj"] is None:
            _cache["obj"] = market_data.MarketDataCache(
                getattr(cfg, "DATA_CACHE_DIR", "data_cache"), cfg.EXCHANGE_TRY_ORDER,
                getattr(cfg, "DATA_FETCH_MAX_REQ_PER_SEC", 6), getattr(cfg, "DATA_FETCH_THREADS", 3))
        return _cache["obj"]


def run_backtest(symbols, days, live_config, overrides=None, progress_cb=None, timeframe=None):
    """
    اجرای کامل بک‌تست، برمی‌گردونه: (conn, meta)
    conn: دیتابیس SQLite در حافظه با همون ساختار جدول trades/equity/signal_log ربات زنده
    meta: جزئیات هر نماد + نتایج جداگانه‌ی بخش اول/دوم بازه
    """
    cfg = build_config(live_config, overrides)
    tf = timeframe or getattr(cfg, "TIMEFRAME", "15m")
    if tf in getattr(cfg, "TIMEFRAME_PROFILES", {}):
        # پروفایل همون تایم‌فریم (تایم‌فریم‌های تایید، کول‌داون، حد زمانی و ...) + بازنویسی‌های کاربر
        cfg = fast_backtest.profile_cfg(cfg, tf)
        for k, v in (overrides or {}).items():
            setattr(cfg, k, v)
        prof = cfg.TIMEFRAME_PROFILES[tf]
        days = min(int(days), int(prof["MAX_DAYS"]))
        symbols = list(symbols)[:int(prof["SCAN_SYMBOLS"])]
    started = datetime.utcnow().isoformat()
    symbols = [s for s in symbols if data_fetcher.is_symbol_allowed(s, cfg)]

    def cb(msg, frac=None):
        if progress_cb:
            progress_cb(msg if frac is None else f"{msg} ({int(frac * 100)}٪)")

    cache = get_cache(cfg)
    plan, data = fast_backtest.load_market_data(cache, symbols, days, cfg, cb,
                                                offline=os.environ.get("TRADINGBOT_OFFLINE") == "1")
    active = [a for a in cfg.ACTIVE_STRATEGIES]
    preps, sym_meta = fast_backtest.prepare_all(plan, data, symbols, cfg, cb, strategies_needed=active)
    del data
    cb("شبیه‌سازی سبد")
    res, P = fast_backtest.run_single(preps, symbols, cfg, record=True)
    conn = sim_engine.to_sqlite(res, cfg.VIRTUAL_BALANCE_START, P.trailing)

    split_ms = int(plan["start_ms"] + (plan["now_ms"] - plan["start_ms"]) * 0.7)
    segments = {
        "split_time": datetime.utcfromtimestamp(split_ms / 1000).isoformat(),
        "first_70": sim_engine.metrics(res["trades"], res["equity"], P.start_balance, None, split_ms),
        "last_30": sim_engine.metrics(res["trades"], res["equity"], P.start_balance, split_ms, None),
        "full": sim_engine.metrics(res["trades"], res["equity"], P.start_balance),
    }
    meta = {"symbols": sym_meta, "start_time": started, "end_time": datetime.utcnow().isoformat(),
            "engine": "fast-v3", "segments": segments, "cache_mb": cache.disk_usage_mb(),
            "timeframe": cfg.TIMEFRAME, "days": days, "entry_mode": getattr(cfg, "ENTRY_MODE", "limit")}
    meta["_cfg"] = cfg
    return conn, meta
