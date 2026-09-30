# -*- coding: utf-8 -*-
"""
اجرای اصلی ربات: هر ۱۵ دقیقه نمادها رو اسکن می‌کنه، سیگنال تولید می‌کنه،
معامله مجازی باز می‌کنه/می‌بندد، و یک داشبورد وب بالا می‌آره.

اجرا: python3 bot.py
"""
import io
import json
import logging
import os
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from datetime import datetime

import psutil
from flask import Flask, jsonify, render_template, request, send_file
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import config
import data_fetcher
import analysis
import strategies
import money
import paper_trader
import backtest
import backtest_analyzer
import fast_backtest
import market_data
import signals_engine
import sim_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tradingbot")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024 * 1024   # آپلود فایل دیتای تاریخی (تا ۳ گیگ)
conn = paper_trader.get_conn(config.DB_PATH)



def _apply_live_defaults_v18():
    """
    یک‌بار (اولین اجرای نسخه‌ی ۱۸): ربات زنده با استراتژی شخصی «خلاف جمعیت دیررس» و تنظیماتش
    (config.CONTRA_LIVE_DEFAULTS) + پاک کردن همه‌ی معاملات/لاگ‌های قبلی تا پنل از صفر شروع کنه.
    """
    if paper_trader.get_setting(conn, "defaults_v18", None) is not None:
        return
    for k, v in config.CONTRA_LIVE_DEFAULTS.items():
        paper_trader.set_setting(conn, k, v)
    for k in ("defaults_v14", "defaults_v16"):
        paper_trader.set_setting(conn, k, "1")
    try:
        cap = float(paper_trader.get_setting(conn, "initial_capital", config.VIRTUAL_BALANCE_START))
    except (TypeError, ValueError):
        cap = float(config.VIRTUAL_BALANCE_START)
    paper_trader.wipe_history(conn, cap)
    latest_analysis.clear()
    paper_trader.set_setting(conn, "defaults_v18", "1")
    log.info(f"[نسخه‌ی ۱۸] استراتژی شخصی «خلاف جمعیت» پیش‌فرض شد و تاریخچه پاک شد (سرمایه {cap}): "
             f"{config.CONTRA_LIVE_DEFAULTS}")


def _apply_live_defaults_v16():
    """یک‌بار: ربات زنده با «الگو + ساختار بازار» و تنظیمات پیش‌فرضش (config.PAT_LIVE_DEFAULTS)."""
    if paper_trader.get_setting(conn, "defaults_v16", None) is not None:
        return
    for k, v in config.PAT_LIVE_DEFAULTS.items():
        paper_trader.set_setting(conn, k, v)
    paper_trader.set_setting(conn, "defaults_v14", "1")
    paper_trader.set_setting(conn, "defaults_v16", "1")
    log.info(f"[نسخه‌ی ۱۶] تنظیمات پیش‌فرض «الگو + ساختار بازار» روی ربات زنده اعمال شد: {config.PAT_LIVE_DEFAULTS}")


def _apply_live_defaults_v14():
    """یک‌بار: ربات زنده با فیبوناچی «حرکت دوم» و تنظیمات پیش‌فرضش (config.FIB_LIVE_DEFAULTS)."""
    if paper_trader.get_setting(conn, "defaults_v14", None) is not None:
        return
    for k, v in config.FIB_LIVE_DEFAULTS.items():
        paper_trader.set_setting(conn, k, v)
    paper_trader.set_setting(conn, "defaults_v14", "1")
    log.info(f"[نسخه‌ی ۱۴] تنظیمات پیش‌فرض فیبوناچی روی ربات زنده اعمال شد: {config.FIB_LIVE_DEFAULTS}")


# پیش‌گرم‌کردن اندازه‌گیری CPU (اولین فراخوانی psutil.cpu_percent همیشه ۰ برمی‌گردونه)
psutil.cpu_percent(interval=None)

# آخرین وضعیت تحلیل هر نماد، برای نمایش در داشبورد
latest_analysis = {}
last_update_time = {"value": None}

# لیست فعلی نمادهای رصدشده (اگه DYNAMIC_SYMBOLS روشن باشه، خودکار آپدیت می‌شه)
active_symbols = {"list": list(config.SYMBOLS)}

# آخرین قیمت شناخته‌شده‌ی هر نماد — برای محاسبه‌ی سود/زیان لحظه‌ای پوزیشن‌های باز
latest_prices = {}


def compute_live_position_metrics(trade, current_price):
    """
    برای یک پوزیشن باز، سود/زیان لحظه‌ای (خام) و درصد پیشرفت به سمت TP/SL رو
    حساب می‌کنه. progress_pct بین -100 (دقیقاً روی حد ضرر) تا +100 (دقیقاً روی
    حد سود) هست؛ صفر یعنی دقیقاً روی نقطه‌ی ورود.
    """
    entry, sl, tp, size, side = trade["entry"], trade["sl"], trade["tp"], trade["size"], trade["side"]
    if current_price is None or size in (None, 0) or entry in (None, 0):
        return {"live_price": None, "unrealized_pnl": None, "unrealized_pnl_pct": None, "progress_pct": None}

    unrealized_pnl = (current_price - entry) * size if side == "LONG" else (entry - current_price) * size
    margin = trade.get("margin") or 0
    unrealized_pnl_pct = (unrealized_pnl / margin * 100) if margin else None

    if unrealized_pnl >= 0:
        reward_dist = abs(tp - entry)
        progress = (unrealized_pnl / (reward_dist * size) * 100) if reward_dist and size else 0
    else:
        risk_dist = abs(entry - sl)
        progress = -(abs(unrealized_pnl) / (risk_dist * size) * 100) if risk_dist and size else 0

    progress = max(-100, min(100, progress))

    return {
        "live_price": current_price,
        "unrealized_pnl": round(unrealized_pnl, 4),
        "unrealized_pnl_pct": round(unrealized_pnl_pct, 2) if unrealized_pnl_pct is not None else None,
        "progress_pct": round(progress, 1),
    }


def get_bot_settings():
    """تنظیمات قابل‌کنترل از پنل: وضعیت روشن/متوقف، درصد ریسک، سرمایه اولیه،
    سطح سخت‌گیری، تریلینگ استاپ، و استراتژی‌های فعال."""
    running = paper_trader.get_setting(conn, "bot_running", "1") == "1"
    risk_pct = float(paper_trader.get_setting(conn, "risk_pct", config.RISK_PER_TRADE_PCT))
    initial_capital = paper_trader.get_setting(conn, "initial_capital", None)
    if initial_capital is None:
        initial_capital = config.VIRTUAL_BALANCE_START
        paper_trader.set_setting(conn, "initial_capital", initial_capital)

    strictness = paper_trader.get_setting(conn, "strictness", config.DEFAULT_STRICTNESS)
    if strictness not in config.STRICTNESS_PRESETS:
        strictness = config.DEFAULT_STRICTNESS
    preset = config.STRICTNESS_PRESETS[strictness]

    trailing_enabled = paper_trader.get_setting(conn, "trailing_enabled",
                                                 "1" if config.USE_TRAILING_SL else "0") == "1"

    # در هر لحظه یک استراتژی فعاله (از پنل انتخاب می‌شه)
    # یک استراتژی یا ترکیب چندتا («a+b»، به ترتیب اولویت)
    strategy = paper_trader.get_setting(conn, "strategy", "+".join(config.ACTIVE_STRATEGIES))
    active_strategies = strategies.parse_strategy(strategy)
    strategy = "+".join(active_strategies)
    combine_mode = "any"
    trail_profile = paper_trader.get_setting(conn, "trail_profile", config.DEFAULT_TRAIL_PROFILE)
    if trail_profile not in config.TRAIL_PROFILES:
        trail_profile = config.DEFAULT_TRAIL_PROFILE
    try:
        risk_usd = float(paper_trader.get_setting(conn, "risk_usd", config.RISK_USD))
    except (TypeError, ValueError):
        risk_usd = float(config.RISK_USD)
    try:
        daily_loss = float(paper_trader.get_setting(conn, "daily_loss", config.DAILY_LOSS_LIMIT_USD))
    except (TypeError, ValueError):
        daily_loss = float(config.DAILY_LOSS_LIMIT_USD)
    try:
        cut_loss_r = float(paper_trader.get_setting(conn, "cut_loss_r", config.CUT_LOSS_R))
    except (TypeError, ValueError):
        cut_loss_r = float(config.CUT_LOSS_R)
    try:
        min_score = float(paper_trader.get_setting(conn, "min_score", config.WC_MIN_SCORE_PCT))
    except (TypeError, ValueError):
        min_score = float(config.WC_MIN_SCORE_PCT)

    def flag(key, default):
        return paper_trader.get_setting(conn, key, "1" if default else "0") == "1"

    min_sl = flag("min_sl", getattr(config, "MIN_SL_PCT", 0) > 0 or getattr(config, "MIN_SL_ATR_MULT", 0) > 0)
    room = flag("room_to_target", getattr(config, "REQUIRE_ROOM_TO_TARGET", False))
    btc_filter = flag("btc_filter", getattr(config, "BTC_REGIME_FILTER", False))
    long_only = flag("long_only", not getattr(config, "ALLOW_SHORT", True))
    htf = flag("htf_enabled", getattr(config, "USE_HTF_CONFIRMATION", False))
    timeframe = paper_trader.get_setting(conn, "timeframe", config.DEFAULT_TIMEFRAME)
    if timeframe not in config.TIMEFRAME_PROFILES:
        timeframe = config.DEFAULT_TIMEFRAME
    entry_mode = paper_trader.get_setting(conn, "entry_mode", config.ENTRY_MODE)
    if entry_mode not in ("limit", "market"):
        entry_mode = config.ENTRY_MODE

    return {
        "running": running,
        "risk_pct": risk_pct,
        "initial_capital": float(initial_capital),
        "strictness": strictness,
        "htf_min_agreement": preset["HTF_MIN_AGREEMENT"],
        "min_rr": preset["MIN_RISK_REWARD"],
        "trailing_enabled": trailing_enabled,
        "active_strategies": active_strategies,
        "combine_mode": combine_mode,
        "min_sl": min_sl,
        "room": room,
        "btc_filter": btc_filter,
        "long_only": long_only,
        "timeframe": timeframe,
        "entry_mode": entry_mode,
        "htf": htf,
        "min_score": min_score,
        "strategy": strategy,
        "trail_profile": trail_profile,
        "risk_usd": risk_usd,
        "risk_mode": config.RISK_MODE,
        "retest": paper_trader.get_setting(conn, "brk_retest", "1" if config.BRK_ENTRY == "retest" else "0") == "1",
        "early_exit": paper_trader.get_setting(conn, "early_exit", "1" if config.EARLY_EXIT else "0") == "1",
        "daily_loss": daily_loss,
        "cut_loss_r": cut_loss_r,
        "pat_structure": paper_trader.get_setting(conn, "pat_structure", config.PAT_STRUCTURE)
        if paper_trader.get_setting(conn, "pat_structure", config.PAT_STRUCTURE) in ("with", "off") else "with",
        "contra_btc": "off" if paper_trader.get_setting(conn, "contra_btc", config.CONTRA_BTC) == "off" else "against",
    }


def live_config_snapshot(settings=None):
    """تنظیمات فعلی ربات زنده به همون فرمتی که «مقایسه‌ی استراتژی‌ها» می‌فهمه."""
    st = settings or get_bot_settings()
    return {"timeframe": st["timeframe"], "active_strategies": st["active_strategies"],
            "combine_mode": st["combine_mode"], "min_score": st["min_score"],
            "free": (st["contra_btc"] == "off") if "contrarian_btc" in st["active_strategies"]
            else st["pat_structure"] == "off",
            "trail_profile": st["trail_profile"], "retest": st["retest"], "early_exit": st["early_exit"],
            "daily_loss": st["daily_loss"] > 0, "cut_loss_r": st["cut_loss_r"], "cut": st["cut_loss_r"],
            "strictness": st["strictness"], "trailing": st["trail_profile"] if st["trailing_enabled"] else False,
            "min_sl": st["min_sl"],
            "room": st["room"], "btc_filter": st["btc_filter"], "long_only": st["long_only"], "htf": st["htf"]}


def refresh_symbols_job():
    if not getattr(config, "DYNAMIC_SYMBOLS", False):
        return
    try:
        top, used_ex = data_fetcher.get_top_symbols(
            config.EXCHANGE_TRY_ORDER,
            quote=config.QUOTE_CURRENCY,
            top_n=config.TOP_N_SYMBOLS,
            exclude_keywords=config.EXCLUDE_KEYWORDS,
            cfg=config,
        )
        if top:
            active_symbols["list"] = top
            log.info(f"لیست نمادها آپدیت شد: {len(top)} نماد (از {used_ex})")
    except Exception as e:
        log.warning(f"آپدیت لیست نمادها ناموفق بود، لیست قبلی حفظ می‌شه: {e}")


def fetch_symbol_df(symbol, timeframe=None, limit=None):
    """فقط کندل‌های بسته‌شده (دقیقاً همون چیزی که بک‌تست هم می‌بینه)."""
    timeframe = timeframe or config.TIMEFRAME
    limit = limit or config.CANDLE_LIMIT
    try:
        df, used_ex = data_fetcher.fetch_closed_ohlcv(
            symbol, timeframe, limit, config.EXCHANGE_TRY_ORDER
        )
        return df
    except Exception as e:
        log.warning(f"دریافت دیتای {symbol} ({timeframe}) ناموفق بود: {e}")
        return None


def get_effective_cfg(settings):
    """یک نسخه‌ی موقت از تنظیمات که سطح سخت‌گیری و استراتژی‌های انتخاب‌شده از پنل
    روش اعمال شده — بدون این‌که به config ماژول اصلی دست بزنه."""
    overrides = {
        "MIN_RISK_REWARD": settings["min_rr"],
        "ACTIVE_STRATEGIES": settings["active_strategies"],
        "STRATEGY_COMBINE_MODE": settings["combine_mode"],
        "WC_MIN_SCORE_PCT": settings["min_score"],
        "MIN_SL_PCT": config.TEST_MIN_SL_PCT if settings["min_sl"] else 0.0,
        "MIN_SL_ATR_MULT": config.TEST_MIN_SL_ATR_MULT if settings["min_sl"] else 0.0,
        "REQUIRE_ROOM_TO_TARGET": settings["room"],
        "ENTRY_MODE": settings["entry_mode"],
        "TRAIL_PROFILE": settings["trail_profile"],
        "RISK_USD": settings["risk_usd"],
        "BRK_ENTRY": "retest" if settings["retest"] else "close",
        "EARLY_EXIT": settings["early_exit"],
        "DAILY_LOSS_LIMIT_USD": settings["daily_loss"],
        "CUT_LOSS_R": settings["cut_loss_r"],
        "PAT_STRUCTURE": settings.get("pat_structure", config.PAT_STRUCTURE),
        "CONTRA_BTC": settings.get("contra_btc", config.CONTRA_BTC),
    }
    # پروفایل تایم‌فریم انتخاب‌شده (تایم‌فریم‌های تایید، کول‌داون، حد زمانی، ...)
    cfg = fast_backtest.profile_cfg(config, settings["timeframe"])
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# روند BTC برای فیلتر رژیم بازار — هر اسکن یک‌بار حساب می‌شه، نه برای هر نماد
btc_regime_state = {"trend": None, "time": 0}


def get_btc_regime(cfg):
    now = time.time()
    tf = getattr(cfg, "BTC_REGIME_TIMEFRAME", config.BTC_REGIME_TIMEFRAME)
    if btc_regime_state["trend"] is not None and btc_regime_state.get("tf") == tf \
            and now - btc_regime_state["time"] < 60:
        return btc_regime_state["trend"]
    df = fetch_symbol_df(config.BTC_REGIME_SYMBOL, timeframe=tf, limit=config.HTF_CANDLE_LIMIT)
    trend = None
    if df is not None and len(df) >= config.SWING_ORDER * 2 + 5:
        trend = analysis.trend_from_df(df, swing_order=config.SWING_ORDER)
    btc_regime_state.update({"trend": trend, "time": now, "tf": tf})
    return trend


def pair_trend_live(symbol, cfg):
    """روند داو نمودار ارز÷BTC (مثل موتور بک‌تست: آخرین HTF_CANDLE_LIMIT کندل هر دو، ادغام روی زمان)."""
    tf = getattr(cfg, "BTC_REGIME_TIMEFRAME", config.BTC_REGIME_TIMEFRAME)
    sdf = fetch_symbol_df(symbol, timeframe=tf, limit=config.HTF_CANDLE_LIMIT)
    bdf = fetch_symbol_df(config.BTC_REGIME_SYMBOL, timeframe=tf, limit=config.HTF_CANDLE_LIMIT)
    if sdf is None or bdf is None:
        return None
    rdf = signals_engine.pair_ratio_df(sdf, bdf)
    if len(rdf) < config.SWING_ORDER * 2 + 5:
        return None
    return analysis.trend_from_df(rdf, swing_order=config.SWING_ORDER)


def check_htf_confirmation(symbol, side, min_agreement, htf_timeframes=None, cfg=None, preset=None):
    """
    تایید چند-تایم‌فریمی: فقط برای سیگنال‌های کاندید صدا زده می‌شه (نه برای هر نماد)،
    پس هزینه‌ی شبکه‌اش تقریباً ناچیزه. روند داو رو توی تایم‌فریم‌های بالاتر چک می‌کنه.
    وزن‌دار (HTF_WEIGHTED): تایم‌فریم بالاتر وزن بیشتر + نمودار ارز÷BTC با وزن BTC_PAIR_WEIGHT؛
    خروجی agree = درصد وزن موافق. وگرنه: شمارش ساده‌ی تایم‌فریم‌های موافق.
    """
    agree, disagree, neutral = 0, 0, 0
    checked = []
    trends = {}
    wanted_trend = "uptrend" if side == "LONG" else "downtrend"
    opposite_trend = "downtrend" if side == "LONG" else "uptrend"

    htf_timeframes = htf_timeframes or config.HTF_TIMEFRAMES
    for tf in htf_timeframes:
        df = fetch_symbol_df(symbol, timeframe=tf, limit=config.HTF_CANDLE_LIMIT)
        if df is None or len(df) < (config.SWING_ORDER * 2 + 5):
            continue
        trend = analysis.trend_from_df(df, swing_order=config.SWING_ORDER)
        trends[tf] = trend
        checked.append(f"{tf}:{trend}")
        if trend == wanted_trend:
            agree += 1
        elif trend == opposite_trend:
            disagree += 1
        else:
            neutral += 1

    if cfg is not None and getattr(cfg, "HTF_WEIGHTED", False) and preset is not None:
        pw = sim_engine.pair_weight_for(symbol, cfg)
        pair = pair_trend_live(symbol, cfg) if pw > 0 else None
        pct = sim_engine.htf_weighted_pct(trends, htf_timeframes, side, pair, pw)
        req_l, req_s = sim_engine.htf_required_pct(preset, config.SHORT_EXTRA_HTF_AGREEMENT)
        req = req_s if side == "SHORT" else req_l
        ok = pct >= req - 1e-9
        pair_txt = f" | ارز/BTC: {pair or 'نامعلوم'} (وزن {pw:g})" if pw > 0 else ""
        detail = f"وزن موافق={pct:.0f}٪ (حداقل لازم={req:.0f}٪){pair_txt} ({', '.join(checked)})"
        return ok, detail, int(round(pct))

    ok = agree >= min_agreement
    detail = f"موافق={agree}/{len(htf_timeframes)} (حداقل لازم={min_agreement}) مخالف={disagree} خنثی={neutral} ({', '.join(checked)})"
    return ok, detail, agree


def compute_xs_picks(settings):
    """
    مومنتوم نسبی هفتگی: فقط بعد از بسته‌شدن کندل روزانه‌ی یکشنبه. XS_UNIVERSE نماد پرحجم رتبه‌بندی
    می‌شن (با همون strategies.xs_rank موتور بک‌تست). خروجی {symbol: "LONG"/"SHORT"}.
    """
    if "xs_momentum" not in settings["active_strategies"] or settings["timeframe"] != "1d":
        return {}
    cfg = get_effective_cfg(settings)
    universe = [s_ for s_ in active_symbols["list"] if data_fetcher.is_symbol_allowed(s_, config)][:config.XS_UNIVERSE]
    rets = {}
    last_open = None
    for sym in universe:
        df = fetch_symbol_df(sym, timeframe="1d", limit=config.XS_LOOKBACK_DAYS + 5)
        if df is None or len(df) <= config.XS_LOOKBACK_DAYS:
            continue
        lo = int(df["timestamp"].values.astype("datetime64[ms]").astype("int64")[-1])
        last_open = lo if last_open is None else max(last_open, lo)
        rets[sym] = strategies.xs_return(df["close"].values, config.XS_LOOKBACK_DAYS)
    if last_open is None or not strategies.xs_is_rebalance(last_open):
        return {}
    picks = strategies.xs_rank(rets, cfg)
    if picks:
        log.info(f"[مومنتوم هفتگی] رتبه‌بندی {len(rets)} نماد: " +
                 "، ".join(f"{k} {'خرید' if v == 'LONG' else 'فروش'}" for k, v in picks.items()))
    return picks


def scan_symbol(symbol, settings, xs_picks=None):
    if not data_fetcher.is_symbol_allowed(symbol, config):
        return  # فیلتر مطلق: طلا و استیبل‌کوین به استیبل‌کوین هیچ‌وقت معامله نمی‌شن

    cfg = get_effective_cfg(settings)
    # فقط کندل‌های بسته‌شده؛ چند کندل اضافه برای حالت ترکیب «تاییدی»
    df = fetch_symbol_df(symbol, timeframe=cfg.TIMEFRAME, limit=config.CANDLE_LIMIT + config.CONFIRM_LOOKBACK_BARS)
    if df is None or len(df) < (config.SWING_ORDER * 2 + 5):
        return
    result = strategies.generate_combined_signal(df, cfg)
    # مومنتوم هفتگی (اگه این نماد در رتبه‌بندی این هفته انتخاب شده) — اولویت به ترتیب استراتژی‌های فعال
    active = settings["active_strategies"]
    if xs_picks and symbol in xs_picks and "xs_momentum" in active:
        window = df.iloc[-config.CANDLE_LIMIT:]
        xs_sig = strategies.generate_xs_signal(window, cfg, xs_picks[symbol])
        cur = result.get("signal")
        use_xs = False
        if xs_sig:
            if cur is None:
                use_xs = True
            else:
                cs = cur.get("strategy")
                use_xs = cs not in active or active.index("xs_momentum") < active.index(cs)
        if use_xs:
            result = dict(result, signal=xs_sig)
    latest_analysis[symbol] = result
    if symbol not in latest_prices:
        latest_prices[symbol] = result["price"]
    # بستن پوزیشن‌ها اینجا انجام نمی‌شه؛ price_check_job هر دقیقه با کندل‌های ۱ دقیقه‌ای
    # (با سایه‌ها) این کار رو دقیق‌تر انجام می‌ده.

    if not result["signal"]:
        return

    sig = result["signal"]
    strategy_name = sig.get("strategy", "unknown")

    score = sig.get("score")
    reasons = "، ".join(strategies.reason_labels(sig.get("reasons", [])))
    if sig.get("vol_ratio"):
        reasons += f" ({sig['vol_ratio']:g}× میانگین)"

    def reject(reason, htf_agree=None):
        paper_trader.log_signal(conn, symbol, sig["side"], strategy_name, sig["entry"], sig["sl"],
                                 sig["tp"], sig["rr"], htf_agree, opened=False, rejection_reason=reason,
                                 score=score, reasons=reasons)

    if not settings["running"]:
        return  # وقتی متوقفه، حتی سیگنال رد‌شده رو لاگ نمی‌کنیم (فقط پایشه، تصمیمی نمی‌گیره)

    if paper_trader.has_open_trade(conn, symbol):
        return  # از قبل پوزیشن باز داره؛ این یه سیگنال جدید مستقل نیست که رد بشه

    # ۱) کول‌داون (مومنتوم هفتگی کول‌داون نداره: هر هفته دوباره رتبه‌بندی می‌شه)
    if strategy_name != "xs_momentum" and paper_trader.is_in_cooldown(conn, symbol, cfg.COOLDOWN_HOURS):
        reject("cooldown")
        return

    # ۱.۵) حد ضرر روزانه: اگه امروز (UTC) به اندازه‌ی حد، ضرر تحقق‌یافته داشتیم، تا فردا معامله‌ی جدید نه
    if settings["daily_loss"] > 0:
        day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        if paper_trader.realized_pnl_since(conn, day_start) <= -settings["daily_loss"]:
            reject("daily_loss_limit")
            return

    # ۲) جهت مجاز و فیلتر رژیم BTC (هر دو اختیاری، از پنل)
    if sig["side"] == "SHORT" and settings["long_only"]:
        reject("side_disabled")
        return
    if strategy_name == strategies.CONTRA_KEY and getattr(cfg, "CONTRA_BTC", "against") != "off":
        # استراتژی شخصی: فقط خلاف روند داو BTC (همون قانون موتور بک‌تست)
        if not strategies.contra_btc_allows(sig["side"], get_btc_regime(cfg)):
            reject("btc_regime")
            return
    elif settings["btc_filter"]:
        btc_trend = get_btc_regime(cfg)
        if (sig["side"] == "LONG" and btc_trend == "downtrend") or (sig["side"] == "SHORT" and btc_trend == "uptrend"):
            reject("btc_regime")
            return

    # ۳) تایید چند-تایم‌فریمی (اختیاری، از پنل) با سطح سخت‌گیری فعلی (SHORT به تاییدیه‌ی بیشتری نیاز داره)
    htf_detail, htf_agree = "خاموش", None
    if settings["htf"]:
        req_long, req_short = sim_engine.htf_required(settings["htf_min_agreement"], len(cfg.HTF_TIMEFRAMES),
                                                      config.SHORT_EXTRA_HTF_AGREEMENT)
        min_agreement = req_short if sig["side"] == "SHORT" else req_long
        htf_ok, htf_detail, htf_agree = check_htf_confirmation(symbol, sig["side"], min_agreement,
                                                               cfg.HTF_TIMEFRAMES, cfg=cfg,
                                                               preset=settings["htf_min_agreement"])
        if not htf_ok:
            log.info(f"[رد شد - عدم تایید تایم‌فریم بالاتر] {symbol} {sig['side']} | {htf_detail}")
            reject("htf_disagreement", htf_agree)
            return

    # ۴) ثبت سفارش (لیمیت در انتظار) یا ورود بازار — با رعایت سرمایه، لوریج ایمن و سقف تنوع
    tf_ms = market_data.TF_MS[cfg.TIMEFRAME]
    # ورود با پولبک همیشه لیمیته (روی سطح شکسته‌شده، با مهلت خودش)
    entry_taker = (settings["entry_mode"] == "market" or bool(sig.get("market_only"))) and not sig.get("limit_only")
    wait_bars = int(sig.get("wait_bars") or cfg.LIMIT_WAIT_BARS)
    entry_price = sig["entry"]
    if entry_taker:
        px = latest_1m_price(symbol)
        if px is None:
            reject("no_price")
            return
        slip = config.TAKER_SLIPPAGE_PCT / 100
        entry_price = px * (1 + slip) if sig["side"] == "LONG" else px * (1 - slip)
        bad = (entry_price <= sig["sl"] or entry_price >= sig["tp"]) if sig["side"] == "LONG" \
            else (entry_price >= sig["sl"] or entry_price <= sig["tp"])
        if bad:
            reject("gap_past_level")
            return
    candle_close_ms = int(df["timestamp"].values.astype("datetime64[ms]").astype("int64")[-1]) + tf_ms
    trade_res = paper_trader.open_trade(
        conn, symbol, sig["side"], entry_price, sig["sl"], sig["tp"],
        settings["risk_pct"], settings["initial_capital"],
        min_notional=config.MIN_NOTIONAL_USD, max_open_positions=config.MAX_OPEN_POSITIONS,
        max_leverage=config.MAX_LEVERAGE, leverage_safety_mult=config.LEVERAGE_SAFETY_MULTIPLIER,
        position_pct_cap=config.MAX_POSITION_PCT_OF_CAPITAL,
        # روندگیر همیشه با تریلینگ شاندلیر خودش مدیریت می‌شه (حد سود ثابت نداره)
        trailing_enabled=(strategy_name.startswith("trend_follow") or
                          (settings["trailing_enabled"] and strategy_name != "xs_momentum")),
        strategy_name=strategy_name,
        pending=not entry_taker, expire_ts=candle_close_ms + wait_bars * tf_ms,
        entry_taker=entry_taker,
        max_hold_min=(config.XS_HOLD_DAYS * 1440 if strategy_name == "xs_momentum" else cfg.MAX_HOLD_MINUTES),
        timeframe=cfg.TIMEFRAME, score=score,
        risk_usd=settings["risk_usd"] if config.RISK_MODE == "usd" else None,
        fees=money.fee_fracs(config.MAKER_FEE_PCT, config.TAKER_FEE_PCT, config.TAKER_SLIPPAGE_PCT, entry_taker)[:2],
    )

    paper_trader.log_signal(conn, symbol, sig["side"], strategy_name, sig["entry"], sig["sl"], sig["tp"],
                             sig["rr"], htf_agree, opened=trade_res["opened"],
                             rejection_reason=None if trade_res["opened"] else trade_res["reason"],
                             score=score, reasons=reasons)

    if trade_res["opened"]:
        cap_note = " (حجم به‌خاطر سقف تنوع/سرمایه‌ی آزاد کوچک‌تر شد)" if trade_res.get("capped") else ""
        trail_on = strategy_name.startswith("trend_follow") or (settings["trailing_enabled"] and strategy_name != "xs_momentum")
        trail_note = " | تریلینگ: روشن" if trail_on else ""
        if strategy_name == "xs_momentum":
            trail_note += f" | نگه‌داری {config.XS_HOLD_DAYS} روز"
        risk_txt = f"ضرر خالص اگه SL بخوره=${trade_res.get('risk_amount', 0):.2f}"
        log.info(
            f"[سیگنال جدید ✅] {symbol} {sig['side']} {strategy_name} امتیاز={score} ({reasons}) ورود={sig['entry']:.4f} "
            f"حدضرر={sig['sl']:.4f} حدسود={sig['tp']:.4f} R:R قیمتی={sig['rr']:.2f} "
            f"{risk_txt} لوریج={trade_res['leverage']}x "
            f"مارجین=${trade_res['margin']:.2f} ارزش‌پوزیشن=${trade_res['notional']:.2f}{cap_note}{trail_note} | HTF: {htf_detail}"
        )
    else:
        log.info(f"[سیگنال رد شد - {trade_res['reason']}] {symbol} {sig['side']}")


def latest_1m_price(symbol):
    try:
        df, _ = data_fetcher.fetch_ohlcv_with_fallback(symbol, "1m", 2, config.EXCHANGE_TRY_ORDER)
        return float(df["close"].iloc[-1])
    except Exception:
        return None


def full_scan_job():
    """هر دقیقه صدا زده می‌شه؛ فقط وقتی یک کندلِ تایم‌فریم انتخاب‌شده تازه بسته شده، اسکن می‌کنه."""
    settings = get_bot_settings()
    tf = settings["timeframe"]
    tf_ms = market_data.TF_MS[tf]
    now_min = int(time.time() // 60) * 60_000
    if now_min % tf_ms != 0:
        return   # هنوز کندل جدیدی از این تایم‌فریم بسته نشده
    n_symbols = int(config.TIMEFRAME_PROFILES[tf]["SCAN_SYMBOLS"])
    try:
        xs_picks = compute_xs_picks(settings)
    except Exception as e:
        log.warning(f"رتبه‌بندی مومنتوم هفتگی با خطا مواجه شد: {e}")
        xs_picks = {}
    for symbol in active_symbols["list"][:n_symbols]:
        try:
            scan_symbol(symbol, settings, xs_picks)
        except Exception as e:
            log.warning(f"اسکن {symbol} با خطا مواجه شد: {e}")
    last_update_time["value"] = datetime.utcnow().isoformat()


def price_check_job():
    """
    هر دقیقه: برای نمادهایی که پوزیشن باز دارن، کندل‌های ۱ دقیقه‌ای جدید (از آخرین کندل
    پردازش‌شده) گرفته و با همون قانون بک‌تست (paper_trader.step_bar، با سایه‌ی کندل)
    روی پوزیشن اعمال می‌شن. اگه ربات مدتی خاموش بوده، کندل‌های اون مدت هم جبران می‌شن.
    """
    settings = get_bot_settings()
    open_symbols = paper_trader.get_open_symbols(conn)
    now_ms = int(time.time() * 1000)
    for symbol in open_symbols:
        try:
            rows = conn.execute(
                "SELECT open_time, last_bar_ts FROM trades WHERE symbol=? AND status IN ('OPEN','PENDING')",
                (symbol,)
            ).fetchall()
            since = now_ms - 5 * 60_000
            for open_time, last_bar_ts in rows:
                if last_bar_ts:
                    since = min(since, int(last_bar_ts) + 1)
                elif open_time:
                    o_ms = paper_trader.utc_ms(datetime.fromisoformat(open_time))
                    since = min(since, o_ms - (o_ms % 60_000))
            since = max(since, now_ms - 3 * 1000 * 60_000)
            raw, _ = data_fetcher.fetch_since(symbol, "1m", since, config.EXCHANGE_TRY_ORDER)
            if not raw:
                continue
            latest_prices[symbol] = float(raw[-1][4])  # قیمت لحظه‌ای (کندل در حال تشکیل)
            closed = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]))
                      for r in raw if int(r[0]) + 60_000 <= now_ms]

            if config.LOG_POSITION_PRICE_HISTORY:
                for trade_id in paper_trader.get_open_trade_ids(conn, symbol):
                    paper_trader.log_price_snapshot(conn, trade_id, latest_prices[symbol])
                paper_trader.commit(conn)

            prof = config.TRAIL_PROFILES[settings["trail_profile"]]
            paper_trader.process_bars(
                conn, symbol, closed, settings["initial_capital"],
                prof["ladder"], prof["beyond"],
                config.MAKER_FEE_PCT, config.TAKER_FEE_PCT, config.TAKER_SLIPPAGE_PCT,
                config.FUNDING_PCT_PER_8H, now_ms=now_ms, trail_floor=config.TRAIL_BREAKEVEN_FLOOR,
                early_bars=config.EARLY_EXIT_BARS if settings["early_exit"] else 0,
                early_min_r=config.EARLY_EXIT_MIN_R,
                profiles={"trend_follow": (config.TREND_TRAIL["ladder"], config.TREND_TRAIL["beyond"],
                                           config.TREND_TRAIL.get("floor_r"))},
                cut_r=settings["cut_loss_r"],
            )
        except Exception as e:
            log.warning(f"چک قیمت {symbol} با خطا مواجه شد: {e}")


scheduler = BackgroundScheduler()
scheduler.add_job(refresh_symbols_job, "interval", hours=config.SYMBOL_REFRESH_HOURS,
                   next_run_time=datetime.now())
# اسکن هر دقیقه ۸ ثانیه بعد از شروع دقیقه صدا زده می‌شه، ولی خودش فقط وقتی یک کندل از
# تایم‌فریم انتخاب‌شده (۱ دقیقه تا ۴ ساعت) تازه بسته شده، کار می‌کنه.
scheduler.add_job(full_scan_job, CronTrigger(second=8), max_instances=1, coalesce=True)
scheduler.add_job(price_check_job, "interval", minutes=config.PRICE_CHECK_INTERVAL_MINUTES,
                   max_instances=1, coalesce=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    settings = get_bot_settings()
    stats = paper_trader.get_stats(conn, settings["initial_capital"])
    balance = round(paper_trader.get_balance(conn, settings["initial_capital"]), 2)
    locked_capital = round(paper_trader.get_locked_capital(conn), 2)
    available_capital = round(balance - locked_capital, 2)
    open_trades = paper_trader.get_open_trades(conn)
    for t in open_trades:
        t.update(compute_live_position_metrics(t, latest_prices.get(t["symbol"])))

    closed_page = max(1, int(request.args.get("closed_page", 1)))
    closed_page_size = min(100, max(5, int(request.args.get("closed_page_size", 20))))
    closed_trades, closed_total = paper_trader.get_closed_trades(
        conn, limit=closed_page_size, offset=(closed_page - 1) * closed_page_size
    )
    equity_curve = paper_trader.get_equity_curve(conn)

    initial_capital = settings["initial_capital"]
    total_return_pct = round((balance - initial_capital) / initial_capital * 100, 2) if initial_capital else 0.0

    symbols_view = []
    for symbol in active_symbols["list"]:
        a = latest_analysis.get(symbol, {})
        if not a:
            continue  # هنوز اسکن نشده، فعلاً نشونش نده
        symbols_view.append({
            "symbol": symbol,
            "trend": a.get("trend", "در حال بارگذاری"),
            "price": a.get("price"),
            "support": a.get("support"),
            "resistance": a.get("resistance"),
            "signal": a.get("signal"),
            "score_long": a.get("score_long"),
            "score_short": a.get("score_short"),
            "box_top": a.get("box_top"),
            "box_bottom": a.get("box_bottom"),
            "vol_ratio": a.get("vol_ratio"),
        })

    system_stats = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": psutil.virtual_memory().percent,
    }

    return jsonify({
        "balance": balance,
        "locked_capital": locked_capital,
        "available_capital": available_capital,
        "initial_capital": initial_capital,
        "total_return_pct": total_return_pct,
        "is_profitable": balance >= initial_capital,
        "bot_running": settings["running"],
        "risk_pct": settings["risk_pct"],
        "allowed_risk_levels": config.ALLOWED_RISK_LEVELS,
        "open_positions_count": len(open_trades),
        "max_open_positions": config.MAX_OPEN_POSITIONS,
        "min_rr": settings["min_rr"],
        "maker_fee_pct": config.MAKER_FEE_PCT,
        "taker_fee_pct": config.TAKER_FEE_PCT,
        "taker_slippage_pct": config.TAKER_SLIPPAGE_PCT,
        "max_leverage": config.MAX_LEVERAGE,
        "max_position_pct": config.MAX_POSITION_PCT_OF_CAPITAL,
        "htf_enabled": settings["htf"],
        "htf": settings["htf"],
        "htf_timeframes": config.HTF_TIMEFRAMES,
        "strictness": settings["strictness"],
        "htf_min_agreement": settings["htf_min_agreement"],
        "short_extra_htf_agreement": config.SHORT_EXTRA_HTF_AGREEMENT,
        "excluded_gold_tokens": config.EXCLUDE_GOLD_TOKENS,
        "excluded_stablecoin_bases": config.STABLECOIN_BASES,
        "strictness_presets": {k: v["label"] for k, v in config.STRICTNESS_PRESETS.items()},
        "trailing_enabled": settings["trailing_enabled"],
        "trailing_ladder": config.TRAIL_PROFILES[settings["trail_profile"]]["ladder"],
        "trailing_beyond_r": config.TRAIL_PROFILES[settings["trail_profile"]]["beyond"],
        "trail_profile": settings["trail_profile"],
        "trail_profiles": {k: v["label"] for k, v in config.TRAIL_PROFILES.items()},
        "strategy": settings["strategy"],
        "pat_structure": settings["pat_structure"],
        "pat_info": {"count": len(__import__("patterns").pattern_names(config.PAT_SET)), "min_sl": config.PAT_MIN_SL_ATR},
        "contra_info": {"count": len(__import__("patterns").pattern_names("reversal")),
                        "min_sl": config.CONTRA_MIN_SL_ATR,
                        "btc_tf": {"1w": "هفتگی", "1d": "روزانه", "4h": "۴ ساعته", "1h": "۱ ساعته"}.get(
                            config.PROFILE_BTC_TIMEFRAME.get(settings["timeframe"], "4h"),
                            config.PROFILE_BTC_TIMEFRAME.get(settings["timeframe"], "4h"))},
        "fib_info": {"impulse": config.FIB_MIN_IMPULSE_ATR, "zone": f"{config.FIB_ZONE_LO:g} تا {config.FIB_ZONE_HI:g}",
                     "vp": config.FIB_VP_BARS},
        "strategy_labels": {**{k: v["short"] for k, v in strategies.STRATEGY_REGISTRY.items()},
                            **strategies.STRATEGY_COMBOS},
        "cut_loss_r": settings["cut_loss_r"],
        "trend_info": {"bars": config.TR_BREAKOUT_BARS, "ma": config.TR_MA, "sl_atr": config.TR_SL_ATR,
                       "trail": config.TREND_TRAIL["label"]},
        "xs_info": {"lookback": config.XS_LOOKBACK_DAYS, "top_k": config.XS_TOP_K, "hold": config.XS_HOLD_DAYS,
                    "short": config.XS_SHORT, "sl_atr": config.XS_SL_ATR, "universe": config.XS_UNIVERSE},
        "risk_usd": settings["risk_usd"],
        "risk_mode": config.RISK_MODE,
        "retest": settings["retest"],
        "early_exit": settings["early_exit"],
        "daily_loss": settings["daily_loss"],
        "early_exit_bars": config.EARLY_EXIT_BARS,
        "early_exit_min_r": config.EARLY_EXIT_MIN_R,
        "retest_wait_bars": config.BRK_RETEST_WAIT_BARS,
        "htf_weighted": config.HTF_WEIGHTED,
        "btc_pair_weight": config.BTC_PAIR_WEIGHT,
        "breakout": {"box_bars": config.BRK_BOX_BARS, "vol_mult": config.BRK_VOL_MULT,
                     "main_trend": config.BRK_MAIN_TREND},
        "active_strategies": settings["active_strategies"],
        "combine_mode": settings["combine_mode"],
        "min_score": settings["min_score"],
        "score_choices": config.WC_SCORE_CHOICES,
        "weights": config.WC_WEIGHTS,
        "component_labels": strategies.COMPONENT_LABELS,
        "min_sl": settings["min_sl"],
        "room": settings["room"],
        "btc_filter": settings["btc_filter"],
        "long_only": settings["long_only"],
        "timeframe": settings["timeframe"],
        "entry_mode": settings["entry_mode"],
        "timeframe_profiles": {k: {"label": v["label"], "max_days": v["MAX_DAYS"], "symbols": v["SCAN_SYMBOLS"],
                                   "htf": v["HTF_TIMEFRAMES"], "max_hold_min": v["MAX_HOLD_MINUTES"]}
                               for k, v in config.TIMEFRAME_PROFILES.items()},
        "funding_pct_8h": config.FUNDING_PCT_PER_8H,
        "test_min_sl_pct": config.TEST_MIN_SL_PCT,
        "test_min_sl_atr": config.TEST_MIN_SL_ATR_MULT,
        "confirm_lookback_bars": config.CONFIRM_LOOKBACK_BARS,
        "available_strategies": {k: v["label"] for k, v in strategies.STRATEGY_REGISTRY.items()},
        "system": system_stats,
        "bot_version": config.BOT_VERSION,
        "stats": stats,
        "symbols": symbols_view,
        "open_trades": open_trades,
        "closed_trades": closed_trades,
        "closed_trades_total": closed_total,
        "closed_page": closed_page,
        "closed_page_size": closed_page_size,
        "equity_curve": [{"time": t, "balance": b} for t, b in equity_curve],
        "last_update": last_update_time["value"],
        "symbol_count": min(len(active_symbols["list"]),
                            config.TIMEFRAME_PROFILES[settings["timeframe"]]["SCAN_SYMBOLS"]),
        "symbols_scanned": len(symbols_view),
    })


@app.route("/api/signal_log")
def api_signal_log():
    page = max(1, int(request.args.get("page", 1)))
    page_size = min(200, max(5, int(request.args.get("page_size", 60))))
    only_rejected = request.args.get("only_rejected", "").lower() == "true"
    signals, total = paper_trader.get_signal_log(
        conn, limit=page_size, offset=(page - 1) * page_size, only_rejected=only_rejected
    )
    return jsonify({"ok": True, "signals": signals, "total": total, "page": page, "page_size": page_size})


@app.route("/api/trade_price_history/<int:trade_id>")
def api_trade_price_history(trade_id):
    return jsonify({"ok": True, "history": paper_trader.get_price_history(conn, trade_id)})


@app.route("/api/control", methods=["POST"])
def api_control():
    """
    کنترل از پنل: روشن/متوقف کردن معامله‌گیری و تغییر درصد ریسک.
    عمداً «سرمایه اولیه» اینجا نیست — اون یک عملیات جدا و آگاهانه‌ست (/api/reset_capital)
    چون خط پایه‌ی محاسبه‌ی بازده رو عوض می‌کنه و نباید به‌صورت جانبی اجرا بشه.
    """
    body = request.get_json(force=True, silent=True) or {}

    if "running" in body:
        paper_trader.set_setting(conn, "bot_running", "1" if body["running"] else "0")
        log.info(f"[کنترل پنل] وضعیت ربات: {'روشن' if body['running'] else 'متوقف (فقط پایش)'}")

    if "risk_pct" in body:
        try:
            risk_val = float(body["risk_pct"])
            if risk_val in config.ALLOWED_RISK_LEVELS:
                paper_trader.set_setting(conn, "risk_pct", risk_val)
                log.info(f"[کنترل پنل] درصد ریسک هر معامله: {risk_val}%")
            else:
                return jsonify({"ok": False, "error": "سطح ریسک مجاز نیست"}), 400
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "مقدار ریسک نامعتبر است"}), 400

    if "strictness" in body:
        if body["strictness"] in config.STRICTNESS_PRESETS:
            paper_trader.set_setting(conn, "strictness", body["strictness"])
            log.info(f"[کنترل پنل] سطح سخت‌گیری: {body['strictness']}")
        else:
            return jsonify({"ok": False, "error": "سطح سخت‌گیری نامعتبر است"}), 400

    if "trailing_enabled" in body:
        paper_trader.set_setting(conn, "trailing_enabled", "1" if body["trailing_enabled"] else "0")
        log.info(f"[کنترل پنل] تریلینگ استاپ: {'روشن' if body['trailing_enabled'] else 'خاموش'} "
                 f"(فقط روی پوزیشن‌های جدید اثر داره)")

    if "strategy" in body:
        parts = str(body["strategy"]).split("+")
        if parts and all(p_ in strategies.STRATEGY_REGISTRY for p_ in parts):
            paper_trader.set_setting(conn, "strategy", "+".join(parts))
            log.info(f"[کنترل پنل] استراتژی: {body['strategy']}")
        else:
            return jsonify({"ok": False, "error": "استراتژی نامعتبر است"}), 400
    if body.get("pat_structure") in ("with", "off"):
        paper_trader.set_setting(conn, "pat_structure", body["pat_structure"])
        log.info(f"[کنترل پنل] ساختار بازار برای الگوها: {body['pat_structure']}")
    if "cut_loss_r" in body:
        try:
            cr = float(body["cut_loss_r"])
            if not 0 <= cr < 1:
                raise ValueError
            paper_trader.set_setting(conn, "cut_loss_r", cr)
            log.info(f"[کنترل پنل] بستن در ‎-{cr:g}R" if cr else "[کنترل پنل] بستن در ‎-xR: خاموش")
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "مقدار «بستن در ‎-xR» باید بین ۰ و ۰.۹۹ باشه"}), 400
    if "trail_profile" in body:
        if body["trail_profile"] in config.TRAIL_PROFILES:
            paper_trader.set_setting(conn, "trail_profile", body["trail_profile"])
            log.info(f"[کنترل پنل] پروفایل تریلینگ: {body['trail_profile']}")
        else:
            return jsonify({"ok": False, "error": "پروفایل تریلینگ نامعتبر است"}), 400
    for key, setting in (("retest", "brk_retest"), ("early_exit", "early_exit")):
        if key in body:
            paper_trader.set_setting(conn, setting, "1" if body[key] else "0")
            log.info(f"[کنترل پنل] {key}: {'روشن' if body[key] else 'خاموش'}")
    if "daily_loss" in body:
        try:
            dl = float(body["daily_loss"])
            if not 0 <= dl <= 10000:
                raise ValueError
            paper_trader.set_setting(conn, "daily_loss", dl)
            log.info(f"[کنترل پنل] حد ضرر روزانه: ${dl:g}" if dl else "[کنترل پنل] حد ضرر روزانه: خاموش")
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "حد ضرر روزانه نامعتبر است"}), 400
    if "risk_usd" in body:
        try:
            ru = float(body["risk_usd"])
            if not 0.5 <= ru <= 1000:
                raise ValueError
            paper_trader.set_setting(conn, "risk_usd", ru)
            log.info(f"[کنترل پنل] ضرر خالص هر معامله: ${ru:g}")
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "مقدار ریسک دلاری نامعتبر است (۰.۵ تا ۱۰۰۰)"}), 400

    if "min_score" in body:
        try:
            ms = float(body["min_score"])
            if not 0 < ms <= 100:
                raise ValueError
            paper_trader.set_setting(conn, "min_score", ms)
            log.info(f"[کنترل پنل] حداقل امتیاز ورود: {ms:g}")
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "امتیاز نامعتبر است"}), 400

    if "timeframe" in body:
        if body["timeframe"] in config.TIMEFRAME_PROFILES:
            paper_trader.set_setting(conn, "timeframe", body["timeframe"])
            log.info(f"[کنترل پنل] تایم‌فریم/پروفایل: {body['timeframe']}")
        else:
            return jsonify({"ok": False, "error": "تایم‌فریم نامعتبر است"}), 400
    if "entry_mode" in body:
        if body["entry_mode"] in ("limit", "market"):
            paper_trader.set_setting(conn, "entry_mode", body["entry_mode"])
        else:
            return jsonify({"ok": False, "error": "مدل ورود نامعتبر است"}), 400

    for key, setting in (("htf", "htf_enabled"), ("min_sl", "min_sl"), ("room", "room_to_target"),
                         ("btc_filter", "btc_filter"), ("long_only", "long_only")):
        if key in body:
            paper_trader.set_setting(conn, setting, "1" if body[key] else "0")
            log.info(f"[کنترل پنل] {key}: {'روشن' if body[key] else 'خاموش'}")

    return jsonify({"ok": True, "settings": get_bot_settings()})


@app.route("/api/reset_capital", methods=["POST"])
def api_reset_capital():
    """
    عملیات جدا و آگاهانه برای تغییر/ریست سرمایه اولیه. تاریخچه‌ی معاملات پاک
    نمی‌شه، فقط خط پایه‌ی موجودی/بازده ریست می‌شه. پنل قبل از این باید تایید
    صریح از کاربر بگیره (چون برگشت‌ناپذیره).
    """
    body = request.get_json(force=True, silent=True) or {}
    try:
        amount = float(body.get("amount"))
        if amount <= 0:
            return jsonify({"ok": False, "error": "سرمایه باید بزرگ‌تر از صفر باشد"}), 400
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "مقدار سرمایه نامعتبر است"}), 400

    if paper_trader.get_open_position_count(conn) > 0:
        return jsonify({
            "ok": False,
            "error": "برای ریست سرمایه، اول همه‌ی پوزیشن‌های باز رو ببند (وگرنه محاسبه‌ی سرمایه‌ی قفل‌شده به‌هم می‌ریزه)."
        }), 400

    paper_trader.reset_capital(conn, amount)
    log.info(f"[ریست سرمایه از پنل] سرمایه اولیه به {amount} تنظیم شد")
    return jsonify({"ok": True, "settings": get_bot_settings()})


@app.route("/api/reset_all", methods=["POST"])
def api_reset_all():
    """شروع از صفر: پاک کردن همه‌ی معاملات (حتی بازها)، لاگ سیگنال‌ها و منحنی موجودی. تنظیمات می‌مونن."""
    body = request.get_json(force=True, silent=True) or {}
    if body.get("confirm") != "RESET":
        return jsonify({"ok": False, "error": "تایید لازمه"}), 400
    try:
        amount = float(body.get("amount") or get_bot_settings()["initial_capital"])
        if amount <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "مقدار سرمایه نامعتبر است"}), 400
    paper_trader.wipe_history(conn, amount)
    latest_analysis.clear()
    log.info(f"[شروع از صفر از پنل] همه‌ی معاملات و لاگ‌ها پاک شد؛ سرمایه {amount}")
    return jsonify({"ok": True, "settings": get_bot_settings()})


@app.route("/api/close_trade", methods=["POST"])
def api_close_trade():
    """بستن دستی یک پوزیشن باز از پنل، با قیمت لحظه‌ای فعلی."""
    body = request.get_json(force=True, silent=True) or {}
    trade_id = body.get("id")
    if not trade_id:
        return jsonify({"ok": False, "error": "شناسه معامله مشخص نشده"}), 400

    row = conn.execute("SELECT symbol FROM trades WHERE id=? AND status IN ('OPEN','PENDING')",
                       (trade_id,)).fetchone()
    if not row:
        return jsonify({"ok": False, "error": "پوزیشن باز پیدا نشد"}), 404
    symbol = row[0]

    try:
        df, _ = data_fetcher.fetch_ohlcv_with_fallback(symbol, "1m", 2, config.EXCHANGE_TRY_ORDER)
        current_price = float(df["close"].iloc[-1])
    except Exception:
        return jsonify({"ok": False, "error": "دریافت قیمت لحظه‌ای ناموفق بود"}), 500

    settings = get_bot_settings()
    result = paper_trader.close_trade_manually(
        conn, trade_id, current_price, settings["initial_capital"],
        config.MAKER_FEE_PCT, config.TAKER_FEE_PCT, config.TAKER_SLIPPAGE_PCT,
    )
    if result["ok"]:
        log.info(f"[بستن دستی از پنل] {result['symbol']} سود/زیان={result['pnl']}")
    return jsonify(result)


@app.route("/api/backup")
def api_backup():
    """دانلود مستقیم یک فایل بکاپ (دیتابیس + تنظیمات) بدون نیاز به SSH."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(config.DB_PATH, arcname="tradingbot.db")
        tar.add("config.py", arcname="config.py")
        # ۵ گزارش آخر مقایسه‌ی استراتژی‌ها (برای تحلیل بعدی)
        try:
            d = _reports_dir()
            reps = sorted([f for f in os.listdir(d) if f.startswith("compare_") and f.endswith(".json")],
                          key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)[:5]
            for f in reps:
                tar.add(os.path.join(d, f), arcname=f"reports/{f}")
        except Exception:
            pass
    buf.seek(0)
    filename = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/gzip")


# ==================== بک‌تست (تحلیل تاریخی ۱-۲ ساله) ====================
# اجرا در پس‌زمینه (نه توی همون درخواست HTTP) چون ممکنه چند دقیقه طول بکشه.
# فقط یک بک‌تست هم‌زمان مجازه تا فشار غیرضروری روی VPS نیفته.

backtest_jobs = {}          # job_id -> {state, progress, result, error, started_at}
backtest_lock = threading.Lock()
backtest_running_job = {"id": None}

# اتصال SQLite در حافظه‌ی هر بک‌تست، برای صفحه‌بندی معاملات/سیگنال‌ها بعد از اتمام
# (فقط چندتای آخر نگه داشته می‌شه تا حافظه‌ی VPS هدر نره)
backtest_connections = {}   # job_id -> sqlite conn
BACKTEST_CONN_KEEP = 3


def _remember_backtest_conn(job_id, conn_bt):
    backtest_connections[job_id] = conn_bt
    if len(backtest_connections) > BACKTEST_CONN_KEEP:
        oldest_id = next(iter(backtest_connections))
        try:
            backtest_connections[oldest_id].close()
        except Exception:
            pass
        del backtest_connections[oldest_id]


def _run_backtest_job(job_id, symbols, days, overrides, timeframe=None):
    def progress_cb(msg):
        backtest_jobs[job_id]["progress"] = msg

    try:
        backtest_jobs[job_id]["state"] = "running"
        conn_bt, meta = backtest.run_backtest(symbols, days, config, overrides, progress_cb, timeframe=timeframe)
        cfg_bt = meta.pop("_cfg", None) or backtest.build_config(config, overrides)
        report = backtest_analyzer.analyze(conn_bt, cfg_bt, meta)
        report["timeframe"] = meta.get("timeframe")
        report["days"] = meta.get("days")
        report["segments"] = meta.get("segments")
        report["engine"] = meta.get("engine")
        report["bot_version"] = meta.get("bot_version")
        backtest_jobs[job_id]["result"] = report
        backtest_jobs[job_id]["state"] = "done"
        backtest_jobs[job_id]["progress"] = "تمام شد"
        _remember_backtest_conn(job_id, conn_bt)
    except Exception as e:
        log.exception(f"بک‌تست {job_id} با خطا متوقف شد")
        backtest_jobs[job_id]["state"] = "error"
        backtest_jobs[job_id]["error"] = str(e)
    finally:
        with backtest_lock:
            if backtest_running_job["id"] == job_id:
                backtest_running_job["id"] = None


@app.route("/api/backtest/start", methods=["POST"])
def api_backtest_start():
    body = request.get_json(force=True, silent=True) or {}
    days = int(body.get("days", 365))
    days = max(30, min(days, 1095))  # بین ۱ ماه تا ۳ سال، محافظ در برابر مقدار نامعقول

    symbols = body.get("symbols")
    if not symbols:
        top_n = int(body.get("top_n", 15))
        top_n = max(1, min(top_n, 250))
        symbols = active_symbols["list"][:top_n] if active_symbols["list"] else list(config.SYMBOLS)[:top_n]

    overrides = body.get("overrides") or {}
    allowed_override_keys = {
        "RISK_PER_TRADE_PCT", "MIN_RISK_REWARD", "PROXIMITY_PCT", "COOLDOWN_HOURS",
        "USE_REJECTION_CONFIRMATION", "USE_HTF_CONFIRMATION", "HTF_MIN_AGREEMENT",
        "MAX_LEVERAGE", "MAX_POSITION_PCT_OF_CAPITAL", "MAX_OPEN_POSITIONS",
        "MAKER_FEE_PCT", "TAKER_FEE_PCT", "TAKER_SLIPPAGE_PCT",
        "USE_TRAILING_SL", "TRAILING_SL_LADDER", "TRAILING_SL_BEYOND_DISTANCE_R",
        "ACTIVE_STRATEGIES", "STRATEGY_COMBINE_MODE",
        "BREAKOUT_LOOKBACK", "BREAKOUT_VOLUME_MULT",
        "MIN_SL_PCT", "MIN_SL_ATR_MULT", "REQUIRE_ROOM_TO_TARGET", "BTC_REGIME_FILTER",
        "ALLOW_LONG", "ALLOW_SHORT", "CONFIRM_LOOKBACK_BARS", "SHORT_EXTRA_HTF_AGREEMENT", "ENTRY_MODE",
        "WC_MIN_SCORE_PCT", "RISK_USD", "RISK_MODE", "TRAIL_PROFILE", "BTC_PAIR_WEIGHT", "HTF_WEIGHTED",
        "BRK_BOX_BARS", "BRK_BOX_MAX_ATR", "BRK_BOX_MIN_TOUCHES", "BRK_VOL_MULT", "BRK_MAX_EXT_ATR",
        "BRK_MAIN_TREND", "BRK_SL_MODE", "BRK_USE_SR", "BRK_ENTRY", "BRK_RETEST_WAIT_BARS",
        "EARLY_EXIT", "EARLY_EXIT_BARS", "EARLY_EXIT_MIN_R", "DAILY_LOSS_LIMIT_USD", "CUT_LOSS_R",
        "TR_BREAKOUT_BARS", "TR_MA", "TR_SL_ATR", "TR_VOL_MULT",
    }
    overrides = {k: v for k, v in overrides.items() if k in allowed_override_keys}

    # میان‌بر راحت: اگه به‌جای پارامترهای تک‌تک، فقط اسم سطح سخت‌گیری داده بشه
    strictness_name = body.get("strictness")
    if strictness_name and strictness_name in config.STRICTNESS_PRESETS:
        preset = config.STRICTNESS_PRESETS[strictness_name]
        overrides.setdefault("HTF_MIN_AGREEMENT", preset["HTF_MIN_AGREEMENT"])
        overrides.setdefault("MIN_RISK_REWARD", preset["MIN_RISK_REWARD"])

    # میان‌برهای ساده‌ی پنل برای بهبودهای قابل‌آزمایش
    if body.get("min_sl"):
        overrides["MIN_SL_PCT"] = config.TEST_MIN_SL_PCT
        overrides["MIN_SL_ATR_MULT"] = config.TEST_MIN_SL_ATR_MULT
    if "room" in body:
        overrides["REQUIRE_ROOM_TO_TARGET"] = bool(body["room"])
    if "btc_filter" in body:
        overrides["BTC_REGIME_FILTER"] = bool(body["btc_filter"])
    if body.get("long_only"):
        overrides["ALLOW_SHORT"] = False
    if body.get("entry_mode") in ("limit", "market"):
        overrides["ENTRY_MODE"] = body["entry_mode"]
    if "htf" in body:
        overrides["USE_HTF_CONFIRMATION"] = bool(body["htf"])
    if body.get("min_score") is not None:
        overrides["WC_MIN_SCORE_PCT"] = float(body["min_score"])
    strat = body.get("strategy") or get_bot_settings()["strategy"]
    overrides["ACTIVE_STRATEGIES"] = strategies.parse_strategy(strat)
    _live = get_bot_settings()
    overrides["PAT_STRUCTURE"] = body.get("pat_structure") if body.get("pat_structure") in ("with", "off") \
        else _live["pat_structure"]
    overrides["CONTRA_BTC"] = body.get("contra_btc") if body.get("contra_btc") in ("against", "off") \
        else _live["contra_btc"]
    if body.get("cut_loss_r") is not None:
        try:
            overrides["CUT_LOSS_R"] = max(0.0, min(0.99, float(body["cut_loss_r"])))
        except (TypeError, ValueError):
            pass
    overrides["STRATEGY_COMBINE_MODE"] = "any"
    if body.get("trail_profile") in config.TRAIL_PROFILES:
        overrides["TRAIL_PROFILE"] = body["trail_profile"]
    if "trailing" in body:
        overrides["USE_TRAILING_SL"] = bool(body["trailing"])
    if body.get("risk_usd") is not None:
        try:
            overrides["RISK_USD"] = max(0.5, min(1000.0, float(body["risk_usd"])))
        except (TypeError, ValueError):
            pass
    if "retest" in body:
        overrides["BRK_ENTRY"] = "retest" if body["retest"] else "close"
    if "early_exit" in body:
        overrides["EARLY_EXIT"] = bool(body["early_exit"])
    if body.get("daily_loss") is not None:
        try:
            overrides["DAILY_LOSS_LIMIT_USD"] = max(0.0, float(body["daily_loss"]))
        except (TypeError, ValueError):
            pass

    if compare_running():
        return jsonify({"ok": False, "error": "الان «مقایسه‌ی استراتژی‌ها» در حال اجراست؛ بعد از تمام شدنش امتحان کن."}), 409
    with backtest_lock:
        if backtest_running_job["id"] is not None:
            return jsonify({"ok": False, "error": "یک بک‌تست دیگه الان در حال اجراست. صبر کن تمام بشه."}), 409
        job_id = uuid.uuid4().hex[:12]
        backtest_running_job["id"] = job_id

    backtest_jobs[job_id] = {
        "state": "queued", "progress": "در صف", "result": None, "error": None,
        "started_at": datetime.utcnow().isoformat(),
        "params": {"days": days, "symbols": symbols, "overrides": overrides},
    }
    timeframe = body.get("timeframe") if body.get("timeframe") in config.TIMEFRAME_PROFILES else None
    backtest_jobs[job_id]["params"]["timeframe"] = timeframe
    t = threading.Thread(target=_run_backtest_job, args=(job_id, symbols, days, overrides, timeframe), daemon=True)
    t.start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/backtest/status/<job_id>")
def api_backtest_status(job_id):
    job = backtest_jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "این job پیدا نشد"}), 404
    return jsonify({
        "ok": True, "state": job["state"], "progress": job["progress"],
        "error": job["error"], "started_at": job["started_at"], "params": job["params"],
    })


@app.route("/api/backtest/result/<job_id>")
def api_backtest_result(job_id):
    job = backtest_jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "این job پیدا نشد"}), 404
    if job["state"] != "done":
        return jsonify({"ok": False, "error": f"هنوز تمام نشده (وضعیت: {job['state']})"}), 400
    return jsonify({"ok": True, "result": job["result"], "params": job["params"]})


@app.route("/api/backtest/jobs")
def api_backtest_jobs():
    """لیست بک‌تست‌های این نشست (فقط در حافظه؛ با ری‌استارت ربات پاک می‌شه)."""
    items = [{"job_id": jid, "state": j["state"], "started_at": j["started_at"], "params": j["params"]}
              for jid, j in sorted(backtest_jobs.items(), key=lambda x: x[1]["started_at"], reverse=True)]
    return jsonify({"ok": True, "jobs": items[:20]})


@app.route("/api/backtest/trades/<job_id>")
def api_backtest_trades(job_id):
    """
    لیست کامل و صفحه‌بندی‌شده‌ی معاملات بسته‌شده‌ی یک بک‌تست تمام‌شده — برای مرور
    تک‌تک معاملات (نه فقط آمار خلاصه)، با امکان دسترسی به قدیمی‌ترها هم.
    """
    conn_bt = backtest_connections.get(job_id)
    if conn_bt is None:
        return jsonify({"ok": False, "error": "نتیجه‌ی این بک‌تست دیگه در دسترس نیست (شاید بک‌تست‌های جدیدتری اجرا شده)"}), 404
    page = max(1, int(request.args.get("page", 1)))
    page_size = min(200, max(10, int(request.args.get("page_size", 30))))
    trades, total = paper_trader.get_closed_trades(conn_bt, limit=page_size, offset=(page - 1) * page_size)
    return jsonify({"ok": True, "trades": trades, "total": total, "page": page, "page_size": page_size})


@app.route("/api/backtest/signals/<job_id>")
def api_backtest_signals(job_id):
    """لیست کامل و صفحه‌بندی‌شده‌ی سیگنال‌های یک بک‌تست (اجراشده + رد‌شده با دلیل)."""
    conn_bt = backtest_connections.get(job_id)
    if conn_bt is None:
        return jsonify({"ok": False, "error": "نتیجه‌ی این بک‌تست دیگه در دسترس نیست (شاید بک‌تست‌های جدیدتری اجرا شده)"}), 404
    page = max(1, int(request.args.get("page", 1)))
    page_size = min(200, max(10, int(request.args.get("page_size", 30))))
    only_rejected = request.args.get("only_rejected", "").lower() == "true"
    signals, total = paper_trader.get_signal_log(
        conn_bt, limit=page_size, offset=(page - 1) * page_size, only_rejected=only_rejected
    )
    return jsonify({"ok": True, "signals": signals, "total": total, "page": page, "page_size": page_size})


# ==================== مقایسه‌ی خودکار استراتژی‌ها ====================
# در یک پروسه‌ی جدا با اولویت پایین اجرا می‌شه (compare.py)، تا محاسبات سنگینش ربات زنده
# و پنل رو کند نکنه. پیشرفت و نتیجه در فایل ذخیره می‌شن، پس بعد از ری‌استارت ربات هم
# گزارش‌های قبلی در دسترس می‌مونن.

compare_proc = {"proc": None, "job_id": None}
compare_lock = threading.Lock()


def _reports_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.REPORTS_DIR)
    os.makedirs(d, exist_ok=True)
    return d


def _pid_alive(pid):
    """پروسه‌ی compare.py با این شماره هنوز زنده‌ست؟ (زامبی = مرده)"""
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            if b"compare.py" not in f.read():
                return False
        with open(f"/proc/{pid}/stat", "r") as f:
            return f.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except Exception:
        return False


def _read_progress(job_id):
    try:
        with open(_progress_path(job_id), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _external_job():
    """کار سنگینی که هنوز زنده‌ست ولی این پروسه‌ی ربات شروعش نکرده (مثلاً قبل از ری‌استارت ربات)."""
    d = _reports_dir()
    now = time.time()
    try:
        names = os.listdir(d)
    except OSError:
        return None
    for name in names:
        if not (name.startswith("progress_") and name.endswith(".json")):
            continue
        path = os.path.join(d, name)
        try:
            if now - os.path.getmtime(path) > 3 * 86400:
                continue
            with open(path, "r", encoding="utf-8") as f:
                st = json.load(f)
        except Exception:
            continue
        if st.get("state") == "running" and _pid_alive(st.get("pid")):
            return st.get("job_id"), st.get("kind", "compare")
    return None


def compare_running():
    p = compare_proc["proc"]
    if p is not None and p.poll() is None:
        return True
    ext = _external_job()
    if ext:
        compare_proc["job_id"], compare_proc["kind"] = ext
        return True
    return False


def _progress_path(job_id):
    return os.path.join(_reports_dir(), f"progress_{job_id}.json")


def _report_path(job_id):
    return os.path.join(_reports_dir(), f"compare_{job_id}.json")


def _safe_job_id(job_id):
    return "".join(ch for ch in str(job_id) if ch.isalnum())[:32]


@app.route("/api/compare/start", methods=["POST"])
def api_compare_start():
    body = request.get_json(force=True, silent=True) or {}
    days = max(60, min(int(body.get("days_override") or body.get("days", 365)), 1095))
    top_n = max(3, min(int(body.get("top_n", 20)), 60))
    grid = body.get("grid") if body.get("grid") in ("full", "focus", "all", "fib", "pattern", "contrarian") \
        else "quick"
    unseen = bool(body.get("unseen"))
    tfs = [t for t in (body.get("timeframes") or ["15m", "1h", "4h"]) if t in config.TIMEFRAME_PROFILES]
    if not tfs:
        return jsonify({"ok": False, "error": "حداقل یک تایم‌فریم انتخاب کن"}), 400
    with compare_lock:
        if compare_running():
            return jsonify({"ok": False, "error": "یک مقایسه‌ی دیگه در حال اجراست.",
                            "job_id": compare_proc["job_id"]}), 409
        if backtest_running_job["id"] is not None:
            return jsonify({"ok": False, "error": "یک بک‌تست تکی در حال اجراست؛ صبر کن تمام بشه."}), 409
        symbols = active_symbols["list"][:top_n] if active_symbols["list"] else list(config.SYMBOLS)[:top_n]
        if unseen:
            # گذشته‌ی دیده‌نشده: ارزهای قدیمی، بازه‌ی ثابت که به UNSEEN_END ختم می‌شه (فقط روزانه)
            symbols, days, tfs = list(config.UNSEEN_SYMBOLS), int(config.UNSEEN_DAYS), ["1d"]
        job_id = uuid.uuid4().hex[:10]
        baseline_path = os.path.join(_reports_dir(), f"baseline_{job_id}.json")
        live_conf = live_config_snapshot()
        default_conf = {"timeframe": "15m", "active_strategies": list(config.ACTIVE_STRATEGIES),
                        "combine_mode": "any", "min_score": float(config.WC_MIN_SCORE_PCT),
                        "strictness": config.DEFAULT_STRICTNESS,
                        "trailing": config.DEFAULT_TRAIL_PROFILE if config.USE_TRAILING_SL else False,
                        "min_sl": False, "room": False, "btc_filter": False, "long_only": False,
                        "htf": bool(config.USE_HTF_CONFIRMATION), "retest": config.BRK_ENTRY == "retest",
                        "early_exit": bool(config.EARLY_EXIT), "daily_loss": config.DAILY_LOSS_LIMIT_USD > 0}
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump([{"name": "تنظیمات فعلی ربات زنده", "config": live_conf},
                       {"name": "تنظیمات پیش‌فرض", "config": default_conf}], f, ensure_ascii=False)
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare.py"),
               "--days", str(days), "--grid", grid, "--job-id", job_id, "--timeframes", ",".join(tfs),
               "--progress-file", _progress_path(job_id), "--baseline-file", baseline_path,
               "--symbols", ",".join(symbols)]
        if unseen:
            cmd += ["--end", config.UNSEEN_END]
        _spawn_compare(cmd, job_id)
    log.info(f"[مقایسه‌ی استراتژی‌ها] شروع شد: {job_id} ({days} روز، {len(symbols)} نماد، {grid}، {tfs})")
    return jsonify({"ok": True, "job_id": job_id})


def _spawn_compare(cmd, job_id, kind="compare"):
    """اجرای compare.py در پروسه‌ی جدا با اولویت پایین (باید داخل compare_lock صدا زده بشه)."""
    if os.environ.get("TRADINGBOT_OFFLINE") == "1":
        cmd.append("--offline")   # فقط برای تست بدون اینترنت
    log_file = open(os.path.join(_reports_dir(), f"{kind}_{job_id}.log"), "a")

    def lower_priority():
        try:
            os.nice(15)
        except Exception:
            pass

    compare_proc["proc"] = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                            preexec_fn=lower_priority, close_fds=True)
    compare_proc["job_id"] = job_id
    compare_proc["kind"] = kind


# ==================== دیتای تاریخی: دریافت یک‌باره، اعتبارسنجی، بکاپ و بازگردانی ====================

@app.route("/api/data/fetch", methods=["POST"])
def api_data_fetch():
    """فقط دانلود/به‌روزرسانی دیتای تاریخی + گزارش اعتبار (بدون بک‌تست)."""
    body = request.get_json(force=True, silent=True) or {}
    days = max(30, min(int(body.get("days", 730)), 1095))
    top_n = max(3, min(int(body.get("top_n", 20)), 60))
    tfs = [t for t in (body.get("timeframes") or ["15m", "1h", "4h"]) if t in config.TIMEFRAME_PROFILES]
    if not tfs:
        return jsonify({"ok": False, "error": "حداقل یک تایم‌فریم انتخاب کن"}), 400
    with compare_lock:
        if compare_running():
            return jsonify({"ok": False, "error": "یک کار دیگه (مقایسه/دریافت دیتا) در حال اجراست.",
                            "job_id": compare_proc["job_id"]}), 409
        if backtest_running_job["id"] is not None:
            return jsonify({"ok": False, "error": "یک بک‌تست تکی در حال اجراست؛ صبر کن تمام بشه."}), 409
        symbols = active_symbols["list"][:top_n] if active_symbols["list"] else list(config.SYMBOLS)[:top_n]
        job_id = uuid.uuid4().hex[:10]
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare.py"),
               "--data-only", "--days", str(days), "--job-id", job_id, "--timeframes", ",".join(tfs),
               "--progress-file", _progress_path(job_id), "--symbols", ",".join(symbols)]
        _spawn_compare(cmd, job_id, kind="data")
    log.info(f"[دیتای تاریخی] دریافت/به‌روزرسانی شروع شد: {job_id} ({days} روز، {len(symbols)} نماد، {tfs})")
    return jsonify({"ok": True, "job_id": job_id})


def _latest_data_report():
    d = _reports_dir()
    files = sorted([f for f in os.listdir(d) if f.startswith("data_") and f.endswith(".json")],
                   key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    if not files:
        return None
    try:
        with open(os.path.join(d, files[0]), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _exports_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    os.makedirs(d, exist_ok=True)
    return d


@app.route("/api/data_cache/export")
def api_data_cache_export():
    """
    کل دیتای تاریخی + گزارش‌ها در یک فایل (برای بکاپ یا انتقال به سرور دیگه). فایل روی دیسک ساخته
    می‌شه و همیشه فقط آخرین نسخه نگه داشته می‌شه تا فضای سرور پر نشه.
    """
    if compare_running():
        return jsonify({"ok": False, "error": "صبر کن کار در حال اجرا (مقایسه/دریافت دیتا) تمام بشه."}), 409
    cache = backtest.get_cache(config)
    d = _exports_dir()
    for f in os.listdir(d):
        try:
            os.remove(os.path.join(d, f))
        except OSError:
            pass
    name = f"tradingbot_data_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.tar.gz"
    path = cache.export_archive(os.path.join(d, name), reports_dir=_reports_dir())
    return send_file(path, as_attachment=True, download_name=name, mimetype="application/gzip")


@app.route("/api/data_cache/import", methods=["POST"])
def api_data_cache_import():
    """بازگردانی/ادغام دیتای تاریخی از فایلی که قبلاً با «دانلود فایل دیتا» گرفته شده."""
    if compare_running() or backtest_running_job["id"] is not None:
        return jsonify({"ok": False, "error": "صبر کن کار در حال اجرا تمام بشه، بعد بازگردانی کن."}), 409
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"ok": False, "error": "فایلی انتخاب نشده"}), 400
    tmp = os.path.join(_exports_dir(), f"upload_{uuid.uuid4().hex[:8]}.tar.gz")
    try:
        f.save(tmp)
        stats = backtest.get_cache(config).import_archive(tmp, reports_dir=_reports_dir())
    except Exception as e:
        log.warning(f"بازگردانی دیتا ناموفق: {e}")
        return jsonify({"ok": False, "error": "فایل معتبر نیست — باید همون فایلی باشه که با «دانلود فایل دیتا» "
                                              "یا دستور export-data ساخته شده (tar.gz)"}), 400
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    log.info(f"[دیتای تاریخی] بازگردانی از فایل: {stats}")
    return jsonify({"ok": True, "stats": stats})


@app.route("/api/compare/status/<job_id>")
def api_compare_status(job_id):
    job_id = _safe_job_id(job_id)
    path = _progress_path(job_id)
    if not os.path.exists(path):
        if os.path.exists(_report_path(job_id)):
            return jsonify({"ok": True, "state": "done", "progress": 1.0, "message": "تمام شد"})
        if compare_proc["job_id"] == job_id and compare_running():
            return jsonify({"ok": True, "state": "running", "progress": 0.0, "message": "در حال شروع"})
        return jsonify({"ok": False, "error": "پیدا نشد"}), 404
    try:
        with open(path, "r", encoding="utf-8") as f:
            st = json.load(f)
    except Exception:
        return jsonify({"ok": True, "state": "running", "progress": 0, "message": "در حال شروع"})
    if st.get("state") == "running":
        p = compare_proc["proc"]
        alive = (compare_proc["job_id"] == job_id and p is not None and p.poll() is None) or _pid_alive(st.get("pid"))
        if not alive:
            if st.get("kind") == "entry" and os.path.isdir(_entry_work(job_id)):
                st["state"] = "stopped"
                st["resumable"] = True
                st["message"] = ("سنجش وسط کار متوقف شد (ری‌استارت ربات، کمبود رم یا ری‌استارت سرور). "
                                 "قسمت‌های انجام‌شده ذخیره شدن — «▶ ادامه» رو بزن تا از همون‌جا ادامه بده.")
            else:
                st["state"] = "error"
                st["message"] = ("کار وسط راه متوقف شد (ری‌استارت ربات/سرور یا کمبود رم). دوباره شروع کن؛ "
                                 "اگه باز تکرار شد، تعداد نمادها رو کمتر کن.")
    try:
        st["stale_sec"] = int((datetime.utcnow() - datetime.fromisoformat(st.get("updated_at") or st.get("started_at"))).total_seconds())
    except Exception:
        pass
    return jsonify({"ok": True, **{k: v for k, v in st.items() if k != "error"}, "error": st.get("error")})


@app.route("/api/compare/result/<job_id>")
def api_compare_result(job_id):
    path = _report_path(_safe_job_id(job_id))
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "گزارش پیدا نشد"}), 404
    with open(path, "r", encoding="utf-8") as f:
        return jsonify({"ok": True, "report": json.load(f)})


@app.route("/api/compare/list")
def api_compare_list():
    items = []
    d = _reports_dir()
    for name in os.listdir(d):
        if not (name.startswith("compare_") and name.endswith(".json")):
            continue
        path = os.path.join(d, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                rep_ = json.load(f)
            m = rep_.get("meta", {})
            rec = rep_.get("recommendation", {})
            items.append({"job_id": m.get("job_id") or name[8:-5], "created_at": m.get("created_at"),
                          "days": m.get("days"), "symbols": len(m.get("symbols", [])),
                          "grid": m.get("grid"), "n_configs": m.get("n_configs"),
                          "timeframes": list((m.get("timeframes") or {}).keys()),
                          "status": rec.get("status"),
                          "best": (rec.get("best") or {}).get("label")})
        except Exception:
            continue
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    running = compare_proc["job_id"] if compare_running() and compare_proc.get("kind", "compare") == "compare" else None
    return jsonify({"ok": True, "reports": items[:30], "running_job": running})


@app.route("/api/compare/apply", methods=["POST"])
def api_compare_apply():
    """اعمال یک تنظیم از نتایج مقایسه روی ربات زنده (فقط روی سیگنال‌ها و پوزیشن‌های بعدی اثر داره)."""
    body = request.get_json(force=True, silent=True) or {}
    c = body.get("config") or {}
    if c.get("strictness") not in config.STRICTNESS_PRESETS:
        return jsonify({"ok": False, "error": "تنظیم نامعتبر"}), 400
    paper_trader.set_setting(conn, "strictness", c["strictness"])
    if c.get("min_score") is not None:
        paper_trader.set_setting(conn, "min_score", float(c["min_score"]))
    paper_trader.set_setting(conn, "trailing_enabled", "1" if c.get("trailing") else "0")
    if isinstance(c.get("trailing"), str) and c["trailing"] in config.TRAIL_PROFILES:
        paper_trader.set_setting(conn, "trail_profile", c["trailing"])
    act = [a for a in (c.get("active_strategies") or []) if a in strategies.STRATEGY_REGISTRY]
    if act:
        paper_trader.set_setting(conn, "strategy", "+".join(act))
    cut = c.get("cut", c.get("cut_loss_r"))
    if cut is not None:
        paper_trader.set_setting(conn, "cut_loss_r", float(cut))
    if "retest" in c:
        paper_trader.set_setting(conn, "brk_retest", "1" if c["retest"] else "0")
    if "free" in c:
        if "contrarian_btc" in act:
            paper_trader.set_setting(conn, "contra_btc", "off" if c["free"] else "against")
        else:
            paper_trader.set_setting(conn, "pat_structure", "off" if c["free"] else "with")
    if "early_exit" in c:
        paper_trader.set_setting(conn, "early_exit", "1" if c["early_exit"] else "0")
    if "daily_loss" in c:
        paper_trader.set_setting(conn, "daily_loss", config.DAILY_LOSS_LIMIT_USD if c["daily_loss"] else 0)
    paper_trader.set_setting(conn, "min_sl", "1" if c.get("min_sl") else "0")
    paper_trader.set_setting(conn, "room_to_target", "1" if c.get("room") else "0")
    paper_trader.set_setting(conn, "btc_filter", "1" if c.get("btc_filter") else "0")
    paper_trader.set_setting(conn, "long_only", "1" if c.get("long_only") else "0")
    if c.get("timeframe") in config.TIMEFRAME_PROFILES:
        paper_trader.set_setting(conn, "timeframe", c["timeframe"])
    paper_trader.set_setting(conn, "htf_enabled", "1" if c.get("htf", True) else "0")
    log.info(f"[اعمال از مقایسه] تنظیمات جدید ربات زنده: {c}")
    return jsonify({"ok": True, "settings": get_bot_settings()})


def _zip_response(files, name):
    """ساخت یک فایل zip در حافظه از (نام داخل zip، محتوا یا مسیر فایل)."""
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for arcname, content in files:
            if isinstance(content, str) and os.path.exists(content):
                z.write(content, arcname)
            elif content is not None:
                z.writestr(arcname, content)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=name, mimetype="application/zip")


@app.route("/api/compare/download/<job_id>")
def api_compare_download(job_id):
    """همه‌ی خروجی‌های یک مقایسه در یک فایل: گزارش کامل، جدول همه‌ی تنظیم‌ها (CSV)، خلاصه‌ی متنی."""
    job_id = _safe_job_id(job_id)
    d = _reports_dir()
    rep_path = _report_path(job_id)
    if not os.path.exists(rep_path):
        return jsonify({"ok": False, "error": "گزارش پیدا نشد"}), 404
    files = [("report.json", rep_path),
             ("all_results.csv", os.path.join(d, f"results_{job_id}.csv")),
             ("summary.txt", os.path.join(d, f"summary_{job_id}.txt")),
             ("config.py", "config.py")]
    return _zip_response(files, f"compare_{job_id}.zip")


# ==================== سنجش کیفیت ورود (entry_study.py) ====================
# هر سیگنال ورود جدا از مدیریت معامله در برابر ورود شانسی سنجیده می‌شه (پروسه‌ی جدا، مثل مقایسه).

def _entry_path(job_id):
    return os.path.join(_reports_dir(), f"entry_{job_id}.json")


@app.route("/api/entry/start", methods=["POST"])
def api_entry_start():
    body = request.get_json(force=True, silent=True) or {}
    days = max(60, min(int(body.get("days", 730)), 1095))
    top_n = max(3, min(int(body.get("top_n", 30)), 100))
    signals = body.get("signals") if body.get("signals") in ("core", "patterns", "funding", "all") else "core"
    tfs = [t for t in (body.get("timeframes") or ["15m", "1h", "4h", "1d"]) if t in config.TIMEFRAME_PROFILES]
    if not tfs:
        return jsonify({"ok": False, "error": "حداقل یک تایم‌فریم انتخاب کن"}), 400
    with compare_lock:
        if compare_running():
            return jsonify({"ok": False, "error": "یک کار دیگه (مقایسه/سنجش/دریافت دیتا) در حال اجراست.",
                            "job_id": compare_proc["job_id"]}), 409
        if backtest_running_job["id"] is not None:
            return jsonify({"ok": False, "error": "یک بک‌تست تکی در حال اجراست؛ صبر کن تمام بشه."}), 409
        symbols = _entry_symbols(top_n)
        job_id = uuid.uuid4().hex[:10]
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare.py"),
               "--entry-study", "--days", str(days), "--job-id", job_id, "--timeframes", ",".join(tfs),
               "--progress-file", _progress_path(job_id), "--symbols", ",".join(symbols), "--signals", signals]
        _spawn_compare(cmd, job_id, kind="entry")
    log.info(f"[سنجش کیفیت ورود] شروع شد: {job_id} ({days} روز، {len(symbols)} نماد، {tfs})")
    return jsonify({"ok": True, "job_id": job_id})


def _entry_work(job_id):
    return os.path.join(_reports_dir(), f"entry_work_{_safe_job_id(job_id)}")


def _entry_params(job_id):
    try:
        with open(os.path.join(_entry_work(job_id), "params.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _spawn_entry_resume(job_id, params):
    """ادامه‌ی سنجش نیمه‌کاره (باید داخل compare_lock صدا زده بشه)."""
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare.py"),
           "--entry-study", "--resume", "--days", str(params["days"]), "--job-id", job_id,
           "--timeframes", ",".join(params["timeframes"]), "--progress-file", _progress_path(job_id),
           "--symbols", ",".join(params["symbols"])]
    _spawn_compare(cmd, job_id, kind="entry")


@app.route("/api/entry/resume", methods=["POST"])
def api_entry_resume():
    body = request.get_json(force=True, silent=True) or {}
    job_id = _safe_job_id(body.get("job_id", ""))
    params = _entry_params(job_id)
    if not params:
        return jsonify({"ok": False, "error": "کار نیمه‌کاره‌ای با این شناسه پیدا نشد"}), 404
    with compare_lock:
        if compare_running():
            return jsonify({"ok": False, "error": "یک کار دیگه در حال اجراست.", "job_id": compare_proc["job_id"]}), 409
        if backtest_running_job["id"] is not None:
            return jsonify({"ok": False, "error": "یک بک‌تست تکی در حال اجراست؛ صبر کن تمام بشه."}), 409
        _spawn_entry_resume(job_id, params)
    log.info(f"[سنجش کیفیت ورود] ادامه از جای قطع‌شده: {job_id}")
    return jsonify({"ok": True, "job_id": job_id})


def _unfinished_entries():
    import re
    out = []
    d = _reports_dir()
    for name in os.listdir(d):
        mm = re.match(r"^entry_work_([A-Za-z0-9]+)$", name)
        if not mm or os.path.exists(_entry_path(mm.group(1))):
            continue
        job_id = mm.group(1)
        params = _entry_params(job_id)
        if not params:
            continue
        done = [tf for tf in params.get("timeframes", [])
                if os.path.exists(os.path.join(_entry_work(job_id), tf, "done.json"))]
        st = _read_progress(job_id) or {}
        out.append({"job_id": job_id, "created_at": params.get("created_at"), "days": params.get("days"),
                    "symbols": len(params.get("symbols", [])), "timeframes": params.get("timeframes", []),
                    "done_timeframes": done, "state": st.get("state"), "progress": st.get("progress", 0),
                    "running": st.get("state") == "running" and _pid_alive(st.get("pid")),
                    "auto_resumes": params.get("auto_resumes", 0)})
    out.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return out


def _auto_resume_entry():
    """
    بعد از روشن شدن ربات: اگه سنجشی وسط کار کشته شده بود (مثلاً با ری‌استارت ربات/سرور)، خودکار از همون‌جا
    ادامه پیدا می‌کنه — حداکثر ۳ بار برای هر سنجش (اگه مدام به‌خاطر کمبود رم قطع بشه، تکرار بی‌پایان نشه).
    """
    time.sleep(90)
    try:
        for u in _unfinished_entries():
            if u["running"] or u["state"] != "running" or int(u.get("auto_resumes", 0)) >= 3:
                continue
            with compare_lock:
                if compare_running() or backtest_running_job["id"] is not None:
                    return
                params = _entry_params(u["job_id"])
                params["auto_resumes"] = int(params.get("auto_resumes", 0)) + 1
                pth = os.path.join(_entry_work(u["job_id"]), "params.json")
                with open(pth + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(params, f, ensure_ascii=False)
                os.replace(pth + ".tmp", pth)
                _spawn_entry_resume(u["job_id"], params)
            log.info(f"[سنجش کیفیت ورود] ادامه‌ی خودکار بعد از ری‌استارت: {u['job_id']} (بار {params['auto_resumes']})")
            return
    except Exception as e:
        log.warning(f"[سنجش کیفیت ورود] ادامه‌ی خودکار ناموفق: {e}")


def _entry_symbols(top_n):
    """نمادهای پرحجم برای سنجش (تا ۱۰۰): اگه لیست ربات کوتاه‌تره، از صرافی گرفته می‌شه."""
    syms = list(active_symbols["list"] or config.SYMBOLS)
    if len(syms) < top_n:
        try:
            more, _ = data_fetcher.get_top_symbols(config.EXCHANGE_TRY_ORDER, quote=config.QUOTE_CURRENCY,
                                                   top_n=top_n + 10, exclude_keywords=config.EXCLUDE_KEYWORDS,
                                                   cfg=config)
            syms += [x for x in more if x not in syms]
        except Exception as e:
            log.warning(f"[سنجش کیفیت ورود] لیست نمادهای بیشتر گرفته نشد: {e}")
    return syms[:top_n]


def _load_entry(job_id):
    path = _entry_path(_safe_job_id(job_id))
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.route("/api/entry/result/<job_id>")
def api_entry_result(job_id):
    rep_ = _load_entry(job_id)
    if rep_ is None:
        return jsonify({"ok": False, "error": "گزارش پیدا نشد"}), 404
    rep_ = dict(rep_)
    rep_.pop("features", None)   # جدول کامل فیلترها سنگینه؛ برای هر سیگنال جدا گرفته می‌شه
    return jsonify({"ok": True, "report": rep_})


@app.route("/api/entry/features/<job_id>")
def api_entry_features(job_id):
    rep_ = _load_entry(job_id)
    if rep_ is None:
        return jsonify({"ok": False, "error": "گزارش پیدا نشد"}), 404
    sig, tf = request.args.get("signal", ""), request.args.get("tf", "")
    rows = [f for f in rep_.get("features", []) if f.get("signal") == sig and f.get("tf") == tf]
    return jsonify({"ok": True, "rows": rows})


@app.route("/api/entry/list")
def api_entry_list():
    import re
    items = []
    d = _reports_dir()
    for name in os.listdir(d):
        mm = re.match(r"^entry_([A-Za-z0-9]+)\.json$", name)
        if not mm:
            continue
        try:
            with open(os.path.join(d, name), "r", encoding="utf-8") as f:
                rep_ = json.load(f)
            m = rep_.get("meta", {})
            items.append({"job_id": mm.group(1), "created_at": m.get("created_at"), "days": m.get("days"),
                          "signal_set": m.get("signal_set", "core"),
                          "symbols": len(m.get("symbols", [])), "timeframes": list((m.get("timeframes") or {}).keys()),
                          "status": (rep_.get("recommendation") or {}).get("status")})
        except Exception:
            continue
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    running = compare_proc["job_id"] if compare_running() and compare_proc.get("kind") == "entry" else None
    return jsonify({"ok": True, "reports": items[:30], "running_job": running, "unfinished": _unfinished_entries()})


ENTRY_README = """راهنمای فایل‌های «سنجش کیفیت ورود»
===================================
summary.txt       خلاصه‌ی فارسی: پایه‌ی شانسی، بهترین ورودها، فیلترهای مفید
signals.csv       هر سیگنال × جهت (long/short/both) × بخش (is=آموزش ۷۰٪ اول، oos=آزمون ۳۰٪ آخر، all=کل)
features.csv      تحلیل فیلترها: هر سیگنال بر اساس هر ویژگی به سطل تقسیم شده
events.csv.gz     تک‌تک ورودها (با سقف ردیف) — برای ساختن و آزمودن فیلترهای ترکیبی
report.json       همه‌ی بالایی‌ها برای پنل
config.py         تنظیمات دقیق همین اجرا

واحد R = ENTRY_STUDY_R_ATR × ATR(14) کندل سیگنال (برای همه‌ی ورودها یکسان).
براکت‌ها: 1_1 = +1R قبل از −1R ، 2_1 = +2R قبل از −1R ، 3_1 = +3R قبل از −1R ، 2_05 = +2R قبل از −0.5R
  اگه هر دو در یک کندل لمس شدن = ضرر. اگه تا H کندل به هیچ‌کدوم نرسید = بازده بسته‌شدن کندل H‌ام.

ستون‌های signals.csv (پیشوند is_/oos_/all_):
  n                تعداد ورود          clusters  تعداد خوشه‌ی مستقل (هفته، یا H کندل اگه بلندتر)
  p_X              درصد برد براکت X بین ورودهای به‌نتیجه‌رسیده (۰ تا ۱)
  base_p_X         همون برای ورود شانسی (همون نمادها، همون دوره، همون جهت‌ها)
  edge_p_X, z_p_X  فاصله با شانس و معناداریش (z بالای ۲ = معنادار)
  ev_net_X         میانگین نتیجه‌ی هر ورود به R بعد از کم کردن کارمزد (ارزش مورد انتظار خالص)
                   ×۵ = دلار در هر معامله با ریسک ۵ دلار
  base_ev_X, edge_ev_X, z_ev_X   همون برای ارزش خالص
  timeout_X        درصد ورودهایی که تا H کندل به هیچ‌کدوم نرسیدن
  fwd_k            میانگین بازده k کندل بعد (به R، هم‌جهت)
  mfe_med/mae_med  میانه‌ی بیشترین سود/ضرر شناور تا H کندل (به R)
  rdist_pct        میانه‌ی فاصله‌ی ۱R به درصد قیمت     cost_r  میانه‌ی کارمزد رفت‌وبرگشت به R
  own_p2, own_ev2  با حد ضرر خود استراتژی: درصد برد و ارزش خالص RR2 (به R خود استراتژی)
  verdict          robust = مزیت پایدار (آموزش z≥۲، آزمون هم بهتر و z≥۱، سود خالص در هر دو)
                   edge_costs = بهتر از شانس ولی کارمزد سودش رو می‌خوره
                   edge_weak_oos = بهتر از شانس ولی سود خالص فقط در یک بخش
                   weak = کمی بهتر، معنادار نیست   none = بدون مزیت   reverse = بدتر از شانس   few = کم‌داده
  sym_better_pct   در چند درصد نمادها درصد برد ۱:۱ از شانس بیشتر بوده

ستون‌های features.csv:
  p1_* = درصد برد ۱:۱ سطل، lift1_* = فاصله با کل سیگنال، ev2_* = ارزش خالص RR2 سطل،
  d_ev2_* = بهبود ارزش RR2 نسبت به کل سیگنال، z_is/z_oos = معناداری بهبود،
  flag: good = در آموزش (z≥۲.۵) و آزمون (z≥۱.۵) هر دو بهتر؛ bad = هر دو بدتر
  tfs_agree = همین فیلتر در چند تایم‌فریم تکرار شده (هرچی بیشتر، قابل‌اعتمادتر)
هشدار: هزاران سطل آزمایش می‌شه؛ چند فیلتر شانسی هم ممکنه good بگیرن. فیلتری که منطق داره و در چند
تایم‌فریم تکرار می‌شه قابل‌اعتمادتره.
"""


@app.route("/api/entry/download/<job_id>")
def api_entry_download(job_id):
    """همه‌ی خروجی‌های سنجش ورود در یک فایل zip."""
    job_id = _safe_job_id(job_id)
    d = _reports_dir()
    if not os.path.exists(_entry_path(job_id)):
        return jsonify({"ok": False, "error": "گزارش پیدا نشد"}), 404
    files = [("README_fa.txt", ENTRY_README),
             ("summary.txt", os.path.join(d, f"entry_summary_{job_id}.txt")),
             ("signals.csv", os.path.join(d, f"entry_signals_{job_id}.csv")),
             ("features.csv", os.path.join(d, f"entry_features_{job_id}.csv")),
             ("events.csv.gz", os.path.join(d, f"entry_events_{job_id}.csv.gz")),
             ("report.json", _entry_path(job_id)),
             ("config.py", "config.py")]
    return _zip_response(files, f"entry_{job_id}.zip")


def _rows_to_csv(conn_bt, query, cols=None):
    import csv
    cur = conn_bt.execute(query)
    names = cols or [c[0] for c in cur.description]
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(names)
    w.writerows(cur.fetchall())
    return "\ufeff" + out.getvalue()


@app.route("/api/backtest/download/<job_id>")
def api_backtest_download(job_id):
    """خروجی کامل یک بک‌تست: گزارش تحلیل، همه‌ی معاملات، همه‌ی سیگنال‌ها (با دلیل رد)، منحنی موجودی."""
    job = backtest_jobs.get(job_id)
    conn_bt = backtest_connections.get(job_id)
    if not job or job.get("state") != "done" or conn_bt is None:
        return jsonify({"ok": False, "error": "نتیجه‌ی این بک‌تست دیگه در دسترس نیست"}), 404
    files = [
        ("report.json", json.dumps({"params": job["params"], "result": job["result"]}, ensure_ascii=False, indent=1)),
        ("trades.csv", _rows_to_csv(conn_bt, "SELECT * FROM trades ORDER BY open_time")),
        ("signals.csv", _rows_to_csv(conn_bt, "SELECT * FROM signal_log ORDER BY time")),
        ("equity.csv", _rows_to_csv(conn_bt, "SELECT time, balance FROM equity ORDER BY id")),
        ("config.py", "config.py"),
    ]
    return _zip_response(files, f"backtest_{job_id}.zip")


@app.route("/api/data_cache")
def api_data_cache():
    cache = backtest.get_cache(config)
    symbols = sorted({k.split("|")[0] for k in cache.index})
    rep_ = _latest_data_report()
    return jsonify({"ok": True, "size_mb": cache.disk_usage_mb(), "symbols": len(symbols),
                    "by_timeframe": cache.summary(), "latest_check": rep_,
                    "running_job": compare_proc["job_id"] if compare_running() else None,
                    "running_kind": compare_proc.get("kind") if compare_running() else None})



if __name__ == "__main__":
    _apply_live_defaults_v14()
    _apply_live_defaults_v16()
    _apply_live_defaults_v18()
    scheduler.start()
    threading.Thread(target=_auto_resume_entry, daemon=True).start()
    log.info("ربات معامله‌گر مجازی استارت شد.")
    app.run(host=config.HOST, port=config.PORT, debug=False, use_reloader=False)
