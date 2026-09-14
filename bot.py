# -*- coding: utf-8 -*-
"""
اجرای اصلی ربات: هر ۱۵ دقیقه نمادها رو اسکن می‌کنه، سیگنال تولید می‌کنه،
معامله مجازی باز می‌کنه/می‌بندد، و یک داشبورد وب روی پورت ۵۰۰۰ بالا می‌آره.

اجرا: python3 bot.py
"""
import logging
from datetime import datetime

from flask import Flask, jsonify, render_template
from apscheduler.schedulers.background import BackgroundScheduler

import config
import data_fetcher
import analysis
import paper_trader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tradingbot")

app = Flask(__name__)
conn = paper_trader.get_conn(config.DB_PATH)

# آخرین وضعیت تحلیل هر نماد، برای نمایش در داشبورد
latest_analysis = {}
last_update_time = {"value": None}

# لیست فعلی نمادهای رصدشده (اگه DYNAMIC_SYMBOLS روشن باشه، خودکار آپدیت می‌شه)
active_symbols = {"list": list(config.SYMBOLS)}


def refresh_symbols_job():
    if not getattr(config, "DYNAMIC_SYMBOLS", False):
        return
    try:
        top, used_ex = data_fetcher.get_top_symbols(
            config.EXCHANGE_TRY_ORDER,
            quote=config.QUOTE_CURRENCY,
            top_n=config.TOP_N_SYMBOLS,
            exclude_keywords=config.EXCLUDE_KEYWORDS,
        )
        if top:
            active_symbols["list"] = top
            log.info(f"لیست نمادها آپدیت شد: {len(top)} نماد (از {used_ex})")
    except Exception as e:
        log.warning(f"آپدیت لیست نمادها ناموفق بود، لیست قبلی حفظ می‌شه: {e}")


def fetch_symbol_df(symbol):
    try:
        df, used_ex = data_fetcher.fetch_ohlcv_with_fallback(
            symbol, config.TIMEFRAME, config.CANDLE_LIMIT, config.EXCHANGE_TRY_ORDER
        )
        return df
    except Exception as e:
        log.warning(f"دریافت دیتای {symbol} ناموفق بود: {e}")
        return None


def scan_symbol(symbol):
    df = fetch_symbol_df(symbol)
    if df is None or len(df) < (config.SWING_ORDER * 2 + 5):
        return
    result = analysis.generate_signal(df, config)
    latest_analysis[symbol] = result

    current_price = result["price"]
    paper_trader.check_and_close_trades(conn, symbol, current_price, config.VIRTUAL_BALANCE_START)

    if result["signal"] and not paper_trader.has_open_trade(conn, symbol):
        sig = result["signal"]
        paper_trader.open_trade(
            conn, symbol, sig["side"], sig["entry"], sig["sl"], sig["tp"],
            config.RISK_PER_TRADE_PCT, config.VIRTUAL_BALANCE_START,
        )
        log.info(
            f"[سیگنال جدید] {symbol} {sig['side']} ورود={sig['entry']:.4f} "
            f"حدضرر={sig['sl']:.4f} حدسود={sig['tp']:.4f} R:R={sig['rr']:.2f}"
        )


def full_scan_job():
    for symbol in active_symbols["list"]:
        try:
            scan_symbol(symbol)
        except Exception as e:
            log.warning(f"اسکن {symbol} با خطا مواجه شد: {e}")
    last_update_time["value"] = datetime.utcnow().isoformat()


def price_check_job():
    # فقط نمادهایی که معامله باز دارن رو چک می‌کنه، نه کل ۲۵۰ نماد
    # (خیلی سبک‌تره و برای VPS ضعیف هم مناسبه)
    open_symbols = paper_trader.get_open_symbols(conn)
    for symbol in open_symbols:
        try:
            df = fetch_symbol_df(symbol)
            if df is not None:
                price = float(df["close"].iloc[-1])
                paper_trader.check_and_close_trades(conn, symbol, price, config.VIRTUAL_BALANCE_START)
        except Exception as e:
            log.warning(f"چک قیمت {symbol} با خطا مواجه شد: {e}")


scheduler = BackgroundScheduler()
scheduler.add_job(refresh_symbols_job, "interval", hours=config.SYMBOL_REFRESH_HOURS,
                   next_run_time=datetime.now())
scheduler.add_job(full_scan_job, "interval", minutes=config.SCAN_INTERVAL_MINUTES,
                   next_run_time=datetime.now())
scheduler.add_job(price_check_job, "interval", minutes=config.PRICE_CHECK_INTERVAL_MINUTES)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    stats = paper_trader.get_stats(conn)
    balance = round(paper_trader.get_balance(conn, config.VIRTUAL_BALANCE_START), 2)
    open_trades = paper_trader.get_open_trades(conn)
    closed_trades = paper_trader.get_closed_trades(conn)
    equity_curve = paper_trader.get_equity_curve(conn)

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
        })

    return jsonify({
        "balance": balance,
        "stats": stats,
        "symbols": symbols_view,
        "open_trades": open_trades,
        "closed_trades": closed_trades,
        "equity_curve": [{"time": t, "balance": b} for t, b in equity_curve],
        "last_update": last_update_time["value"],
        "timeframe": config.TIMEFRAME,
        "min_rr": config.MIN_RISK_REWARD,
        "symbol_count": len(active_symbols["list"]),
        "symbols_scanned": len(symbols_view),
    })


if __name__ == "__main__":
    scheduler.start()
    log.info("ربات معامله‌گر مجازی استارت شد.")
    app.run(host=config.HOST, port=config.PORT, debug=False, use_reloader=False)
