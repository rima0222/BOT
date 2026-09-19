# -*- coding: utf-8 -*-
"""
اجرای اصلی ربات: هر ۱۵ دقیقه نمادها رو اسکن می‌کنه، سیگنال تولید می‌کنه،
معامله مجازی باز می‌کنه/می‌بندد، و یک داشبورد وب روی پورت ۵۰۰۰ بالا می‌آره.

اجرا: python3 bot.py
"""
import logging
from datetime import datetime

from flask import Flask, jsonify, render_template, request
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


def get_bot_settings():
    """تنظیمات قابل‌کنترل از پنل: وضعیت روشن/متوقف، درصد ریسک، سرمایه اولیه."""
    running = paper_trader.get_setting(conn, "bot_running", "1") == "1"
    risk_pct = float(paper_trader.get_setting(conn, "risk_pct", config.RISK_PER_TRADE_PCT))
    initial_capital = paper_trader.get_setting(conn, "initial_capital", None)
    if initial_capital is None:
        initial_capital = config.VIRTUAL_BALANCE_START
        paper_trader.set_setting(conn, "initial_capital", initial_capital)
    return {
        "running": running,
        "risk_pct": risk_pct,
        "initial_capital": float(initial_capital),
    }


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


def scan_symbol(symbol, settings):
    df = fetch_symbol_df(symbol)
    if df is None or len(df) < (config.SWING_ORDER * 2 + 5):
        return
    result = analysis.generate_signal(df, config)
    latest_analysis[symbol] = result

    current_price = result["price"]
    paper_trader.check_and_close_trades(conn, symbol, current_price, settings["initial_capital"])

    # فقط وقتی ربات "روشن" باشه پوزیشن جدید باز می‌کنه؛
    # وقتی متوقفه، همچنان قیمت‌ها رو رصد و پوزیشن‌های باز رو مدیریت می‌کنه
    if settings["running"] and result["signal"] and not paper_trader.has_open_trade(conn, symbol):
        sig = result["signal"]
        paper_trader.open_trade(
            conn, symbol, sig["side"], sig["entry"], sig["sl"], sig["tp"],
            settings["risk_pct"], settings["initial_capital"],
        )
        log.info(
            f"[سیگنال جدید] {symbol} {sig['side']} ورود={sig['entry']:.4f} "
            f"حدضرر={sig['sl']:.4f} حدسود={sig['tp']:.4f} R:R={sig['rr']:.2f} "
            f"ریسک={settings['risk_pct']}%"
        )


def full_scan_job():
    settings = get_bot_settings()
    for symbol in active_symbols["list"]:
        try:
            scan_symbol(symbol, settings)
        except Exception as e:
            log.warning(f"اسکن {symbol} با خطا مواجه شد: {e}")
    last_update_time["value"] = datetime.utcnow().isoformat()


def price_check_job():
    # همیشه اجرا می‌شه (حتی وقتی ربات متوقفه) چون باید پوزیشن‌های باز رو مدیریت کنه
    settings = get_bot_settings()
    open_symbols = paper_trader.get_open_symbols(conn)
    for symbol in open_symbols:
        try:
            df = fetch_symbol_df(symbol)
            if df is not None:
                price = float(df["close"].iloc[-1])
                paper_trader.check_and_close_trades(conn, symbol, price, settings["initial_capital"])
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
    settings = get_bot_settings()
    stats = paper_trader.get_stats(conn)
    balance = round(paper_trader.get_balance(conn, settings["initial_capital"]), 2)
    open_trades = paper_trader.get_open_trades(conn)
    closed_trades = paper_trader.get_closed_trades(conn)
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
        })

    return jsonify({
        "balance": balance,
        "initial_capital": initial_capital,
        "total_return_pct": total_return_pct,
        "is_profitable": balance >= initial_capital,
        "bot_running": settings["running"],
        "risk_pct": settings["risk_pct"],
        "allowed_risk_levels": config.ALLOWED_RISK_LEVELS,
        "open_positions_count": len(open_trades),
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


@app.route("/api/control", methods=["POST"])
def api_control():
    """کنترل از پنل: روشن/متوقف کردن معامله‌گیری، تغییر درصد ریسک، تغییر سرمایه اولیه."""
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

    if "initial_capital" in body:
        try:
            amount = float(body["initial_capital"])
            if amount <= 0:
                return jsonify({"ok": False, "error": "سرمایه باید بزرگ‌تر از صفر باشد"}), 400
            paper_trader.reset_capital(conn, amount)
            log.info(f"[کنترل پنل] سرمایه اولیه به {amount} تنظیم شد")
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "مقدار سرمایه نامعتبر است"}), 400

    return jsonify({"ok": True, "settings": get_bot_settings()})


if __name__ == "__main__":
    scheduler.start()
    log.info("ربات معامله‌گر مجازی استارت شد.")
    app.run(host=config.HOST, port=config.PORT, debug=False, use_reloader=False)
