# -*- coding: utf-8 -*-
"""
اجرای اصلی ربات: هر ۱۵ دقیقه نمادها رو اسکن می‌کنه، سیگنال تولید می‌کنه،
معامله مجازی باز می‌کنه/می‌بندد، و یک داشبورد وب بالا می‌آره.

اجرا: python3 bot.py
"""
import io
import logging
import tarfile
from datetime import datetime

import psutil
from flask import Flask, jsonify, render_template, request, send_file
from apscheduler.schedulers.background import BackgroundScheduler

import config
import data_fetcher
import analysis
import paper_trader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tradingbot")

app = Flask(__name__)
conn = paper_trader.get_conn(config.DB_PATH)

# پیش‌گرم‌کردن اندازه‌گیری CPU (اولین فراخوانی psutil.cpu_percent همیشه ۰ برمی‌گردونه)
psutil.cpu_percent(interval=None)

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


def fetch_symbol_df(symbol, timeframe=None, limit=None):
    timeframe = timeframe or config.TIMEFRAME
    limit = limit or config.CANDLE_LIMIT
    try:
        df, used_ex = data_fetcher.fetch_ohlcv_with_fallback(
            symbol, timeframe, limit, config.EXCHANGE_TRY_ORDER
        )
        return df
    except Exception as e:
        log.warning(f"دریافت دیتای {symbol} ({timeframe}) ناموفق بود: {e}")
        return None


def check_htf_confirmation(symbol, side):
    """
    تایید چند-تایم‌فریمی: فقط برای سیگنال‌های کاندید صدا زده می‌شه (نه برای هر نماد)،
    پس هزینه‌ی شبکه‌اش تقریباً ناچیزه. روند رو توی تایم‌فریم‌های بالاتر چک می‌کنه و
    می‌شمره چندتاشون هم‌جهت با سیگنال هستن.
    """
    if not getattr(config, "USE_HTF_CONFIRMATION", False):
        return True, "غیرفعال"

    agree, disagree, neutral = 0, 0, 0
    checked = []
    wanted_trend = "uptrend" if side == "LONG" else "downtrend"
    opposite_trend = "downtrend" if side == "LONG" else "uptrend"

    for tf in config.HTF_TIMEFRAMES:
        df = fetch_symbol_df(symbol, timeframe=tf, limit=config.HTF_CANDLE_LIMIT)
        if df is None or len(df) < (config.SWING_ORDER * 2 + 5):
            continue
        trend = analysis.trend_from_df(df, swing_order=config.SWING_ORDER)
        checked.append(f"{tf}:{trend}")
        if trend == wanted_trend:
            agree += 1
        elif trend == opposite_trend:
            disagree += 1
        else:
            neutral += 1

    ok = agree >= config.HTF_MIN_AGREEMENT
    detail = f"موافق={agree} مخالف={disagree} خنثی={neutral} ({', '.join(checked)})"
    return ok, detail


def scan_symbol(symbol, settings):
    df = fetch_symbol_df(symbol)
    if df is None or len(df) < (config.SWING_ORDER * 2 + 5):
        return
    result = analysis.generate_signal(df, config)
    latest_analysis[symbol] = result

    current_price = result["price"]
    paper_trader.check_and_close_trades(
        conn, symbol, current_price, settings["initial_capital"], config.COST_PCT_PER_SIDE
    )

    if not (settings["running"] and result["signal"] and not paper_trader.has_open_trade(conn, symbol)):
        return

    sig = result["signal"]

    # ۱) کول‌داون: بعد از بسته‌شدن آخرین معامله‌ی این نماد، به‌اندازه‌ی کافی صبر شده؟
    if paper_trader.is_in_cooldown(conn, symbol, config.COOLDOWN_HOURS):
        return

    # ۲) تایید چند-تایم‌فریمی: روند تایم‌فریم‌های بالاتر هم باید هم‌جهت باشه
    htf_ok, htf_detail = check_htf_confirmation(symbol, sig["side"])
    if not htf_ok:
        log.info(f"[رد شد - عدم تایید تایم‌فریم بالاتر] {symbol} {sig['side']} | {htf_detail}")
        return

    # ۳) باز کردن پوزیشن مجازی (با رعایت واقعی سرمایه، لوریج ایمن و سقف تنوع)
    trade_res = paper_trader.open_trade(
        conn, symbol, sig["side"], sig["entry"], sig["sl"], sig["tp"],
        settings["risk_pct"], settings["initial_capital"],
        min_notional=config.MIN_NOTIONAL_USD, max_open_positions=config.MAX_OPEN_POSITIONS,
        max_leverage=config.MAX_LEVERAGE, leverage_safety_mult=config.LEVERAGE_SAFETY_MULTIPLIER,
        position_pct_cap=config.MAX_POSITION_PCT_OF_CAPITAL,
    )

    if trade_res["opened"]:
        cap_note = " (حجم به‌خاطر سقف تنوع/سرمایه‌ی آزاد کوچک‌تر شد)" if trade_res.get("capped") else ""
        log.info(
            f"[سیگنال جدید ✅] {symbol} {sig['side']} ورود={sig['entry']:.4f} "
            f"حدضرر={sig['sl']:.4f} حدسود={sig['tp']:.4f} R:R={sig['rr']:.2f} "
            f"ریسک={settings['risk_pct']}% لوریج={trade_res['leverage']}x "
            f"مارجین=${trade_res['margin']:.2f} ارزش‌پوزیشن=${trade_res['notional']:.2f}{cap_note} | HTF: {htf_detail}"
        )
    else:
        log.info(f"[سیگنال رد شد - {trade_res['reason']}] {symbol} {sig['side']}")


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
    # فقط نمادهایی که معامله باز دارن رو چک می‌کنه، نه کل لیست بزرگ نمادها (سبک می‌مونه)
    settings = get_bot_settings()
    open_symbols = paper_trader.get_open_symbols(conn)
    for symbol in open_symbols:
        try:
            df = fetch_symbol_df(symbol)
            if df is not None:
                price = float(df["close"].iloc[-1])
                paper_trader.check_and_close_trades(
                    conn, symbol, price, settings["initial_capital"], config.COST_PCT_PER_SIDE
                )
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
    locked_capital = round(paper_trader.get_locked_capital(conn), 2)
    available_capital = round(balance - locked_capital, 2)
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
        "min_rr": config.MIN_RISK_REWARD,
        "cost_pct_per_side": config.COST_PCT_PER_SIDE,
        "max_leverage": config.MAX_LEVERAGE,
        "max_position_pct": config.MAX_POSITION_PCT_OF_CAPITAL,
        "htf_enabled": config.USE_HTF_CONFIRMATION,
        "htf_timeframes": config.HTF_TIMEFRAMES,
        "system": system_stats,
        "stats": stats,
        "symbols": symbols_view,
        "open_trades": open_trades,
        "closed_trades": closed_trades,
        "equity_curve": [{"time": t, "balance": b} for t, b in equity_curve],
        "last_update": last_update_time["value"],
        "timeframe": config.TIMEFRAME,
        "symbol_count": len(active_symbols["list"]),
        "symbols_scanned": len(symbols_view),
    })


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


@app.route("/api/close_trade", methods=["POST"])
def api_close_trade():
    """بستن دستی یک پوزیشن باز از پنل، با قیمت لحظه‌ای فعلی."""
    body = request.get_json(force=True, silent=True) or {}
    trade_id = body.get("id")
    if not trade_id:
        return jsonify({"ok": False, "error": "شناسه معامله مشخص نشده"}), 400

    row = conn.execute("SELECT symbol FROM trades WHERE id=? AND status='OPEN'", (trade_id,)).fetchone()
    if not row:
        return jsonify({"ok": False, "error": "پوزیشن باز پیدا نشد"}), 404
    symbol = row[0]

    df = fetch_symbol_df(symbol)
    if df is None:
        return jsonify({"ok": False, "error": "دریافت قیمت لحظه‌ای ناموفق بود"}), 500
    current_price = float(df["close"].iloc[-1])

    settings = get_bot_settings()
    result = paper_trader.close_trade_manually(
        conn, trade_id, current_price, settings["initial_capital"], config.COST_PCT_PER_SIDE
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
    buf.seek(0)
    filename = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/gzip")


if __name__ == "__main__":
    scheduler.start()
    log.info("ربات معامله‌گر مجازی استارت شد.")
    app.run(host=config.HOST, port=config.PORT, debug=False, use_reloader=False)
