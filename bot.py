# -*- coding: utf-8 -*-
"""
اجرای اصلی ربات: هر ۱۵ دقیقه نمادها رو اسکن می‌کنه، سیگنال تولید می‌کنه،
معامله مجازی باز می‌کنه/می‌بندد، و یک داشبورد وب بالا می‌آره.

اجرا: python3 bot.py
"""
import io
import json
import logging
import tarfile
import threading
import uuid
from datetime import datetime

import psutil
from flask import Flask, jsonify, render_template, request, send_file
from apscheduler.schedulers.background import BackgroundScheduler

import config
import data_fetcher
import analysis
import strategies
import paper_trader
import backtest
import backtest_analyzer

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

    active_strategies_raw = paper_trader.get_setting(conn, "active_strategies", None)
    if active_strategies_raw:
        try:
            active_strategies = json.loads(active_strategies_raw)
        except Exception:
            active_strategies = list(config.ACTIVE_STRATEGIES)
    else:
        active_strategies = list(config.ACTIVE_STRATEGIES)

    combine_mode = paper_trader.get_setting(conn, "strategy_combine_mode", config.STRATEGY_COMBINE_MODE)

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


def get_effective_cfg(settings):
    """یک نسخه‌ی موقت از تنظیمات که سطح سخت‌گیری و استراتژی‌های انتخاب‌شده از پنل
    روش اعمال شده — بدون این‌که به config ماژول اصلی دست بزنه."""
    overrides = {
        "MIN_RISK_REWARD": settings["min_rr"],
        "ACTIVE_STRATEGIES": settings["active_strategies"],
        "STRATEGY_COMBINE_MODE": settings["combine_mode"],
    }
    return backtest.build_config(config, overrides)


def check_htf_confirmation(symbol, side, min_agreement):
    """
    تایید چند-تایم‌فریمی: فقط برای سیگنال‌های کاندید صدا زده می‌شه (نه برای هر نماد)،
    پس هزینه‌ی شبکه‌اش تقریباً ناچیزه. روند رو توی تایم‌فریم‌های بالاتر چک می‌کنه و
    می‌شمره چندتاشون هم‌جهت با سیگنال هستن.
    """
    if not getattr(config, "USE_HTF_CONFIRMATION", False):
        return True, "غیرفعال", 0

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

    ok = agree >= min_agreement
    detail = f"موافق={agree}/{len(config.HTF_TIMEFRAMES)} (حداقل لازم={min_agreement}) مخالف={disagree} خنثی={neutral} ({', '.join(checked)})"
    return ok, detail, agree


def scan_symbol(symbol, settings):
    df = fetch_symbol_df(symbol)
    if df is None or len(df) < (config.SWING_ORDER * 2 + 5):
        return
    cfg = get_effective_cfg(settings)
    result = strategies.generate_combined_signal(df, cfg)
    latest_analysis[symbol] = result

    current_price = result["price"]
    latest_prices[symbol] = current_price
    paper_trader.check_and_close_trades(
        conn, symbol, current_price, settings["initial_capital"],
        config.MAKER_FEE_PCT, config.TAKER_FEE_PCT, config.TAKER_SLIPPAGE_PCT,
    )

    if not result["signal"]:
        return

    sig = result["signal"]
    strategy_name = sig.get("strategy", "unknown")

    def reject(reason, htf_agree=None):
        paper_trader.log_signal(conn, symbol, sig["side"], strategy_name, sig["entry"], sig["sl"],
                                 sig["tp"], sig["rr"], htf_agree, opened=False, rejection_reason=reason)

    if not settings["running"]:
        return  # وقتی متوقفه، حتی سیگنال رد‌شده رو لاگ نمی‌کنیم (فقط پایشه، تصمیمی نمی‌گیره)

    if paper_trader.has_open_trade(conn, symbol):
        return  # از قبل پوزیشن باز داره؛ این یه سیگنال جدید مستقل نیست که رد بشه

    # ۱) کول‌داون
    if paper_trader.is_in_cooldown(conn, symbol, config.COOLDOWN_HOURS):
        reject("cooldown")
        return

    # ۲) تایید چند-تایم‌فریمی با سطح سخت‌گیری فعلی
    htf_ok, htf_detail, htf_agree = check_htf_confirmation(symbol, sig["side"], settings["htf_min_agreement"])
    if not htf_ok:
        log.info(f"[رد شد - عدم تایید تایم‌فریم بالاتر] {symbol} {sig['side']} | {htf_detail}")
        reject("htf_disagreement", htf_agree)
        return

    # ۳) باز کردن پوزیشن مجازی (با رعایت واقعی سرمایه، لوریج ایمن و سقف تنوع)
    trade_res = paper_trader.open_trade(
        conn, symbol, sig["side"], sig["entry"], sig["sl"], sig["tp"],
        settings["risk_pct"], settings["initial_capital"],
        min_notional=config.MIN_NOTIONAL_USD, max_open_positions=config.MAX_OPEN_POSITIONS,
        max_leverage=config.MAX_LEVERAGE, leverage_safety_mult=config.LEVERAGE_SAFETY_MULTIPLIER,
        position_pct_cap=config.MAX_POSITION_PCT_OF_CAPITAL,
        trailing_enabled=settings["trailing_enabled"], strategy_name=strategy_name,
    )

    paper_trader.log_signal(conn, symbol, sig["side"], strategy_name, sig["entry"], sig["sl"], sig["tp"],
                             sig["rr"], htf_agree, opened=trade_res["opened"],
                             rejection_reason=None if trade_res["opened"] else trade_res["reason"])

    if trade_res["opened"]:
        cap_note = " (حجم به‌خاطر سقف تنوع/سرمایه‌ی آزاد کوچک‌تر شد)" if trade_res.get("capped") else ""
        trail_note = " | تریلینگ: روشن" if settings["trailing_enabled"] else ""
        log.info(
            f"[سیگنال جدید ✅] {symbol} {sig['side']} استراتژی={strategy_name} ورود={sig['entry']:.4f} "
            f"حدضرر={sig['sl']:.4f} حدسود={sig['tp']:.4f} R:R={sig['rr']:.2f} "
            f"ریسک={settings['risk_pct']}% لوریج={trade_res['leverage']}x "
            f"مارجین=${trade_res['margin']:.2f} ارزش‌پوزیشن=${trade_res['notional']:.2f}{cap_note}{trail_note} | HTF: {htf_detail}"
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
                latest_prices[symbol] = price

                if settings["trailing_enabled"]:
                    paper_trader.update_trailing_stops(
                        conn, symbol, price, config.TRAILING_SL_LADDER, config.TRAILING_SL_BEYOND_DISTANCE_R
                    )

                if config.LOG_POSITION_PRICE_HISTORY:
                    for trade_id in paper_trader.get_open_trade_ids(conn, symbol):
                        paper_trader.log_price_snapshot(conn, trade_id, price)
                    paper_trader.commit(conn)

                paper_trader.check_and_close_trades(
                    conn, symbol, price, settings["initial_capital"],
                    config.MAKER_FEE_PCT, config.TAKER_FEE_PCT, config.TAKER_SLIPPAGE_PCT,
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
    for t in open_trades:
        t.update(compute_live_position_metrics(t, latest_prices.get(t["symbol"])))
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
        "min_rr": settings["min_rr"],
        "maker_fee_pct": config.MAKER_FEE_PCT,
        "taker_fee_pct": config.TAKER_FEE_PCT,
        "taker_slippage_pct": config.TAKER_SLIPPAGE_PCT,
        "max_leverage": config.MAX_LEVERAGE,
        "max_position_pct": config.MAX_POSITION_PCT_OF_CAPITAL,
        "htf_enabled": config.USE_HTF_CONFIRMATION,
        "htf_timeframes": config.HTF_TIMEFRAMES,
        "strictness": settings["strictness"],
        "htf_min_agreement": settings["htf_min_agreement"],
        "strictness_presets": {k: v["label"] for k, v in config.STRICTNESS_PRESETS.items()},
        "trailing_enabled": settings["trailing_enabled"],
        "trailing_ladder": config.TRAILING_SL_LADDER,
        "trailing_beyond_r": config.TRAILING_SL_BEYOND_DISTANCE_R,
        "active_strategies": settings["active_strategies"],
        "combine_mode": settings["combine_mode"],
        "available_strategies": {k: v["label"] for k, v in strategies.STRATEGY_REGISTRY.items()},
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


@app.route("/api/signal_log")
def api_signal_log():
    limit = int(request.args.get("limit", 100))
    return jsonify({"ok": True, "signals": paper_trader.get_signal_log(conn, limit=min(limit, 500))})


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

    if "active_strategies" in body:
        chosen = [s for s in body["active_strategies"] if s in strategies.STRATEGY_REGISTRY]
        if not chosen:
            return jsonify({"ok": False, "error": "حداقل یک استراتژی باید فعال باشه"}), 400
        paper_trader.set_setting(conn, "active_strategies", json.dumps(chosen))
        log.info(f"[کنترل پنل] استراتژی‌های فعال: {chosen}")

    if "combine_mode" in body:
        if body["combine_mode"] in ("any", "all"):
            paper_trader.set_setting(conn, "strategy_combine_mode", body["combine_mode"])
            log.info(f"[کنترل پنل] حالت ترکیب استراتژی‌ها: {body['combine_mode']}")
        else:
            return jsonify({"ok": False, "error": "حالت ترکیب نامعتبر است"}), 400

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
    buf.seek(0)
    filename = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/gzip")


# ==================== بک‌تست (تحلیل تاریخی ۱-۲ ساله) ====================
# اجرا در پس‌زمینه (نه توی همون درخواست HTTP) چون ممکنه چند دقیقه طول بکشه.
# فقط یک بک‌تست هم‌زمان مجازه تا فشار غیرضروری روی VPS نیفته.

backtest_jobs = {}          # job_id -> {state, progress, result, error, started_at}
backtest_lock = threading.Lock()
backtest_running_job = {"id": None}


def _run_backtest_job(job_id, symbols, days, overrides):
    def progress_cb(msg):
        backtest_jobs[job_id]["progress"] = msg

    try:
        backtest_jobs[job_id]["state"] = "running"
        conn_bt, meta = backtest.run_backtest(symbols, days, config, overrides, progress_cb)
        cfg_bt = backtest.build_config(config, overrides)
        report = backtest_analyzer.analyze(conn_bt, cfg_bt, meta)
        backtest_jobs[job_id]["result"] = report
        backtest_jobs[job_id]["state"] = "done"
        backtest_jobs[job_id]["progress"] = "تمام شد"
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
        top_n = max(1, min(top_n, 60))
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
    }
    overrides = {k: v for k, v in overrides.items() if k in allowed_override_keys}

    # میان‌بر راحت: اگه به‌جای پارامترهای تک‌تک، فقط اسم سطح سخت‌گیری داده بشه
    strictness_name = body.get("strictness")
    if strictness_name and strictness_name in config.STRICTNESS_PRESETS:
        preset = config.STRICTNESS_PRESETS[strictness_name]
        overrides.setdefault("HTF_MIN_AGREEMENT", preset["HTF_MIN_AGREEMENT"])
        overrides.setdefault("MIN_RISK_REWARD", preset["MIN_RISK_REWARD"])

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
    t = threading.Thread(target=_run_backtest_job, args=(job_id, symbols, days, overrides), daemon=True)
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


if __name__ == "__main__":
    scheduler.start()
    log.info("ربات معامله‌گر مجازی استارت شد.")
    app.run(host=config.HOST, port=config.PORT, debug=False, use_reloader=False)
