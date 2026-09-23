# -*- coding: utf-8 -*-
"""
موتور بک‌تست.

اصل طراحی: این موتور نباید هیچ منطق جدا و موازی با ربات زنده داشته باشه، وگرنه
نتیجه‌ی بک‌تست هیچ ربطی به رفتار واقعی ربات نداره. برای همین:
  - سیگنال‌دهی از همون analysis.py (determine_trend, get_support_resistance,
    is_rejection_candle, compute_atr) استفاده می‌کنه.
  - باز/بستن پوزیشن، مارجین، لوریج، کارمزد، کول‌داون، سقف پوزیشن همه از همون
    paper_trader.py میان (روی یک دیتابیس SQLite ایزوله در حافظه، کاملاً جدا از
    دیتابیس زنده‌ی ربات).
  - تنها تفاوت: داده‌ی قیمت به‌جای صرافی زنده، از دیتای تاریخی گام‌به‌گام میاد.

نکته‌ی حیاتی: هیچ محاسبه‌ای نباید از دیتای *بعد* از لحظه‌ی فعلیِ شبیه‌سازی استفاده
کنه (no lookahead bias). برای همین سوینگ‌ها فقط با یک تاخیر تاییدی (SWING_ORDER
کندل) "دیده‌شده" حساب می‌شن — دقیقاً همون رفتاری که ربات زنده روی دیتای در حال
تکمیل داره.
"""
import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ccxt

import analysis
import paper_trader

log = logging.getLogger("backtest")


def _get_exchange(name):
    ex_class = getattr(ccxt, name)
    return ex_class({"enableRateLimit": True, "timeout": 20000})


def fetch_full_history(symbol, timeframe, since_ms, exchange_order, limit_per_call=1000):
    """دریافت کل تاریخچه‌ی یک نماد از since_ms تا الان، با صفحه‌بندی خودکار."""
    last_err = None
    for ex_name in exchange_order:
        try:
            ex = _get_exchange(ex_name)
            all_rows = []
            cursor = since_ms
            now_ms = int(time.time() * 1000)
            guard = 0
            while cursor < now_ms:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit_per_call)
                if not batch:
                    break
                all_rows.extend(batch)
                last_ts = batch[-1][0]
                if last_ts <= cursor:
                    break
                cursor = last_ts + 1
                guard += 1
                if guard > 5000:  # محافظ در برابر حلقه‌ی بی‌نهایت غیرمنتظره
                    break
            if not all_rows:
                continue
            df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df.drop_duplicates(subset="timestamp", inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df, ex_name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"دریافت تاریخچه‌ی {symbol} ({timeframe}) ناموفق بود: {last_err}")


def _precompute(df, cfg):
    """پیش‌محاسبه‌ی وکتوریزه (یک‌بار برای کل سری) به‌جای محاسبه‌ی تکراری در هر گام."""
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    opens = df["open"].values.astype(float)
    volumes = df["volume"].values.astype(float)

    order = cfg.SWING_ORDER
    win = 2 * order + 1
    roll_max = pd.Series(highs).rolling(win, center=True).max().values
    roll_min = pd.Series(lows).rolling(win, center=True).min().values
    is_sh = (~np.isnan(roll_max)) & (highs == roll_max)
    is_sl = (~np.isnan(roll_min)) & (lows == roll_min)

    atr = analysis.compute_atr(df, cfg.ATR_PERIOD).values

    change = closes - opens
    up_vol = pd.Series(np.where(change > 0, volumes, np.nan))
    down_vol = pd.Series(np.where(change < 0, volumes, np.nan))
    up_vol_mean = up_vol.rolling(20, min_periods=1).mean().values
    down_vol_mean = down_vol.rolling(20, min_periods=1).mean().values

    return {
        "is_sh": is_sh, "is_sl": is_sl, "atr": atr,
        "up_vol_mean": up_vol_mean, "down_vol_mean": down_vol_mean,
        "highs": highs, "lows": lows, "closes": closes, "opens": opens,
        "sh_positions": np.where(is_sh)[0], "sl_positions": np.where(is_sl)[0],
    }


def _htf_trend_series(pre, order):
    """روند تئوری داو برای هر کندلِ یک تایم‌فریم بالاتر، با همون تاخیر تاییدی."""
    n = len(pre["is_sh"])
    trends = np.full(n, "sideways", dtype=object)
    sh_pos, sl_pos = pre["sh_positions"], pre["sl_positions"]
    sh_ptr = sl_ptr = 0
    for t in range(n):
        limit = t - order
        while sh_ptr < len(sh_pos) and sh_pos[sh_ptr] <= limit:
            sh_ptr += 1
        while sl_ptr < len(sl_pos) and sl_pos[sl_ptr] <= limit:
            sl_ptr += 1
        if sh_ptr < 2 or sl_ptr < 2:
            continue
        h1, h2 = pre["highs"][sh_pos[sh_ptr - 2]], pre["highs"][sh_pos[sh_ptr - 1]]
        l1, l2 = pre["lows"][sl_pos[sl_ptr - 2]], pre["lows"][sl_pos[sl_ptr - 1]]
        if h2 > h1 and l2 > l1:
            trends[t] = "uptrend"
        elif h2 < h1 and l2 < l1:
            trends[t] = "downtrend"
    return trends


def _simulate_symbol(conn, symbol, df, htf_frames, cfg, progress_cb=None):
    pre = _precompute(df, cfg)
    n = len(df)
    order = cfg.SWING_ORDER
    timestamps = df["timestamp"].values

    # --- پیش‌محاسبه‌ی روند هر تایم‌فریم بالاتر + هم‌ترازسازی با تایم‌فریم اصلی (merge_asof وکتوریزه) ---
    htf_trend_at_bar = {}
    if cfg.USE_HTF_CONFIRMATION:
        for tf, df_htf in htf_frames.items():
            if len(df_htf) < order * 2 + 5:
                continue
            pre_htf = _precompute(df_htf, cfg)
            trend_series = _htf_trend_series(pre_htf, order)
            tdf = pd.DataFrame({"timestamp": df_htf["timestamp"], "trend": trend_series})
            merged = pd.merge_asof(
                df[["timestamp"]], tdf.sort_values("timestamp"), on="timestamp", direction="backward"
            )
            htf_trend_at_bar[tf] = merged["trend"].fillna("sideways").values

    warmup = max(order * 2 + 5, cfg.ATR_PERIOD + 1, 25)
    sh_ptr = sl_ptr = 0
    sh_pos, sl_pos = pre["sh_positions"], pre["sl_positions"]
    trades_opened = 0
    trailing_enabled = getattr(cfg, "USE_TRAILING_SL", False)
    ladder = getattr(cfg, "TRAILING_SL_LADDER", [])
    beyond_r = getattr(cfg, "TRAILING_SL_BEYOND_DISTANCE_R", 0.7)
    maker_fee = getattr(cfg, "MAKER_FEE_PCT", 0.0)
    taker_fee = getattr(cfg, "TAKER_FEE_PCT", 0.0)
    taker_slip = getattr(cfg, "TAKER_SLIPPAGE_PCT", 0.0)

    for t in range(warmup, n):
        current_price = float(pre["closes"][t])
        current_time = timestamps[t]
        sim_time = pd.Timestamp(current_time).to_pydatetime()

        # ۱) اول همیشه پوزیشن‌های باز رو مدیریت کن (دقیقاً مثل زنده): تریلینگ، بعد چک بسته‌شدن
        if trailing_enabled:
            paper_trader.update_trailing_stops(conn, symbol, current_price, ladder, beyond_r, as_of=sim_time)
        paper_trader.check_and_close_trades(
            conn, symbol, current_price, cfg.VIRTUAL_BALANCE_START,
            maker_fee, taker_fee, taker_slip, as_of=sim_time,
        )
        if paper_trader.has_open_trade(conn, symbol):
            continue

        # ۲) سوینگ‌های "دیده‌شده تا این لحظه" — پیشروی تک‌جهته (two-pointer)، بدون بازمحاسبه
        limit = t - order
        while sh_ptr < len(sh_pos) and sh_pos[sh_ptr] <= limit:
            sh_ptr += 1
        while sl_ptr < len(sl_pos) and sl_pos[sl_ptr] <= limit:
            sl_ptr += 1
        if sh_ptr < 2 or sl_ptr < 2:
            continue

        swing_highs = [(int(sh_pos[i]), float(pre["highs"][sh_pos[i]])) for i in (sh_ptr - 2, sh_ptr - 1)]
        swing_lows = [(int(sl_pos[i]), float(pre["lows"][sl_pos[i]])) for i in (sl_ptr - 2, sl_ptr - 1)]
        trend = analysis.determine_trend(swing_highs, swing_lows)
        if trend == "sideways":
            continue

        atr = pre["atr"][t]
        if np.isnan(atr) or atr <= 0:
            continue

        all_sh_prices = pre["highs"][sh_pos[:sh_ptr]]
        all_sl_prices = pre["lows"][sl_pos[:sl_ptr]]
        nearest_support, nearest_resistance = analysis.get_support_resistance(
            [(0, p) for p in all_sh_prices], [(0, p) for p in all_sl_prices],
            current_price, cfg.SR_CLUSTER_PCT
        )

        uvm, dvm = pre["up_vol_mean"][t], pre["down_vol_mean"][t]
        vol_up_ok = (not np.isnan(uvm)) and (not np.isnan(dvm)) and uvm > dvm
        vol_down_ok = (not np.isnan(uvm)) and (not np.isnan(dvm)) and dvm > uvm

        signal = None
        candle_row_df = df.iloc[t:t + 1]

        if trend == "uptrend" and nearest_support and vol_up_ok:
            dist_pct = (current_price - nearest_support) / nearest_support * 100
            rej_ok = (not cfg.USE_REJECTION_CONFIRMATION) or analysis.is_rejection_candle(
                candle_row_df, "LONG", nearest_support, cfg.PROXIMITY_PCT
            )
            if 0 <= dist_pct <= cfg.PROXIMITY_PCT and rej_ok:
                entry = current_price
                sl = nearest_support - atr * cfg.ATR_SL_BUFFER
                risk = entry - sl
                if risk > 0:
                    tp_min = entry + risk * cfg.MIN_RISK_REWARD
                    tp = max(tp_min, nearest_resistance) if nearest_resistance else tp_min
                    rr = (tp - entry) / risk
                    if rr >= cfg.MIN_RISK_REWARD:
                        signal = {"side": "LONG", "entry": entry, "sl": sl, "tp": tp, "rr": rr}

        if signal is None and trend == "downtrend" and nearest_resistance and vol_down_ok:
            dist_pct = (nearest_resistance - current_price) / nearest_resistance * 100
            rej_ok = (not cfg.USE_REJECTION_CONFIRMATION) or analysis.is_rejection_candle(
                candle_row_df, "SHORT", nearest_resistance, cfg.PROXIMITY_PCT
            )
            if 0 <= dist_pct <= cfg.PROXIMITY_PCT and rej_ok:
                entry = current_price
                sl = nearest_resistance + atr * cfg.ATR_SL_BUFFER
                risk = sl - entry
                if risk > 0:
                    tp_min = entry - risk * cfg.MIN_RISK_REWARD
                    tp = min(tp_min, nearest_support) if nearest_support else tp_min
                    rr = (entry - tp) / risk
                    if rr >= cfg.MIN_RISK_REWARD:
                        signal = {"side": "SHORT", "entry": entry, "sl": sl, "tp": tp, "rr": rr}

        if not signal:
            continue

        strategy_name = "dow_support_resistance"

        if paper_trader.is_in_cooldown(conn, symbol, cfg.COOLDOWN_HOURS, as_of=sim_time):
            paper_trader.log_signal(conn, symbol, signal["side"], strategy_name, signal["entry"], signal["sl"],
                                     signal["tp"], signal["rr"], None, opened=False,
                                     rejection_reason="cooldown", as_of=sim_time)
            continue

        htf_agree = None
        if cfg.USE_HTF_CONFIRMATION and htf_trend_at_bar:
            wanted = "uptrend" if signal["side"] == "LONG" else "downtrend"
            htf_agree = sum(1 for arr in htf_trend_at_bar.values() if arr[t] == wanted)
            if htf_agree < cfg.HTF_MIN_AGREEMENT:
                paper_trader.log_signal(conn, symbol, signal["side"], strategy_name, signal["entry"], signal["sl"],
                                         signal["tp"], signal["rr"], htf_agree, opened=False,
                                         rejection_reason="htf_disagreement", as_of=sim_time)
                continue

        res = paper_trader.open_trade(
            conn, symbol, signal["side"], signal["entry"], signal["sl"], signal["tp"],
            cfg.RISK_PER_TRADE_PCT, cfg.VIRTUAL_BALANCE_START,
            min_notional=cfg.MIN_NOTIONAL_USD, max_open_positions=cfg.MAX_OPEN_POSITIONS,
            max_leverage=cfg.MAX_LEVERAGE, leverage_safety_mult=cfg.LEVERAGE_SAFETY_MULTIPLIER,
            position_pct_cap=cfg.MAX_POSITION_PCT_OF_CAPITAL,
            trailing_enabled=trailing_enabled, strategy_name=strategy_name, as_of=sim_time,
        )

        paper_trader.log_signal(conn, symbol, signal["side"], strategy_name, signal["entry"], signal["sl"],
                                 signal["tp"], signal["rr"], htf_agree, opened=res["opened"],
                                 rejection_reason=None if res["opened"] else res["reason"], as_of=sim_time)

        if res["opened"]:
            trades_opened += 1
            # چند فیلد اضافه که فقط توی بک‌تست برای تحلیل عمیق‌تر لازمه (اختیاری، اگه ستونش نبود صرف‌نظر کن)
            if htf_agree is not None:
                try:
                    trade_id = conn.execute(
                        "SELECT id FROM trades WHERE status='OPEN' AND symbol=? ORDER BY id DESC LIMIT 1", (symbol,)
                    ).fetchone()
                    if trade_id:
                        conn.execute("UPDATE trades SET rr_planned=?, htf_agree=? WHERE id=?",
                                     (signal["rr"], htf_agree, trade_id[0]))
                        conn.commit()
                except Exception:
                    pass

        if progress_cb and t % 2000 == 0:
            progress_cb(f"{symbol}: گام {t}/{n}")

    return trades_opened


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


def run_backtest(symbols, days, live_config, overrides=None, progress_cb=None):
    """
    اجرای کامل بک‌تست روی چند نماد، برمی‌گردونه: (conn, meta)
    conn: دیتابیس SQLite در حافظه با همون ساختار جدول trades/equity ربات زنده
    meta: دیکشنری شامل جزئیات هر نماد (تعداد کندل، خطا در صورت وجود، و ...)
    """
    cfg = build_config(live_config, overrides)
    since_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    conn = paper_trader.get_conn(":memory:")
    # ستون‌های اضافه‌ی مخصوص بک‌تست (برای تحلیل عمیق‌تر، ربطی به ربات زنده نداره)
    for col, coltype in (("rr_planned", "REAL"), ("htf_agree", "INTEGER")):
        try:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {coltype}")
            conn.commit()
        except Exception:
            pass
    sim_start_time = datetime.utcfromtimestamp(since_ms / 1000)
    paper_trader.reset_capital(conn, cfg.VIRTUAL_BALANCE_START, as_of=sim_start_time)

    meta = {"symbols": {}, "start_time": datetime.utcnow().isoformat()}

    for idx, symbol in enumerate(symbols):
        if progress_cb:
            progress_cb(f"دریافت دیتای {symbol} ({idx + 1}/{len(symbols)})")
        try:
            df_main, used_ex = fetch_full_history(symbol, cfg.TIMEFRAME, since_ms, cfg.EXCHANGE_TRY_ORDER)
        except Exception as e:
            meta["symbols"][symbol] = {"ok": False, "error": str(e)}
            continue

        if len(df_main) < cfg.SWING_ORDER * 2 + 25:
            meta["symbols"][symbol] = {"ok": False, "error": "دیتای کافی نیست"}
            continue

        htf_frames = {}
        if cfg.USE_HTF_CONFIRMATION:
            for tf in cfg.HTF_TIMEFRAMES:
                try:
                    df_htf, _ = fetch_full_history(symbol, tf, since_ms, cfg.EXCHANGE_TRY_ORDER)
                    htf_frames[tf] = df_htf
                except Exception:
                    continue

        if progress_cb:
            progress_cb(f"شبیه‌سازی {symbol} ({idx + 1}/{len(symbols)}) — {len(df_main)} کندل")

        try:
            trades_opened = _simulate_symbol(conn, symbol, df_main, htf_frames, cfg, progress_cb)
            meta["symbols"][symbol] = {
                "ok": True, "candles": len(df_main), "trades_opened": trades_opened,
                "from": str(df_main["timestamp"].iloc[0]), "to": str(df_main["timestamp"].iloc[-1]),
            }
        except Exception as e:
            meta["symbols"][symbol] = {"ok": False, "error": str(e)}

    meta["end_time"] = datetime.utcnow().isoformat()
    return conn, meta
