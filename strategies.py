# -*- coding: utf-8 -*-
"""
فریم‌ورک استراتژی قابل‌ترکیب.

هر استراتژی یک تابع generate(df, cfg) -> signal_dict|None هست. توی
config.ACTIVE_STRATEGIES مشخص می‌شه کدوم‌ها فعالن، و config.STRATEGY_COMBINE_MODE
مشخص می‌کنه چطور ترکیب بشن:
  - "any": هر کدوم از استراتژی‌های فعال سیگنال داد، همون قبول می‌شه (اولین سیگنال
    غیر خالی که پیدا بشه، به ترتیب لیست ACTIVE_STRATEGIES).
  - "all": باید همه‌ی استراتژی‌های فعال روی یک جهت (LONG/SHORT) توافق داشته باشن؛
    وگرنه سیگنالی صادر نمی‌شه. این حالت طبیعتاً خیلی سخت‌گیرتره.

برای اضافه‌کردن یک استراتژی جدید: یک تابع generate_xxx(df, cfg) بنویس که همون
فرمت خروجی analysis.generate_signal رو برگردونه (دیکشنری با trend/price/signal)،
و اون رو به STRATEGY_REGISTRY اضافه کن. بعد اسمش رو به config.ACTIVE_STRATEGIES اضافه کن.
"""
import numpy as np
import pandas as pd

import analysis
import indicators


def generate_dow_support_resistance(df, cfg):
    """استراتژی اصلی: تئوری داو + حمایت/مقاومت + کندل تاییدی (پیاده‌سازی در analysis.py)."""
    return analysis.generate_signal(df, cfg)


def generate_breakout(df, cfg):
    """
    استراتژی مکمل: بریک‌اوت از سقف/کف N کندل اخیر، با تایید حجم.
    فلسفه‌ی این استراتژی با استراتژی اصلی فرق داره: اصلی دنبال "برگشت از سطح"
    (mean-reversion-ish در نقطه‌ی ورود) می‌گرده؛ این یکی دنبال "ادامه‌ی حرکت بعد از
    شکست سطح" (ادامه‌دهنده‌ی روند) می‌گرده. ترکیبشون می‌تونه دو نوع فرصت متفاوت
    رو پوشش بده.
    """
    lookback = getattr(cfg, "BREAKOUT_LOOKBACK", 40)
    vol_mult = getattr(cfg, "BREAKOUT_VOLUME_MULT", 1.5)
    if len(df) < lookback + 5:
        return {"trend": "sideways", "price": float(df["close"].iloc[-1]), "support": None,
                "resistance": None, "atr": None, "signal": None, "strategy": "breakout"}

    current = df.iloc[-1]
    window = df.iloc[-(lookback + 1):-1]
    recent_high = window["high"].max()
    recent_low = window["low"].min()
    avg_vol = window["volume"].mean()

    atr_series = analysis.compute_atr(df, cfg.ATR_PERIOD)
    atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else None
    current_price = float(current["close"])

    result = {"trend": None, "price": current_price, "support": float(recent_low),
              "resistance": float(recent_high), "atr": atr, "signal": None, "strategy": "breakout"}

    if not atr or atr <= 0:
        return result

    vol_ok = current["volume"] > avg_vol * vol_mult

    # بریک‌اوت صعودی: بستن کندل بالای سقف N کندل اخیر + حجم بالا
    if current["close"] > recent_high and vol_ok:
        # همون سطح شکسته‌شده، حالا حمایت جدید
        sig = analysis.finalize_signal("LONG", current_price, float(recent_high) - atr * cfg.ATR_SL_BUFFER,
                                       None, atr, cfg, tp_uses_level=False)
        if sig:
            result["trend"] = "uptrend"
            result["signal"] = sig
            return result

    # بریک‌اوت نزولی: بستن کندل زیر کف N کندل اخیر + حجم بالا
    if current["close"] < recent_low and vol_ok:
        sig = analysis.finalize_signal("SHORT", current_price, float(recent_low) + atr * cfg.ATR_SL_BUFFER,
                                       None, atr, cfg, tp_uses_level=False)
        if sig:
            result["trend"] = "downtrend"
            result["signal"] = sig
            return result

    result["trend"] = "sideways"
    return result


def generate_volume_spike(df, cfg):
    """
    استراتژی مستقل «افزایش حجم»: کندل با حجم به‌طرز غیرعادی بالا (چند برابر میانگین)
    که جهت‌دار هم باشه (صعودی/نزولی واضح)، به‌تنهایی یک سیگنال ورود در جهت همون
    کندل تولید می‌کنه — مستقل از روند داو یا حمایت/مقاومت. فلسفه‌ش اینه: حجم
    ناگهانی نشونه‌ی ورود پول بزرگه، صرف‌نظر از این‌که قیمت کجای نمودار باشه.
    """
    lookback = getattr(cfg, "VOLUME_SPIKE_LOOKBACK", 30)
    spike_mult = getattr(cfg, "VOLUME_SPIKE_MULT", 2.5)
    body_min_pct = getattr(cfg, "VOLUME_SPIKE_MIN_BODY_PCT", 0.5)

    if len(df) < lookback + 5:
        return {"trend": "sideways", "price": float(df["close"].iloc[-1]), "support": None,
                "resistance": None, "atr": None, "signal": None, "strategy": "volume_spike"}

    current = df.iloc[-1]
    window = df.iloc[-(lookback + 1):-1]
    avg_vol = window["volume"].mean()
    current_price = float(current["close"])

    atr_series = analysis.compute_atr(df, cfg.ATR_PERIOD)
    atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else None

    result = {"trend": "sideways", "price": current_price, "support": None, "resistance": None,
              "atr": atr, "signal": None, "strategy": "volume_spike"}

    if not atr or atr <= 0 or avg_vol <= 0:
        return result

    is_spike = current["volume"] > avg_vol * spike_mult
    body = abs(current["close"] - current["open"])
    candle_range = current["high"] - current["low"]
    body_ratio = body / candle_range if candle_range > 0 else 0
    directional = body_ratio >= body_min_pct  # بدنه‌ی کندل باید حداقل نصف کل رنج رو بگیره (سایه کم)

    if not (is_spike and directional):
        return result

    if current["close"] > current["open"]:
        result["trend"] = "uptrend"
        result["signal"] = analysis.finalize_signal(
            "LONG", current_price, float(current["low"]) - atr * cfg.ATR_SL_BUFFER, None, atr, cfg,
            tp_uses_level=False)
    else:
        result["trend"] = "downtrend"
        result["signal"] = analysis.finalize_signal(
            "SHORT", current_price, float(current["high"]) + atr * cfg.ATR_SL_BUFFER, None, atr, cfg,
            tp_uses_level=False)

    return result


def _candle_third_close(o, h, l, c):
    """رنج کندل رو به سه قسمت مساوی تقسیم می‌کنه و می‌گه بسته‌شدن کجا افتاده: high/mid/low."""
    rng = h - l
    if rng <= 0:
        return "mid"
    pos = (c - l) / rng
    if pos >= 2 / 3:
        return "high"
    if pos <= 1 / 3:
        return "low"
    return "mid"


def generate_candle_setup(df, cfg):
    """
    استراتژی «کندل ستاپ» (روش پرایس‌اکشن TST / BOF — همون چارچوبی که در آموزش‌های
    فارسی پرایس‌اکشن مثل سبک آرشیا عزیزپور و ال‌بروکس تدریس می‌شه):

    - ستاپ TST (Test & Reversal): قیمت دقیقاً به حمایت/مقاومت می‌رسه (بدون رد شدن)
      و کندلی که اونجا تشکیل می‌شه با "قدرت مخالف" بسته می‌شه — یعنی نزدیک حمایت
      در یک سوم بالایی رنج خودش بسته بشه (خریداران غالب شدن)، یا نزدیک مقاومت در
      یک سوم پایینی بسته بشه (فروشندگان غالب شدن).
    - ستاپ BOF (Break & Fail): قیمت کمی از سطح رد می‌شه ولی همون کندل یا کندل بعدی
      با ضعف برمی‌گرده و دوباره داخل محدوده بسته می‌شه.

    این دقیق‌تر از فیلتر rejection candle داخل استراتژی اصلیه، چون از تقسیم‌بندی
    سه‌قسمتی (نه فقط نصف) و بررسی محل دقیق نفوذ به سطح استفاده می‌کنه.
    """
    swing_order = cfg.SWING_ORDER
    if len(df) < swing_order * 2 + 10:
        return {"trend": "sideways", "price": float(df["close"].iloc[-1]), "support": None,
                "resistance": None, "atr": None, "signal": None, "strategy": "candle_setup"}

    swing_highs, swing_lows = analysis.find_swings(df, order=swing_order)
    current_price = float(df["close"].iloc[-1])
    nearest_support, nearest_resistance = analysis.get_support_resistance(
        swing_highs, swing_lows, current_price, cfg.SR_CLUSTER_PCT
    )
    atr_series = analysis.compute_atr(df, cfg.ATR_PERIOD)
    atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else None

    result = {"trend": "sideways", "price": current_price, "support": nearest_support,
              "resistance": nearest_resistance, "atr": atr, "signal": None, "strategy": "candle_setup"}

    if not atr or atr <= 0:
        return result

    last = df.iloc[-1]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    proximity = cfg.PROXIMITY_PCT

    # --- نزدیک حمایت: به‌دنبال ستاپ صعودی (TST یا BOF) ---
    if nearest_support:
        touched = l <= nearest_support * (1 + proximity / 100)
        broke_slightly = l < nearest_support  # BOF: کمی از سطح رد شده
        closed_back_above = c >= nearest_support  # ولی با ضعف برگشته بالا
        close_pos = _candle_third_close(o, h, l, c)

        is_tst = touched and not broke_slightly and close_pos == "high"
        is_bof = touched and broke_slightly and closed_back_above and close_pos in ("high", "mid")

        if is_tst or is_bof:
            sig = analysis.finalize_signal("LONG", current_price, l - atr * cfg.ATR_SL_BUFFER,
                                           nearest_resistance, atr, cfg, tp_uses_level=False)
            if sig:
                sig["setup_type"] = "TST" if is_tst else "BOF"
                result["trend"] = "uptrend"
                result["signal"] = sig
                return result

    # --- نزدیک مقاومت: به‌دنبال ستاپ نزولی (TST یا BOF) ---
    if nearest_resistance:
        touched = h >= nearest_resistance * (1 - proximity / 100)
        broke_slightly = h > nearest_resistance
        closed_back_below = c <= nearest_resistance
        close_pos = _candle_third_close(o, h, l, c)

        is_tst = touched and not broke_slightly and close_pos == "low"
        is_bof = touched and broke_slightly and closed_back_below and close_pos in ("low", "mid")

        if is_tst or is_bof:
            sig = analysis.finalize_signal("SHORT", current_price, h + atr * cfg.ATR_SL_BUFFER,
                                           nearest_support, atr, cfg, tp_uses_level=False)
            if sig:
                sig["setup_type"] = "TST" if is_tst else "BOF"
                result["trend"] = "downtrend"
                result["signal"] = sig
                return result

    return result


# ==================== استراتژی‌های شناخته‌شده‌ی اندیکاتوری ====================
# همه با همون خروجی استاندارد و همون مرحله‌ی نهایی (analysis.finalize_signal)، یعنی
# حداقل R:R، حداقل فاصله‌ی SL و بقیه‌ی قواعد مشترک روی این‌ها هم اعمال می‌شه.

def _empty(df, name):
    return {"trend": "sideways", "price": float(df["close"].iloc[-1]), "support": None,
            "resistance": None, "atr": None, "signal": None, "strategy": name}


def _prep(df, cfg, name, min_len=30):
    """آرایه‌ها + ATR آخرین کندل؛ اگه دیتا یا ATR نامعتبر بود، None."""
    if len(df) < min_len:
        return None, _empty(df, name)
    atr_series = analysis.compute_atr(df, cfg.ATR_PERIOD)
    atr = atr_series.iloc[-1]
    res = _empty(df, name)
    if pd.isna(atr) or atr <= 0:
        return None, res
    res["atr"] = float(atr)
    a = {k: df[k].values.astype(float) for k in ("open", "high", "low", "close", "volume")}
    a["atr"] = float(atr)
    a["atr_series"] = atr_series.values
    return a, res


def _finish(res, long_sig, short_sig, cfg, a):
    """اول خرید، اگه نهایی نشد فروش (مثل بقیه‌ی استراتژی‌ها)."""
    if long_sig is not None:
        sig = analysis.finalize_signal("LONG", float(a["close"][-1]), long_sig, None, a["atr"], cfg, False)
        if sig:
            res["trend"], res["signal"] = "uptrend", sig
            return res
    if short_sig is not None:
        sig = analysis.finalize_signal("SHORT", float(a["close"][-1]), short_sig, None, a["atr"], cfg, False)
        if sig:
            res["trend"], res["signal"] = "downtrend", sig
    return res


def generate_ema_cross(df, cfg):
    """کراس EMA سریع/کند در جهت EMA روند. حد ضرر پشت کف/سقف چند کندل اخیر."""
    a, res = _prep(df, cfg, "ema_cross")
    if a is None:
        return res
    c, h, l = a["close"], a["high"], a["low"]
    ef = indicators.ema(c, cfg.EMA_CROSS_FAST)
    es = indicators.ema(c, cfg.EMA_CROSS_SLOW)
    et = indicators.ema(c, cfg.EMA_CROSS_TREND)
    buf = a["atr"] * cfg.ATR_SL_BUFFER
    L = cfg.SWING_STOP_LOOKBACK
    long_sl = short_sl = None
    if ef[-1] > es[-1] and ef[-2] <= es[-2] and c[-1] > et[-1]:
        long_sl = float(indicators.last_min(l, L)[-1]) - buf
    elif ef[-1] < es[-1] and ef[-2] >= es[-2] and c[-1] < et[-1]:
        short_sl = float(indicators.last_max(h, L)[-1]) + buf
    return _finish(res, long_sl, short_sl, cfg, a)


def generate_ema_pullback(df, cfg):
    """در روند (EMA سریع بالای کند)، قیمت به EMA سریع برمی‌گرده و با کندل هم‌جهت ازش جدا می‌شه."""
    a, res = _prep(df, cfg, "ema_pullback")
    if a is None:
        return res
    o, c, h, l = a["open"], a["close"], a["high"], a["low"]
    ef = indicators.ema(c, cfg.EMA_PB_FAST)
    es = indicators.ema(c, cfg.EMA_PB_SLOW)
    buf = a["atr"] * cfg.ATR_SL_BUFFER
    L = cfg.SWING_STOP_LOOKBACK
    long_sl = short_sl = None
    if ef[-1] > es[-1] and c[-1] > es[-1] and l[-1] <= ef[-1] and c[-1] > ef[-1] and c[-1] > o[-1]:
        long_sl = float(indicators.last_min(l, L)[-1]) - buf
    if ef[-1] < es[-1] and c[-1] < es[-1] and h[-1] >= ef[-1] and c[-1] < ef[-1] and c[-1] < o[-1]:
        short_sl = float(indicators.last_max(h, L)[-1]) + buf
    return _finish(res, long_sl, short_sl, cfg, a)


def generate_bb_reversion(df, cfg):
    """بازگشت به داخل باند بولینگر بعد از بسته شدن بیرون باند، با RSI اشباع (میانگین‌گرا)."""
    a, res = _prep(df, cfg, "bb_reversion")
    if a is None:
        return res
    c, h, l = a["close"], a["high"], a["low"]
    mid, up, lo = indicators.bollinger(c, cfg.BB_PERIOD, cfg.BB_STD)
    rsi = indicators.rsi_sma(c, cfg.RSI_PERIOD)
    buf = a["atr"] * cfg.ATR_SL_BUFFER
    long_sl = short_sl = None
    if c[-2] < lo[-2] and c[-1] > lo[-1] and rsi[-2] < cfg.BB_RSI_LOW:
        long_sl = float(min(l[-2], l[-1])) - buf
    if c[-2] > up[-2] and c[-1] < up[-1] and rsi[-2] > cfg.BB_RSI_HIGH:
        short_sl = float(max(h[-2], h[-1])) + buf
    return _finish(res, long_sl, short_sl, cfg, a)


def generate_rsi_pullback(df, cfg):
    """در جهت روند (میانگین ۱۰۰)، RSI از ناحیه‌ی اصلاح برمی‌گرده."""
    a, res = _prep(df, cfg, "rsi_pullback")
    if a is None:
        return res
    c, h, l = a["close"], a["high"], a["low"]
    trend = indicators.sma(c, cfg.RSI_PB_TREND_SMA)
    rsi = indicators.rsi_sma(c, cfg.RSI_PERIOD)
    buf = a["atr"] * cfg.ATR_SL_BUFFER
    L = cfg.SWING_STOP_LOOKBACK
    long_sl = short_sl = None
    if c[-1] > trend[-1] and rsi[-2] < cfg.RSI_PB_LONG_CROSS <= rsi[-1]:
        long_sl = float(indicators.last_min(l, L)[-1]) - buf
    if c[-1] < trend[-1] and rsi[-2] > cfg.RSI_PB_SHORT_CROSS >= rsi[-1]:
        short_sl = float(indicators.last_max(h, L)[-1]) + buf
    return _finish(res, long_sl, short_sl, cfg, a)


def generate_squeeze_breakout(df, cfg):
    """فشردگی نوسان (باند بولینگر داخل کانال کلتنر) و بعد شکست با حجم — شروع حرکت‌های بزرگ."""
    a, res = _prep(df, cfg, "squeeze_breakout")
    if a is None:
        return res
    c, v = a["close"], a["volume"]
    mid, up, lo = indicators.bollinger(c, cfg.BB_PERIOD, cfg.BB_STD)
    atr_s = a["atr_series"]
    kc_up = mid + cfg.SQUEEZE_KC_MULT * atr_s
    kc_lo = mid - cfg.SQUEEZE_KC_MULT * atr_s
    with np.errstate(invalid="ignore"):
        sq = (up < kc_up) & (lo > kc_lo)
    n_sq = int(sq[-1 - cfg.SQUEEZE_LOOKBACK:-1].sum())
    avg_v = indicators.prev_mean(v, 20)[-1]
    vol_ok = v[-1] > avg_v * cfg.SQUEEZE_VOLUME_MULT
    buf = a["atr"] * cfg.ATR_SL_BUFFER
    long_sl = short_sl = None
    if n_sq >= cfg.SQUEEZE_MIN_BARS and vol_ok:
        if c[-1] > up[-1]:
            long_sl = float(mid[-1]) - buf
        elif c[-1] < lo[-1]:
            short_sl = float(mid[-1]) + buf
    return _finish(res, long_sl, short_sl, cfg, a)


def generate_donchian_trend(df, cfg):
    """شکست کانال دانچیان در جهت روند، حد ضرر ۲×ATR (سبک معامله‌گران لاک‌پشت)."""
    a, res = _prep(df, cfg, "donchian_trend")
    if a is None:
        return res
    c, h, l = a["close"], a["high"], a["low"]
    hh = indicators.prev_max(h, cfg.DONCHIAN_PERIOD)
    ll = indicators.prev_min(l, cfg.DONCHIAN_PERIOD)
    trend = indicators.sma(c, cfg.DONCHIAN_TREND_SMA)
    dist = a["atr"] * cfg.DONCHIAN_ATR_STOP
    long_sl = short_sl = None
    if c[-1] > hh[-1] and c[-1] > trend[-1]:
        long_sl = float(c[-1]) - dist
    elif c[-1] < ll[-1] and c[-1] < trend[-1]:
        short_sl = float(c[-1]) + dist
    return _finish(res, long_sl, short_sl, cfg, a)


STRATEGY_REGISTRY = {
    "dow_support_resistance": {
        "fn": generate_dow_support_resistance,
        "label": "داو + حمایت/مقاومت (اصلی)",
    },
    "breakout": {
        "fn": generate_breakout,
        "label": "شکست (بریک‌اوت) با تایید حجم",
    },
    "volume_spike": {
        "fn": generate_volume_spike,
        "label": "افزایش ناگهانی حجم",
    },
    "candle_setup": {
        "fn": generate_candle_setup,
        "label": "کندل ستاپ (TST/BOF پرایس‌اکشن)",
    },
    "ema_cross": {"fn": generate_ema_cross, "label": "کراس EMA با فیلتر روند"},
    "ema_pullback": {"fn": generate_ema_pullback, "label": "پولبک به EMA در روند"},
    "bb_reversion": {"fn": generate_bb_reversion, "label": "بازگشت از باند بولینگر + RSI"},
    "rsi_pullback": {"fn": generate_rsi_pullback, "label": "پولبک RSI در جهت روند"},
    "squeeze_breakout": {"fn": generate_squeeze_breakout, "label": "شکست بعد از فشردگی (Squeeze)"},
    "donchian_trend": {"fn": generate_donchian_trend, "label": "شکست کانال دانچیان (لاک‌پشت)"},
}


def _window(df, end_offset, size):
    """پنجره‌ی دقیقاً size کندلی که به کندلِ end_offset تا از آخر ختم می‌شه (۰ = آخرین کندل)."""
    end = len(df) - end_offset
    if end <= 0:
        return None
    return df.iloc[max(0, end - size):end]


def _run_strategy(name, window_df, cfg):
    if window_df is None or len(window_df) == 0:
        return None
    try:
        return STRATEGY_REGISTRY[name]["fn"](window_df, cfg)
    except Exception:
        return None


def generate_combined_signal(df, cfg):
    """
    اجرای همه‌ی استراتژی‌های فعال روی آخرین پنجره‌ی CANDLE_LIMIT کندلی، و ترکیبشون طبق
    cfg.STRATEGY_COMBINE_MODE:
      - "any": اولین استراتژی (به ترتیب لیست) که سیگنال داد.
      - "all": همه‌ی استراتژی‌ها دقیقاً روی همین کندل و هم‌جهت.
      - "confirm": استراتژی اول ماشه‌ست؛ بقیه باید در CONFIRM_LOOKBACK_BARS کندل اخیر
        (شامل همین کندل) حداقل یک سیگنال هم‌جهت داده باشن.
    df می‌تونه بلندتر از CANDLE_LIMIT باشه (برای حالت confirm لازمه)؛ هر ارزیابی دقیقاً
    روی یک پنجره‌ی CANDLE_LIMIT کندلی انجام می‌شه — درست مثل موتور بک‌تست.
    """
    active = [s for s in getattr(cfg, "ACTIVE_STRATEGIES", ["dow_support_resistance"])
              if s in STRATEGY_REGISTRY]
    if not active:
        active = ["dow_support_resistance"]
    size = getattr(cfg, "CANDLE_LIMIT", len(df))
    last_window = _window(df, 0, size)

    results = {name: _run_strategy(name, last_window, cfg) for name in active}

    primary = results.get(active[0]) or next((r for r in results.values() if r), {
        "trend": "sideways", "price": float(df["close"].iloc[-1]),
        "support": None, "resistance": None, "atr": None, "signal": None,
    })

    combine_mode = getattr(cfg, "STRATEGY_COMBINE_MODE", "any")

    if combine_mode == "all":
        signals = [r["signal"] for r in results.values() if r and r.get("signal")]
        if len(signals) == len(active) and len(signals) > 0:
            sides = {s["side"] for s in signals}
            if len(sides) == 1:
                chosen = dict(min(signals, key=lambda s: s["rr"]))  # محافظه‌کارترین
                chosen["strategy"] = "+".join(active)
                primary = dict(primary)
                primary["signal"] = chosen
                return primary
        primary = dict(primary)
        primary["signal"] = None
        return primary

    if combine_mode == "confirm":
        trig = results.get(active[0])
        primary = dict(primary)
        primary["signal"] = None
        if not trig or not trig.get("signal"):
            return primary
        side = trig["signal"]["side"]
        lookback = max(1, int(getattr(cfg, "CONFIRM_LOOKBACK_BARS", 8)))
        for name in active[1:]:
            confirmed = False
            for k in range(lookback):
                r = results.get(name) if k == 0 else _run_strategy(name, _window(df, k, size), cfg)
                if r and r.get("signal") and r["signal"]["side"] == side:
                    confirmed = True
                    break
            if not confirmed:
                return primary
        out = dict(trig)
        out["signal"] = dict(trig["signal"])
        out["signal"]["strategy"] = "|".join(active)
        return out

    # حالت "any": اولین استراتژی‌ای که سیگنال داده رو قبول کن
    for name in active:
        r = results.get(name)
        if r and r.get("signal"):
            out = dict(r)
            out["signal"] = dict(r["signal"])
            out["signal"]["strategy"] = name
            return out

    primary = dict(primary)
    primary["signal"] = None
    primary["strategy"] = None
    return primary
