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
import pandas as pd
import analysis


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
