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
        entry = current_price
        sl = recent_high - atr * cfg.ATR_SL_BUFFER  # همون سطح شکسته‌شده، حالا حمایت جدید
        risk = entry - sl
        if risk > 0:
            tp = entry + risk * cfg.MIN_RISK_REWARD
            rr = (tp - entry) / risk
            if rr >= cfg.MIN_RISK_REWARD:
                result["trend"] = "uptrend"
                result["signal"] = {"side": "LONG", "entry": entry, "sl": sl, "tp": tp, "rr": rr}
                return result

    # بریک‌اوت نزولی: بستن کندل زیر کف N کندل اخیر + حجم بالا
    if current["close"] < recent_low and vol_ok:
        entry = current_price
        sl = recent_low + atr * cfg.ATR_SL_BUFFER
        risk = sl - entry
        if risk > 0:
            tp = entry - risk * cfg.MIN_RISK_REWARD
            rr = (entry - tp) / risk
            if rr >= cfg.MIN_RISK_REWARD:
                result["trend"] = "downtrend"
                result["signal"] = {"side": "SHORT", "entry": entry, "sl": sl, "tp": tp, "rr": rr}
                return result

    result["trend"] = "sideways"
    return result


STRATEGY_REGISTRY = {
    "dow_support_resistance": {
        "fn": generate_dow_support_resistance,
        "label": "داو + حمایت/مقاومت (اصلی)",
    },
    "breakout": {
        "fn": generate_breakout,
        "label": "بریک‌اوت با تایید حجم",
    },
}


def generate_combined_signal(df, cfg):
    """
    اجرای همه‌ی استراتژی‌های فعال روی یک دیتافریم، و ترکیبشون طبق
    cfg.STRATEGY_COMBINE_MODE. خروجی رو با یک فیلد "strategy" مشخص می‌کنه که
    کدوم استراتژی سیگنال رو صادر کرده (برای تحلیل بعدی).
    """
    active = [s for s in getattr(cfg, "ACTIVE_STRATEGIES", ["dow_support_resistance"])
              if s in STRATEGY_REGISTRY]
    if not active:
        active = ["dow_support_resistance"]

    results = {}
    for name in active:
        try:
            results[name] = STRATEGY_REGISTRY[name]["fn"](df, cfg)
        except Exception:
            results[name] = None

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
                # میانگین سطوح ورود/خروج بین استراتژی‌های توافق‌کننده (محافظه‌کارانه)
                chosen = min(signals, key=lambda s: s["rr"])  # کم‌ریسک‌ترین (محافظه‌کارترین) رو انتخاب کن
                chosen = dict(chosen)
                chosen["strategy"] = "+".join(active)
                primary = dict(primary)
                primary["signal"] = chosen
                return primary
        primary = dict(primary)
        primary["signal"] = None
        return primary

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
