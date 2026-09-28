# -*- coding: utf-8 -*-
"""
استراتژی ربات: «ترکیبی وزن‌دار» — فقط همین یک استراتژی.

پنج جزء، هر کدوم با یک وزن (config.WC_WEIGHTS)، برای خرید و فروش جدا امتیاز می‌دن:
  ۱) تئوری داو: روند صعودی (سقف و کف بالاتر) یا نزولی (سقف و کف پایین‌تر)
  ۲) حمایت/مقاومت با تایید کندل بسته: کندل لمس پشت سطح بسته می‌شه، کندل بعدی تایید می‌کنه
  ۳) سه میانگین متحرک ساده ۷، ۲۵، ۹۹: چیدمان کامل = امتیاز کامل، چیدمان نیمه = نصف
  ۴) RSI: مومنتوم هم‌جهت (خرید بین ۵۰ و ۷۵، فروش بین ۲۵ و ۵۰)
  ۵) سایکل بازار (فازهای داو): میانگین ۹۹ شیب‌دار + حجم رو به افزایش = فاز رشد یا ریزش

امتیاز = مجموع وزن اجزای برقرار ÷ مجموع همه‌ی وزن‌ها × ۱۰۰. اگه امتیاز یک جهت به حداقل
امتیاز ورود (WC_MIN_SCORE_PCT) برسه و کندل آخر هم‌جهت بسته شده باشه (سبز برای خرید،
قرمز برای فروش)، سیگنال صادر می‌شه. حد ضرر پشت حمایت/مقاومت (اگه نزدیکش بودیم) یا پشت
کف/سقف ۱۰ کندل اخیر، با بافر ATR؛ حد سود با حداقل R:R. بقیه‌ی قواعد مشترک (حداقل فاصله‌ی
SL، فضای تا هدف، ...) از analysis.finalize_signal میان.

موتور بک‌تست (signals_engine.py) دقیقاً همین محاسبات رو برای کل تاریخچه انجام می‌ده و
tests/test_engine.py برابری‌شون رو کندل به کندل چک می‌کنه.
"""
import numpy as np
import pandas as pd

import analysis
import indicators

COMPONENTS = ("dow", "sr", "sma", "rsi", "cycle")
COMPONENT_LABELS = {"dow": "روند داو", "sr": "حمایت/مقاومت (تایید کندل)", "sma": "۳SMA (۷،۲۵،۹۹)", "rsi": "RSI",
                    "cycle": "سایکل (فاز بازار)"}


def score_total(cfg):
    total = 0.0
    for k in COMPONENTS:
        total = total + float(cfg.WC_WEIGHTS.get(k, 0.0))
    return total


def generate_weighted_confluence(df, cfg):
    name = "weighted_confluence"
    price = float(df["close"].iloc[-1])
    res = {"trend": "sideways", "price": price, "support": None, "resistance": None, "atr": None,
           "signal": None, "strategy": name, "score_long": 0.0, "score_short": 0.0,
           "parts_long": [], "parts_short": []}
    if len(df) < 30:
        return res

    swing_highs, swing_lows = analysis.find_swings(df, order=cfg.SWING_ORDER)
    trend = analysis.determine_trend(swing_highs, swing_lows)
    support, resistance = analysis.get_support_resistance(swing_highs, swing_lows, price, cfg.SR_CLUSTER_PCT)
    res.update({"trend": trend, "support": support, "resistance": resistance})

    atr_series = analysis.compute_atr(df, cfg.ATR_PERIOD)
    atr = atr_series.iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return res
    atr = float(atr)
    res["atr"] = atr

    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)

    s_fast = indicators.sma(c, cfg.SMA_FAST)[-1]
    s_mid = indicators.sma(c, cfg.SMA_MID)[-1]
    s_slow_series = indicators.sma(c, cfg.SMA_SLOW)
    s_slow = s_slow_series[-1]
    rsi = indicators.rsi_sma(c, cfg.RSI_PERIOD)[-1]
    k = cfg.CYCLE_SLOPE_BARS
    slope = s_slow_series[-1] - s_slow_series[-1 - k] if len(c) > k else np.nan
    vol_fast = indicators.sma(v, cfg.CYCLE_VOL_FAST)[-1]
    vol_slow = indicators.sma(v, cfg.CYCLE_VOL_SLOW)[-1]
    vol_up = bool(vol_fast >= vol_slow)   # NaN → False

    prox = cfg.PROXIMITY_PCT
    if getattr(cfg, "SR_CONFIRM", False):
        # کندل لمس (یکی مونده به آخر) سطح رو لمس کرده و پشتش بسته شده؛ کندل آخر (بسته‌شده) تایید کرده
        mx = cfg.SR_CONFIRM_MAX_DIST_PCT
        ref_l = h[-2] if cfg.SR_CONFIRM_BREAK == "high" else c[-2]
        ref_s = l[-2] if cfg.SR_CONFIRM_BREAK == "high" else c[-2]
        near_sup = bool(support) and l[-2] <= support * (1 + prox / 100) and c[-2] >= support \
            and c[-1] > o[-1] and c[-1] > ref_l and 0 <= (price - support) / support * 100 <= mx
        near_res = bool(resistance) and h[-2] >= resistance * (1 - prox / 100) and c[-2] <= resistance \
            and c[-1] < o[-1] and c[-1] < ref_s and 0 <= (resistance - price) / resistance * 100 <= mx
        sup_stop = min(support, l[-2]) if support else None
        res_stop = max(resistance, h[-2]) if resistance else None
    else:
        near_sup = bool(support) and 0 <= (price - support) / support * 100 <= prox \
            and l[-1] <= support * (1 + prox / 100)
        near_res = bool(resistance) and 0 <= (resistance - price) / resistance * 100 <= prox \
            and h[-1] >= resistance * (1 - prox / 100)
        sup_stop, res_stop = support, resistance

    W = cfg.WC_WEIGHTS
    lo_l, hi_l = cfg.RSI_LONG_ZONE
    lo_s, hi_s = cfg.RSI_SHORT_ZONE
    on_long = {
        "dow": trend == "uptrend",
        "sr": near_sup,
        "sma": 1.0 if (s_fast > s_mid > s_slow) else (0.5 if (s_fast > s_mid and price > s_slow) else 0.0),
        "rsi": lo_l < rsi < hi_l,
        "cycle": bool(slope > 0) and vol_up,
    }
    on_short = {
        "dow": trend == "downtrend",
        "sr": near_res,
        "sma": 1.0 if (s_fast < s_mid < s_slow) else (0.5 if (s_fast < s_mid and price < s_slow) else 0.0),
        "rsi": lo_s < rsi < hi_s,
        "cycle": bool(slope < 0) and vol_up,
    }

    def score(on):
        s = 0.0
        for comp in COMPONENTS:
            s = s + float(W.get(comp, 0.0)) * float(on[comp])
        return s / score_total(cfg) * 100

    sc_l, sc_s = score(on_long), score(on_short)
    res["score_long"], res["score_short"] = round(sc_l, 1), round(sc_s, 1)
    res["parts_long"] = [comp for comp in COMPONENTS if on_long[comp]]
    res["parts_short"] = [comp for comp in COMPONENTS if on_short[comp]]

    min_score = float(cfg.WC_MIN_SCORE_PCT) - 1e-9
    buf = atr * cfg.ATR_SL_BUFFER
    L = cfg.SWING_STOP_LOOKBACK
    if c[-1] > o[-1] and sc_l >= min_score:
        sl = (sup_stop - buf) if near_sup else (float(indicators.last_min(l, L)[-1]) - buf)
        sig = analysis.finalize_signal("LONG", price, sl, resistance, atr, cfg, tp_uses_level=False)
        if sig:
            sig.update({"score": round(sc_l, 1), "reasons": res["parts_long"]})
            res["signal"] = sig
            return res
    if c[-1] < o[-1] and sc_s >= min_score:
        sl = (res_stop + buf) if near_res else (float(indicators.last_max(h, L)[-1]) + buf)
        sig = analysis.finalize_signal("SHORT", price, sl, support, atr, cfg, tp_uses_level=False)
        if sig:
            sig.update({"score": round(sc_s, 1), "reasons": res["parts_short"]})
            res["signal"] = sig
    return res


STRATEGY_REGISTRY = {
    "weighted_confluence": {
        "fn": generate_weighted_confluence,
        "label": "ترکیبی وزن‌دار: داو + حمایت/مقاومت + سایکل + RSI + ۳SMA",
    },
}


def _window(df, end_offset, size):
    """پنجره‌ی دقیقاً size کندلی که به کندلِ end_offset تا از آخر ختم می‌شه (۰ = آخرین کندل)."""
    end = len(df) - end_offset
    if end <= 0:
        return None
    return df.iloc[max(0, end - size):end]


def generate_combined_signal(df, cfg):
    """اجرای استراتژی روی آخرین پنجره‌ی CANDLE_LIMIT کندلی (دقیقاً مثل موتور بک‌تست)."""
    size = getattr(cfg, "CANDLE_LIMIT", len(df))
    window = _window(df, 0, size)
    try:
        r = generate_weighted_confluence(window, cfg)
    except Exception:
        r = None
    if not r:
        return {"trend": "sideways", "price": float(df["close"].iloc[-1]), "support": None,
                "resistance": None, "atr": None, "signal": None, "strategy": None}
    if r.get("signal"):
        r = dict(r)
        r["signal"] = dict(r["signal"])
        r["signal"]["strategy"] = "weighted_confluence"
    return r
