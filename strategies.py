# -*- coding: utf-8 -*-
"""
استراتژی‌های ربات (در هر لحظه فقط یکی فعاله، از پنل انتخاب می‌شه):
  - «ترکیبی وزن‌دار» (weighted_confluence) — پایین توضیح داده شده
  - «شکست باکس» (box_breakout) — generate_box_breakout


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


BREAKOUT_LABELS = {"box": "شکست باکس", "sr": "شکست حمایت/مقاومت", "vol": "حجم", "retest": "ورود با پولبک",
                   "trend_breakout": "شکست سقف/کف ۲۰ کندل هم‌جهت روند (روندگیر)",
                   "xs_top": "جزو قوی‌ترین‌های هفته (مومنتوم)", "xs_bottom": "جزو ضعیف‌ترین‌های هفته (مومنتوم)"}


class LimitCfg:
    """همون تنظیمات، ولی ورود همیشه لیمیت (برای حساب کارمزد ورود با پولبک که همیشه لیمیته)."""

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k):
        if k == "ENTRY_MODE":
            return "limit"
        return getattr(self._cfg, k)


def engine_name(cfg):
    """اسم کلید موتور بک‌تست برای استراتژی فعال (شکست باکس با ورود پولبک کلید جدا داره)."""
    return engine_key(active_strategy(cfg), cfg)


def generate_box_breakout(df, cfg):
    """
    شکست باکس (رنج) یا حمایت/مقاومت، فقط با کندل بسته‌شده، با تایید اجباری حجم.
    باکس = BRK_BOX_BARS کندلِ قبل از کندل آخر؛ کندل آخر (بسته‌شده) کندل شکسته.
    """
    name = "box_breakout"
    price = float(df["close"].iloc[-1])
    res = {"trend": "sideways", "price": price, "support": None, "resistance": None, "atr": None,
           "signal": None, "strategy": name, "box_top": None, "box_bottom": None, "vol_ratio": None}
    N = int(cfg.BRK_BOX_BARS)
    if len(df) < max(30, N + 2):
        return res

    swing_highs, swing_lows = analysis.find_swings(df, order=cfg.SWING_ORDER)
    trend = analysis.determine_trend(swing_highs, swing_lows)
    support, resistance = analysis.get_support_resistance(swing_highs, swing_lows, price, cfg.SR_CLUSTER_PCT)
    res.update({"trend": trend, "support": support, "resistance": resistance})

    atr_series = analysis.compute_atr(df, cfg.ATR_PERIOD)
    atr, atr_prev = atr_series.iloc[-1], atr_series.iloc[-2]
    if pd.isna(atr) or atr <= 0 or pd.isna(atr_prev) or atr_prev <= 0:
        return res
    atr, atr_prev = float(atr), float(atr_prev)
    res["atr"] = atr

    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)

    # --- باکس: N کندل قبل از کندل شکست ---
    top = float(indicators.prev_max(h, N)[-1])
    bot = float(indicators.prev_min(l, N)[-1])
    height = top - bot
    box_ok = height > 0 and height <= cfg.BRK_BOX_MAX_ATR * atr_prev
    if box_ok:
        tol = cfg.BRK_TOUCH_TOL * height
        n_top = int((h[-N - 1:-1] >= top - tol).sum())
        n_bot = int((l[-N - 1:-1] <= bot + tol).sum())
        box_ok = n_top >= cfg.BRK_BOX_MIN_TOUCHES and n_bot >= cfg.BRK_BOX_MIN_TOUCHES
    res["box_top"], res["box_bottom"] = (top, bot) if box_ok else (None, None)

    # --- حجم: کندل شکست در برابر میانگین حجم باکس (اجباری) ---
    vol_avg = float(indicators.prev_mean(v, N)[-1])
    vol_ok = vol_avg > 0 and v[-1] >= cfg.BRK_VOL_MULT * vol_avg
    res["vol_ratio"] = round(v[-1] / vol_avg, 2) if vol_avg > 0 else None

    # --- سطح حمایت/مقاومت نسبت به کندل قبل از شکست ---
    sup_p, res_p = analysis.get_support_resistance(swing_highs, swing_lows, float(c[-2]), cfg.SR_CLUSTER_PCT)
    ext = cfg.BRK_MAX_EXT_ATR * atr
    use_sr = bool(getattr(cfg, "BRK_USE_SR", True))
    mode = getattr(cfg, "BRK_MAIN_TREND", "not_against")
    buf = atr * cfg.ATR_SL_BUFFER
    sl_mid = getattr(cfg, "BRK_SL_MODE", "candle") == "mid"
    retest = getattr(cfg, "BRK_ENTRY", "close") == "retest"
    off = cfg.BRK_RETEST_OFFSET_ATR * atr if retest else 0.0
    rsl = cfg.BRK_RETEST_SL_ATR * atr if retest else 0.0
    wait = int(getattr(cfg, "BRK_RETEST_WAIT_BARS", 1))

    def trend_ok(want, against):
        if mode == "with":
            return trend == want
        if mode == "not_against":
            return trend != against
        return True

    if c[-1] > o[-1] and vol_ok and trend_ok("uptrend", "downtrend"):
        box_l = box_ok and c[-1] > top and c[-1] - top <= ext
        sr_l = use_sr and bool(res_p) and c[-1] > res_p and c[-1] - res_p <= ext
        if box_l or sr_l:
            if retest:
                lv = top if box_l else res_p
                sig = analysis.finalize_signal("LONG", min(lv + off, price), lv - rsl, None, atr, LimitCfg(cfg),
                                               tp_uses_level=False)
            else:
                if sl_mid:
                    sl = ((top + bot) / 2.0 if box_l else res_p) - buf
                else:
                    sl = l[-1] - buf
                sig = analysis.finalize_signal("LONG", price, sl, None, atr, cfg, tp_uses_level=False)
            if sig:
                why = (["box"] if box_l else []) + (["sr"] if sr_l else [])
                sig.update({"score": None, "reasons": why + ["vol"] + (["retest"] if retest else []),
                            "vol_ratio": res["vol_ratio"], "limit_only": retest, "wait_bars": wait if retest else None})
                res["signal"] = sig
                return res
    if c[-1] < o[-1] and vol_ok and trend_ok("downtrend", "uptrend"):
        box_s = box_ok and c[-1] < bot and bot - c[-1] <= ext
        sr_s = use_sr and bool(sup_p) and c[-1] < sup_p and sup_p - c[-1] <= ext
        if box_s or sr_s:
            if retest:
                lv = bot if box_s else sup_p
                sig = analysis.finalize_signal("SHORT", max(lv - off, price), lv + rsl, None, atr, LimitCfg(cfg),
                                               tp_uses_level=False)
            else:
                if sl_mid:
                    sl = ((top + bot) / 2.0 if box_s else sup_p) + buf
                else:
                    sl = h[-1] + buf
                sig = analysis.finalize_signal("SHORT", price, sl, None, atr, cfg, tp_uses_level=False)
            if sig:
                why = (["box"] if box_s else []) + (["sr"] if sr_s else [])
                sig.update({"score": None, "reasons": why + ["vol"] + (["retest"] if retest else []),
                            "vol_ratio": res["vol_ratio"], "limit_only": retest, "wait_bars": wait if retest else None})
                res["signal"] = sig
    return res


class TrendCfg:
    """تنظیمات روندگیر: حد سود عملاً غیرفعال (TR_TP_R) و ورود همیشه بازار (برای حساب کارمزد)."""

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k):
        if k == "ENTRY_MODE":
            return "market"
        if k == "MIN_RISK_REWARD":
            return self._cfg.TR_TP_R
        return getattr(self._cfg, k)


def generate_trend_follow(df, cfg):
    """
    روندگیر: شکست سقف/کف TR_BREAKOUT_BARS کندل اخیر با کندل بسته‌شده (اولین بسته‌شدن بیرون)،
    هم‌جهت با میانگین TR_MA و روند داو (خلاف جهت نباشه). حد ضرر TR_SL_ATR×ATR؛ بدون حد سود ثابت
    (خروج با تریلینگ شاندلیر config.TREND_TRAIL). ورود با سفارش بازار.
    """
    name = "trend_follow"
    price = float(df["close"].iloc[-1])
    res = {"trend": "sideways", "price": price, "support": None, "resistance": None, "atr": None,
           "signal": None, "strategy": name}
    N = int(cfg.TR_BREAKOUT_BARS)
    if len(df) < max(30, N + 2):
        return res
    swing_highs, swing_lows = analysis.find_swings(df, order=cfg.SWING_ORDER)
    trend = analysis.determine_trend(swing_highs, swing_lows)
    support, resistance = analysis.get_support_resistance(swing_highs, swing_lows, price, cfg.SR_CLUSTER_PCT)
    res.update({"trend": trend, "support": support, "resistance": resistance})
    atr = analysis.compute_atr(df, cfg.ATR_PERIOD).iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return res
    atr = float(atr)
    res["atr"] = atr
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    top_s = indicators.prev_max(h, N)
    bot_s = indicators.prev_min(l, N)
    top, bot, top_p, bot_p = top_s[-1], bot_s[-1], top_s[-2], bot_s[-2]
    ma = indicators.sma(c, int(cfg.TR_MA))[-1]
    vm = float(getattr(cfg, "TR_VOL_MULT", 0.0) or 0.0)
    if vm > 0:
        va = float(indicators.prev_mean(v, N)[-1])
        vol_ok = va > 0 and v[-1] >= vm * va
    else:
        vol_ok = True
    dist = cfg.TR_SL_ATR * atr
    if c[-1] > o[-1] and c[-1] > top and c[-2] <= top_p and c[-1] > ma and trend != "downtrend" and vol_ok:
        sig = analysis.finalize_signal("LONG", price, price - dist, None, atr, TrendCfg(cfg), tp_uses_level=False)
        if sig:
            sig.update({"score": None, "reasons": ["trend_breakout"], "market_only": True})
            res["signal"] = sig
            return res
    if c[-1] < o[-1] and c[-1] < bot and c[-2] >= bot_p and c[-1] < ma and trend != "uptrend" and vol_ok:
        sig = analysis.finalize_signal("SHORT", price, price + dist, None, atr, TrendCfg(cfg), tp_uses_level=False)
        if sig:
            sig.update({"score": None, "reasons": ["trend_breakout"], "market_only": True})
            res["signal"] = sig
    return res


# ==================== فیبوناچی «حرکت دوم» (fib_phase) ====================
FIB_COMPONENTS = ("zone", "rsi", "vol", "vp", "sr", "dow")
FIB_LABELS = {"zone": "ناحیه‌ی طلایی فیبوناچی", "rsi": "RSI خنک‌شده و برگشته", "vol": "حجم اصلاح کمتر از حرکت اول",
              "vp": "گره‌ی پرحجم (پروفایل حجم) در ناحیه", "sr": "سقف/کف شکسته‌شده در ناحیه", "dow": "روند داو هم‌جهت"}


def volume_profile_hvn(h, l, c, v, a, b, bins, hvn_frac):
    """پروفایل حجم کندل‌های a..b: حجم هر کندل در خونه‌ی قیمت معمولش ((سقف+کف+بسته)/۳).
    خروجی: (کف، پهنای هر خونه، ماسک گره‌های پرحجم) یا None"""
    hh = h[a:b + 1]
    ll = l[a:b + 1]
    lo = float(ll.min())
    hi = float(hh.max())
    if not hi > lo:
        return None
    tp = (hh + ll + c[a:b + 1]) / 3.0
    idx = np.minimum(((tp - lo) / (hi - lo) * bins).astype(np.int64), bins - 1)
    vol = np.bincount(idx, weights=v[a:b + 1], minlength=bins)
    mx = float(vol.max())
    if not mx > 0:
        return None
    return lo, (hi - lo) / bins, vol >= hvn_frac * mx


def fib_eval(side, o, h, l, c, v, atr, rsi, trend, t, s, sh, sl, cfg):
    """
    ستاپ «حرکت دوم» فیبوناچی روی کندل t (بسته‌شده). مشترک بین ربات زنده و موتور بک‌تست.
    sh/sl: اندیس سوینگ‌های سقف/کف تاییدشده داخل پنجره (صعودی). s: شروع پنجره.
    trend: روند داو پنجره (۱، ۱-، ۰). خروجی: None یا {"sl", "score", "parts", "retr", ...}
    """
    a_t = float(atr[t])
    if not (a_t > 0) or t - 1 < s:
        return None
    long = side == "LONG"
    piv, prev = (sh, sl) if long else (sl, sh)
    if len(piv) < 2 or len(prev) < 1:
        return None
    P = int(piv[-1])           # انتهای حرکت اول (سقف برای خرید، کف برای فروش)
    Pp = int(piv[-2])          # سقف/کف قبلی
    j = int(np.searchsorted(prev, P)) - 1
    if j < 0:
        return None
    A = int(prev[j])           # شروع حرکت اول
    if long:
        top, bot = float(h[P]), float(l[A])
        if cfg.FIB_REQUIRE_BREAK and not (top > float(h[Pp])):
            return None
    else:
        top, bot = float(h[A]), float(l[P])
        if cfg.FIB_REQUIRE_BREAK and not (bot < float(l[Pp])):
            return None
    imp = top - bot
    if not (imp >= cfg.FIB_MIN_IMPULSE_ATR * a_t) or t <= P + 1:
        return None
    # حرکت دوم هنوز شروع نشده: از بعد از P تا کندل قبل، قیمت از انتهای حرکت اول رد نشده
    if long:
        if float(h[P + 1:t].max()) >= top or not (c[t] < top):
            return None
        p = P + 1 + int(np.argmin(l[P + 1:t + 1]))
        retr = (top - float(l[p])) / imp
    else:
        if float(l[P + 1:t].min()) <= bot or not (c[t] > bot):
            return None
        p = P + 1 + int(np.argmax(h[P + 1:t + 1]))
        retr = (float(h[p]) - bot) / imp
    if not (cfg.FIB_ZONE_LO <= retr <= cfg.FIB_ZONE_HI) or t - p >= cfg.FIB_TRIGGER_BARS:
        return None
    # کندل برگشت (تایید با کندل بسته‌شده)
    if long:
        if not (c[t] > o[t] and c[t] > h[t - 1]):
            return None
        stop = float(l[p]) - cfg.FIB_SL_BUFFER_ATR * a_t
        z_hi, z_lo = top - cfg.FIB_ZONE_LO * imp, top - cfg.FIB_ZONE_HI * imp
    else:
        if not (c[t] < o[t] and c[t] < l[t - 1]):
            return None
        stop = float(h[p]) + cfg.FIB_SL_BUFFER_ATR * a_t
        z_lo, z_hi = bot + cfg.FIB_ZONE_LO * imp, bot + cfg.FIB_ZONE_HI * imp

    parts = {}
    parts["zone"] = 1.0 if retr >= cfg.FIB_GOLDEN else 0.5
    rp = float(rsi[p]) if p - s >= int(cfg.RSI_PERIOD) else float("nan")
    r_now, r_prev = float(rsi[t]), float(rsi[t - 1])
    if long:
        ok = rp <= cfg.FIB_RSI_LONG_MAX and r_now > r_prev
    else:
        ok = rp >= 100.0 - cfg.FIB_RSI_LONG_MAX and r_now < r_prev
    parts["rsi"] = 1.0 if ok else 0.0
    parts["vol"] = 1.0 if float(np.mean(v[P + 1:p + 1])) < float(np.mean(v[A + 1:P + 1])) else 0.0
    a_vp = max(s, t - int(cfg.FIB_VP_BARS) + 1)
    vp = volume_profile_hvn(h, l, c, v, a_vp, t, int(cfg.FIB_VP_BINS), float(cfg.FIB_VP_HVN))
    hit = False
    if vp is not None:
        lo_vp, w_vp, hvn = vp
        for k in np.flatnonzero(hvn).tolist():
            b_lo = lo_vp + k * w_vp
            if b_lo <= z_hi and b_lo + w_vp >= z_lo:
                hit = True
                break
    parts["vp"] = 1.0 if hit else 0.0
    lvl = float(h[Pp]) if long else float(l[Pp])
    tol = cfg.FIB_SR_TOL_ATR * a_t
    parts["sr"] = 1.0 if (z_lo - tol <= lvl <= z_hi + tol) else 0.0
    parts["dow"] = 1.0 if trend == (1 if long else -1) else 0.0
    wsum = 0.0
    acc = 0.0
    for k in FIB_COMPONENTS:
        w = float(cfg.FIB_WEIGHTS.get(k, 0.0))
        wsum = wsum + w
        acc = acc + w * parts[k]
    score = acc / wsum * 100.0 if wsum > 0 else 0.0
    return {"sl": stop, "score": score, "parts": parts, "retr": retr, "impulse_start": A, "impulse_end": P,
            "pullback": p}


def generate_fib_phase(df, cfg):
    """فیبوناچی «حرکت دوم» روی آخرین کندل بسته‌شده‌ی پنجره (config: FIB_*)."""
    name = "fib_phase"
    price = float(df["close"].iloc[-1])
    res = {"trend": "sideways", "price": price, "support": None, "resistance": None, "atr": None,
           "signal": None, "strategy": name, "score_long": 0.0, "score_short": 0.0}
    n = len(df)
    if n < 50:
        return res
    swing_highs, swing_lows = analysis.find_swings(df, order=cfg.SWING_ORDER)
    trend = analysis.determine_trend(swing_highs, swing_lows)
    support, resistance = analysis.get_support_resistance(swing_highs, swing_lows, price, cfg.SR_CLUSTER_PCT)
    res.update({"trend": trend, "support": support, "resistance": resistance})
    atr_s = analysis.compute_atr(df, cfg.ATR_PERIOD).values
    atr = atr_s[-1]
    if not (atr > 0):
        return res
    res["atr"] = float(atr)
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    rsi = indicators.rsi_sma(c, cfg.RSI_PERIOD)
    sh = np.array([i for i, _ in swing_highs], dtype=np.int64)
    sl = np.array([i for i, _ in swing_lows], dtype=np.int64)
    tr = {"uptrend": 1, "downtrend": -1}.get(trend, 0)
    min_score = float(cfg.WC_MIN_SCORE_PCT) - 1e-9
    for side in ("LONG", "SHORT"):
        st = fib_eval(side, o, h, l, c, v, atr_s, rsi, tr, n - 1, 0, sh, sl, cfg)
        if st is None:
            continue
        res["score_long" if side == "LONG" else "score_short"] = round(st["score"], 1)
        if st["score"] < min_score:
            continue
        sig = analysis.finalize_signal(side, price, st["sl"], None, float(atr), cfg, tp_uses_level=False)
        if sig:
            sig.update({"score": st["score"], "reasons": ["fib_" + k for k in FIB_COMPONENTS if st["parts"][k] > 0],
                        "fib_retr": round(st["retr"], 3)})
            res["signal"] = sig
            return res
    return res


# ==================== الگو + ساختار بازار (pattern_structure) ====================

def pattern_setup(side, c, atr, trend, t, sig, cfg):
    """مشترک بین ربات زنده و موتور بک‌تست. sig = خروجی patterns.pattern_signals. None یا (حد ضرر، شماره‌ی الگو)."""
    long = side == "LONG"
    slv, nm = (sig[0], sig[1]) if long else (sig[2], sig[3])
    if nm[t] < 0 or not (atr[t] > 0):
        return None
    if getattr(cfg, "PAT_STRUCTURE", "with") == "with" and trend != (1 if long else -1):
        return None
    d = float(cfg.PAT_MIN_SL_ATR) * float(atr[t])
    stop = min(float(slv[t]), float(c[t]) - d) if long else max(float(slv[t]), float(c[t]) + d)
    return stop, int(nm[t])


def generate_pattern_structure(df, cfg):
    """الگوی کندلی/کلاسیک + ساختار بازار داو، روی آخرین کندل بسته‌شده‌ی پنجره."""
    import patterns
    name = "pattern_structure"
    price = float(df["close"].iloc[-1])
    res = {"trend": "sideways", "price": price, "support": None, "resistance": None, "atr": None,
           "signal": None, "strategy": name}
    n = len(df)
    if n < 50:
        return res
    swing_highs, swing_lows = analysis.find_swings(df, order=cfg.SWING_ORDER)
    trend = analysis.determine_trend(swing_highs, swing_lows)
    support, resistance = analysis.get_support_resistance(swing_highs, swing_lows, price, cfg.SR_CLUSTER_PCT)
    res.update({"trend": trend, "support": support, "resistance": resistance})
    atr_s = analysis.compute_atr(df, cfg.ATR_PERIOD).values
    if not (atr_s[-1] > 0):
        return res
    res["atr"] = float(atr_s[-1])
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    names = patterns.pattern_names(getattr(cfg, "PAT_SET", "all"))
    sig = patterns.pattern_signals(o, h, l, c, atr_s, int(cfg.SWING_ORDER), names)
    tr = {"uptrend": 1, "downtrend": -1}.get(trend, 0)
    for side in ("LONG", "SHORT"):
        st = pattern_setup(side, c, atr_s, tr, n - 1, sig, cfg)
        if st is None:
            continue
        out = analysis.finalize_signal(side, price, st[0], None, float(atr_s[-1]), cfg, tp_uses_level=False)
        if out:
            out.update({"score": None, "reasons": ["pattern:" + names[st[1]], "structure"],
                        "pattern": patterns.label(names[st[1]])})
            res["signal"] = out
            return res
    return res


# ==================== مومنتوم نسبی هفتگی (cross-sectional momentum) ====================
DAY_MS = 86_400_000


def xs_is_rebalance(open_ms):
    """کندل روزانه‌ی یکشنبه (UTC) — بسته‌شدنش (دوشنبه ۰۰:۰۰) لحظه‌ی رتبه‌بندی هفتگیه."""
    return (int(open_ms) // DAY_MS + 3) % 7 == 6


def xs_return(closes, lookback):
    """بازده lookback کندل اخیر (کندل آخر نسبت به lookback کندل قبلش)."""
    if len(closes) <= lookback:
        return None
    return float(closes[-1]) / float(closes[-1 - lookback]) - 1.0


def xs_rank(rets, cfg):
    """
    rets: {symbol: بازده}. خروجی {symbol: "LONG"/"SHORT"} — XS_TOP_K تای قوی‌تر خرید، ضعیف‌ترها فروش.
    مشترک بین ربات زنده و بک‌تست (ترتیب قطعی: برابرها با اسم نماد).
    """
    items = [(sym, r) for sym, r in rets.items() if r is not None and r == r]
    if len(items) < int(cfg.XS_MIN_UNIVERSE):
        return {}
    k = int(cfg.XS_TOP_K)
    top = sorted(items, key=lambda x: (-x[1], x[0]))[:k]
    picks = {sym: "LONG" for sym, _ in top}
    if getattr(cfg, "XS_SHORT", True):
        for sym, _ in sorted(items, key=lambda x: (x[1], x[0]))[:k]:
            if sym not in picks:
                picks[sym] = "SHORT"
    return picks


class XsCfg(TrendCfg):
    """مومنتوم: ورود بازار، حد سود عملاً غیرفعال (خروج با پایان هفته یا حد ضرر محافظ)."""


def generate_xs_signal(df, cfg, side):
    """سیگنال مومنتوم برای نمادی که در رتبه‌بندی هفتگی انتخاب شده (روی آخرین کندل بسته‌شده)."""
    price = float(df["close"].iloc[-1])
    atr = analysis.compute_atr(df, cfg.ATR_PERIOD).iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return None
    atr = float(atr)
    dist = cfg.XS_SL_ATR * atr
    sl = price - dist if side == "LONG" else price + dist
    sig = analysis.finalize_signal(side, price, sl, None, atr, XsCfg(cfg), tp_uses_level=False)
    if sig:
        sig.update({"score": None, "reasons": ["xs_top" if side == "LONG" else "xs_bottom"], "market_only": True,
                    "strategy": "xs_momentum"})
    return sig


def _xs_placeholder(df, cfg):
    """مومنتوم نسبی به رتبه‌بندی همه‌ی نمادها نیاز داره (bot.full_scan_job)، نه یک نماد تنها."""
    return {"trend": "sideways", "price": float(df["close"].iloc[-1]), "support": None, "resistance": None,
            "atr": None, "signal": None, "strategy": "xs_momentum"}


STRATEGY_REGISTRY = {
    "weighted_confluence": {
        "fn": generate_weighted_confluence,
        "label": "ترکیبی وزن‌دار: داو + حمایت/مقاومت + سایکل + RSI + ۳SMA",
        "short": "ترکیبی وزن‌دار",
    },
    "box_breakout": {
        "fn": generate_box_breakout,
        "label": "شکست باکس و حمایت/مقاومت (با کندل بسته + تایید حجم + روند داو)",
        "short": "شکست باکس",
    },
    "trend_follow": {
        "fn": generate_trend_follow,
        "label": "روندگیر: شکست سقف/کف ۲۰ کندل، بدون سقف سود، تریلینگ شاندلیر",
        "short": "روندگیر",
    },
    "fib_phase": {
        "fn": generate_fib_phase,
        "label": "فیبوناچی «حرکت دوم»: اصلاح به ناحیه‌ی طلایی + امتیاز (RSI، حجم، پروفایل حجم، سطح، داو)",
        "short": "فیبوناچی حرکت دوم",
    },
    "pattern_structure": {
        "fn": generate_pattern_structure,
        "label": "الگو + ساختار بازار: الگوی کندلی/کلاسیک هم‌جهت با سقف و کف‌های داو (HH/HL، LH/LL)، RR2 + تریلینگ",
        "short": "الگو + ساختار بازار",
    },
    "xs_momentum": {
        "fn": _xs_placeholder,
        "label": "مومنتوم نسبی هفتگی: خرید قوی‌ترین‌ها، فروش ضعیف‌ترین‌ها (فقط روزانه)",
        "short": "مومنتوم هفتگی (فقط روزانه)",
    },
}

# ترکیب‌های آماده (چند استراتژی با هم؛ اولی اولویت داره اگه هم‌زمان سیگنال بدن)
STRATEGY_COMBOS = {
    "trend_follow+box_breakout": "ترکیب: روندگیر + شکست باکس",
    "trend_follow+weighted_confluence": "ترکیب: روندگیر + ترکیبی وزن‌دار",
    "trend_follow+box_breakout+weighted_confluence": "ترکیب: هر سه استراتژی",
    "xs_momentum+trend_follow": "ترکیب: مومنتوم هفتگی + روندگیر (فقط روزانه)",
}


def parse_strategy(value):
    """«a+b» → [a, b] (فقط استراتژی‌های معتبر)."""
    names = [x for x in str(value or "").split("+") if x in STRATEGY_REGISTRY]
    return names or ["box_breakout"]


def active_strategy(cfg):
    act = list(getattr(cfg, "ACTIVE_STRATEGIES", []) or [])
    return act[0] if act and act[0] in STRATEGY_REGISTRY else "weighted_confluence"


def active_strategies(cfg):
    act = [a for a in (getattr(cfg, "ACTIVE_STRATEGIES", []) or []) if a in STRATEGY_REGISTRY]
    return act or ["weighted_confluence"]


def engine_key(name, cfg):
    if name == "box_breakout" and getattr(cfg, "BRK_ENTRY", "close") == "retest":
        return "box_breakout@retest"
    if name == "pattern_structure" and getattr(cfg, "PAT_STRUCTURE", "with") == "off":
        return "pattern_structure@free"
    return name


def engine_names(cfg):
    """کلیدهای موتور بک‌تست برای همه‌ی استراتژی‌های فعال (به ترتیب اولویت)."""
    return [engine_key(n, cfg) for n in active_strategies(cfg)]


def reason_labels(reasons):
    out = []
    for r in reasons or []:
        rs = str(r)
        if rs.startswith("pattern:"):
            import patterns
            out.append(patterns.label(rs[8:]))
            continue
        if rs == "structure":
            out.append("ساختار داو هم‌جهت")
            continue
        out.append(COMPONENT_LABELS.get(r) or BREAKOUT_LABELS.get(r) or FIB_LABELS.get(rs[4:]) or r)
    return out


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
    # چند استراتژی فعال: به ترتیب اولویت؛ اولین سیگنال برنده‌ست (مثل موتور بک‌تست)
    r = None
    name = None
    for nm in active_strategies(cfg):
        try:
            rr = STRATEGY_REGISTRY[nm]["fn"](window, cfg)
        except Exception:
            rr = None
        if rr is None:
            continue
        if r is None:
            r, name = rr, nm
        if rr.get("signal"):
            r, name = rr, nm
            break
    if not r:
        return {"trend": "sideways", "price": float(df["close"].iloc[-1]), "support": None,
                "resistance": None, "atr": None, "signal": None, "strategy": None}
    if r.get("signal"):
        r = dict(r)
        r["signal"] = dict(r["signal"])
        r["signal"]["strategy"] = name
    return r
