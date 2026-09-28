# -*- coding: utf-8 -*-
"""
موتور سیگنال وکتوریزه برای بک‌تست سریع — «دقیقاً» معادل استراتژی‌های ربات زنده.

ربات زنده در هر اسکن، هر استراتژی رو روی یک پنجره‌ی CANDLE_LIMIT کندلیِ بسته‌شده
اجرا می‌کنه (strategies.py). اجرای همون توابع برای تک‌تک کندل‌های ۲ سال (۷۰ هزار کندل
برای هر نماد) خیلی کنده. این ماژول همون محاسبات رو برای کل سری یک‌جا انجام می‌ده،
با این تضمین که برای هر کندل t دقیقاً همون چیزی رو می‌بینه که ربات زنده روی پنجره‌ی
[t-CANDLE_LIMIT+1, t] می‌دید:
  - سوینگ‌ها: فقط سوینگ‌هایی که داخل همون پنجره تایید شدن (نه کل تاریخچه)
  - حمایت/مقاومت: با همون تابع خوشه‌بندی analysis._cluster_levels
  - مرحله‌ی نهایی سیگنال: همون قواعد analysis.finalize_signal (نسخه‌ی آرایه‌ای)
هیچ داده‌ای از آینده استفاده نمی‌شه (no lookahead). تست‌های tests/test_engine.py
این معادل‌بودن رو روی کندل‌های تصادفی با خود توابع زنده مقایسه و تایید می‌کنن.

خروجی هر استراتژی «کاندیدهای ساختاری» تُنُکه (فقط کندل‌هایی که شرایط پایه رو دارن)،
تا بشه با هزاران ترکیب تنظیمات مختلف بدون محاسبه‌ی دوباره آزمایششون کرد.
"""
import bisect

import numpy as np
import pandas as pd

import analysis
import indicators as ind
import market_data

STRATEGY_NAMES = ["weighted_confluence"]
LONG, SHORT = 1, -1


class Series:
    """آرایه‌های یک نماد در یک تایم‌فریم (فقط کندل‌های بسته‌شده)."""

    def __init__(self, arr, timeframe):
        arr = np.asarray(arr)
        self.timeframe = timeframe
        self.ts = arr[:, 0].astype(np.int64)
        self.o = np.ascontiguousarray(arr[:, 1], dtype=np.float64)
        self.h = np.ascontiguousarray(arr[:, 2], dtype=np.float64)
        self.l = np.ascontiguousarray(arr[:, 3], dtype=np.float64)
        self.c = np.ascontiguousarray(arr[:, 4], dtype=np.float64)
        self.v = np.ascontiguousarray(arr[:, 5], dtype=np.float64)
        self.close_ts = market_data.close_times_ms(self.ts, timeframe)
        self.n = len(self.ts)

    def slice_until(self, n_keep):
        """نسخه‌ی کوتاه‌شده (برای تست عدم نگاه به آینده)."""
        arr = np.column_stack([self.ts, self.o, self.h, self.l, self.c, self.v])[:n_keep]
        return Series(arr, self.timeframe)

    def to_df(self, start=0, end=None):
        end = self.n if end is None else end
        return pd.DataFrame({
            "timestamp": pd.to_datetime(self.ts[start:end], unit="ms"),
            "open": self.o[start:end], "high": self.h[start:end], "low": self.l[start:end],
            "close": self.c[start:end], "volume": self.v[start:end],
        })


def swing_positions(h, l, order):
    """معادل وکتوریزه‌ی analysis.find_swings روی کل سری."""
    win = 2 * order + 1
    rmax = pd.Series(h).rolling(win, center=True).max().values
    rmin = pd.Series(l).rolling(win, center=True).min().values
    is_sh = (~np.isnan(rmax)) & (h == rmax)
    is_sl = (~np.isnan(rmin)) & (l == rmin)
    return np.flatnonzero(is_sh), np.flatnonzero(is_sl)


def window_swing_bounds(n, window, order, sh_pos, sl_pos):
    """برای هر کندل t: بازه‌ی ایندکس سوینگ‌هایی که داخل پنجره‌ی ختم‌شده به t تایید شدن."""
    t = np.arange(n)
    s = np.maximum(0, t - window + 1)
    lo, hi = s + order, t - order
    a_h = np.searchsorted(sh_pos, lo, "left")
    b_h = np.searchsorted(sh_pos, hi, "right")
    a_l = np.searchsorted(sl_pos, lo, "left")
    b_l = np.searchsorted(sl_pos, hi, "right")
    return a_h, b_h, a_l, b_l, (t - s + 1)


def dow_trend_array(h, l, sh_pos, sl_pos, a_h, b_h, a_l, b_l):
    """روند داو برای هر کندل (۱ صعودی، ۱- نزولی، ۰ خنثی) — معادل analysis.determine_trend."""
    n = len(a_h)
    cnt_h, cnt_l = b_h - a_h, b_l - a_l
    ok = (cnt_h >= 2) & (cnt_l >= 2)
    trend = np.zeros(n, dtype=np.int8)
    if not ok.any() or len(sh_pos) < 2 or len(sl_pos) < 2:
        return trend
    ih2 = sh_pos[np.clip(b_h - 1, 0, len(sh_pos) - 1)]
    ih1 = sh_pos[np.clip(b_h - 2, 0, len(sh_pos) - 1)]
    il2 = sl_pos[np.clip(b_l - 1, 0, len(sl_pos) - 1)]
    il1 = sl_pos[np.clip(b_l - 2, 0, len(sl_pos) - 1)]
    h1, h2, l1, l2 = h[ih1], h[ih2], l[il1], l[il2]
    trend[ok & (h2 > h1) & (l2 > l1)] = 1
    trend[ok & (h2 < h1) & (l2 < l1)] = -1
    return trend


def htf_trend_series(series, order, window):
    """
    روند هر کندل تایم‌فریم بالاتر، دقیقاً مثل analysis.trend_from_df روی آخرین
    `window` کندل بسته‌شده. اگه کندل کافی نبود (کمتر از order*2+5)، ۲ برمی‌گرده یعنی
    «بررسی‌نشده» (ربات زنده هم اون تایم‌فریم رو نمی‌شمره).
    """
    sh_pos, sl_pos = swing_positions(series.h, series.l, order)
    a_h, b_h, a_l, b_l, wlen = window_swing_bounds(series.n, window, order, sh_pos, sl_pos)
    trend = dow_trend_array(series.h, series.l, sh_pos, sl_pos, a_h, b_h, a_l, b_l)
    trend = trend.copy()
    trend[wlen < order * 2 + 5] = 2
    return trend


def align_to(main_close_ts, htf_series, htf_trend):
    """روند آخرین کندل بسته‌شده‌ی تایم‌فریم بالاتر در لحظه‌ی بسته‌شدن هر کندل اصلی."""
    idx = np.searchsorted(htf_series.close_ts, main_close_ts, "right") - 1
    out = np.full(len(main_close_ts), 2, dtype=np.int8)
    ok = idx >= 0
    out[ok] = htf_trend[idx[ok]]
    return out


def _support_resistance_arrays(c, h, l, sh_pos, sl_pos, a_h, b_h, a_l, b_l, cluster_pct, need):
    """نزدیک‌ترین حمایت/مقاومت برای هر کندلِ لازم (دقیقاً با analysis._cluster_levels)."""
    n = len(c)
    sup = np.full(n, np.nan)
    res = np.full(n, np.nan)
    key_h = key_l = None
    lv_h = lv_l = []
    hl = h.tolist()
    ll = l.tolist()
    for t in np.flatnonzero(need).tolist():
        kh = (int(a_h[t]), int(b_h[t]))
        if kh != key_h:
            key_h = kh
            lv_h = analysis._cluster_levels([hl[p] for p in sh_pos[kh[0]:kh[1]].tolist()], cluster_pct)
        kl = (int(a_l[t]), int(b_l[t]))
        if kl != key_l:
            key_l = kl
            lv_l = analysis._cluster_levels([ll[p] for p in sl_pos[kl[0]:kl[1]].tolist()], cluster_pct)
        price = c[t]
        i = bisect.bisect_right(lv_h, price)
        if i < len(lv_h):
            res[t] = lv_h[i]
        j = bisect.bisect_left(lv_l, price) - 1
        if j >= 0:
            sup[t] = lv_l[j]
    return sup, res


class Structural:
    """کاندیدهای ساختاری یک شاخه (خرید یا فروش) — تنک، به‌همراه امتیاز هر کندل."""
    __slots__ = ("idx", "sl", "level", "atr", "tp_uses_level", "score")

    def __init__(self, idx, sl, level, atr, tp_uses_level, score):
        self.idx = idx.astype(np.int64)
        self.sl = sl
        self.level = level
        self.atr = atr
        self.tp_uses_level = tp_uses_level
        self.score = score


def _prev(a):
    """مقدار کندل قبلی برای هر اندیس (اندیس ۰ = NaN)."""
    p = np.full(len(a), np.nan)
    p[1:] = a[:-1]
    return p


def compute_structural(series, cfg, first_idx=0):
    """
    استراتژی ترکیبی وزن‌دار برای کل سری — معادل دقیق strategies.generate_weighted_confluence
    روی پنجره‌ی CANDLE_LIMIT کندلیِ ختم‌شده به هر کندل. خروجی:
    {"weighted_confluence": (Structural_long, Structural_short)}
    """
    import strategies as st_mod
    n = series.n
    o, h, l, c, v = series.o, series.h, series.l, series.c, series.v
    order = cfg.SWING_ORDER
    W = int(getattr(cfg, "CANDLE_LIMIT", 300))
    prox = cfg.PROXIMITY_PCT
    buf_mult = cfg.ATR_SL_BUFFER

    sh_pos, sl_pos = swing_positions(h, l, order)
    a_h, b_h, a_l, b_l, wlen = window_swing_bounds(n, W, order, sh_pos, sl_pos)
    trend = dow_trend_array(h, l, sh_pos, sl_pos, a_h, b_h, a_l, b_l)

    df = pd.DataFrame({"high": h, "low": l, "close": c})
    atr = analysis.compute_atr(df, cfg.ATR_PERIOD).values
    atr_ok = (~np.isnan(atr)) & (atr > 0)

    in_range = np.zeros(n, dtype=bool)
    in_range[first_idx:] = True
    ok = in_range & (wlen >= order * 2 + 5) & (wlen >= 30) & atr_ok

    sup, res = _support_resistance_arrays(c, h, l, sh_pos, sl_pos, a_h, b_h, a_l, b_l, cfg.SR_CLUSTER_PCT, ok)
    has_sup = ~np.isnan(sup) & (sup != 0)
    has_res = ~np.isnan(res) & (res != 0)

    s_fast = ind.sma(c, cfg.SMA_FAST)
    s_mid = ind.sma(c, cfg.SMA_MID)
    s_slow = ind.sma(c, cfg.SMA_SLOW)
    rsi = ind.rsi_sma(c, cfg.RSI_PERIOD)
    k = cfg.CYCLE_SLOPE_BARS
    slope = np.full(n, np.nan)
    slope[k:] = s_slow[k:] - s_slow[:-k]
    vol_up = ind.sma(v, cfg.CYCLE_VOL_FAST) >= ind.sma(v, cfg.CYCLE_VOL_SLOW)

    with np.errstate(invalid="ignore", divide="ignore"):
        if getattr(cfg, "SR_CONFIRM", False):
            mx = cfg.SR_CONFIRM_MAX_DIST_PCT
            pl, ph, pc = _prev(l), _prev(h), _prev(c)
            ref_l = ph if cfg.SR_CONFIRM_BREAK == "high" else pc
            ref_s = pl if cfg.SR_CONFIRM_BREAK == "high" else pc
            near_sup = has_sup & (pl <= sup * (1 + prox / 100)) & (pc >= sup) & (c > o) & (c > ref_l) \
                & ((c - sup) / sup * 100 >= 0) & ((c - sup) / sup * 100 <= mx)
            near_res = has_res & (ph >= res * (1 - prox / 100)) & (pc <= res) & (c < o) & (c < ref_s) \
                & ((res - c) / res * 100 >= 0) & ((res - c) / res * 100 <= mx)
            sup_stop = np.minimum(sup, pl)
            res_stop = np.maximum(res, ph)
        else:
            near_sup = has_sup & ((c - sup) / sup * 100 >= 0) & ((c - sup) / sup * 100 <= prox) & (l <= sup * (1 + prox / 100))
            near_res = has_res & ((res - c) / res * 100 >= 0) & ((res - c) / res * 100 <= prox) & (h >= res * (1 - prox / 100))
            sup_stop, res_stop = sup, res
        sma_l = np.where((s_fast > s_mid) & (s_mid > s_slow), 1.0, np.where((s_fast > s_mid) & (c > s_slow), 0.5, 0.0))
        sma_s = np.where((s_fast < s_mid) & (s_mid < s_slow), 1.0, np.where((s_fast < s_mid) & (c < s_slow), 0.5, 0.0))
        lo_l, hi_l = cfg.RSI_LONG_ZONE
        lo_s, hi_s = cfg.RSI_SHORT_ZONE
        rsi_l = (rsi > lo_l) & (rsi < hi_l)
        rsi_s = (rsi > lo_s) & (rsi < hi_s)
        cyc_l = (slope > 0) & vol_up
        cyc_s = (slope < 0) & vol_up

    Wt = cfg.WC_WEIGHTS
    total = st_mod.score_total(cfg)
    on_l = {"dow": (trend == 1).astype(float), "sr": near_sup.astype(float), "sma": sma_l,
            "rsi": rsi_l.astype(float), "cycle": cyc_l.astype(float)}
    on_s = {"dow": (trend == -1).astype(float), "sr": near_res.astype(float), "sma": sma_s,
            "rsi": rsi_s.astype(float), "cycle": cyc_s.astype(float)}
    score_l = np.zeros(n)
    score_s = np.zeros(n)
    for comp in st_mod.COMPONENTS:
        w = float(Wt.get(comp, 0.0))
        score_l = score_l + w * on_l[comp]
        score_s = score_s + w * on_s[comp]
    score_l = score_l / total * 100
    score_s = score_s / total * 100

    floor = min([float(x) for x in getattr(cfg, "WC_SCORE_CHOICES", [])] + [float(cfg.WC_MIN_SCORE_PCT)]) - 1e-9
    buf = atr * buf_mult
    L = cfg.SWING_STOP_LOOKBACK
    lmin = ind.last_min(l, L)
    hmax = ind.last_max(h, L)
    sl_long = np.where(near_sup, sup_stop - buf, lmin - buf)
    sl_short = np.where(near_res, res_stop + buf, hmax + buf)
    Lm = ok & (c > o) & (score_l >= floor)
    Sm = ok & (c < o) & (score_s >= floor)

    def mk(mask, sl, level, score):
        idx = np.flatnonzero(mask)
        return Structural(idx, sl[idx], level[idx], atr[idx], False, score[idx])

    return {"weighted_confluence": (mk(Lm, sl_long, res, score_l), mk(Sm, sl_short, sup, score_s))}


def finalize_vec(side, entry, sl, level, atr, tp_uses_level, min_rr, min_sl_pct, min_sl_atr, room):
    """نسخه‌ی آرایه‌ای analysis.finalize_signal — با همون ترتیب عملیات محاسباتی."""
    with np.errstate(invalid="ignore", divide="ignore"):
        if side == LONG:
            risk = entry - sl
        else:
            risk = sl - entry
        valid = risk > 0
        req = np.maximum(entry * min_sl_pct / 100.0, np.nan_to_num(atr) * min_sl_atr)
        widen = valid & (risk < req)
        if widen.any():
            sl = sl.copy()
            risk = risk.copy()
            if side == LONG:
                sl[widen] = entry[widen] - req[widen]
                risk[widen] = entry[widen] - sl[widen]
            else:
                sl[widen] = entry[widen] + req[widen]
                risk[widen] = sl[widen] - entry[widen]
        has_level = (~np.isnan(level)) & (level != 0)
        if side == LONG:
            tp_min = entry + risk * min_rr
            tp = np.where(has_level, np.maximum(tp_min, level), tp_min) if tp_uses_level else tp_min
            rr = (tp - entry) / risk
        else:
            tp_min = entry - risk * min_rr
            tp = np.where(has_level, np.minimum(tp_min, level), tp_min) if tp_uses_level else tp_min
            rr = (entry - tp) / risk
        valid &= rr >= min_rr - 1e-9
        if room:
            room_v = (level - entry) if side == LONG else (entry - level)
            valid &= (~has_level) | (room_v >= min_rr * risk - 1e-9)
    return valid, sl, tp, rr


class Finalized:
    """سیگنال نهایی (تنک، مرتب بر اساس ایندکس کندل)."""
    __slots__ = ("idx", "side", "sl", "tp", "rr", "score")

    def __init__(self, idx, side, sl, tp, rr, score):
        self.idx, self.side, self.sl, self.tp, self.rr, self.score = idx, side, sl, tp, rr, score

    @staticmethod
    def empty():
        z = np.zeros(0)
        return Finalized(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8), z, z, z, z)


def finalize_strategy(structural_pair, close, variant):
    """حداقل امتیاز + مرحله‌ی نهایی روی دو شاخه‌ی خرید/فروش؛ خرید اولویت داره (مثل کد زنده)."""
    min_score = float(variant.get("min_score", 0.0)) - 1e-9
    parts = []
    for side, st in ((LONG, structural_pair[0]), (SHORT, structural_pair[1])):
        if len(st.idx) == 0:
            continue
        entry = close[st.idx]
        valid, sl, tp, rr = finalize_vec(side, entry, st.sl, st.level, st.atr, st.tp_uses_level,
                                         variant["min_rr"], variant["min_sl_pct"], variant["min_sl_atr"],
                                         variant["room"])
        valid &= st.score >= min_score
        if valid.any():
            parts.append((st.idx[valid], np.full(int(valid.sum()), side, dtype=np.int8),
                          sl[valid], tp[valid], rr[valid], st.score[valid]))
    if not parts:
        return Finalized.empty()
    cols = [np.concatenate([p[i] for p in parts]) for i in range(6)]
    idx, side, sl, tp, rr, score = cols
    order = np.lexsort((-side, idx))
    idx, side, sl, tp, rr, score = idx[order], side[order], sl[order], tp[order], rr[order], score[order]
    keep = np.ones(len(idx), dtype=bool)
    keep[1:] = idx[1:] != idx[:-1]
    return Finalized(idx[keep], side[keep], sl[keep], tp[keep], rr[keep], score[keep])


def combine(finals, active, mode="any", lookback=8):
    """فقط یک استراتژی داریم؛ خروجی نهایی همون سیگنال‌های نهایی‌شده‌ست."""
    name = "weighted_confluence"
    f = finals.get(name)
    if f is None:
        return None
    return {"idx": f.idx, "side": f.side, "sl": f.sl, "tp": f.tp, "rr": f.rr, "score": f.score,
            "label": [name] * len(f.idx)}
