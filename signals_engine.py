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

STRATEGY_NAMES = ["dow_support_resistance", "breakout", "volume_spike", "candle_setup",
                  "ema_cross", "ema_pullback", "bb_reversion", "rsi_pullback", "squeeze_breakout", "donchian_trend"]
INDICATOR_STRATEGIES = STRATEGY_NAMES[4:]
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
    """کاندیدهای ساختاری یک شاخه (خرید یا فروش) از یک استراتژی — تنک."""
    __slots__ = ("idx", "sl", "level", "atr", "tp_uses_level")

    def __init__(self, idx, sl, level, atr, tp_uses_level):
        self.idx = idx.astype(np.int64)
        self.sl = sl
        self.level = level
        self.atr = atr
        self.tp_uses_level = tp_uses_level


def compute_structural(series, cfg, first_idx=0):
    """
    همه‌ی کاندیدهای ساختاری هر استراتژی برای این سری. خروجی:
    {strategy_name: (Structural_long, Structural_short)}
    """
    n = series.n
    o, h, l, c, v = series.o, series.h, series.l, series.c, series.v
    order = cfg.SWING_ORDER
    W = int(getattr(cfg, "CANDLE_LIMIT", 300))
    prox = cfg.PROXIMITY_PCT
    buf = cfg.ATR_SL_BUFFER

    sh_pos, sl_pos = swing_positions(h, l, order)
    a_h, b_h, a_l, b_l, wlen = window_swing_bounds(n, W, order, sh_pos, sl_pos)
    trend = dow_trend_array(h, l, sh_pos, sl_pos, a_h, b_h, a_l, b_l)

    df = pd.DataFrame({"high": h, "low": l, "close": c})
    atr = analysis.compute_atr(df, cfg.ATR_PERIOD).values
    atr_ok = (~np.isnan(atr)) & (atr > 0)

    in_range = np.zeros(n, dtype=bool)
    in_range[first_idx:] = True
    base_ok = in_range & (wlen >= order * 2 + 5)   # همون گیت len(df) در scan_symbol

    # حجم هم‌جهت داو: میانگین حجم کندل‌های صعودی/نزولی در ۲۰ کندل اخیر
    change = c - o
    up_mean = pd.Series(np.where(change > 0, v, np.nan)).rolling(20, min_periods=1).mean().values
    dn_mean = pd.Series(np.where(change < 0, v, np.nan)).rolling(20, min_periods=1).mean().values
    both = (~np.isnan(up_mean)) & (~np.isnan(dn_mean))
    vol_up = both & (up_mean > dn_mean)
    vol_dn = both & (dn_mean > up_mean)

    active = set(getattr(cfg, "_ENGINE_STRATEGIES", STRATEGY_NAMES))
    need_dow_sr = base_ok & atr_ok & (((trend == 1) & vol_up) | ((trend == -1) & vol_dn))
    need_candle_sr = base_ok & atr_ok & (wlen >= order * 2 + 10)
    need = np.zeros(n, dtype=bool)
    if "dow_support_resistance" in active:
        need |= need_dow_sr
    if "candle_setup" in active:
        need |= need_candle_sr
    sup, res = _support_resistance_arrays(c, h, l, sh_pos, sl_pos, a_h, b_h, a_l, b_l, cfg.SR_CLUSTER_PCT, need)
    has_sup = ~np.isnan(sup) & (sup != 0)
    has_res = ~np.isnan(res) & (res != 0)

    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1.0), np.nan)
    out = {}

    def mk(mask, sl, level, tp_uses_level):
        idx = np.flatnonzero(mask)
        lv = level[idx] if level is not None else np.full(len(idx), np.nan)
        return Structural(idx, sl[idx], lv, atr[idx], tp_uses_level)

    # ---------- داو + حمایت/مقاومت ----------
    if "dow_support_resistance" in active:
        with np.errstate(invalid="ignore", divide="ignore"):
            dist_l = (c - sup) / sup * 100
            dist_s = (res - c) / res * 100
        L = need_dow_sr & (trend == 1) & has_sup & vol_up & (dist_l >= 0) & (dist_l <= prox)
        S = need_dow_sr & (trend == -1) & has_res & vol_dn & (dist_s >= 0) & (dist_s <= prox)
        if cfg.USE_REJECTION_CONFIRMATION:
            with np.errstate(invalid="ignore"):
                rej_l = (rng > 0) & (l <= sup * (1 + prox / 100)) & (c > o) & (pos >= 0.5)
                rej_s = (rng > 0) & (h >= res * (1 - prox / 100)) & (c < o) & (((h - c) / np.where(rng > 0, rng, 1.0)) >= 0.5)
            L &= rej_l
            S &= rej_s
        out["dow_support_resistance"] = (
            mk(L, sup - atr * buf, res, True),
            mk(S, res + atr * buf, sup, True),
        )

    # ---------- بریک‌اوت ----------
    if "breakout" in active:
        lb = int(getattr(cfg, "BREAKOUT_LOOKBACK", 40))
        mult = getattr(cfg, "BREAKOUT_VOLUME_MULT", 1.5)
        prev_high = pd.Series(h).rolling(lb).max().shift(1).values
        prev_low = pd.Series(l).rolling(lb).min().shift(1).values
        avg_vol = pd.Series(v).rolling(lb).mean().shift(1).values
        ok = base_ok & atr_ok & (wlen >= lb + 5)
        with np.errstate(invalid="ignore"):
            vol_ok = v > avg_vol * mult
            L = ok & vol_ok & (c > prev_high)
            S = ok & vol_ok & (c < prev_low)
        out["breakout"] = (
            mk(L, prev_high - atr * buf, None, False),
            mk(S, prev_low + atr * buf, None, False),
        )

    # ---------- افزایش ناگهانی حجم ----------
    if "volume_spike" in active:
        lb = int(getattr(cfg, "VOLUME_SPIKE_LOOKBACK", 30))
        mult = getattr(cfg, "VOLUME_SPIKE_MULT", 2.5)
        body_min = getattr(cfg, "VOLUME_SPIKE_MIN_BODY_PCT", 0.5)
        avg_vol = pd.Series(v).rolling(lb).mean().shift(1).values
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = base_ok & atr_ok & (wlen >= lb + 5) & (avg_vol > 0)
            spike = v > avg_vol * mult
            ratio = np.where(rng > 0, np.abs(c - o) / np.where(rng > 0, rng, 1.0), 0.0)
            directional = ratio >= body_min
        base = ok & spike & directional
        L = base & (c > o)
        S = base & ~(c > o)
        out["volume_spike"] = (
            mk(L, l - atr * buf, None, False),
            mk(S, h + atr * buf, None, False),
        )

    # ---------- کندل ستاپ (TST / BOF) ----------
    if "candle_setup" in active:
        third_high = pos >= 2 / 3
        third_low = pos <= 1 / 3
        third_high = np.where(np.isnan(pos), False, third_high)
        third_low = np.where(np.isnan(pos), False, third_low)
        third_mid = ~third_high & ~third_low
        with np.errstate(invalid="ignore"):
            t_l = l <= sup * (1 + prox / 100)
            b_l_ = l < sup
            back_l = c >= sup
            L = need_candle_sr & has_sup & ((t_l & ~b_l_ & third_high) | (t_l & b_l_ & back_l & (third_high | third_mid)))
            t_s = h >= res * (1 - prox / 100)
            b_s = h > res
            back_s = c <= res
            S = need_candle_sr & has_res & ((t_s & ~b_s & third_low) | (t_s & b_s & back_s & (third_low | third_mid)))
        out["candle_setup"] = (
            mk(L, l - atr * buf, res, False),
            mk(S, h + atr * buf, sup, False),
        )

    new_active = [x for x in INDICATOR_STRATEGIES if x in active]
    if new_active:
        out.update(_indicator_structural(series, cfg, new_active, atr, atr_ok, base_ok, wlen, W))
    return out


def _shift1(x):
    y = np.empty_like(x, dtype=np.float64)
    y[0] = np.nan
    y[1:] = x[:-1]
    return y


def _indicator_structural(series, cfg, names, atr, atr_ok, base_ok, wlen, W):
    """نسخه‌ی وکتوریزه‌ی استراتژی‌های اندیکاتوری strategies.py (همون قواعد، همون پنجره)."""
    n = series.n
    o, h, l, c, v = series.o, series.h, series.l, series.c, series.v
    t = np.arange(n)
    s = np.maximum(0, t - W + 1)
    tp = np.maximum(t - 1, s)          # کندل قبلی داخل همون پنجره
    ok = base_ok & atr_ok & (wlen >= 30)
    buf = atr * cfg.ATR_SL_BUFFER
    L = cfg.SWING_STOP_LOOKBACK
    lmin = ind.last_min(l, L)
    hmax = ind.last_max(h, L)
    out = {}
    nan = np.full(n, np.nan)

    def mk(mask, sl):
        idx = np.flatnonzero(mask)
        return Structural(idx, sl[idx], np.full(len(idx), np.nan), atr[idx], False)

    def ema_win(span):
        full = ind.ema(c, span)
        return ind.ema_window_at(c, full, span, t, s), ind.ema_window_at(c, full, span, tp, s)

    with np.errstate(invalid="ignore"):
        if "ema_cross" in names:
            ef, ef_p = ema_win(cfg.EMA_CROSS_FAST)
            es_, es_p = ema_win(cfg.EMA_CROSS_SLOW)
            et, _ = ema_win(cfg.EMA_CROSS_TREND)
            Lm = ok & (ef > es_) & (ef_p <= es_p) & (c > et)
            Sm = ok & ~Lm & (ef < es_) & (ef_p >= es_p) & (c < et)
            out["ema_cross"] = (mk(Lm, lmin - buf), mk(Sm, hmax + buf))

        if "ema_pullback" in names:
            ef, _ = ema_win(cfg.EMA_PB_FAST)
            es_, _ = ema_win(cfg.EMA_PB_SLOW)
            Lm = ok & (ef > es_) & (c > es_) & (l <= ef) & (c > ef) & (c > o)
            Sm = ok & (ef < es_) & (c < es_) & (h >= ef) & (c < ef) & (c < o)
            out["ema_pullback"] = (mk(Lm, lmin - buf), mk(Sm, hmax + buf))

        if "bb_reversion" in names or "squeeze_breakout" in names:
            mid, up, lo = ind.bollinger(c, cfg.BB_PERIOD, cfg.BB_STD)
        if "bb_reversion" in names or "rsi_pullback" in names:
            rsi = ind.rsi_sma(c, cfg.RSI_PERIOD)
            rsi_p = _shift1(rsi)
        c_p = _shift1(c)

        if "bb_reversion" in names:
            lo_p, up_p = _shift1(lo), _shift1(up)
            Lm = ok & (c_p < lo_p) & (c > lo) & (rsi_p < cfg.BB_RSI_LOW)
            Sm = ok & (c_p > up_p) & (c < up) & (rsi_p > cfg.BB_RSI_HIGH)
            l_p, h_p = _shift1(l), _shift1(h)
            out["bb_reversion"] = (mk(Lm, np.minimum(l_p, l) - buf), mk(Sm, np.maximum(h_p, h) + buf))

        if "rsi_pullback" in names:
            tr = ind.sma(c, cfg.RSI_PB_TREND_SMA)
            Lm = ok & (c > tr) & (rsi_p < cfg.RSI_PB_LONG_CROSS) & (cfg.RSI_PB_LONG_CROSS <= rsi)
            Sm = ok & (c < tr) & (rsi_p > cfg.RSI_PB_SHORT_CROSS) & (cfg.RSI_PB_SHORT_CROSS >= rsi)
            out["rsi_pullback"] = (mk(Lm, lmin - buf), mk(Sm, hmax + buf))

        if "squeeze_breakout" in names:
            kc_up = mid + cfg.SQUEEZE_KC_MULT * atr
            kc_lo = mid - cfg.SQUEEZE_KC_MULT * atr
            sq = ((up < kc_up) & (lo > kc_lo)).astype(np.float64)
            n_sq = pd.Series(sq).rolling(cfg.SQUEEZE_LOOKBACK, min_periods=1).sum().shift(1).values
            avg_v = ind.prev_mean(v, 20)
            base = ok & (n_sq >= cfg.SQUEEZE_MIN_BARS) & (v > avg_v * cfg.SQUEEZE_VOLUME_MULT)
            Lm = base & (c > up)
            Sm = base & ~(c > up) & (c < lo)
            out["squeeze_breakout"] = (mk(Lm, mid - buf), mk(Sm, mid + buf))

        if "donchian_trend" in names:
            hh = ind.prev_max(h, cfg.DONCHIAN_PERIOD)
            ll = ind.prev_min(l, cfg.DONCHIAN_PERIOD)
            tr = ind.sma(c, cfg.DONCHIAN_TREND_SMA)
            dist = atr * cfg.DONCHIAN_ATR_STOP
            Lm = ok & (c > hh) & (c > tr)
            Sm = ok & ~Lm & (c < ll) & (c < tr)
            out["donchian_trend"] = (mk(Lm, c - dist), mk(Sm, c + dist))
    return out


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
    """سیگنال نهایی یک استراتژی (تنک، مرتب بر اساس ایندکس کندل)."""
    __slots__ = ("idx", "side", "sl", "tp", "rr")

    def __init__(self, idx, side, sl, tp, rr):
        self.idx, self.side, self.sl, self.tp, self.rr = idx, side, sl, tp, rr

    @staticmethod
    def empty():
        z = np.zeros(0)
        return Finalized(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8), z, z, z)


def finalize_strategy(structural_pair, close, variant):
    """اعمال مرحله‌ی نهایی روی دو شاخه‌ی خرید/فروش و ادغام (خرید اولویت داره، مثل کد زنده)."""
    parts = []
    for side, st in ((LONG, structural_pair[0]), (SHORT, structural_pair[1])):
        if len(st.idx) == 0:
            continue
        entry = close[st.idx]
        valid, sl, tp, rr = finalize_vec(side, entry, st.sl, st.level, st.atr, st.tp_uses_level,
                                         variant["min_rr"], variant["min_sl_pct"], variant["min_sl_atr"],
                                         variant["room"])
        if valid.any():
            parts.append((st.idx[valid], np.full(int(valid.sum()), side, dtype=np.int8),
                          sl[valid], tp[valid], rr[valid]))
    if not parts:
        return Finalized.empty()
    idx = np.concatenate([p[0] for p in parts])
    side = np.concatenate([p[1] for p in parts])
    sl = np.concatenate([p[2] for p in parts])
    tp = np.concatenate([p[3] for p in parts])
    rr = np.concatenate([p[4] for p in parts])
    # اگه روی یک کندل هم خرید و هم فروش معتبر بود، خرید (شاخه‌ی اول) انتخاب می‌شه
    order = np.lexsort((-side, idx))
    idx, side, sl, tp, rr = idx[order], side[order], sl[order], tp[order], rr[order]
    keep = np.ones(len(idx), dtype=bool)
    keep[1:] = idx[1:] != idx[:-1]
    return Finalized(idx[keep], side[keep], sl[keep], tp[keep], rr[keep])


def combine(finals, active, mode, lookback=8):
    """
    ترکیب سیگنال استراتژی‌ها (معادل strategies.generate_combined_signal).
    finals: dict name -> Finalized ; خروجی: (idx, side, sl, tp, rr, label_list_per_row)
    """
    active = [a for a in active if a in finals]
    if not active:
        return None
    if mode == "any" or len(active) == 1:
        taken = {}
        for name in active:
            f = finals[name]
            for k in range(len(f.idx)):
                t = int(f.idx[k])
                if t not in taken:
                    taken[t] = (int(f.side[k]), f.sl[k], f.tp[k], f.rr[k],
                                name if (mode == "any" or len(active) == 1) else name)
        if not taken:
            return _empty_combined()
        ts = np.array(sorted(taken), dtype=np.int64)
        rows = [taken[t] for t in ts.tolist()]
        if mode == "all" and len(active) == 1:
            rows = [(r[0], r[1], r[2], r[3], active[0]) for r in rows]
        if mode == "confirm" and len(active) == 1:
            rows = [(r[0], r[1], r[2], r[3], active[0]) for r in rows]
        return _pack(ts, rows)

    if mode == "all":
        base = finals[active[0]]
        maps = [dict(zip(finals[a].idx.tolist(), range(len(finals[a].idx)))) for a in active]
        out_t, rows = [], []
        label = "+".join(active)
        for k0, t in enumerate(base.idx.tolist()):
            side0 = int(base.side[k0])
            best = (base.sl[k0], base.tp[k0], base.rr[k0])
            ok = True
            for ai in range(1, len(active)):
                k = maps[ai].get(t)
                if k is None:
                    ok = False
                    break
                f = finals[active[ai]]
                if int(f.side[k]) != side0:
                    ok = False
                    break
                if f.rr[k] < best[2]:
                    best = (f.sl[k], f.tp[k], f.rr[k])
            if ok:
                out_t.append(t)
                rows.append((side0, best[0], best[1], best[2], label))
        if not out_t:
            return _empty_combined()
        return _pack(np.array(out_t, dtype=np.int64), rows)

    if mode == "confirm":
        trig = finals[active[0]]
        label = "|".join(active)
        conf_idx = {}
        for a in active[1:]:
            f = finals[a]
            conf_idx[a] = {LONG: f.idx[f.side == LONG], SHORT: f.idx[f.side == SHORT]}
        out_t, rows = [], []
        for k0, t in enumerate(trig.idx.tolist()):
            side0 = int(trig.side[k0])
            ok = True
            for a in active[1:]:
                arr = conf_idx[a][side0]
                j = np.searchsorted(arr, t, "right") - 1
                if j < 0 or arr[j] < t - lookback + 1:
                    ok = False
                    break
            if ok:
                out_t.append(t)
                rows.append((side0, trig.sl[k0], trig.tp[k0], trig.rr[k0], label))
        if not out_t:
            return _empty_combined()
        return _pack(np.array(out_t, dtype=np.int64), rows)

    raise ValueError(f"حالت ترکیب ناشناخته: {mode}")


def _empty_combined():
    z = np.zeros(0)
    return {"idx": np.zeros(0, dtype=np.int64), "side": np.zeros(0, dtype=np.int8),
            "sl": z, "tp": z, "rr": z, "label": []}


def _pack(ts, rows):
    return {
        "idx": ts,
        "side": np.array([r[0] for r in rows], dtype=np.int8),
        "sl": np.array([r[1] for r in rows], dtype=np.float64),
        "tp": np.array([r[2] for r in rows], dtype=np.float64),
        "rr": np.array([r[3] for r in rows], dtype=np.float64),
        "label": [r[4] for r in rows],
    }
