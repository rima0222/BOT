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
import money

STRATEGY_NAMES = ["weighted_confluence", "box_breakout", "box_breakout@retest", "trend_follow", "fib_phase",
                  "pattern_structure", "pattern_structure@free", "contrarian_btc", "contrarian_btc@any"]
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


def pair_ratio_series(sym_arr, btc_arr, tf):
    """نمودار ارز÷BTC: فقط کندل‌هایی که هر دو دارن؛ باز/سقف/کف/بسته همه = نسبت قیمت بسته‌شدن
    (نمودار خطی؛ ربات زنده همین رو با pair_ratio_df می‌سازه)."""
    sym_arr = np.asarray(sym_arr)
    btc_arr = np.asarray(btc_arr)
    common, i1, i2 = np.intersect1d(sym_arr[:, 0].astype(np.int64), btc_arr[:, 0].astype(np.int64),
                                    assume_unique=True, return_indices=True)
    if len(common) == 0:
        return None
    r = sym_arr[i1, 4] / btc_arr[i2, 4]
    arr = np.column_stack([common.astype(np.float64), r, r, r, r, np.zeros(len(r))])
    return Series(arr, tf)


def pair_ratio_df(sym_df, btc_df):
    """نسخه‌ی ربات زنده (روی دیتافریم) — همون ساخت نمودار نسبت."""
    m = sym_df[["timestamp", "close"]].merge(btc_df[["timestamp", "close"]], on="timestamp", suffixes=("_s", "_b"))
    r = m["close_s"].values / m["close_b"].values
    return pd.DataFrame({"timestamp": m["timestamp"], "open": r, "high": r, "low": r, "close": r,
                         "volume": np.zeros(len(r))})


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
    __slots__ = ("idx", "sl", "level", "atr", "tp_uses_level", "score", "use_score", "entry")

    def __init__(self, idx, sl, level, atr, tp_uses_level, score, use_score=True, entry=None):
        self.idx = idx.astype(np.int64)
        self.sl = sl
        self.level = level
        self.atr = atr
        self.tp_uses_level = tp_uses_level
        self.score = score
        self.use_score = use_score
        self.entry = entry   # قیمت ورود (لیمیت پولبک)؛ None = قیمت بسته‌شدن کندل سیگنال


def _prev(a):
    """مقدار کندل قبلی برای هر اندیس (اندیس ۰ = NaN)."""
    p = np.full(len(a), np.nan)
    p[1:] = a[:-1]
    return p


def compute_structural(series, cfg, first_idx=0):
    """
    استراتژی‌ها برای کل سری — معادل دقیق strategies.generate_* روی پنجره‌ی CANDLE_LIMIT
    کندلیِ ختم‌شده به هر کندل. فقط استراتژی‌های لازم (cfg._ENGINE_STRATEGIES) حساب می‌شن.
    خروجی: {name: (Structural_long, Structural_short)}
    """
    n = series.n
    h, l, c = series.h, series.l, series.c
    order = cfg.SWING_ORDER
    W = int(getattr(cfg, "CANDLE_LIMIT", 300))
    sh_pos, sl_pos = swing_positions(h, l, order)
    a_h, b_h, a_l, b_l, wlen = window_swing_bounds(n, W, order, sh_pos, sl_pos)
    trend = dow_trend_array(h, l, sh_pos, sl_pos, a_h, b_h, a_l, b_l)
    df = pd.DataFrame({"high": h, "low": l, "close": c})
    atr = analysis.compute_atr(df, cfg.ATR_PERIOD).values
    atr_ok = (~np.isnan(atr)) & (atr > 0)
    in_range = np.zeros(n, dtype=bool)
    in_range[first_idx:] = True
    ctx = {"sh_pos": sh_pos, "sl_pos": sl_pos, "bounds": (a_h, b_h, a_l, b_l), "wlen": wlen, "trend": trend,
           "atr": atr, "atr_ok": atr_ok, "in_range": in_range}
    names = list(getattr(cfg, "_ENGINE_STRATEGIES", None) or STRATEGY_NAMES)
    out = {}
    if "weighted_confluence" in names:
        out["weighted_confluence"] = _structural_weighted(series, cfg, ctx)
    if any(n.startswith("box_breakout") for n in names):
        out.update(_structural_breakout(series, cfg, ctx))
    if "trend_follow" in names:
        out["trend_follow"] = _structural_trend(series, cfg, ctx)
    if "fib_phase" in names:
        out["fib_phase"] = _structural_fib(series, cfg, ctx, W)
    if any(nm.startswith("pattern_structure") for nm in names):
        out.update(_structural_pattern(series, cfg, ctx))
    if any(nm.startswith("contrarian_btc") for nm in names):
        out.update(_structural_contrarian(series, cfg, ctx))
    return out


def _structural_contrarian(series, cfg, ctx):
    """
    معادل strategies.generate_contrarian_btc (الگوهای برگشتی، بدون فیلتر ساختار، حداقل حد ضرر CONTRA_MIN_SL_ATR).
    شرط «خلاف روند BTC» اینجا نیست — مثل ربات زنده، بعد از سیگنال در sim_engine.run_portfolio روی کلید
    «contrarian_btc» اعمال می‌شه. «contrarian_btc@any» همون سیگنال‌ها بدون این شرط (کنترل آزمایش).
    """
    import patterns
    import strategies
    c = series.c
    atr, in_range, trend = ctx["atr"], ctx["in_range"], ctx["trend"]
    names = patterns.pattern_names("reversal")
    sig = patterns.pattern_signals(series.o, series.h, series.l, c, atr, int(cfg.SWING_ORDER), names)
    base = in_range & ctx["atr_ok"] & (ctx["wlen"] >= 50)
    pcfg = strategies._ContraPatCfg(cfg)
    pair = []
    for side, arr_i in (("LONG", 1), ("SHORT", 3)):
        idx_l, sl_l = [], []
        for t in np.flatnonzero(base & (sig[arr_i] >= 0)).tolist():
            st = strategies.pattern_setup(side, c, atr, int(trend[t]), t, sig, pcfg)
            if st is not None:
                idx_l.append(t)
                sl_l.append(st[0])
        idx = np.array(idx_l, dtype=np.int64)
        pair.append(Structural(idx, np.array(sl_l, dtype=np.float64), np.full(len(idx), np.nan), atr[idx],
                               False, np.full(len(idx), np.nan), use_score=False))
    return {"contrarian_btc": tuple(pair), "contrarian_btc@any": tuple(pair)}


def _structural_pattern(series, cfg, ctx):
    """
    معادل strategies.generate_pattern_structure: همون کتابخونه‌ی الگو روی کل سری (الگوها فقط به کندل‌های
    گذشته‌ی نزدیک و سوینگ‌های تاییدشده وابسته‌ان) + همون تابع مشترک strategies.pattern_setup.
    دو کلید: با فیلتر ساختار داو و بدون اون (@free).
    """
    import patterns
    import strategies
    c = series.c
    atr, in_range, trend = ctx["atr"], ctx["in_range"], ctx["trend"]
    wlen = ctx["wlen"]
    names = patterns.pattern_names(getattr(cfg, "PAT_SET", "all"))
    sig = patterns.pattern_signals(series.o, series.h, series.l, c, atr, int(cfg.SWING_ORDER), names)
    base = in_range & ctx["atr_ok"] & (wlen >= 50)
    out = {}
    for key, mode in (("pattern_structure", "with"), ("pattern_structure@free", "off")):
        class _C:
            PAT_STRUCTURE = mode
            PAT_MIN_SL_ATR = cfg.PAT_MIN_SL_ATR
        pair = []
        for side, arr_i in (("LONG", 1), ("SHORT", 3)):
            idx_l, sl_l = [], []
            for t in np.flatnonzero(base & (sig[arr_i] >= 0)).tolist():
                st = strategies.pattern_setup(side, c, atr, int(trend[t]), t, sig, _C)
                if st is not None:
                    idx_l.append(t)
                    sl_l.append(st[0])
            idx = np.array(idx_l, dtype=np.int64)
            pair.append(Structural(idx, np.array(sl_l, dtype=np.float64), np.full(len(idx), np.nan), atr[idx],
                                   False, np.full(len(idx), np.nan), use_score=False))
        out[key] = tuple(pair)
    return out


def _structural_fib(series, cfg, ctx, W):
    """
    معادل strategies.generate_fib_phase. پیش‌فیلتر وکتوریزه (کندل برگشتی لازم)، بعد برای هر کاندید
    دقیقاً همون تابع مشترک strategies.fib_eval با سوینگ‌های تاییدشده‌ی همون پنجره صدا زده می‌شه.
    """
    import strategies
    n = series.n
    o, h, l, c, v = series.o, series.h, series.l, series.c, series.v
    atr, atr_ok, in_range, trend = ctx["atr"], ctx["atr_ok"], ctx["in_range"], ctx["trend"]
    sh_pos, sl_pos = ctx["sh_pos"], ctx["sl_pos"]
    a_h, b_h, a_l, b_l = ctx["bounds"]
    wlen = ctx["wlen"]
    rsi = ind.rsi_sma(c, int(cfg.RSI_PERIOD))
    h_p, l_p = _prev(h), _prev(l)
    with np.errstate(invalid="ignore"):
        base = in_range & atr_ok & (wlen >= 50)
        cand_l = base & (c > o) & (c > h_p)
        cand_s = base & (c < o) & (c < l_p)
    cols = {1: ([], [], []), -1: ([], [], [])}
    for side_code, cand, name in ((1, cand_l, "LONG"), (-1, cand_s, "SHORT")):
        for t in np.flatnonzero(cand).tolist():
            s0 = max(0, t - W + 1)
            st = strategies.fib_eval(name, o, h, l, c, v, atr, rsi, int(trend[t]), t, s0,
                                     sh_pos[a_h[t]:b_h[t]], sl_pos[a_l[t]:b_l[t]], cfg)
            if st is not None:
                cols[side_code][0].append(t)
                cols[side_code][1].append(st["sl"])
                cols[side_code][2].append(st["score"])

    def mk(side_code):
        idx = np.array(cols[side_code][0], dtype=np.int64)
        return Structural(idx, np.array(cols[side_code][1], dtype=np.float64), np.full(len(idx), np.nan), atr[idx],
                          False, np.array(cols[side_code][2], dtype=np.float64), use_score=True)

    return (mk(1), mk(-1))


def _structural_trend(series, cfg, ctx):
    """معادل strategies.generate_trend_follow."""
    n = series.n
    o, h, l, c, v = series.o, series.h, series.l, series.c, series.v
    N = int(cfg.TR_BREAKOUT_BARS)
    wlen, trend, atr, atr_ok, in_range = ctx["wlen"], ctx["trend"], ctx["atr"], ctx["atr_ok"], ctx["in_range"]
    ok = in_range & (wlen >= max(30, N + 2)) & atr_ok
    top = ind.prev_max(h, N)
    bot = ind.prev_min(l, N)
    top_p, bot_p, c_p = _prev(top), _prev(bot), _prev(c)
    ma = ind.sma(c, int(cfg.TR_MA))
    with np.errstate(invalid="ignore"):
        vm = float(getattr(cfg, "TR_VOL_MULT", 0.0) or 0.0)
        if vm > 0:
            va = ind.prev_mean(v, N)
            vol_ok = (va > 0) & (v >= vm * va)
        else:
            vol_ok = np.ones(n, dtype=bool)
        Lm = ok & (c > o) & (c > top) & (c_p <= top_p) & (c > ma) & (trend != -1) & vol_ok
        Sm = ok & (c < o) & (c < bot) & (c_p >= bot_p) & (c < ma) & (trend != 1) & vol_ok
        dist = cfg.TR_SL_ATR * atr
        sl_long = c - dist
        sl_short = c + dist
    nan_lv = np.full(n, np.nan)
    nan_sc = np.full(n, np.nan)

    def mk(mask, sl):
        idx = np.flatnonzero(mask)
        return Structural(idx, sl[idx], nan_lv[idx], atr[idx], False, nan_sc[idx], use_score=False)

    return (mk(Lm, sl_long), mk(Sm, sl_short))


def _structural_weighted(series, cfg, ctx):
    """معادل strategies.generate_weighted_confluence."""
    import strategies as st_mod
    n = series.n
    o, h, l, c, v = series.o, series.h, series.l, series.c, series.v
    order = cfg.SWING_ORDER
    prox = cfg.PROXIMITY_PCT
    buf_mult = cfg.ATR_SL_BUFFER
    sh_pos, sl_pos = ctx["sh_pos"], ctx["sl_pos"]
    a_h, b_h, a_l, b_l = ctx["bounds"]
    wlen, trend, atr, atr_ok, in_range = ctx["wlen"], ctx["trend"], ctx["atr"], ctx["atr_ok"], ctx["in_range"]
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
        return Structural(idx, sl[idx], level[idx], atr[idx], False, score[idx], use_score=True)

    return (mk(Lm, sl_long, res, score_l), mk(Sm, sl_short, sup, score_s))


def _structural_breakout(series, cfg, ctx):
    """معادل strategies.generate_box_breakout."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = series.n
    o, h, l, c, v = series.o, series.h, series.l, series.c, series.v
    N = int(cfg.BRK_BOX_BARS)
    sh_pos, sl_pos = ctx["sh_pos"], ctx["sl_pos"]
    a_h, b_h, a_l, b_l = ctx["bounds"]
    wlen, trend, atr, atr_ok, in_range = ctx["wlen"], ctx["trend"], ctx["atr"], ctx["atr_ok"], ctx["in_range"]
    atr_prev = _prev(atr)
    ok = in_range & (wlen >= max(30, N + 2)) & atr_ok & (~np.isnan(atr_prev)) & (atr_prev > 0)

    top = ind.prev_max(h, N)
    bot = ind.prev_min(l, N)
    with np.errstate(invalid="ignore"):
        height = top - bot
        box_ok = (height > 0) & (height <= cfg.BRK_BOX_MAX_ATR * atr_prev)
        tol = cfg.BRK_TOUCH_TOL * height
        n_top = np.zeros(n, dtype=np.int64)
        n_bot = np.zeros(n, dtype=np.int64)
        if n > N:
            hw = sliding_window_view(h, N)[:n - N]
            lw = sliding_window_view(l, N)[:n - N]
            n_top[N:] = (hw >= (top[N:] - tol[N:])[:, None]).sum(axis=1)
            n_bot[N:] = (lw <= (bot[N:] + tol[N:])[:, None]).sum(axis=1)
        box_ok &= (n_top >= cfg.BRK_BOX_MIN_TOUCHES) & (n_bot >= cfg.BRK_BOX_MIN_TOUCHES)

        vol_avg = ind.prev_mean(v, N)
        vol_ok = (vol_avg > 0) & (v >= cfg.BRK_VOL_MULT * vol_avg)
        mode = getattr(cfg, "BRK_MAIN_TREND", "not_against")
        if mode == "with":
            tr_l, tr_s = trend == 1, trend == -1
        elif mode == "not_against":
            tr_l, tr_s = trend != -1, trend != 1
        else:
            tr_l = tr_s = np.ones(n, dtype=bool)
        cand_l = ok & (c > o) & vol_ok & tr_l
        cand_s = ok & (c < o) & vol_ok & tr_s

        use_sr = bool(getattr(cfg, "BRK_USE_SR", True))
        ext = cfg.BRK_MAX_EXT_ATR * atr
        if use_sr:
            sup_p, res_p = _support_resistance_arrays(_prev(c), h, l, sh_pos, sl_pos, a_h, b_h, a_l, b_l,
                                                      cfg.SR_CLUSTER_PCT, cand_l | cand_s)
            has_res = ~np.isnan(res_p) & (res_p != 0)
            has_sup = ~np.isnan(sup_p) & (sup_p != 0)
        else:
            sup_p = res_p = np.full(n, np.nan)
            has_res = has_sup = np.zeros(n, dtype=bool)
        box_l = box_ok & (c > top) & (c - top <= ext)
        sr_l = has_res & (c > res_p) & (c - res_p <= ext)
        box_s = box_ok & (c < bot) & (bot - c <= ext)
        sr_s = has_sup & (c < sup_p) & (sup_p - c <= ext)
        Lm = cand_l & (box_l | sr_l)
        Sm = cand_s & (box_s | sr_s)

        buf = atr * cfg.ATR_SL_BUFFER
        if getattr(cfg, "BRK_SL_MODE", "candle") == "mid":
            mid = (top + bot) / 2.0
            sl_long = np.where(box_l, mid, res_p) - buf
            sl_short = np.where(box_s, mid, sup_p) + buf
        else:
            sl_long = l - buf
            sl_short = h + buf

        # ورود با پولبک: لیمیت روی سطح شکسته‌شده (+کمی فاصله، ولی نه بدتر از قیمت فعلی)
        off = cfg.BRK_RETEST_OFFSET_ATR * atr
        rsl = cfg.BRK_RETEST_SL_ATR * atr
        lv_l = np.where(box_l, top, res_p)
        lv_s = np.where(box_s, bot, sup_p)
        ent_rl = np.minimum(lv_l + off, c)
        ent_rs = np.maximum(lv_s - off, c)
        sl_rl = lv_l - rsl
        sl_rs = lv_s + rsl

    nan_lv = np.full(n, np.nan)
    nan_sc = np.full(n, np.nan)

    def mk(mask, sl, entry=None):
        idx = np.flatnonzero(mask)
        return Structural(idx, sl[idx], nan_lv[idx], atr[idx], False, nan_sc[idx], use_score=False,
                          entry=None if entry is None else entry[idx])

    return {"box_breakout": (mk(Lm, sl_long), mk(Sm, sl_short)),
            "box_breakout@retest": (mk(Lm, sl_rl, ent_rl), mk(Sm, sl_rs, ent_rs))}


def finalize_vec(side, entry, sl, level, atr, tp_uses_level, min_rr, min_sl_pct, min_sl_atr, room, net=None):
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
            tp_min = money.net_tp(True, entry, sl, min_rr, *net) if net else entry + risk * min_rr
            tp = np.where(has_level, np.maximum(tp_min, level), tp_min) if tp_uses_level else tp_min
            rr = (tp - entry) / risk
        else:
            tp_min = money.net_tp(False, entry, sl, min_rr, *net) if net else entry - risk * min_rr
            tp = np.where(has_level, np.minimum(tp_min, level), tp_min) if tp_uses_level else tp_min
            rr = (entry - tp) / risk
        valid &= rr >= min_rr - 1e-9
        if room:
            room_v = (level - entry) if side == LONG else (entry - level)
            valid &= (~has_level) | (room_v >= min_rr * risk - 1e-9)
    return valid, sl, tp, rr


class Finalized:
    """سیگنال نهایی (تنک، مرتب بر اساس ایندکس کندل)."""
    __slots__ = ("idx", "side", "sl", "tp", "rr", "score", "entry")

    def __init__(self, idx, side, sl, tp, rr, score, entry):
        self.idx, self.side, self.sl, self.tp, self.rr, self.score = idx, side, sl, tp, rr, score
        self.entry = entry

    @staticmethod
    def empty():
        z = np.zeros(0)
        return Finalized(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8), z, z, z, z, z)


def finalize_strategy(structural_pair, close, variant):
    """حداقل امتیاز + مرحله‌ی نهایی روی دو شاخه‌ی خرید/فروش؛ خرید اولویت داره (مثل کد زنده)."""
    min_score = float(variant.get("min_score", 0.0)) - 1e-9
    parts = []
    for side, st in ((LONG, structural_pair[0]), (SHORT, structural_pair[1])):
        if len(st.idx) == 0:
            continue
        entry = close[st.idx] if st.entry is None else st.entry
        valid, sl, tp, rr = finalize_vec(side, entry, st.sl, st.level, st.atr, st.tp_uses_level,
                                         variant["min_rr"], variant["min_sl_pct"], variant["min_sl_atr"],
                                         variant["room"], variant.get("net"))
        if st.use_score:
            valid &= st.score >= min_score
        if valid.any():
            parts.append((st.idx[valid], np.full(int(valid.sum()), side, dtype=np.int8),
                          sl[valid], tp[valid], rr[valid], st.score[valid], entry[valid]))
    if not parts:
        return Finalized.empty()
    cols = [np.concatenate([p[i] for p in parts]) for i in range(7)]
    order = np.lexsort((-cols[1], cols[0]))
    idx, side, sl, tp, rr, score, ent = [c[order] for c in cols]
    keep = np.ones(len(idx), dtype=bool)
    keep[1:] = idx[1:] != idx[:-1]
    return Finalized(idx[keep], side[keep], sl[keep], tp[keep], rr[keep], score[keep], ent[keep])


def combine(finals, active, mode="any", lookback=8):
    """
    یک یا چند استراتژی فعال: سیگنال‌های همه با هم، و اگه چندتا روی یک کندل سیگنال دادن، اولی
    (به ترتیب اولویت در active) برنده‌ست — دقیقاً مثل strategies.generate_combined_signal.
    """
    names = [a for a in (active or ["weighted_confluence"]) if a in finals]
    if not names:
        return None
    if len(names) == 1:
        f = finals[names[0]]
        return {"idx": f.idx, "side": f.side, "sl": f.sl, "tp": f.tp, "rr": f.rr, "score": f.score,
                "entry": f.entry, "label": [names[0]] * len(f.idx)}
    cols = {k: [] for k in ("idx", "side", "sl", "tp", "rr", "score", "entry", "pri")}
    for pri, name in enumerate(names):
        f = finals[name]
        for k in ("idx", "side", "sl", "tp", "rr", "score", "entry"):
            cols[k].append(getattr(f, k))
        cols["pri"].append(np.full(len(f.idx), pri, dtype=np.int32))
    m = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((m["pri"], m["idx"]))
    m = {k: v[order] for k, v in m.items()}
    keep = np.ones(len(m["idx"]), dtype=bool)
    keep[1:] = m["idx"][1:] != m["idx"][:-1]
    m = {k: v[keep] for k, v in m.items()}
    return {"idx": m["idx"], "side": m["side"], "sl": m["sl"], "tp": m["tp"], "rr": m["rr"], "score": m["score"],
            "entry": m["entry"], "label": [names[int(p)] for p in m["pri"].tolist()]}
