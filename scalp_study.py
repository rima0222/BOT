# -*- coding: utf-8 -*-
"""
سنجش اسکلپ چند-تایم‌فریمی روی کندل‌های ۱ دقیقه‌ای واقعی: روند از ۱۵ دقیقه و ۱ ساعته، ورود روی ۱ دقیقه.

هر معامله: ضرر خالص دقیقاً ۱$ و سود خالص RR دلار (با همه‌ی کارمزدها، مثل move_study)، حد ضرر ثابت درصدی.
روی هر ارز در هر لحظه حداکثر یک پوزیشن (تا قبلی تموم نشه، ورود جدید نه) — پس «تعداد معامله در روز» واقعیه.

روند (فقط کندل‌های بسته‌شده‌ی تایم‌فریم بالاتر، بدون نگاه به آینده):
  صعودی = SMA7 > SMA25 > SMA99 هم روی ۱۵ دقیقه و هم روی ۱ ساعته؛ نزولی برعکس.

نوع‌های ورود (برای مقایسه):
  random      — هر ۳۰ دقیقه، هر دو جهت، بدون هیچ فیلتر (مبنا)
  trend       — هر ۳۰ دقیقه، فقط در جهت روند (اثر خالص روند)
  pullback    — روند + روی ۱ دقیقه قیمت زیر SMA20 رفته و حالا با کندل سبز بالای SMA7 بسته شده (ادامه‌ی روند)
  touch15     — روند + سایه‌ی کندل ۱ دقیقه از خط زرد ۱۵ دقیقه (SMA7) رد شده و بالاش بسته شده (ستاپ کاربر)
  breakout    — روند + بسته‌شدن ۱ دقیقه بالای سقف ۳۰ کندل قبل (شکست در جهت روند)
"""
import numpy as np

STOP_PCTS = [0.3, 0.5, 0.7]
RR_LIST = [2.0, 3.0]
VARIANTS = ["random", "trend", "pullback", "touch15", "breakout"]
MAX_HOURS = 12
SAMPLE_EVERY = 30      # برای random و trend
IS_FRACTION = 0.7
RISK_USD = 1.0


def fee_model(cfg):
    fe = float(cfg.MAKER_FEE_PCT) / 100.0
    f_sl = (float(cfg.TAKER_FEE_PCT) + float(cfg.TAKER_SLIPPAGE_PCT)) / 100.0
    f_tp = float(cfg.MAKER_FEE_PCT) / 100.0
    f_mk = (float(cfg.TAKER_FEE_PCT) + float(cfg.TAKER_SLIPPAGE_PCT)) / 100.0
    fund_h = float(getattr(cfg, "FUNDING_PCT_PER_8H", 0.0) or 0.0) / 100.0 / 8.0
    return fe, f_sl, f_tp, f_mk, fund_h


def levels(d, rr, fees):
    fe, f_sl, f_tp, _, _ = fees
    n = RISK_USD / (d + fe + f_sl)
    return n, rr * RISK_USD / n + fe + f_tp


def sma(x, k):
    out = np.full(len(x), np.nan)
    if len(x) >= k:
        cs = np.cumsum(np.r_[0.0, x])
        out[k - 1:] = (cs[k:] - cs[:-k]) / k
    return out


def resample(arr, minutes):
    """کندل‌های کامل تایم‌فریم بالاتر از ۱ دقیقه‌ای: [ts_open, o, h, l, c]."""
    ms = minutes * 60_000
    ts = arr[:, 0].astype(np.int64)
    g = ts // ms
    cut = np.flatnonzero(np.diff(g)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(ts)]
    full = (ends - starts) == minutes
    starts, ends = starts[full], ends[full]
    o = arr[starts, 1]
    h = np.array([arr[s:e, 2].max() for s, e in zip(starts, ends)])
    l = np.array([arr[s:e, 3].min() for s, e in zip(starts, ends)])
    c = arr[ends - 1, 4]
    return np.column_stack([(g[starts] * ms).astype(np.float64), o, h, l, c])


def htf_trend(arr1, minutes):
    """روند SMA7>25>99 روی تایم‌فریم بالاتر، هم‌تراز با بسته‌شدن هر کندل ۱ دقیقه (فقط کندل‌های بسته‌شده)."""
    r = resample(arr1, minutes)
    n1 = len(arr1)
    if len(r) < 110:
        return np.zeros(n1, dtype=np.int8), np.full(n1, np.nan)
    c = r[:, 4]
    s7, s25, s99 = sma(c, 7), sma(c, 25), sma(c, 99)
    tr = np.zeros(len(r), dtype=np.int8)
    tr[(s7 > s25) & (s25 > s99)] = 1
    tr[(s7 < s25) & (s25 < s99)] = -1
    close_ts = r[:, 0] + minutes * 60_000
    c1 = arr1[:, 0] + 60_000
    idx = np.searchsorted(close_ts, c1, "right") - 1
    out = np.zeros(n1, dtype=np.int8)
    s7a = np.full(n1, np.nan)
    ok = idx >= 0
    out[ok] = tr[idx[ok]]
    s7a[ok] = s7[idx[ok]]
    return out, s7a


def signals(arr1):
    """خروجی: {variant: (idx آرایه، side آرایه)} — ورود روی بسته‌شدن کندل idx."""
    o, h, l, c = arr1[:, 1], arr1[:, 2], arr1[:, 3], arr1[:, 4]
    n = len(c)
    t15, s7_15 = htf_trend(arr1, 15)
    t60, _ = htf_trend(arr1, 60)
    trend = np.where((t15 == t60), t15, 0).astype(np.int8)
    warm = 99 * 60 + 5
    base = np.arange(n) >= warm
    out = {}
    samp = np.zeros(n, dtype=bool)
    samp[warm::SAMPLE_EVERY] = True
    ri = np.flatnonzero(samp)
    out["random"] = (np.r_[ri, ri], np.r_[np.ones(len(ri), np.int8), -np.ones(len(ri), np.int8)])
    ti = np.flatnonzero(samp & (trend != 0))
    out["trend"] = (ti, trend[ti])
    m7, m20 = sma(c, 7), sma(c, 20)
    pc, pm20 = np.r_[np.nan, c[:-1]], np.r_[np.nan, m20[:-1]]
    with np.errstate(invalid="ignore"):
        pl = base & (trend == 1) & (pc < pm20) & (c > m7) & (c > o)
        ps = base & (trend == -1) & (pc > pm20) & (c < m7) & (c < o)
        tl = base & (trend == 1) & (l <= s7_15) & (np.minimum(o, c) >= s7_15)
        ts_ = base & (trend == -1) & (h >= s7_15) & (np.maximum(o, c) <= s7_15)
        hh = np.full(n, np.nan)
        ll = np.full(n, np.nan)
        if n > 31:
            from numpy.lib.stride_tricks import sliding_window_view
            hh[30:] = sliding_window_view(h[:-1], 30).max(axis=1)[:n - 30]
            ll[30:] = sliding_window_view(l[:-1], 30).min(axis=1)[:n - 30]
        bl = base & (trend == 1) & (c > hh)
        bs = base & (trend == -1) & (c < ll)
    for name, lm, sm in (("pullback", pl, ps), ("touch15", tl, ts_), ("breakout", bl, bs)):
        il, is_ = np.flatnonzero(lm), np.flatnonzero(sm)
        idx = np.r_[il, is_]
        side = np.r_[np.ones(len(il), np.int8), -np.ones(len(is_), np.int8)]
        order = np.argsort(idx, kind="stable")
        out[name] = (idx[order], side[order])
    return out


def outcomes(arr1, idx, side, d, rr, fees, max_hours=MAX_HOURS):
    """برای هر ورود: (pnl$, اندیس کندل خروج). اگه حد ضرر و سود در یک کندل خوردن، ضرر (بدبینانه)."""
    fe, f_sl, f_tp, f_mk, fund_h = fees
    N, u = levels(d, rr, fees)
    h, l, c = arr1[:, 2], arr1[:, 3], arr1[:, 4]
    n = len(c)
    W = int(max_hours * 60)
    pnl = np.zeros(len(idx))
    ex = np.zeros(len(idx), dtype=np.int64)
    for k, (t, sd) in enumerate(zip(idx.tolist(), side.tolist())):
        e = c[t]
        end = min(n, t + 1 + W)
        if t + 1 >= n:
            pnl[k], ex[k] = 0.0, t
            continue
        hi = h[t + 1:end] / e - 1.0
        lo = l[t + 1:end] / e - 1.0
        if sd == 1:
            slh, tph = lo <= -d, hi >= u
        else:
            slh, tph = hi >= d, lo <= -u
        i_sl = int(np.argmax(slh)) if slh.any() else len(hi)
        i_tp = int(np.argmax(tph)) if tph.any() else len(hi)
        if i_sl >= len(hi) and i_tp >= len(hi):
            j = len(hi)
            last = c[end - 1] / e - 1.0
            p = N * sd * last - N * (fe + f_mk)
        elif i_sl <= i_tp:
            j = i_sl + 1
            p = -RISK_USD
        else:
            j = i_tp + 1
            p = rr * RISK_USD
        pnl[k] = p - N * fund_h * j / 60.0
        ex[k] = t + j
    return pnl, ex


def sequential(idx, ex):
    """یک پوزیشن در هر لحظه روی هر ارز: ورودهایی که وسط پوزیشن قبلی‌ان حذف می‌شن."""
    keep = np.zeros(len(idx), dtype=bool)
    busy = -1
    for k, t in enumerate(idx.tolist()):
        if t > busy:
            keep[k] = True
            busy = int(ex[k])
    return keep


def run_symbol(arr1, fees):
    """خروجی: {(variant, d, rr): لیست (زمان ms، pnl، دقیقه‌ها)}"""
    sig = signals(arr1)
    out = {}
    ts = arr1[:, 0]
    for v in VARIANTS:
        idx, side = sig[v]
        for d in STOP_PCTS:
            for rr in RR_LIST:
                if not len(idx):
                    out[(v, d, rr)] = []
                    continue
                pnl, ex = outcomes(arr1, idx, side, d / 100.0, rr, fees)
                if v == "random":
                    keep = np.ones(len(idx), dtype=bool)     # مبنا: همه‌ی نمونه‌ها (هم‌پوشانی مهم نیست)
                else:
                    keep = sequential(idx, ex)
                out[(v, d, rr)] = list(zip(ts[idx[keep]].tolist(), pnl[keep].tolist(),
                                           (ex[keep] - idx[keep]).tolist()))
    return out


def summarize(per_symbol, days, split_ms):
    rows = []
    for v in VARIANTS:
        for d in STOP_PCTS:
            for rr in RR_LIST:
                allv = [x for s in per_symbol.values() for x in s.get((v, d, rr), [])]
                if not allv:
                    continue
                t = np.array([x[0] for x in allv])
                p = np.array([x[1] for x in allv])
                m = np.array([x[2] for x in allv], dtype=float)
                day = (t // 86_400_000).astype(np.int64)
                ud = np.unique(day)
                daily = np.array([p[day == x].sum() for x in ud])
                se = daily.std(ddof=1) / np.sqrt(len(daily)) if len(daily) > 2 else np.nan
                is_m, oos_m = t < split_ms, t >= split_ms
                tpd = len(p) / max(1.0, days)
                avg = float(p.mean())
                rows.append({
                    "variant": v, "stop_pct": d, "rr": rr, "trades": int(len(p)),
                    "trades_per_day": round(tpd, 1),
                    "win_pct": round(float((p > 0).mean() * 100), 1),
                    "breakeven_pct": round(100.0 / (1.0 + rr), 1),
                    "avg_usd": round(avg, 4),
                    "avg_is": round(float(p[is_m].mean()), 4) if is_m.any() else None,
                    "avg_oos": round(float(p[oos_m].mean()), 4) if oos_m.any() else None,
                    "daily_usd": round(float(daily.mean()), 2),
                    "daily_t": round(float(daily.mean() / se), 2) if se and se > 0 else None,
                    "median_minutes": round(float(np.median(m)), 0),
                    "risk_for_20": round(20.0 / (avg * tpd), 2) if avg > 0 and tpd > 0 else None,
                })
    return rows
