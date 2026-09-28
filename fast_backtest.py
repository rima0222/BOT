# -*- coding: utf-8 -*-
"""
هماهنگ‌کننده‌ی بک‌تست سریع: دیتا (از کش) → آماده‌سازی هر نماد → کاندیدها → سبد.

جریان کار:
  1) load_market_data: دیتای لازم (تایم‌فریم اصلی + تایم‌فریم‌های بالاتر + BTC) از کش
     محلی خونده می‌شه؛ فقط کم‌وکسرها از صرافی دانلود می‌شن (موازی).
  2) prepare_symbol: یک‌بار برای هر نماد، همه‌ی محاسبات سنگین (سوینگ، حمایت/مقاومت،
     ATR، روند تایم‌فریم‌های بالاتر هم‌تراز با هر کندل، روند BTC) انجام می‌شه.
  3) هر «تنظیم» (ترکیب استراتژی، سخت‌گیری، تریلینگ، فیلترها) فقط یک مرحله‌ی ارزون
     نهایی‌سازی + یک اجرای سبد لازم داره — برای همین می‌شه هزاران تنظیم رو در چند
     دقیقه مقایسه کرد.
"""
import time

import numpy as np

import market_data
import sim_engine
import signals_engine as se

DAY_MS = 86_400_000


def _resample(series_arr, src_tf, dst_tf):
    """ساخت کندل تایم‌فریم بالاتر از کندل‌های کوچک‌تر (فقط گروه‌های کامل و بسته‌شده)."""
    src_ms = market_data.TF_MS[src_tf]
    dst_ms = market_data.TF_MS[dst_tf]
    if dst_ms % src_ms != 0 or len(series_arr) == 0:
        return None
    arr = np.asarray(series_arr)
    ts = arr[:, 0].astype(np.int64)
    grp = ts // dst_ms
    starts = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    ends = np.r_[starts[1:], len(ts)]
    per = dst_ms // src_ms
    # گروه اول ممکنه ناقص شروع شده باشه (قبل از شروع دیتا)؛ دور ریخته می‌شه
    first_ok = 0 if ts[0] % dst_ms == 0 else 1
    last_close = ts[-1] + src_ms
    rows = []
    for gi in range(first_ok, len(starts)):
        s, e = starts[gi], ends[gi]
        g_open = grp[s] * dst_ms
        if g_open + dst_ms > last_close:
            break  # هنوز بسته نشده
        if e - s < max(1, per // 2):
            continue  # دیتای خیلی ناقص
        rows.append((g_open, arr[s, 1], arr[s:e, 2].max(), arr[s:e, 3].min(), arr[e - 1, 4], arr[s:e, 5].sum()))
    if not rows:
        return None
    return np.array(rows, dtype=np.float64)


def profile_cfg(base_cfg, timeframe):
    """یک نسخه از تنظیمات با پروفایل یک تایم‌فریم (۱ دقیقه تا ۴ ساعته) اعمال‌شده."""
    import backtest
    prof = base_cfg.TIMEFRAME_PROFILES[timeframe]
    over = {"TIMEFRAME": timeframe, "HTF_TIMEFRAMES": list(prof["HTF_TIMEFRAMES"]),
            "COOLDOWN_HOURS": prof["COOLDOWN_HOURS"], "MAX_HOLD_MINUTES": prof["MAX_HOLD_MINUTES"],
            "LIMIT_WAIT_BARS": prof["LIMIT_WAIT_BARS"],
            "BTC_REGIME_TIMEFRAME": getattr(base_cfg, "PROFILE_BTC_TIMEFRAME", {}).get(timeframe, "4h")}
    return backtest.build_config(base_cfg, over)


def plan_jobs(symbols, days, cfg, now_ms=None):
    now_ms = now_ms or int(time.time() * 1000)
    start_ms = now_ms - int(days) * DAY_MS
    main_tf = cfg.TIMEFRAME
    main_ms = market_data.TF_MS[main_tf]
    htf_limit = int(getattr(cfg, "HTF_CANDLE_LIMIT", 120))
    warm_main = (int(getattr(cfg, "CANDLE_LIMIT", 300)) + int(getattr(cfg, "CONFIRM_LOOKBACK_BARS", 8)) + 10) * main_ms

    resampled, fetched = [], []
    # دیتای تایم‌فریم‌های بالاتر وقتی لازمه که تایید HTF روشن باشه، یا مقایسه‌ی خودکار (که هر دو حالت رو تست می‌کنه)
    need_htf = getattr(cfg, "_NEED_HTF", None)
    if need_htf is None:
        need_htf = getattr(cfg, "USE_HTF_CONFIRMATION", True)
    htfs = list(getattr(cfg, "HTF_TIMEFRAMES", [])) if need_htf else []
    for tf in htfs:
        tf_ms = market_data.TF_MS.get(tf)
        # ۱ساعته و ۴ساعته رو از خود کندل‌های ۱۵ دقیقه‌ای می‌سازیم (دقیقاً همون سقف/کف‌ها،
        # ولی بدون ده‌ها درخواست اضافه)
        if tf_ms and tf_ms % main_ms == 0 and tf_ms <= 4 * 3_600_000:
            resampled.append(tf)
            warm_main = max(warm_main, (htf_limit + 5) * tf_ms)
        else:
            fetched.append(tf)

    jobs = []
    for sym in symbols:
        jobs.append((sym, main_tf, start_ms - warm_main))
        for tf in fetched:
            jobs.append((sym, tf, start_ms - (htf_limit + 5) * market_data.approx_tf_ms(tf)))
    btc_job = None
    if getattr(cfg, "BTC_REGIME_FILTER_DATA", True):
        btc_tf = getattr(cfg, "BTC_REGIME_TIMEFRAME", "4h")
        btc_job = (getattr(cfg, "BTC_REGIME_SYMBOL", "BTC/USDT"), btc_tf,
                   start_ms - (htf_limit + 5) * market_data.approx_tf_ms(btc_tf) - warm_main)
        jobs.append(btc_job)
    return {"start_ms": start_ms, "now_ms": now_ms, "jobs": jobs, "resampled": resampled,
            "fetched": fetched, "btc_job": btc_job, "main_since_ms": start_ms - warm_main}


def load_market_data(cache, symbols, days, cfg, progress_cb=None, offline=False):
    plan = plan_jobs(symbols, days, cfg)

    def cb(done, total, sym, tf, res):
        if progress_cb:
            status = "✓" if not isinstance(res, Exception) else "✗"
            progress_cb(f"دیتا {done}/{total}: {sym} {tf} {status}", done / max(1, total))

    if offline:
        data = {}
        for sym, tf, _ in plan["jobs"]:
            arr = cache.load(sym, tf)
            data[(sym, tf)] = arr if arr is not None else RuntimeError("در کش نیست")
    else:
        data = cache.ensure_many(plan["jobs"], cb)
    return plan, data


class SymbolPrep:
    def __init__(self, symbol, series, structural, htf_long, htf_short, btc_trend, first_idx, cfg):
        self.symbol = symbol
        self.series = series
        self.structural = structural
        self.htf_long = htf_long
        self.htf_short = htf_short
        self.btc_trend = btc_trend
        self.first_idx = first_idx
        self.paths = sim_engine.PathSim(series, getattr(cfg, "TRAILING_SL_LADDER", []),
                                        getattr(cfg, "TRAILING_SL_BEYOND_DISTANCE_R", 0.6))
        self._finals = {}

    def finals(self, name, variant):
        key = (name, variant["min_rr"], variant["min_sl_pct"], variant["min_sl_atr"], variant["room"],
               variant.get("min_score", 0.0))
        f = self._finals.get(key)
        if f is None:
            f = se.finalize_strategy(self.structural[name], self.series.c, variant)
            self._finals[key] = f
        return f


def prepare_symbol(symbol, main_arr, htf_arrays, btc_arr, start_ms, cfg, strategies_needed=None):
    series = se.Series(main_arr, cfg.TIMEFRAME)
    first_idx = int(np.searchsorted(series.close_ts, start_ms, "left"))
    if strategies_needed:
        cfg._ENGINE_STRATEGIES = list(strategies_needed)
    structural = se.compute_structural(series, cfg, first_idx)
    series.v = None   # حجم دیگه لازم نیست؛ آزادسازی رم

    order = cfg.SWING_ORDER
    htf_limit = int(getattr(cfg, "HTF_CANDLE_LIMIT", 120))
    htf_long = np.zeros(series.n, dtype=np.int8)
    htf_short = np.zeros(series.n, dtype=np.int8)
    for tf, arr in htf_arrays.items():
        if arr is None or len(arr) == 0:
            continue
        hs = se.Series(arr, tf)
        tr = se.htf_trend_series(hs, order, htf_limit)
        al = se.align_to(series.close_ts, hs, tr)
        htf_long += (al == 1)
        htf_short += (al == -1)

    btc_trend = np.zeros(series.n, dtype=np.int8)
    if btc_arr is not None and len(btc_arr):
        bs = se.Series(btc_arr, getattr(cfg, "BTC_REGIME_TIMEFRAME", "4h"))
        tr = se.htf_trend_series(bs, order, htf_limit)
        al = se.align_to(series.close_ts, bs, tr)
        btc_trend = np.where(al == 2, 0, al).astype(np.int8)
    return SymbolPrep(symbol, series, structural, htf_long, htf_short, btc_trend, first_idx, cfg)


def prepare_all(plan, data, symbols, cfg, progress_cb=None, strategies_needed=None):
    preps, meta = {}, {}
    btc_arr = None
    if plan.get("btc_job"):
        b = data.get((plan["btc_job"][0], plan["btc_job"][1]))
        btc_arr = None if isinstance(b, Exception) else b
    for i, sym in enumerate(symbols):
        main = data.get((sym, cfg.TIMEFRAME))
        if main is None or isinstance(main, Exception):
            meta[sym] = {"ok": False, "error": str(main) if main is not None else "دیتا نیست"}
            continue
        main = np.asarray(main)
        # فقط بازه‌ی لازم (شروع بک‌تست منهای گرم‌کردن) — کش ممکنه دیتای خیلی قدیمی‌تر هم داشته باشه
        main = np.array(main[(main[:, 0] >= plan["main_since_ms"]) & (main[:, 0] < plan["now_ms"])])
        if len(main) and (main[:, 1:5] <= 0).any():
            meta[sym] = {"ok": False, "error": "قیمت صفر/منفی در دیتا"}
            continue
        if len(main) < cfg.SWING_ORDER * 2 + 25:
            meta[sym] = {"ok": False, "error": "دیتای کافی نیست"}
            continue
        # اعتبار دیتا در کل بازه‌ی خواسته‌شده (مثلاً واقعاً ۲ سال کامل؟)
        q = market_data.data_quality(main, cfg.TIMEFRAME, plan["start_ms"], plan["now_ms"])
        min_cov = float(getattr(cfg, "MIN_DATA_COVERAGE_PCT", 0) or 0)
        if min_cov and q["coverage_pct"] < min_cov:
            why = ("این ارز بعد از شروع بازه لیست شده" if q["listed_after_start"]
                   else f"وقفه‌ی دیتا ({q['gaps']} وقفه، بزرگ‌ترین {q['max_gap_hours']} ساعت)")
            meta[sym] = {"ok": False, "quality": q,
                         "error": f"پوشش دیتا {q['coverage_pct']}٪ (کمتر از {min_cov:g}٪) — {why}"}
            continue
        htf_arrays = {}
        for tf in plan["resampled"]:
            htf_arrays[tf] = _resample(main, cfg.TIMEFRAME, tf)
        for tf in plan["fetched"]:
            a = data.get((sym, tf))
            htf_arrays[tf] = None if (a is None or isinstance(a, Exception)) else np.asarray(a)
        try:
            prep = prepare_symbol(sym, main, htf_arrays, btc_arr, plan["start_ms"], cfg, strategies_needed)
        except Exception as e:  # نماد خراب نباید کل بک‌تست رو متوقف کنه
            meta[sym] = {"ok": False, "error": f"خطای آماده‌سازی: {e}"}
            continue
        if prep.first_idx >= prep.series.n - 1:
            meta[sym] = {"ok": False, "error": "در بازه‌ی انتخابی دیتا نداره"}
            continue
        preps[sym] = prep
        s = prep.series
        meta[sym] = {"ok": True, "candles": int(s.n - prep.first_idx),
                     "from": str(np.datetime64(int(s.ts[prep.first_idx]), "ms")),
                     "to": str(np.datetime64(int(s.ts[-1]), "ms")), "quality": q}
        if progress_cb:
            progress_cb(f"آماده‌سازی {sym} ({i + 1}/{len(symbols)})", (i + 1) / max(1, len(symbols)))
    return preps, meta


def variant_from_cfg(cfg):
    return {"min_rr": float(cfg.MIN_RISK_REWARD), "min_sl_pct": float(getattr(cfg, "MIN_SL_PCT", 0.0)),
            "min_sl_atr": float(getattr(cfg, "MIN_SL_ATR_MULT", 0.0)),
            "room": bool(getattr(cfg, "REQUIRE_ROOM_TO_TARGET", False)),
            "min_score": float(getattr(cfg, "WC_MIN_SCORE_PCT", 0.0))}


def candidates_for(preps, active, mode, variant, lookback):
    out = {}
    for sym, p in preps.items():
        finals = {name: p.finals(name, variant) for name in active if name in p.structural}
        c = se.combine(finals, active, mode, lookback)
        if c is not None:
            out[sym] = c
    return out


def run_single(preps, symbols, cfg, record=True):
    """اجرای یک تنظیم کامل (برای بک‌تست تکی پنل) با ثبت همه‌ی سیگنال‌ها."""
    active = ["weighted_confluence"]
    variant = variant_from_cfg(cfg)
    cands = candidates_for(preps, active, cfg.STRATEGY_COMBINE_MODE, variant,
                           int(getattr(cfg, "CONFIRM_LOOKBACK_BARS", 8)))
    order = [s for s in symbols if s in preps]
    merged = sim_engine.merge_candidates(cands, preps, order)
    P = sim_engine.SimParams(cfg, cfg.HTF_MIN_AGREEMENT, getattr(cfg, "USE_TRAILING_SL", False),
                             allow_long=getattr(cfg, "ALLOW_LONG", True), allow_short=getattr(cfg, "ALLOW_SHORT", True),
                             btc_filter=getattr(cfg, "BTC_REGIME_FILTER", False))
    return sim_engine.run_portfolio(merged, preps, order, P, record=record), P
