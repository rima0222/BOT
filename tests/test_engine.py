# -*- coding: utf-8 -*-
"""
تست‌های دقت موتور بک‌تست سریع (بدون نیاز به اینترنت، با دیتای مصنوعی):
  1) سیگنال هر استراتژی در موتور وکتوریزه == خروجی تابع زنده روی همون پنجره
  2) ترکیب استراتژی‌ها (any / all / confirm) == strategies.generate_combined_signal
  3) روند تایم‌فریم بالاتر == analysis.trend_from_df
  4) مسیر معامله (SL/TP/تریلینگ با سایه) == حلقه‌ی کندل‌به‌کندل paper_trader.step_bar
  5) عدم نگاه به آینده: کوتاه کردن دیتا، معاملات قبل از نقطه‌ی برش رو عوض نمی‌کنه
اجرا:  python3 tests/test_engine.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import analysis  # noqa: E402
import backtest  # noqa: E402
import config  # noqa: E402
import fast_backtest  # noqa: E402
import paper_trader  # noqa: E402
import signals_engine as se  # noqa: E402
import sim_engine  # noqa: E402
import strategies  # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  ✗", msg)


def synth(n, seed=1, start_ms=1_600_000_000_000, tf_ms=900_000, price=100.0):
    """قیمت مصنوعی با روندهای متناوب، نوسان متغیر و جهش‌های حجم."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.0012, n // 200 + 1), 200)[:n]
    vol = np.repeat(rng.uniform(0.002, 0.008, n // 300 + 1), 300)[:n]
    rets = drift + rng.normal(0, 1, n) * vol
    close = price * np.exp(np.cumsum(rets))
    open_ = np.r_[price, close[:-1]] * (1 + rng.normal(0, 0.0005, n))
    wick = np.abs(rng.normal(0, 1, (n, 2))) * vol[:, None] * close[:, None] * 0.6
    high = np.maximum(open_, close) + wick[:, 0]
    low = np.minimum(open_, close) - wick[:, 1]
    # گرد کردن به تیک قیمت (تا سقف/کف‌های برابر هم پیش بیاد، مثل دیتای واقعی)
    tick = price * 1e-4
    open_, high, low, close = [np.round(x / tick) * tick for x in (open_, high, low, close)]
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = rng.lognormal(10, 0.4, n) * (1 + (rng.random(n) < 0.03) * rng.uniform(2, 6, n))
    ts = start_ms + np.arange(n, dtype=np.int64) * tf_ms
    return np.column_stack([ts, open_, high, low, close, volume]).astype(np.float64)


def make_cfg(**over):
    return backtest.build_config(config, over)


def close_enough(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


VARIANTS = [
    dict(MIN_RISK_REWARD=2.5),
    dict(MIN_RISK_REWARD=2.0, MIN_SL_PCT=0.6, MIN_SL_ATR_MULT=1.0),
    dict(MIN_RISK_REWARD=3.0, REQUIRE_ROOM_TO_TARGET=True),
    dict(MIN_RISK_REWARD=2.5, MIN_SL_PCT=0.8, REQUIRE_ROOM_TO_TARGET=True, PROXIMITY_PCT=0.8),
]


def test_strategy_equivalence():
    print("۱) معادل‌بودن سیگنال تک‌تک استراتژی‌ها با کد زنده")
    arr = synth(2600, seed=7)
    series = se.Series(arr, "15m")
    rng = np.random.default_rng(3)
    total_sig = 0
    for vi, over in enumerate(VARIANTS):
        cfg = make_cfg(**over)
        W = cfg.CANDLE_LIMIT
        structural = se.compute_structural(series, cfg, 0)
        variant = fast_backtest.variant_from_cfg(cfg)
        finals = {name: se.finalize_strategy(structural[name], series.c, variant) for name in se.STRATEGY_NAMES}
        # همه‌ی کندل‌هایی که موتور سیگنال داده + نمونه‌ی تصادفی از بقیه
        test_ts = set()
        for f in finals.values():
            test_ts.update(f.idx[f.idx >= W - 1].tolist())
        test_ts.update(rng.integers(W - 1, series.n, 250).tolist())
        mism = 0
        for t in sorted(test_ts):
            window = series.to_df(t - W + 1, t + 1)
            for name in se.STRATEGY_NAMES:
                live = strategies.STRATEGY_REGISTRY[name]["fn"](window, cfg).get("signal")
                f = finals[name]
                k = np.searchsorted(f.idx, t)
                eng = None
                if k < len(f.idx) and f.idx[k] == t:
                    eng = {"side": "LONG" if f.side[k] == 1 else "SHORT", "sl": f.sl[k], "tp": f.tp[k], "rr": f.rr[k]}
                if (live is None) != (eng is None):
                    mism += 1
                    if mism <= 5:
                        check(False, f"[v{vi}] {name} t={t}: زنده={live} موتور={eng}")
                    continue
                if live is not None:
                    total_sig += 1
                    same = (live["side"] == eng["side"] and close_enough(live["sl"], eng["sl"])
                            and close_enough(live["tp"], eng["tp"]) and close_enough(live["rr"], eng["rr"]))
                    if not same:
                        mism += 1
                        if mism <= 5:
                            check(False, f"[v{vi}] {name} t={t}: مقادیر فرق دارن زنده={live} موتور={eng}")
        check(mism == 0, f"[v{vi}] {mism} مورد عدم تطابق")
    print(f"  ✓ {total_sig} سیگنال بررسی شد")
    check(total_sig > 50, "تعداد سیگنال برای تست معتبر خیلی کمه")


def test_combine_equivalence():
    print("۲) معادل‌بودن ترکیب استراتژی‌ها (any / all / confirm)")
    arr = synth(2200, seed=11)
    series = se.Series(arr, "15m")
    combos = [
        (["dow_support_resistance", "volume_spike"], "any"),
        (["candle_setup", "breakout", "dow_support_resistance"], "any"),
        (["dow_support_resistance", "candle_setup"], "all"),
        (["volume_spike", "breakout"], "all"),
        (["dow_support_resistance", "volume_spike"], "confirm"),
        (["candle_setup", "volume_spike", "dow_support_resistance"], "confirm"),
    ]
    rng = np.random.default_rng(5)
    checked = 0
    for active, mode in combos:
        cfg = make_cfg(ACTIVE_STRATEGIES=active, STRATEGY_COMBINE_MODE=mode, CONFIRM_LOOKBACK_BARS=6,
                       MIN_SL_PCT=0.3)
        W, K = cfg.CANDLE_LIMIT, cfg.CONFIRM_LOOKBACK_BARS
        structural = se.compute_structural(series, cfg, 0)
        variant = fast_backtest.variant_from_cfg(cfg)
        finals = {n: se.finalize_strategy(structural[n], series.c, variant) for n in active}
        comb = se.combine(finals, active, mode, K)
        eng = {int(t): (int(s), sl, tp, rr, lb) for t, s, sl, tp, rr, lb in
               zip(comb["idx"], comb["side"], comb["sl"], comb["tp"], comb["rr"], comb["label"])}
        ts = set(t for t in eng if t >= W + K)
        for f in finals.values():
            ts.update(int(x) for x in f.idx if x >= W + K)
        ts.update(rng.integers(W + K, series.n, 120).tolist())
        mism = 0
        for t in sorted(ts):
            df = series.to_df(t - W - K + 1, t + 1)
            live = strategies.generate_combined_signal(df, cfg).get("signal")
            e = eng.get(t)
            if (live is None) != (e is None):
                mism += 1
                if mism <= 3:
                    check(False, f"{mode} {active} t={t}: زنده={live} موتور={e}")
                continue
            if live:
                checked += 1
                same = (live["side"] == ("LONG" if e[0] == 1 else "SHORT") and close_enough(live["sl"], e[1])
                        and close_enough(live["tp"], e[2]) and live["strategy"] == e[4])
                if not same:
                    mism += 1
                    if mism <= 3:
                        check(False, f"{mode} {active} t={t}: مقادیر فرق دارن {live} / {e}")
        check(mism == 0, f"{mode} {active}: {mism} عدم تطابق")
    print(f"  ✓ {checked} سیگنال ترکیبی بررسی شد")


def test_htf_equivalence():
    print("۳) معادل‌بودن روند تایم‌فریم بالاتر")
    arr = synth(900, seed=21, tf_ms=14_400_000)
    hs = se.Series(arr, "4h")
    tr = se.htf_trend_series(hs, config.SWING_ORDER, config.HTF_CANDLE_LIMIT)
    mism = 0
    for j in range(5, hs.n):
        s = max(0, j - config.HTF_CANDLE_LIMIT + 1)
        df = hs.to_df(s, j + 1)
        if len(df) < config.SWING_ORDER * 2 + 5:
            live = 2
        else:
            t = analysis.trend_from_df(df, swing_order=config.SWING_ORDER)
            live = {"uptrend": 1, "downtrend": -1}.get(t, 0)
        if live != tr[j]:
            mism += 1
    check(mism == 0, f"روند HTF: {mism} عدم تطابق")
    # هم‌ترازی زمانی: کندل ۴ساعته فقط بعد از بسته شدنش دیده می‌شه
    main = se.Series(synth(4000, seed=2), "15m")
    al_idx = np.searchsorted(hs.close_ts, main.close_ts, "right") - 1
    ok = al_idx >= 0
    check(bool((hs.close_ts[al_idx[ok]] <= main.close_ts[ok]).all()), "HTF از آینده استفاده کرده!")
    print("  ✓ انجام شد")


def test_path_equivalence():
    print("۴) معادل‌بودن مسیر معامله با مدیریت کندل‌به‌کندل زنده")
    arr = synth(6000, seed=33)
    s = se.Series(arr, "15m")
    ladder = config.TRAILING_SL_LADDER
    beyond = config.TRAILING_SL_BEYOND_DISTANCE_R
    sim = sim_engine.PathSim(s, ladder, beyond)
    rng = np.random.default_rng(9)
    mism = 0
    n_checked = 0
    for _ in range(1500):
        t0 = int(rng.integers(10, s.n - 50))
        side = 1 if rng.random() < 0.5 else -1
        entry = s.c[t0]
        dist = entry * rng.uniform(0.002, 0.03)
        sl = entry - dist if side == 1 else entry + dist
        tp = entry + dist * 2.5 if side == 1 else entry - dist * 2.5
        for trailing, hold in ((False, 0), (True, 0), (False, 7), (True, 12)):
            r = sim._run(t0, side, entry, sl, tp, trailing, hold)
            # مرجع: حلقه‌ی step_bar
            side_s = "LONG" if side == 1 else "SHORT"
            cur_sl, peak = sl, entry
            ref = None
            last_k = min(s.n, t0 + 1 + hold) if hold else s.n
            for k in range(t0 + 1, last_k):
                hit, price, kind, level, cur_sl, peak = paper_trader.step_bar(
                    side_s, entry, sl, cur_sl, tp, peak, trailing, s.o[k], s.h[k], s.l[k], s.c[k], ladder, beyond)
                if hit:
                    ref = (k, price, kind, level)
                    break
            if ref is None:
                ref = (last_k - 1, s.c[last_k - 1], "TIME" if last_k < s.n else "END", None)
            n_checked += 1
            same = (r[0] == ref[0] and close_enough(r[1], ref[1], 1e-12) and r[2] == ref[2]
                    and ((r[3] is None and ref[3] is None) or (r[3] is not None and ref[3] is not None
                                                               and close_enough(r[3], ref[3], 1e-12))))
            if not same:
                mism += 1
                if mism <= 5:
                    check(False, f"مسیر t0={t0} side={side} trailing={trailing}: موتور={r[:4]} مرجع={ref}")
    check(mism == 0, f"مسیر معامله: {mism} عدم تطابق از {n_checked}")
    print(f"  ✓ {n_checked} مسیر بررسی شد")


def _build_preps(arrs, cfg, start_ms, cut_ms=None):
    preps = {}
    for sym, a in arrs.items():
        main = a if cut_ms is None else a[a[:, 0] + 900_000 <= cut_ms]
        htf = {"1h": fast_backtest._resample(main, "15m", "1h"), "4h": fast_backtest._resample(main, "15m", "4h")}
        preps[sym] = fast_backtest.prepare_symbol(sym, main, htf, None, start_ms, cfg)
    return preps


def test_no_lookahead_and_portfolio():
    print("۵) عدم نگاه به آینده + ترتیب زمانی سبد")
    arrs = {f"S{i}/USDT": synth(5000, seed=100 + i, price=10 + i * 7) for i in range(4)}
    cfg = make_cfg(ACTIVE_STRATEGIES=["dow_support_resistance", "volume_spike", "candle_setup"],
                   STRATEGY_COMBINE_MODE="any", HTF_TIMEFRAMES=["1h", "4h"], HTF_MIN_AGREEMENT=1,
                   SHORT_EXTRA_HTF_AGREEMENT=0, USE_TRAILING_SL=True, MAX_OPEN_POSITIONS=3)
    check(sim_engine.htf_required(4, 5, 1) == (4, 5) and sim_engine.htf_required(4, 3, 1) == (3, 3),
          "تبدیل سخت‌گیری HTF به تعداد تایم‌فریم اشتباهه")
    start_ms = int(arrs["S0/USDT"][400, 0])
    full = _build_preps(arrs, cfg, start_ms)
    res_full, _ = fast_backtest.run_single(full, list(arrs), cfg)
    cut_ms = int(arrs["S0/USDT"][3200, 0])
    part = _build_preps(arrs, cfg, start_ms, cut_ms)
    res_part, _ = fast_backtest.run_single(part, list(arrs), cfg)

    def key(t):
        return (t["symbol"], t["open_time"], t["side"], round(t["entry"], 10), round(t["pnl"], 8), t["close_time"])

    # معاملاتی که سیگنالشون نزدیک نقطه‌ی برش بوده (سفارش لیمیت هنوز منتظر پر شدن) کنار گذاشته می‌شن
    margin_ms = 900_000 * (cfg.LIMIT_WAIT_BARS + 1)
    closed_before = [key(t) for t in res_full["trades"] if t["close_time"] < cut_ms - margin_ms]
    part_before = [key(t) for t in res_part["trades"] if t["close_time"] < cut_ms - margin_ms
                   and t["exit_type"] != "END"]
    check(len(res_full["trades"]) > 20, f"تعداد معامله‌ی تست خیلی کمه ({len(res_full['trades'])})")
    check(closed_before == part_before,
          f"نگاه به آینده! {len(closed_before)} در برابر {len(part_before)} معامله قبل از برش")
    # هیچ‌وقت بیش از سقف پوزیشن باز هم‌زمان نداریم، و هر نماد حداکثر یک پوزیشن
    events = []
    for t in res_full["trades"]:
        events.append((t["open_time"], 1, t["symbol"]))
        events.append((t["close_time"], -1, t["symbol"]))
    events.sort(key=lambda e: (e[0], e[1]))
    open_now, per_sym, max_open = 0, {}, 0
    ok_sym = True
    for tm, d, sym in events:
        open_now += d
        per_sym[sym] = per_sym.get(sym, 0) + d
        if per_sym[sym] > 1:
            ok_sym = False
        max_open = max(max_open, open_now)
    check(max_open <= cfg.MAX_OPEN_POSITIONS, f"سقف پوزیشن رعایت نشده ({max_open})")
    check(ok_sym, "دو پوزیشن هم‌زمان روی یک نماد!")
    print(f"  ✓ {len(res_full['trades'])} معامله، حداکثر {max_open} پوزیشن هم‌زمان")
    # حالت ورود بازار و حد زمانی هم اجرا بشن و معامله‌ی معتبر بدن
    cfg2 = make_cfg(ACTIVE_STRATEGIES=["ema_pullback", "bb_reversion", "donchian_trend"], STRATEGY_COMBINE_MODE="any",
                    HTF_TIMEFRAMES=["1h", "4h"], HTF_MIN_AGREEMENT=1, ENTRY_MODE="market", MAX_HOLD_MINUTES=180)
    res_m, _ = fast_backtest.run_single(_build_preps(arrs, cfg2, start_ms), list(arrs), cfg2)
    types = {t["exit_type"] for t in res_m["trades"]}
    check(len(res_m["trades"]) > 20 and "TIME" in types, f"ورود بازار/حد زمانی کار نکرد: {len(res_m['trades'])} {types}")
    check(all(t["bars"] <= 12 + 1 for t in res_m["trades"] if t["exit_type"] != "END"), "حد زمانی رعایت نشده")
    print(f"  ✓ ورود بازار + حد زمانی: {len(res_m['trades'])} معامله، انواع خروج: {sorted(types)}")


def test_speed():
    print("۶) سرعت (۲ سال کندل ۱۵ دقیقه‌ای برای یک نماد)")
    arr = synth(70_080, seed=77)
    cfg = make_cfg()
    t0 = time.time()
    htf = {"1h": fast_backtest._resample(arr, "15m", "1h"), "4h": fast_backtest._resample(arr, "15m", "4h")}
    prep = fast_backtest.prepare_symbol("X/USDT", arr, htf, None, int(arr[300, 0]), cfg)
    t1 = time.time()
    print(f"  آماده‌سازی: {t1 - t0:.2f} ثانیه")
    check(t1 - t0 < 30, "آماده‌سازی خیلی کنده")
    return prep


if __name__ == "__main__":
    t_start = time.time()
    test_strategy_equivalence()
    test_combine_equivalence()
    test_htf_equivalence()
    test_path_equivalence()
    test_no_lookahead_and_portfolio()
    test_speed()
    print()
    if FAILS:
        print(f"❌ {len(FAILS)} خطا")
        sys.exit(1)
    print(f"✅ همه‌ی تست‌ها موفق ({time.time() - t_start:.1f} ثانیه)")
