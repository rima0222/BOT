# -*- coding: utf-8 -*-
"""
تست‌های «سنجش کیفیت ورود» (بدون اینترنت):
  1) نتیجه‌ی مسیر (براکت‌ها، MFE/MAE، بازده آینده) == حلقه‌ی ساده‌ی کندل‌به‌کندل
  2) پر شدن سفارش پولبک + «کندل پر شدن فقط حد ضرر»
  3) عدم نگاه به آینده: ویژگی‌ها و سیگنال‌ها با کوتاه کردن دیتا عوض نمی‌شن
  4) بازار کاملاً تصادفی: پایه ≈ ۵۰٪ و تقریباً هیچ سیگنالی «مزیت پایدار» نمی‌گیره
  5) بازار با روندهای ماندگار (مزیت کاشته‌شده): سیگنال‌های روندی مزیت پایدار می‌گیرن
  6) خطای استاندارد خوشه‌ای == فرمول مرجع
  7) ذخیره‌ی مرحله‌ای روی دیسک + خوندن دوباره == اجرای یک‌جا (پایه‌ی «ادامه از جای قطع‌شده»)
  8) الگوهای کندلی/کلاسیک: شکل‌های ساخته‌شده‌ی دستی درست شناخته می‌شن + بدون نگاه به آینده
  9) بک‌تست ساده‌ی حلقه‌ای مستقل (باز کردن پوزیشن روی هر الگو، حد ضرر خود الگو، RR2) == عدد سنجش؛
     حد ضرر همه‌ی الگوها سمت درست قیمت ورود
اجرا:  python3 tests/test_entry_study.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import entry_study as es  # noqa: E402
import fast_backtest  # noqa: E402
from test_engine import synth, make_cfg  # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  ✗", msg)


def random_walk(n, seed, start_ms=1_600_000_000_000, tf_ms=900_000, price=50.0, vol=0.004):
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, vol, n)))
    open_ = np.r_[price, close[:-1]]
    wick = np.abs(rng.normal(0, vol * 0.6, (n, 2))) * close[:, None]
    high = np.maximum(open_, close) + wick[:, 0]
    low = np.minimum(open_, close) - wick[:, 1]
    volume = rng.lognormal(10, 0.4, n)
    ts = start_ms + np.arange(n, dtype=np.int64) * tf_ms
    return np.column_stack([ts, open_, high, low, close, volume]).astype(np.float64)


def brute(h, l, c, p0, e, sd, r, H, tp, sl, stop_only):
    up_best = 0.0
    dn_best = 0.0
    res = None
    for k in range(H):
        j = p0 + k
        up = ((h[j] - e) if sd > 0 else (e - l[j])) / r
        dn = ((e - l[j]) if sd > 0 else (h[j] - e)) / r
        if k == 0 and stop_only:
            up = -np.inf
        dn_best = max(dn_best, dn)
        up_best = max(up_best, up)
        if res is None:
            if dn >= sl - 1e-12:
                res = (-1, -sl)
            elif up >= tp - 1e-12:
                res = (1, tp)
    if res is None:
        res = (0, (c[p0 + H - 1] - e) * (1 if sd > 0 else -1) / r)
    return res, up_best, dn_best


def test_path_brute():
    print("۱) نتیجه‌ی مسیر == حلقه‌ی ساده")
    a = synth(3000, seed=11)
    h, l, c = a[:, 2], a[:, 3], a[:, 4]
    rng = np.random.default_rng(3)
    m = 400
    H = 40
    p0 = rng.integers(1, 3000 - H - 1, m)
    side = np.where(rng.random(m) < 0.5, 1, -1).astype(np.int8)
    entry = c[p0 - 1] * (1 + rng.normal(0, 0.001, m))
    rdist = c[p0 - 1] * rng.uniform(0.002, 0.02, m)
    stop_only = rng.random(m) < 0.3
    out, res, mfe, mae = es.path_outcomes(h, l, c, p0, entry, side, rdist, H, es.BRACKETS, stop_only)
    bad = 0
    for i in range(m):
        for k, (tp, sl) in enumerate(es.BRACKETS):
            (o, rv), ub, db = brute(h, l, c, int(p0[i]), entry[i], side[i], rdist[i], H, tp, sl, stop_only[i])
            if o != out[i, k] or abs(rv - res[i, k]) > 1e-9:
                bad += 1
        if abs(max(ub, 0) - mfe[i]) > 1e-9 or abs(max(db, 0) - mae[i]) > 1e-9:
            bad += 1
    fwd = es.forward_returns(c, p0, entry, side, rdist, [1, 5, H])
    ref = np.column_stack([(c[p0 + k - 1] - entry) * side / rdist for k in (1, 5, H)])
    check(bad == 0 and np.allclose(fwd, ref), f"مسیر: {bad} عدم تطابق")
    print(f"  ✓ {m} ورود × {len(es.BRACKETS)} براکت (با کندل پر شدن فقط-حد‌ضرر) یکسان")


def test_retest_fill_and_lookahead():
    print("۲+۳) پر شدن پولبک + عدم نگاه به آینده")
    cfg = make_cfg(HTF_TIMEFRAMES=["1h", "4h"], BRK_MAIN_TREND="off", BRK_BOX_MAX_ATR=6.0, BRK_BOX_MIN_TOUCHES=1,
                   ENTRY_STUDY_HORIZON={"15m": 48})
    a = synth(6000, seed=21)
    start_ms = int(a[400, 0])
    htf = {"1h": fast_backtest._resample(a, "15m", "1h"), "4h": fast_backtest._resample(a, "15m", "4h")}
    prep = fast_backtest.prepare_symbol("X/USDT", a, htf, None, start_ms, cfg, keep_volume=True)
    # پر شدن پولبک: مقایسه با حلقه‌ی ساده (همون قانون موتور: قیمت باید «از» لیمیت رد بشه)
    v = fast_backtest.variant_from_cfg(cfg)
    f = prep.finals("box_breakout@retest", v)
    s = prep.series
    got_any, got_j = es.retest_fill(s.h, s.l, f.idx.astype(np.int64), f.side, f.entry, cfg.BRK_RETEST_WAIT_BARS)
    bad = 0
    for q, (i, sd, en) in enumerate(zip(f.idx.tolist(), f.side.tolist(), f.entry.tolist())):
        j = None
        for k in range(1, cfg.BRK_RETEST_WAIT_BARS + 1):
            if i + k >= s.n:
                break
            if (s.l[i + k] < en) if sd > 0 else (s.h[i + k] > en):
                j = i + k
                break
        if (j is not None) != bool(got_any[q]) or (j is not None and j != int(got_j[q])):
            bad += 1
    st = es.Study(cfg, "15m", int(a[4000, 0]), ["X/USDT"])
    st.add_symbol(prep, v, None, None)
    raw_n = sum(st.counts.get(("box_breakout@retest", sd), {}).get("signals", 0) for sd in (1, -1))
    got = sum(st.counts.get(("box_breakout@retest", sd), {}).get("filled", 0) for sd in (1, -1))
    check(bad == 0 and len(f.idx) > 20 and 0 < got <= raw_n, f"پولبک: {bad} عدم تطابق، {got}/{raw_n} پر شده")
    print(f"  ✓ پر شدن پولبک == حلقه‌ی ساده ({len(f.idx)} سیگنال، {int(got_any.sum())} پر شده)")

    # عدم نگاه به آینده: ویژگی‌ها و سیگنال‌های ساده روی دیتای کوتاه‌شده == دیتای کامل (تا نقطه‌ی برش)
    F = es.compute_features(prep, cfg)
    R = es.raw_signals(prep, F)
    bad = 0
    for cut in (4500, 5200):
        a2 = a[:cut]
        htf2 = {"1h": fast_backtest._resample(a2, "15m", "1h"), "4h": fast_backtest._resample(a2, "15m", "4h")}
        p2 = fast_backtest.prepare_symbol("X/USDT", a2, htf2, None, start_ms, cfg, keep_volume=True)
        F2 = es.compute_features(p2, cfg)
        R2 = es.raw_signals(p2, F2)
        for k in ("trend", "rsi", "vol_ratio", "bb_ratio", "mom20", "ma_dist", "body", "htf_l", "htf_s", "atr"):
            x, y = F[k][:cut], F2[k]
            same = np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True)
            bad += int((~same).sum())
        for name in R2:
            for sd in (0, 1):
                bad += int((R[name][sd][:cut] != R2[name][sd]).sum())
    check(bad == 0, f"نگاه به آینده: {bad} اختلاف")
    print(f"  ✓ ویژگی‌ها و {len(R)} سیگنال ساده با کوتاه کردن دیتا عوض نمی‌شن")


def _study(arrs, cfg, tf="15m", signal_set="core"):
    syms = list(arrs)
    start_ms = int(arrs[syms[0]][400, 0])
    end_ms = int(arrs[syms[0]][-1, 0])
    split = int(start_ms + (end_ms - start_ms) * 0.7)
    st = es.Study(cfg, tf, split, syms, signal_set=signal_set)
    v = fast_backtest.variant_from_cfg(cfg)
    btc_prep = None
    for sym in syms:
        a = arrs[sym]
        htf = {"1h": fast_backtest._resample(a, "15m", "1h"), "4h": fast_backtest._resample(a, "15m", "4h")}
        p = fast_backtest.prepare_symbol(sym, a, htf, None, start_ms, cfg, keep_volume=True)
        if btc_prep is None:
            btc_prep = es.btc_context(p)
            st.add_symbol(p, v, None, None, is_btc=True, base_samples=3000)
        else:
            st.add_symbol(p, v, btc_prep, None, base_samples=3000)
    st.finish()
    return st, es.analyze(st), es.analyze_features(st)


def test_random_walk_no_edge():
    print("۴) بازار کاملاً تصادفی: هیچ مزیت پایدار الکی")
    cfg = make_cfg(HTF_TIMEFRAMES=["1h", "4h"], ENTRY_STUDY_HORIZON={"15m": 48})
    arrs = {f"R{i}/USDT": random_walk(9000, seed=900 + i, price=20 + 3 * i) for i in range(12)}
    st, rows, frows = _study(arrs, cfg, signal_set="all")
    npat = len({r["signal"] for r in rows if r["signal"].startswith(("cdl_", "pat_", "tal_"))})
    import patterns as _p
    check(npat >= (30 if _p.TALIB_NAMES else 18), f"الگوها در سنجش نیومدن ({npat})")   # الگوهای گپ‌دار در سری بدون گپ پیش نمیان
    base = [r for r in rows if r["signal"] == "base" and r["side"] != "both"]
    p_base = [r["all"]["p"][0] for r in base]
    robust = [r for r in rows if r.get("verdict") == "robust"]
    edge_both = [r for r in rows if r.get("verdict") not in ("base", "few") and
                 r["is"].get("z_p", [0])[0] >= 2 and r["oos"].get("z_p", [0])[0] >= 1]
    tested = [r for r in rows if r.get("verdict") not in ("base", "few")]
    good = [f for f in frows if f["flag"] == "good"]
    check(all(abs(p - 0.5) < 0.02 for p in p_base), f"پایه‌ی تصادفی باید ≈۵۰٪ باشه: {p_base}")
    check(len(robust) == 0, f"در بازار تصادفی {len(robust)} مزیت پایدار الکی: "
                            f"{[(r['signal'], r['side']) for r in robust]}")
    check(len(edge_both) <= max(2, len(tested) // 20), f"مزیت معنادار الکی زیاد: {len(edge_both)} از {len(tested)}")
    check(len(good) <= max(3, len(frows) // 100), f"فیلتر مفید الکی زیاد: {len(good)} از {len(frows)}")
    print(f"  ✓ پایه {[round(p * 100, 1) for p in p_base]}٪؛ {len(tested)} ردیف آزمون‌شده، مزیت پایدار {len(robust)}، "
          f"معنادار در هر دو بخش (بدون کارمزد) {len(edge_both)}؛ فیلتر «مفید» الکی {len(good)} از {len(frows)}")
    return len(good), len(frows)


def test_planted_edge():
    print("۵) بازار با روندهای ماندگار: سیگنال‌های روندی باید مزیت نشون بدن")
    cfg = make_cfg(HTF_TIMEFRAMES=["1h", "4h"], ENTRY_STUDY_HORIZON={"15m": 48})
    arrs = {f"T{i}/USDT": synth(9000, seed=300 + i, price=20 + 3 * i) for i in range(10)}
    st, rows, _ = _study(arrs, cfg)
    by = {(r["signal"], r["side"]): r for r in rows}
    ok = [k for k in ("donch55", "trend_follow") if by.get((k, "both"), {}).get("verdict") in ("robust",
                                                                                              "edge_weak_oos",
                                                                                              "edge_costs")]
    check(len(ok) == 2, f"مزیت کاشته‌شده پیدا نشد: {[(k, by.get((k, 'both'), {}).get('verdict')) for k in ('donch55', 'trend_follow')]}")
    r = by[("donch55", "both")]
    print(f"  ✓ شکست ۵۵ کندل: +1R/−1R آموزش {r['is']['p'][0] * 100:.1f}٪ (پایه {r['is']['base_p'][0] * 100:.1f}٪)، "
          f"آزمون {r['oos']['p'][0] * 100:.1f}٪ — حکم: {r['verdict_text']}")


def test_cluster_se():
    print("۶) خطای استاندارد خوشه‌ای")
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 500)
    g = rng.integers(0, 40, 500)
    m, se = es.cluster_mean_se(x, g)
    G = len(np.unique(g))
    ref = np.sqrt(sum(((x[g == k] - x.mean()).sum()) ** 2 for k in np.unique(g))) / len(x) * np.sqrt(G / (G - 1))
    check(abs(m - x.mean()) < 1e-12 and abs(se - ref) < 1e-12, "خطای خوشه‌ای اشتباهه")
    # هر ورود یک خوشه ≈ خطای معمولی
    m2, se2 = es.cluster_mean_se(x, np.arange(500))
    check(abs(se2 - x.std(ddof=0) / np.sqrt(500) * np.sqrt(500 / 499)) < 1e-12, "خطای خوشه‌ای تکی اشتباهه")
    print("  ✓ درسته")


def test_checkpoint_roundtrip():
    print("۷) ذخیره‌ی مرحله‌ای == اجرای یک‌جا")
    import json
    import tempfile
    cfg = make_cfg(HTF_TIMEFRAMES=["1h", "4h"], ENTRY_STUDY_HORIZON={"15m": 48})
    arrs = {f"C{i}/USDT": synth(5000, seed=40 + i, price=10 + 2 * i) for i in range(5)}
    syms = list(arrs)
    start_ms = int(arrs[syms[0]][400, 0])
    split = int(start_ms + (int(arrs[syms[0]][-1, 0]) - start_ms) * 0.7)
    v = None
    a = es.Study(cfg, "15m", split, syms)
    b = es.Study(cfg, "15m", split, syms)
    paths = []
    with tempfile.TemporaryDirectory() as d:
        for sym in syms:
            ar = arrs[sym]
            htf = {"1h": fast_backtest._resample(ar, "15m", "1h"), "4h": fast_backtest._resample(ar, "15m", "4h")}
            p = fast_backtest.prepare_symbol(sym, ar, htf, None, start_ms, cfg, keep_volume=True)
            v = v or fast_backtest.variant_from_cfg(cfg)
            a.add_symbol(p, v)
            b.add_symbol(p, v, keep_part=False)
            pth = os.path.join(d, sym.replace("/", "_") + ".npz")
            es.save_symbol(pth, sym, b.last_part, b._cur, {"ok": True})
            paths.append(pth)
        a.finish()
        metas, n = b.assemble(paths)
    ja = json.dumps([es.row_to_json(r) for r in es.analyze(a)], sort_keys=True)
    jb = json.dumps([es.row_to_json(r) for r in es.analyze(b)], sort_keys=True)
    fa = json.dumps(es.analyze_features(a), sort_keys=True)
    fb = json.dumps(es.analyze_features(b), sort_keys=True)
    check(ja == jb and fa == fb and a.counts == b.counts and n == len(a.E["sig"]) and len(metas) == len(syms),
          "ذخیره‌ی مرحله‌ای با اجرای یک‌جا فرق داره")
    print(f"  ✓ {n} ورود؛ جدول سیگنال‌ها، فیلترها و شمارش‌ها یکسان")


def _bars(rows, t0=1_600_000_000_000, tf_ms=900_000):
    a = np.array(rows, dtype=np.float64)
    ts = t0 + np.arange(len(a)) * tf_ms
    return np.column_stack([ts, a, np.full(len(a), 1000.0)])


def test_patterns():
    print("۸) الگوهای کندلی و کلاسیک")
    import patterns
    # ۲۰ کندل آروم (ATR ≈ ۱) + ریزش ۶ کندلی + الگو
    flat = [(100, 100.5, 99.5, 100)] * 20
    fall = [(100 - k, 100.3 - k, 98.8 - k, 99 - k) for k in range(6)]     # کندل‌های قرمز رو به پایین
    cases = {
        "cdl_hammer": ([(94, 94.2, 91.0, 94.1)], 1),                    # سایه‌ی پایین بلند
        "cdl_engulfing": ([(94.5, 94.6, 93.4, 93.6), (93.5, 95.2, 93.3, 95.0)], 1),
        "cdl_star": ([(95, 95.1, 92.9, 93.0), (92.9, 93.1, 92.5, 92.8), (92.9, 94.8, 92.8, 94.6)], 1),
    }
    bad = 0
    for name, (tail, side) in cases.items():
        arr = _bars(flat + fall + tail)
        o, h, l, c = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
        atr = np.full(len(c), 1.0)
        ml, ms, _, _ = patterns.candle_patterns(o, h, l, c, atr)[name]
        if not ml[-1] or ms[-1]:
            bad += 1
            print("   الگو شناخته نشد:", name)
    check(bad == 0, f"الگوی کندلی: {bad} شکل درست شناخته نشد")
    # کف دوقلو: دو کف هم‌سطح، سقف میانی، شکست بالای سقف میانی
    seq = []
    for p in list(np.linspace(110, 100, 8)) + list(np.linspace(100, 106, 8)) + list(np.linspace(106, 100.2, 8)) \
            + list(np.linspace(100.2, 105.5, 8)) + [106.8]:
        seq.append((p, p + 0.4, p - 0.4, p))
    arr = _bars(seq)
    atr = np.full(len(arr), 1.0)
    d = patterns.chart_patterns(arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], atr, order=3)["pat_double"]
    check(len(d["L"][0]) == 1 and d["L"][0][0] == len(arr) - 1, f"کف دوقلو شناخته نشد: {d['L'][0]}")
    # بدون نگاه به آینده روی سری تصادفی
    a = synth(8000, seed=12)
    import analysis
    import pandas as pd
    atr = analysis.compute_atr(pd.DataFrame({"high": a[:, 2], "low": a[:, 3], "close": a[:, 4]}), 14).values
    full = patterns.all_patterns(a[:, 1], a[:, 2], a[:, 3], a[:, 4], atr)
    mism = 0
    for cut in (5000, 6666):
        part = patterns.all_patterns(a[:cut, 1], a[:cut, 2], a[:cut, 3], a[:cut, 4], atr[:cut])
        for k in full:
            for (sd, i1, s1), (_, i2, s2) in zip(full[k], part[k]):
                m = i1 < cut
                if not (np.array_equal(i1[m], i2) and np.allclose(s1[m], s2)):
                    mism += 1
    n_all = sum(len(x[1]) for v in full.values() for x in v)
    check(mism == 0 and n_all > 1000, f"الگوها: نگاه به آینده {mism}، تعداد {n_all}")
    print(f"  ✓ شکل‌های دستی درست؛ {len(full)} جفت‌الگو، {n_all} مورد روی سری تصادفی، بدون نگاه به آینده")


def test_naive_pattern_backtest():
    print("۹) بک‌تست ساده‌ی مستقل الگوها == سنجش + حد ضرر سمت درست")
    import patterns
    cfg = make_cfg(HTF_TIMEFRAMES=["1h", "4h"], ENTRY_STUDY_HORIZON={"15m": 48})
    arrs = {f"V{i}/USDT": synth(6000, seed=210 + i, price=20 + i) for i in range(3)}
    syms = list(arrs)
    start = int(arrs[syms[0]][400, 0])
    split = int(start + (int(arrs[syms[0]][-1, 0]) - start) * 0.7)
    st = es.Study(cfg, "15m", split, syms, signal_set="patterns")
    v = fast_backtest.variant_from_cfg(cfg)
    preps = {}
    for sym in syms:
        a = arrs[sym]
        htf = {"1h": fast_backtest._resample(a, "15m", "1h"), "4h": fast_backtest._resample(a, "15m", "4h")}
        preps[sym] = fast_backtest.prepare_symbol(sym, a, htf, None, start, cfg, keep_volume=True)
        st.add_symbol(preps[sym], v, None, None)
    st.finish()
    E, H, gap = st.E, st.H, st.gap
    bad, wrong_side, checked = 0, 0, 0
    for name in ("tal_engulfing", "cdl_hammer", "pat_double", "pat_flag", "pat_wedge", "cdl_inside_break"):
        evs, wins, losses = [], 0, 0
        for sym in syms:
            p = preps[sym]
            ser = p.series
            o, h, l, c = ser.o, ser.h, ser.l, ser.c
            atr = fast_backtest._prep_atr(p)
            for sd, idx, slv in patterns.all_patterns(o, h, l, c, atr, cfg.SWING_ORDER)[name]:
                wrong_side += int(((slv >= c[idx]) if sd > 0 else (slv <= c[idx])).sum())
                ok = (idx >= p.first_idx) & np.isfinite(atr[idx]) & (atr[idx] > 0)
                last = -10 ** 9
                for i, stop in zip(idx[ok].tolist(), slv[ok].tolist()):
                    if i - last < gap:
                        continue
                    last = i
                    if i + H > ser.n - 1:
                        continue
                    e = c[i]
                    risk = abs(e - stop)
                    tp = e + 2 * risk if sd > 0 else e - 2 * risk
                    res = None
                    for k in range(i + 1, i + 1 + H):
                        if (sd > 0 and l[k] <= stop) or (sd < 0 and h[k] >= stop):
                            res = -1.0
                            break
                        if (sd > 0 and h[k] >= tp) or (sd < 0 and l[k] <= tp):
                            res = 2.0
                            break
                    if res is None:
                        res = (c[i + H] - e) * sd / risk
                    wins += res == 2.0
                    losses += res == -1.0
                    evs.append(res - st.cost_market * e / risk)
        m = E["sig"] == st.code[name]
        own = np.isfinite(E["own_res"][m])
        oo = E["own_out"][m][own]
        rs = oo != 0
        same = (len(evs) == int(own.sum()) and abs((oo[rs] == 1).mean() - wins / max(1, wins + losses)) < 1e-9
                and abs(np.mean(E["own_res"][m][own].astype(float) - E["own_cost"][m][own]) - np.mean(evs)) < 1e-3)
        bad += int(not same)
        checked += len(evs)
    check(bad == 0 and wrong_side == 0 and checked > 500,
          f"بک‌تست ساده: {bad} الگو فرق داشت، {wrong_side} حد ضرر سمت اشتباه، {checked} معامله")
    print(f"  ✓ {checked} معامله در ۶ الگو: وین‌ریت و سود هر معامله دقیقاً یکسان؛ هیچ حد ضرری سمت اشتباه نیست")


if __name__ == "__main__":
    t0 = time.time()
    test_path_brute()
    test_retest_fill_and_lookahead()
    test_cluster_se()
    test_random_walk_no_edge()
    test_planted_edge()
    test_checkpoint_roundtrip()
    test_patterns()
    test_naive_pattern_backtest()
    print()
    if FAILS:
        print(f"❌ {len(FAILS)} خطا")
        sys.exit(1)
    print(f"✅ همه‌ی تست‌های سنجش ورود موفق ({time.time() - t0:.1f} ثانیه)")
