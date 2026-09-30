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
import money  # noqa: E402
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
    dict(MIN_RISK_REWARD=2.0, WC_MIN_SCORE_PCT=70),
    dict(MIN_RISK_REWARD=2.0, WC_MIN_SCORE_PCT=50, MIN_SL_PCT=0.6, MIN_SL_ATR_MULT=1.0),
    dict(MIN_RISK_REWARD=2.0, WC_MIN_SCORE_PCT=60, REQUIRE_ROOM_TO_TARGET=True),
    dict(MIN_RISK_REWARD=2.5, WC_MIN_SCORE_PCT=80, MIN_SL_PCT=0.8, REQUIRE_ROOM_TO_TARGET=True, PROXIMITY_PCT=0.8),
    dict(MIN_RISK_REWARD=2.0, WC_MIN_SCORE_PCT=50, WC_WEIGHTS={"dow": 1.0, "sr": 3.0, "sma": 2.5, "rsi": 0.7, "cycle": 1.3}),
    dict(MIN_RISK_REWARD=2.0, WC_MIN_SCORE_PCT=50, SR_CONFIRM_BREAK="close", SR_CONFIRM_MAX_DIST_PCT=0.8,
         WC_WEIGHTS={"dow": 1.0, "sr": 4.0, "sma": 1.0, "rsi": 1.0, "cycle": 1.0}),
    dict(MIN_RISK_REWARD=2.0, WC_MIN_SCORE_PCT=60, SR_CONFIRM=False),
    # شکست باکس با تنظیمات مختلف + حد سود بدون/با کارمزد + ورود بازار
    dict(MIN_RISK_REWARD=2.0, BRK_SL_MODE="mid", BRK_MAIN_TREND="with", BRK_VOL_MULT=1.2),
    dict(MIN_RISK_REWARD=2.0, BRK_USE_SR=False, BRK_MAIN_TREND="off", BRK_BOX_MAX_ATR=6.0, BRK_BOX_MIN_TOUCHES=1,
         TP_NET_OF_FEES=False),
    dict(MIN_RISK_REWARD=3.0, ENTRY_MODE="market", BRK_BOX_BARS=12, BRK_MAX_EXT_ATR=0.8, MIN_SL_PCT=0.5),
]


def live_cfg_for(name, cfg):
    """تابع زنده و تنظیمات معادل هر کلید موتور (box_breakout@retest = شکست باکس با ورود پولبک)."""
    base = name.split("@")[0]
    c = backtest.build_config(cfg, {"BRK_ENTRY": "retest" if name.endswith("@retest") else "close",
                                    "PAT_STRUCTURE": "off" if name.endswith("@free") else "with",
                                    "CONTRA_BTC": "off" if name.endswith("@any") else "against"})
    return strategies.STRATEGY_REGISTRY[base]["fn"], c


def test_strategy_equivalence():
    print("۱) معادل‌بودن سیگنال تک‌تک استراتژی‌ها با کد زنده")
    arr = synth(2600, seed=7)
    series = se.Series(arr, "15m")
    rng = np.random.default_rng(3)
    total_sig = 0
    per_name = {n: 0 for n in se.STRATEGY_NAMES}
    for vi, over in enumerate(VARIANTS):
        cfg = make_cfg(**over)
        W = cfg.CANDLE_LIMIT
        structural = se.compute_structural(series, cfg, 0)
        variant = fast_backtest.variant_from_cfg(cfg)
        live_cfgs = {name: live_cfg_for(name, cfg) for name in se.STRATEGY_NAMES}
        prep_like = fast_backtest.SymbolPrep("T", series, structural, None, None, None, 0, cfg)
        finals = {name: prep_like.finals(name, variant) for name in se.STRATEGY_NAMES}
        # همه‌ی کندل‌هایی که موتور سیگنال داده + نمونه‌ی تصادفی از بقیه
        test_ts = set()
        for f in finals.values():
            test_ts.update(f.idx[f.idx >= W - 1].tolist())
        test_ts.update(rng.integers(W - 1, series.n, 250).tolist())
        mism = 0
        for t in sorted(test_ts):
            window = series.to_df(t - W + 1, t + 1)
            for name in se.STRATEGY_NAMES:
                fn, lcfg = live_cfgs[name]
                live = fn(window, lcfg).get("signal")
                f = finals[name]
                k = np.searchsorted(f.idx, t)
                eng = None
                if k < len(f.idx) and f.idx[k] == t:
                    eng = {"side": "LONG" if f.side[k] == 1 else "SHORT", "sl": f.sl[k], "tp": f.tp[k], "rr": f.rr[k],
                           "score": f.score[k], "entry": f.entry[k]}
                if (live is None) != (eng is None):
                    mism += 1
                    if mism <= 5:
                        check(False, f"[v{vi}] {name} t={t}: زنده={live} موتور={eng}")
                    continue
                if live is not None:
                    total_sig += 1
                    per_name[name] += 1
                    same = (live["side"] == eng["side"] and close_enough(live["sl"], eng["sl"])
                            and close_enough(live["entry"], eng["entry"])
                            and close_enough(live["tp"], eng["tp"]) and close_enough(live["rr"], eng["rr"])
                            and (live.get("score") is None and np.isnan(eng["score"])
                                 or live.get("score") is not None and abs(live["score"] - eng["score"]) < 0.051))
                    if not same:
                        mism += 1
                        if mism <= 5:
                            check(False, f"[v{vi}] {name} t={t}: مقادیر فرق دارن زنده={live} موتور={eng}")
        check(mism == 0, f"[v{vi}] {mism} مورد عدم تطابق")
    print(f"  ✓ {total_sig} سیگنال بررسی شد {per_name}")
    check(all(v > 50 for v in per_name.values()), f"تعداد سیگنال برای تست معتبر خیلی کمه {per_name}")


def test_combined_live_wrapper():
    print("۲) سیگنال از مسیر کامل ربات زنده (generate_combined_signal با پنجره‌ی بلندتر)")
    arr = synth(1800, seed=11)
    series = se.Series(arr, "15m")
    for name in se.STRATEGY_NAMES:
        cfg = make_cfg(WC_MIN_SCORE_PCT=0 if name == "fib_phase" else 60, MIN_SL_PCT=0.3,
                       ACTIVE_STRATEGIES=[name.split("@")[0]],
                       BRK_ENTRY="retest" if name.endswith("@retest") else "close",
                       PAT_STRUCTURE="off" if name.endswith("@free") else "with",
                       CONTRA_BTC="off" if name.endswith("@any") else "against")
        check(strategies.engine_name(cfg) == name, f"engine_name اشتباه برای {name}")
        W = cfg.CANDLE_LIMIT
        st = se.compute_structural(series, cfg, 0)
        f = fast_backtest.SymbolPrep("T", series, st, None, None, None, 0, cfg).finals(
            name, fast_backtest.variant_from_cfg(cfg))
        eng = {int(t): (int(sd), sl, tp) for t, sd, sl, tp in zip(f.idx, f.side, f.sl, f.tp)}
        mism, checked = 0, 0
        for t in range(W + 10, series.n, 2):
            df = series.to_df(t - W - 9, t + 1)     # ۱۰ کندل بیشتر از پنجره، مثل ربات زنده
            r = strategies.generate_combined_signal(df, cfg)
            live = r.get("signal")
            e = eng.get(t)
            if (live is None) != (e is None) or (live and (("LONG" if e[0] == 1 else "SHORT") != live["side"]
                                                           or not close_enough(live["sl"], e[1])
                                                           or not close_enough(live["tp"], e[2])
                                                           or live["strategy"] != name.split("@")[0])):
                mism += 1
            elif live:
                checked += 1
        check(mism == 0, f"مسیر زنده {name}: {mism} عدم تطابق")
        check(checked > 5, f"مسیر زنده {name}: سیگنال خیلی کم ({checked})")
        print(f"  ✓ {name}: {checked} سیگنال")
    # ترکیب چند استراتژی: اولویت با اولی
    cfg = make_cfg(WC_MIN_SCORE_PCT=60, ACTIVE_STRATEGIES=["trend_follow", "weighted_confluence"])
    W = cfg.CANDLE_LIMIT
    st = se.compute_structural(series, cfg, 0)
    prep = fast_backtest.SymbolPrep("T", series, st, None, None, None, 0, cfg)
    v = fast_backtest.variant_from_cfg(cfg)
    comb = se.combine({n: prep.finals(n, v) for n in strategies.engine_names(cfg)}, strategies.engine_names(cfg))
    eng = {int(t): (lab, int(sd)) for t, sd, lab in zip(comb["idx"], comb["side"], comb["label"])}
    mism, checked = 0, 0
    for t in range(W + 10, series.n, 2):
        live = strategies.generate_combined_signal(series.to_df(t - W - 9, t + 1), cfg).get("signal")
        e = eng.get(t)
        if (live is None) != (e is None) or (live and (live["strategy"] != e[0] or
                                                       ("LONG" if e[1] == 1 else "SHORT") != live["side"])):
            mism += 1
        elif live:
            checked += 1
    check(mism == 0 and checked > 5, f"ترکیب استراتژی‌ها: {mism} عدم تطابق، {checked} سیگنال")
    print(f"  ✓ ترکیب روندگیر + ترکیبی وزن‌دار: {checked} سیگنال")


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
    sim = sim_engine.PathSim(s, ladder, beyond, config.TRAIL_PROFILES)
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
        floor = money.breakeven_stop(side == 1, entry, 0.0002, 0.0009)
        cases = [(False, 0, None), (True, 0, None), (False, 7, None), (True, 12, None), (True, 0, floor),
                 (True, 9, floor)] + [(name, 0, floor) for name in config.TRAIL_PROFILES]
        for trailing, hold, fl in cases:
            r = sim._run(t0, side, entry, sl, tp, trailing, hold, fl)
            if isinstance(trailing, str):
                lad, bey = config.TRAIL_PROFILES[trailing]["ladder"], config.TRAIL_PROFILES[trailing]["beyond"]
            else:
                lad, bey = ladder, beyond
            # مرجع: حلقه‌ی step_bar
            side_s = "LONG" if side == 1 else "SHORT"
            cur_sl, peak = sl, entry
            ref = None
            last_k = min(s.n, t0 + 1 + hold) if hold else s.n
            for k in range(t0 + 1, last_k):
                hit, price, kind, level, cur_sl, peak = paper_trader.step_bar(
                    side_s, entry, sl, cur_sl, tp, peak, bool(trailing), s.o[k], s.h[k], s.l[k], s.c[k], lad, bey,
                    fl)
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
                    check(False, f"مسیر t0={t0} side={side} trailing={trailing} floor={fl}: موتور={r[:4]} مرجع={ref}")
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
    for name, over in (("weighted_confluence", {}),
                       ("box_breakout", {"BRK_MAIN_TREND": "off", "BRK_BOX_MAX_ATR": 6.0, "TRAIL_PROFILE": "tight"}),
                       ("trend_follow", {"CUT_LOSS_R": 0.6}),
                       ("trend_follow+box_breakout", {"BRK_MAIN_TREND": "off", "BRK_BOX_MAX_ATR": 6.0})):
        _lookahead_and_portfolio(name, over)


def _lookahead_and_portfolio(name, over):
    print(f"۵) عدم نگاه به آینده + ترتیب زمانی سبد ({name})")
    arrs = {f"S{i}/USDT": synth(5000, seed=100 + i, price=10 + i * 7) for i in range(4)}
    cfg = make_cfg(WC_MIN_SCORE_PCT=50, HTF_TIMEFRAMES=["1h", "4h"], HTF_MIN_AGREEMENT=1,
                   SHORT_EXTRA_HTF_AGREEMENT=0, USE_TRAILING_SL=True, MAX_OPEN_POSITIONS=3,
                   ACTIVE_STRATEGIES=name.split("+"), **over)
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
    cfg2 = make_cfg(WC_MIN_SCORE_PCT=50, HTF_TIMEFRAMES=["1h", "4h"], HTF_MIN_AGREEMENT=1, ENTRY_MODE="market",
                    MAX_HOLD_MINUTES=180, ACTIVE_STRATEGIES=name.split("+"),
                    **dict(over, BRK_ENTRY="close", EARLY_EXIT=False))
    res_m, _ = fast_backtest.run_single(_build_preps(arrs, cfg2, start_ms), list(arrs), cfg2)
    types = {t["exit_type"] for t in res_m["trades"]}
    check(len(res_m["trades"]) > 20 and "TIME" in types, f"ورود بازار/حد زمانی کار نکرد: {len(res_m['trades'])} {types}")
    # R هر معامله = سود خالص ÷ ضرر خالص برنامه‌ریزی‌شده؛ باخت بدون گپ باید دقیقاً ۱R- باشه
    sl_r = [t["R"] for t in res_full["trades"] if t["exit_type"] == "SL"]
    check((len(sl_r) > 0 or over.get("CUT_LOSS_R")) and all(r <= -0.999 for r in sl_r),
          f"R باخت‌ها اشتباهه: {sl_r[:5]}")
    tp_r = [t["R"] for t in res_full["trades"] if t["exit_type"] == "TP"]
    check(all(r >= 1.99 for r in tp_r), f"R بردها اشتباهه: {tp_r[:5]}")
    strat_names = {t["strategy"] for t in res_full["trades"]}
    if "+" in name:
        check(len(strat_names) >= 2, f"ترکیب استراتژی‌ها: فقط {strat_names}")
    if "trend_follow" in name:
        trend_tr = [t for t in res_full["trades"] if t["strategy"] == "trend_follow"]
        check(all(t["exit_type"] != "TP" for t in trend_tr), "روندگیر نباید حد سود ثابت داشته باشه")
        big = max((t["R"] for t in trend_tr), default=0)
        print(f"  ✓ روندگیر: {len(trend_tr)} معامله، بزرگ‌ترین برد {big:.1f}R، انواع خروج "
              f"{sorted({t['exit_type'] for t in trend_tr})}")
    if over.get("CUT_LOSS_R"):
        cuts = [t["R"] for t in res_full["trades"] if t["exit_type"] == "CUT"]
        check(len(cuts) > 0 and all(-0.75 < r < -0.55 for r in cuts), f"بستن در ‎-0.6R اشتباهه: {cuts[:5]}")
    # (روندگیر عمداً قبل از ۱R سود حد ضرر شاندلیرش می‌تونه زیر ورود باشه؛ اینجا فقط بقیه)
    tr_r = [t["R"] for t in res_full["trades"] if t["exit_type"] in ("TRAIL_SL", "BREAKEVEN")
            and not t["strategy"].startswith("trend_follow")]
    check(all(r > -0.2 for r in tr_r), f"تریلینگ با ضرر محسوس بسته شده: {sorted(tr_r)[:5]}")
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


def test_money_management():
    print("۷) مدیریت سرمایه: ضرر خالص ثابت دلاری و سود خالص = R:R × ضرر (بعد از کارمزد)")
    cfg = make_cfg(RISK_MODE="usd", RISK_USD=5.0, MIN_RISK_REWARD=2.0, TP_NET_OF_FEES=True)
    rng = np.random.default_rng(5)
    bad = 0
    for _ in range(400):
        side = "LONG" if rng.random() < 0.5 else "SHORT"
        entry_taker = rng.random() < 0.3
        c = make_cfg(RISK_MODE="usd", RISK_USD=5.0, MIN_RISK_REWARD=2.0, TP_NET_OF_FEES=True,
                     ENTRY_MODE="market" if entry_taker else "limit")
        entry = float(rng.uniform(0.01, 60000))
        dist = entry * float(rng.uniform(0.002, 0.05))
        sl = entry - dist if side == "LONG" else entry + dist
        sig = analysis.finalize_signal(side, entry, sl, None, dist, c, tp_uses_level=False)
        fi, fs, fm = money.cfg_fee_fracs(c)
        pos = paper_trader.compute_position_size(side, entry, sig["sl"], 0.5, 10**9, 0, 0, max_leverage=1, min_notional=0.0,
                                                 position_pct_cap=100, risk_usd=5.0, fees=(fi, fs))
        size = pos["size"]
        mk, tk, sp = c.MAKER_FEE_PCT, c.TAKER_FEE_PCT, c.TAKER_SLIPPAGE_PCT

        def net(exit_price, exit_type):
            gross = (exit_price - entry) * size if side == "LONG" else (entry - exit_price) * size
            fee = paper_trader._fee_for_exit(entry, exit_price, size, exit_type, mk, tk, sp, entry_taker=entry_taker)[0]
            return gross - fee
        loss, win = net(sig["sl"], "SL"), net(sig["tp"], "TP")
        be = money.breakeven_stop(side == "LONG", entry, fi, fs)
        if not (abs(loss + 5.0) < 1e-6 and abs(win - 10.0) < 1e-6 and abs(net(be, "TRAIL_SL")) < 1e-6):
            bad += 1
            if bad <= 3:
                check(False, f"{side} ورود={entry} ضرر={loss} سود={win} سربه‌سر={net(be, 'TRAIL_SL')}")
    check(bad == 0, f"مدیریت سرمایه: {bad} مورد اشتباه")
    print("  ✓ ضرر خالص = ۵$، سود خالص = ۱۰$، سربه‌سر تریلینگ = ۰$ (۴۰۰ حالت تصادفی)")


def test_htf_weighted():
    print("۸) تایید HTF وزن‌دار + نمودار ارز÷BTC == محاسبه‌ی ربات زنده")
    main = synth(3000, seed=41)
    btc_main = synth(3000, seed=42, price=60000.0)
    tfs = ["1h", "4h"]
    cfg = make_cfg(HTF_TIMEFRAMES=tfs, BTC_REGIME_TIMEFRAME="4h", BTC_PAIR_WEIGHT=3.0, HTF_WEIGHTED=True)
    htf = {tf: fast_backtest._resample(main, "15m", tf) for tf in tfs}
    btc4 = fast_backtest._resample(btc_main, "15m", "4h")
    prep = fast_backtest.prepare_symbol("X/USDT", main, htf, btc4, int(main[400, 0]), cfg)
    pw = sim_engine.pair_weight_for("X/USDT", cfg)
    check(pw == 3.0 and sim_engine.pair_weight_for("BTC/USDT", cfg) == 0.0, "وزن نسبت به BTC اشتباهه")
    hs4 = se.Series(htf["4h"], "4h")
    bs4 = se.Series(btc4, "4h")
    pair_seen = 0
    rng = np.random.default_rng(2)
    mism = 0
    for t in rng.integers(400, prep.series.n, 200).tolist():
        close_t = prep.series.close_ts[t]
        trends = {}
        for tf in tfs:
            hs = se.Series(htf[tf], tf)
            j = int(np.searchsorted(hs.close_ts, close_t, "right")) - 1
            if j < 0:
                continue
            df = hs.to_df(max(0, j - cfg.HTF_CANDLE_LIMIT + 1), j + 1)
            if len(df) < cfg.SWING_ORDER * 2 + 5:
                continue
            trends[tf] = analysis.trend_from_df(df, swing_order=cfg.SWING_ORDER)
        # نمودار ارز÷BTC مثل ربات زنده: آخرین HTF_CANDLE_LIMIT کندل بسته‌شده‌ی هر دو، ادغام روی زمان
        pair = None
        j = int(np.searchsorted(hs4.close_ts, close_t, "right")) - 1
        jb = int(np.searchsorted(bs4.close_ts, close_t, "right")) - 1
        if j >= 0 and jb >= 0:
            N = cfg.HTF_CANDLE_LIMIT
            rdf = se.pair_ratio_df(hs4.to_df(max(0, j - N + 1), j + 1), bs4.to_df(max(0, jb - N + 1), jb + 1))
            if len(rdf) >= cfg.SWING_ORDER * 2 + 5:
                pair = analysis.trend_from_df(rdf, swing_order=cfg.SWING_ORDER)
                pair_seen += pair in ("uptrend", "downtrend")
        pl = sim_engine.htf_weighted_pct(trends, tfs, "LONG", pair, pw)
        ps = sim_engine.htf_weighted_pct(trends, tfs, "SHORT", pair, pw)
        if not (close_enough(pl, prep.htf_wlong[t]) and close_enough(ps, prep.htf_wshort[t])):
            mism += 1
    check(mism == 0, f"HTF وزن‌دار: {mism} عدم تطابق")
    check(pair_seen > 50, f"روند نسبت به BTC خیلی کم دیده شد ({pair_seen})")
    check(sim_engine.htf_required_pct(3, 1) == (60.0, 80.0), "درصد لازم HTF اشتباهه")
    print("  ✓ انجام شد")


def test_live_process_bars():
    print("۹) مدیریت پوزیشن ربات زنده (process_bars) == موتور بک‌تست (با خروج زودهنگام، تریلینگ، کف سربه‌سر)")
    from datetime import datetime
    arr = synth(4000, seed=61, tf_ms=60_000)
    s = se.Series(arr, "1m")
    mk, tk, sp = 0.02, 0.06, 0.03
    rng = np.random.default_rng(4)
    mism, n = 0, 0
    kinds = set()
    profiles = dict(config.TRAIL_PROFILES, trend=config.TREND_TRAIL)
    for prof in list(profiles) + [False]:
        lad = profiles[prof]["ladder"] if prof else []
        bey = profiles[prof]["beyond"] if prof else 0.5
        flr = profiles[prof].get("floor_r") if prof else None
        sim = sim_engine.PathSim(s, lad, bey, profiles)
        for _ in range(120):
            t0 = int(rng.integers(10, s.n - 400))
            side = 1 if rng.random() < 0.5 else -1
            entry = float(s.c[t0])
            dist = entry * float(rng.uniform(0.002, 0.02))
            sl = entry - dist if side == 1 else entry + dist
            tp = entry + dist * 2.1 if side == 1 else entry - dist * 2.1
            early = (int(rng.integers(2, 8)), float(rng.choice([0.2, 0.3, 0.5])))
            fi, fs, _ = money.fee_fracs(mk, tk, sp, False)
            floor = money.breakeven_stop(side == 1, entry, fi, fs) if prof else None
            cut = float(rng.choice([0.0, 0.0, 0.6]))
            pending = rng.random() < 0.5
            if pending:
                # سفارش لیمیت: ورود کمی بهتر از قیمت فعلی، پر شدن فقط اگه قیمت ازش رد بشه (مثل run_portfolio)
                entry = entry * (0.999 if side == 1 else 1.001)
                sl = entry - dist if side == 1 else entry + dist
                tp = entry + dist * 2.1 if side == 1 else entry - dist * 2.1
                floor = money.breakeven_stop(side == 1, entry, fi, fs) if prof else None
                wait = 30
                win = s.l[t0 + 1:t0 + 1 + wait] < entry if side == 1 else s.h[t0 + 1:t0 + 1 + wait] > entry
                if not win.any():
                    continue
                k_fill = t0 + 1 + int(np.argmax(win))
                r = sim._run(k_fill - 1, side, entry, sl, tp, prof, 0, floor, early, fill_bar=True, cut=cut or None)
            else:
                r = sim._run(t0, side, entry, sl, tp, prof, 0, floor, early, cut=cut or None)
            conn = paper_trader.get_conn(":memory:")
            as_of = datetime.utcfromtimestamp(s.close_ts[t0] / 1000)
            paper_trader.open_trade(conn, "X/USDT", "LONG" if side == 1 else "SHORT", entry, sl, tp, 1.0, 1e9,
                                    min_notional=0, max_leverage=1, position_pct_cap=100, trailing_enabled=bool(prof),
                                    as_of=as_of, timeframe="1m", pending=pending,
                                    strategy_name="trend_follow" if prof == "trend" else "x",
                                    expire_ts=int(s.close_ts[t0]) + 30 * 60_000 if pending else None)
            bars = [(int(s.ts[k]), s.o[k], s.h[k], s.l[k], s.c[k]) for k in range(t0 + 1, s.n)]
            paper_trader.process_bars(conn, "X/USDT", bars, 1e9, lad if prof != "trend" else [], bey, mk, tk, sp, 0.0,
                                      now_ms=int(s.close_ts[-1]), trail_floor=bool(prof),
                                      early_bars=early[0], early_min_r=early[1],
                                      profiles={"trend_follow": (lad, bey, flr)}, cut_r=cut)
            row = conn.execute("SELECT close_price, exit_type, close_time FROM trades").fetchone()
            n += 1
            if r[2] == "END":
                ok = row[1] is None
            else:
                cut_px = (entry - cut * dist if side == 1 else entry + cut * dist) if cut else None
                exp_type = ("CUT" if (cut_px is not None and r[2] == "STOP" and close_enough(r[3], cut_px, 1e-12))
                            else paper_trader.classify_exit_level(r[2], r[3], sl, entry, bool(prof)))
                exp_close = datetime.utcfromtimestamp(s.close_ts[r[0]] / 1000).isoformat()
                ok = (row[1] == exp_type and close_enough(row[0], r[1], 1e-12) and row[2] == exp_close)
                kinds.add(exp_type)
            if not ok:
                mism += 1
                if mism <= 5:
                    check(False, f"process_bars t0={t0} side={side} prof={prof} early={early}: موتور={r[:4]} زنده={row}")
    check(mism == 0, f"process_bars: {mism} عدم تطابق از {n}")
    check({"EARLY", "SL", "TP", "CUT", "TRAIL_SL"} <= kinds, f"انواع خروج کافی تست نشد: {kinds}")
    print(f"  ✓ {n} معامله، انواع خروج: {sorted(kinds)}")


def test_daily_loss_limit():
    print("۱۰) حد ضرر روزانه")
    arrs = {f"S{i}/USDT": synth(5000, seed=300 + i, price=10 + i * 5) for i in range(5)}
    base = dict(HTF_TIMEFRAMES=["1h", "4h"], USE_HTF_CONFIRMATION=False, ACTIVE_STRATEGIES=["box_breakout"],
                BRK_MAIN_TREND="off", BRK_BOX_MAX_ATR=6.0, MAX_OPEN_POSITIONS=5, COOLDOWN_HOURS=0,
                EARLY_EXIT=False, USE_TRAILING_SL=False)
    start_ms = int(arrs["S0/USDT"][400, 0])
    res_off, _ = fast_backtest.run_single(_build_preps(arrs, make_cfg(**base, DAILY_LOSS_LIMIT_USD=0), start_ms),
                                          list(arrs), make_cfg(**base, DAILY_LOSS_LIMIT_USD=0))
    cfg = make_cfg(**base, DAILY_LOSS_LIMIT_USD=6.0)
    res_on, _ = fast_backtest.run_single(_build_preps(arrs, cfg, start_ms), list(arrs), cfg)
    DAY = 86_400_000
    bad = 0
    for t in res_on["trades"]:
        d = t["open_time"] // DAY
        realized = sum(x["pnl"] for x in res_on["trades"] if x["close_time"] // DAY == d and x["close_time"] <= t["open_time"])
        if realized <= -6.0:
            bad += 1
    rej = sum(1 for x in res_on["signals"] if x["reason"] == "daily_loss_limit")
    check(bad == 0, f"حد ضرر روزانه رعایت نشده ({bad})")
    check(rej > 0 and len(res_on["trades"]) < len(res_off["trades"]), f"حد ضرر روزانه اثری نداشت ({rej})")
    print(f"  ✓ {rej} سیگنال به‌خاطر حد ضرر روزانه رد شد ({len(res_off['trades'])} → {len(res_on['trades'])} معامله)")


def test_cut_loss_whatif():
    print("۱۱) تحلیل «اگه در ‎-xR می‌بستیم» (MAE) == شبیه‌سازی واقعی با حد ضرر ‎-xR")
    s = se.Series(synth(6000, seed=77), "15m")
    sim = sim_engine.PathSim(s, [], 0.5, None)
    rng = np.random.default_rng(12)
    bad, n_touch, n = 0, 0, 0
    for _ in range(800):
        t0 = int(rng.integers(10, s.n - 300))
        side = 1 if rng.random() < 0.5 else -1
        entry = float(s.c[t0])
        dist = entry * float(rng.uniform(0.003, 0.02))
        sl = entry - dist if side == 1 else entry + dist
        tp = entry + dist * 2.1 if side == 1 else entry - dist * 2.1
        fb = rng.random() < 0.5
        x = float(rng.choice([0.4, 0.6, 0.8]))
        r = sim._run(t0, side, entry, sl, tp, False, 0, None, None, fb)
        et = paper_trader.classify_exit_level(r[2], r[3], sl, entry, False)
        mae = sim_engine.trade_mae_r(s, t0, r[0], side, entry, sl, et)
        sl_cut = entry - x * dist if side == 1 else entry + x * dist
        rc = sim._run(t0, side, entry, sl_cut, tp, False, 0, None, None, fb)
        n += 1
        if mae >= x - 1e-12:
            n_touch += 1
            ok = rc[2] == "STOP" and rc[0] <= r[0] and close_enough(rc[3], sl_cut)
        else:
            ok = rc[0] == r[0] and rc[2] == r[2] and close_enough(rc[1], r[1], 1e-12)
        if not ok:
            bad += 1
            if bad <= 3:
                check(False, f"t0={t0} side={side} x={x} mae={mae:.3f} اصلی={r[:4]} با‌حدضرر={rc[:4]}")
    check(bad == 0 and n_touch > 50, f"MAE: {bad} عدم تطابق، {n_touch} لمس از {n}")
    print(f"  ✓ {n} معامله ({n_touch} به ‎-xR رسیدن)")


def test_xs_momentum():
    print("۱۲) مومنتوم نسبی هفتگی (روزانه): رتبه‌بندی موتور == رتبه‌بندی ربات زنده + خروج آخر هفته")
    DAY = 86_400_000
    start = (1_600_000_000_000 // DAY) * DAY
    arrs = {f"M{i:02d}/USDT": synth(900, seed=700 + i, start_ms=start, tf_ms=DAY, price=5 + i * 3) for i in range(12)}
    cfg = make_cfg(TIMEFRAME="1d", ACTIVE_STRATEGIES=["xs_momentum"], USE_HTF_CONFIRMATION=False, HTF_TIMEFRAMES=[],
                   COOLDOWN_HOURS=48, EARLY_EXIT=False, DAILY_LOSS_LIMIT_USD=0, MAX_OPEN_POSITIONS=20,
                   CUT_LOSS_R=0.0)
    start_ms = int(arrs["M00/USDT"][350, 0])
    preps = {sym: fast_backtest.prepare_symbol(sym, a, {}, None, start_ms, cfg) for sym, a in arrs.items()}
    v = fast_backtest.variant_from_cfg(cfg)
    xs = fast_backtest.xs_finals(preps, v)
    eng = {(sym, int(preps[sym].series.ts[i]), int(sd)) for sym, f in xs.items() for i, sd in zip(f.idx, f.side)}
    # مرجع: مثل ربات زنده — هر یکشنبه، بازده ۲۸ روز هر نماد با دیتافریم، رتبه با strategies.xs_rank، سیگنال با
    # strategies.generate_xs_signal روی پنجره‌ی CANDLE_LIMIT
    live = set()
    full = {sym: se.Series(a, "1d") for sym, a in arrs.items()}
    s0 = preps["M00/USDT"].series
    mism_sig = 0
    for i in range(preps["M00/USDT"].first_idx, s0.n):
        if not strategies.xs_is_rebalance(int(s0.ts[i])):
            continue
        rets = {}
        for sym in preps:
            df = full[sym].to_df(max(0, i - cfg.CANDLE_LIMIT + 1), i + 1)
            rets[sym] = strategies.xs_return(df["close"].values, cfg.XS_LOOKBACK_DAYS)
        for sym, side in strategies.xs_rank(rets, cfg).items():
            p = preps[sym]
            sig = strategies.generate_xs_signal(full[sym].to_df(max(0, i - cfg.CANDLE_LIMIT + 1), i + 1), cfg, side)
            if sig:
                live.add((sym, int(s0.ts[i]), 1 if side == "LONG" else -1))
                f = xs[sym]
                k = int(np.searchsorted(f.idx, i))
                if not (k < len(f.idx) and f.idx[k] == i and close_enough(f.sl[k], sig["sl"])
                        and close_enough(f.tp[k], sig["tp"])):
                    mism_sig += 1
    check(eng == live and len(eng) > 100, f"رتبه‌بندی مومنتوم: موتور {len(eng)} زنده {len(live)} "
                                          f"اختلاف {len(eng ^ live)}")
    check(mism_sig == 0, f"حد ضرر/سود مومنتوم: {mism_sig} عدم تطابق")
    res, P = fast_backtest.run_single(preps, list(arrs), cfg)
    tr = res["trades"]
    types = {t["exit_type"] for t in tr}
    held_ok = all(t["bars"] <= cfg.XS_HOLD_DAYS + 1 for t in tr if t["exit_type"] != "END")
    check(len(tr) > 100 and "TIME" in types and held_ok, f"مومنتوم: {len(tr)} معامله، {types}، نگه‌داری درست={held_ok}")
    longs = sum(1 for t in tr if t["side"] == "LONG")
    print(f"  ✓ {len(eng)} انتخاب هفتگی یکسان؛ {len(tr)} معامله ({longs} خرید)، انواع خروج {sorted(types)}")


def test_fib_dense():
    print("۱۳) فیبوناچی حرکت دوم: همه‌ی کندل‌ها (بدون حداقل امتیاز) موتور == ربات زنده، + اجرای سبد")
    total = 0
    mism = 0
    for seed, over in ((71, {}), (72, {"FIB_ZONE_LO": 0.382, "FIB_REQUIRE_BREAK": False, "FIB_MIN_IMPULSE_ATR": 1.5}),
                       (73, {"FIB_WEIGHTS": {"zone": 2.0, "rsi": 1.0, "vol": 0.5, "vp": 1.5, "sr": 1.0, "dow": 0.0},
                             "FIB_TRIGGER_BARS": 5})):
        arr = synth(1800, seed=seed)
        series = se.Series(arr, "15m")
        cfg = make_cfg(WC_MIN_SCORE_PCT=0, ACTIVE_STRATEGIES=["fib_phase"], **over)
        W = cfg.CANDLE_LIMIT
        structural = se.compute_structural(series, cfg, 0)
        variant = fast_backtest.variant_from_cfg(cfg)
        prep_like = fast_backtest.SymbolPrep("T", series, structural, None, None, None, 0, cfg)
        f = prep_like.finals("fib_phase", variant)
        eng = {int(i): k for k, i in enumerate(f.idx)}
        for t in range(W - 1, series.n):
            live = strategies.generate_fib_phase(series.to_df(t - W + 1, t + 1), cfg).get("signal")
            k = eng.get(t)
            if (live is None) != (k is None):
                mism += 1
                if mism <= 3:
                    check(False, f"فیبوناچی t={t}: زنده={live} موتور={'-' if k is None else f.sl[k]}")
                continue
            if live is not None:
                total += 1
                if not (live["side"] == ("LONG" if f.side[k] == 1 else "SHORT") and close_enough(live["sl"], f.sl[k])
                        and close_enough(live["tp"], f.tp[k]) and abs(live["score"] - f.score[k]) < 1e-9):
                    mism += 1
    check(mism == 0 and total >= 30, f"فیبوناچی: {mism} عدم تطابق از {total} سیگنال")
    # اجرای کامل سبد با فیبوناچی (+ بستن در ‎-0.5R) — بدون خطا و با معامله
    arrs = {f"F{i}/USDT": synth(4000, seed=90 + i, price=10 + i * 5) for i in range(3)}
    cfg = make_cfg(WC_MIN_SCORE_PCT=0, ACTIVE_STRATEGIES=["fib_phase"], HTF_TIMEFRAMES=["1h", "4h"],
                   USE_HTF_CONFIRMATION=False, CUT_LOSS_R=0.5, USE_TRAILING_SL=False)
    preps = _build_preps(arrs, cfg, int(arrs["F0/USDT"][400, 0]))
    res, _ = fast_backtest.run_single(preps, list(arrs), cfg)
    m = sim_engine.metrics(res["trades"], res["equity"], cfg.VIRTUAL_BALANCE_START)
    check(len(res["trades"]) > 5 and "avg_week_usd" in m, f"سبد فیبوناچی: {len(res['trades'])} معامله")
    print(f"  ✓ {total} سیگنال روی همه‌ی کندل‌ها یکسان؛ سبد: {len(res['trades'])} معامله، "
          f"میانگین هفتگی {m['avg_week_usd']}$، {m['pos_weeks_pct']}٪ هفته‌ها مثبت")


def test_pattern_structure_dense():
    print("۱۴) الگو + ساختار بازار: همه‌ی کندل‌ها موتور == ربات زنده (با/بدون فیلتر ساختار، چند مجموعه الگو)")
    total = mism = 0
    for seed, over in ((81, {}), (82, {"PAT_STRUCTURE": "off", "PAT_SET": "own+chart"}),
                       (83, {"PAT_SET": "talib", "PAT_MIN_SL_ATR": 1.0})):
        arr = synth(1500, seed=seed)
        series = se.Series(arr, "15m")
        cfg = make_cfg(ACTIVE_STRATEGIES=["pattern_structure"], **over)
        key = strategies.engine_name(cfg)
        W = cfg.CANDLE_LIMIT
        structural = se.compute_structural(series, cfg, 0)
        prep_like = fast_backtest.SymbolPrep("T", series, structural, None, None, None, 0, cfg)
        f = prep_like.finals(key, fast_backtest.variant_from_cfg(cfg))
        eng = {int(i): k for k, i in enumerate(f.idx)}
        for t in range(W - 1, series.n):
            live = strategies.generate_pattern_structure(series.to_df(t - W + 1, t + 1), cfg).get("signal")
            k = eng.get(t)
            if (live is None) != (k is None):
                mism += 1
                if mism <= 3:
                    check(False, f"الگو+ساختار t={t}: زنده={live} موتور={'-' if k is None else f.sl[k]}")
                continue
            if live is not None:
                total += 1
                if not (live["side"] == ("LONG" if f.side[k] == 1 else "SHORT") and close_enough(live["sl"], f.sl[k])
                        and close_enough(live["tp"], f.tp[k])):
                    mism += 1
    check(mism == 0 and total >= 100, f"الگو+ساختار: {mism} عدم تطابق از {total} سیگنال")
    arrs = {f"P{i}/USDT": synth(3000, seed=95 + i, price=10 + i * 5) for i in range(3)}
    cfg = make_cfg(ACTIVE_STRATEGIES=["pattern_structure"], HTF_TIMEFRAMES=["1h", "4h"], USE_HTF_CONFIRMATION=False,
                   USE_TRAILING_SL=True, TRAIL_PROFILE="balanced")
    preps = _build_preps(arrs, cfg, int(arrs["P0/USDT"][400, 0]))
    res, _ = fast_backtest.run_single(preps, list(arrs), cfg)
    types = sorted({t["exit_type"] for t in res["trades"]})
    check(len(res["trades"]) > 20, f"سبد الگو+ساختار: {len(res['trades'])} معامله")
    print(f"  ✓ {total} سیگنال روی همه‌ی کندل‌ها یکسان؛ سبد با تریلینگ: {len(res['trades'])} معامله، خروج‌ها {types}")


def test_contrarian_dense():
    print("۱۵) استراتژی شخصی «خلاف جمعیت»: همه‌ی کندل‌ها موتور == ربات زنده، + شرط BTC در سبد == قانون زنده")
    total = mism = 0
    for seed, over in ((111, {}), (112, {"CONTRA_MIN_SL_ATR": 1.5}), (113, {"CONTRA_BTC": "off"})):
        arr = synth(1500, seed=seed)
        series = se.Series(arr, "15m")
        cfg = make_cfg(ACTIVE_STRATEGIES=["contrarian_btc"], **over)
        key = strategies.engine_name(cfg)
        check(key == ("contrarian_btc@any" if over.get("CONTRA_BTC") == "off" else "contrarian_btc"), f"کلید {key}")
        W = cfg.CANDLE_LIMIT
        structural = se.compute_structural(series, cfg, 0)
        prep_like = fast_backtest.SymbolPrep("T", series, structural, None, None, None, 0, cfg)
        f = prep_like.finals(key, fast_backtest.variant_from_cfg(cfg))
        eng = {int(i): k for k, i in enumerate(f.idx)}
        for t in range(W - 1, series.n):
            live = strategies.generate_contrarian_btc(series.to_df(t - W + 1, t + 1), cfg).get("signal")
            k = eng.get(t)
            if (live is None) != (k is None):
                mism += 1
                if mism <= 3:
                    check(False, f"خلاف جمعیت t={t}: زنده={live} موتور={'-' if k is None else f.sl[k]}")
                continue
            if live is not None:
                total += 1
                if not (live["side"] == ("LONG" if f.side[k] == 1 else "SHORT") and close_enough(live["sl"], f.sl[k])
                        and close_enough(live["tp"], f.tp[k])):
                    mism += 1
    check(mism == 0 and total >= 50, f"خلاف جمعیت: {mism} عدم تطابق از {total} سیگنال")

    # سبد با BTC: هر معامله‌ی contrarian_btc باید خلاف روند BTC همون لحظه باشه، و روند BTC موتور == ربات زنده
    btc = synth(6000, seed=120, price=30000.0)
    btc4 = fast_backtest._resample(btc, "15m", "4h")
    arrs = {f"C{i}/USDT": synth(6000, seed=121 + i, price=10 + i * 5) for i in range(4)}
    res_n = {}
    for mode in ("against", "off"):
        cfg = make_cfg(ACTIVE_STRATEGIES=["contrarian_btc"], HTF_TIMEFRAMES=["1h", "4h"], USE_HTF_CONFIRMATION=False,
                       USE_TRAILING_SL=True, TRAIL_PROFILE="strict", CUT_LOSS_R=0.5, CONTRA_BTC=mode,
                       BTC_REGIME_TIMEFRAME="4h")
        preps = {}
        for sym, a in arrs.items():
            htf = {"1h": fast_backtest._resample(a, "15m", "1h"), "4h": fast_backtest._resample(a, "15m", "4h")}
            preps[sym] = fast_backtest.prepare_symbol(sym, a, htf, btc4, int(a[400, 0]), cfg)
        res, _ = fast_backtest.run_single(preps, list(arrs), cfg)
        res_n[mode] = len(res["trades"])
        if mode == "against":
            bs = se.Series(btc4, "4h")
            bad = btc_mism = 0
            for tr in res["trades"]:
                p = preps[tr["symbol"]]
                i = int(tr["sig_idx"])
                b = int(p.btc_trend[i])
                if not strategies.contra_btc_allows(tr["side"], b):
                    bad += 1
                # روند BTC به روش ربات زنده (آخرین HTF_CANDLE_LIMIT کندل ۴ساعته‌ی بسته‌شده)
                ct = p.series.close_ts[i]
                j = int(np.searchsorted(bs.close_ts, ct, "right")) - 1
                df = bs.to_df(max(0, j - config.HTF_CANDLE_LIMIT + 1), j + 1)
                lt = analysis.trend_from_df(df, swing_order=config.SWING_ORDER) \
                    if len(df) >= config.SWING_ORDER * 2 + 5 else None
                if strategies.contra_btc_allows(tr["side"], lt) != strategies.contra_btc_allows(tr["side"], b):
                    btc_mism += 1
            check(bad == 0, f"خلاف جمعیت: {bad} معامله هم‌جهت BTC باز شده")
            check(btc_mism == 0, f"روند BTC موتور ≠ زنده در {btc_mism} معامله")
            types = sorted({t["exit_type"] for t in res["trades"]})
    check(res_n["against"] > 5 and res_n["off"] > res_n["against"],
          f"سبد خلاف جمعیت: با شرط {res_n['against']}، بدون شرط {res_n['off']} معامله")
    print(f"  ✓ {total} سیگنال روی همه‌ی کندل‌ها یکسان؛ سبد: با شرط BTC {res_n['against']} معامله، "
          f"بدون شرط {res_n['off']}؛ خروج‌ها {types}")


def test_wipe_history():
    print("۱۶) شروع از صفر: پاک شدن معاملات/لاگ‌ها و برگشت موجودی")
    import sqlite3
    conn = paper_trader.get_conn(":memory:")
    conn.execute("INSERT INTO trades (symbol, status, pnl) VALUES ('X/USDT', 'CLOSED', 5.0)")
    conn.execute("INSERT INTO trades (symbol, status) VALUES ('Y/USDT', 'OPEN')")
    conn.execute("INSERT INTO signal_log (symbol, side) VALUES ('X/USDT', 'LONG')")
    paper_trader.record_equity(conn, 123.0)
    paper_trader.set_setting(conn, "strategy", "contrarian_btc")
    paper_trader.wipe_history(conn, 100.0)
    n = [conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("trades", "signal_log")]
    check(n == [0, 0], f"بعد از پاک‌سازی: {n}")
    check(paper_trader.get_balance(conn, 100.0) == 100.0, "موجودی به سرمایه‌ی اولیه برنگشت")
    check(paper_trader.get_setting(conn, "strategy", None) == "contrarian_btc", "تنظیمات نباید پاک بشن")
    check(paper_trader.get_open_position_count(conn) == 0, "پوزیشن باز مونده")
    print("  ✓ انجام شد")


if __name__ == "__main__":
    t_start = time.time()
    test_strategy_equivalence()
    test_combined_live_wrapper()
    test_htf_equivalence()
    test_path_equivalence()
    test_no_lookahead_and_portfolio()
    test_speed()
    test_money_management()
    test_htf_weighted()
    test_live_process_bars()
    test_daily_loss_limit()
    test_cut_loss_whatif()
    test_xs_momentum()
    test_fib_dense()
    test_pattern_structure_dense()
    test_contrarian_dense()
    test_wipe_history()
    print()
    if FAILS:
        print(f"❌ {len(FAILS)} خطا")
        sys.exit(1)
    print(f"✅ همه‌ی تست‌ها موفق ({time.time() - t_start:.1f} ثانیه)")
