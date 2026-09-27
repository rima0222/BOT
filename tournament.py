# -*- coding: utf-8 -*-
"""
مقایسه‌ی خودکار استراتژی‌ها («تورنمنت») روی ۱-۲ سال دیتای تاریخی.

روش (سخت‌گیرانه، برای جلوگیری از خودفریبی):
  - بازه به دو بخش تقسیم می‌شه: ۷۰٪ اول «آموزش» (IS) و ۳۰٪ آخر «آزمون» (OOS).
  - همه‌ی ترکیب‌ها روی کل بازه اجرا می‌شن، ولی «انتخاب» فقط با نتیجه‌ی بخش آموزش انجام
    می‌شه. بخش آزمون، دیتاییه که انتخاب هیچ‌وقت ندیده‌ش — اگه برنده اون‌جا هم سودده
    بمونه، احتمال واقعی بودن لبه‌ی معاملاتی خیلی بیشتره.
  - معیار رتبه‌بندی: «کران پایین ۹۵٪ میانگین R هر معامله» (نه وین‌ریت، نه سود خام).
    این معیار هم کیفیت (میانگین R بعد از کارمزد) و هم تعداد نمونه و ثبات (انحراف معیار)
    رو با هم لحاظ می‌کنه؛ ترکیبی که فقط چند معامله‌ی شانسی داشته، امتیاز بالا نمی‌گیره.
  - اثر هر «بهبود» (تریلینگ، حداقل فاصله‌ی SL، فضای تا هدف، فیلتر BTC، فقط خرید، سطح
    سخت‌گیری) به‌صورت «جفتی» سنجیده می‌شه: دو تنظیمی که فقط در همون یک مورد فرق دارن
    مقایسه می‌شن، و درصد دفعاتی که اون بهبود نتیجه رو بهتر کرده گزارش می‌شه.
"""
import itertools
import math
import time
from collections import defaultdict

import numpy as np

import fast_backtest
import sim_engine
import signals_engine as se

DOW, BRK, VOL, CDL = "dow_support_resistance", "breakout", "volume_spike", "candle_setup"
LEVEL_STRATEGIES = {DOW, CDL}   # استراتژی‌هایی که سطح مقابل دارن (فیلتر «فضای تا هدف» فقط روی این‌ها اثر داره)

STRATEGY_LABELS = {DOW: "داو+حمایت/مقاومت", BRK: "بریک‌اوت", VOL: "افزایش حجم", CDL: "کندل‌ستاپ"}


def strategy_sets(grid):
    singles = [((s,), "any") for s in (DOW, BRK, VOL, CDL)]
    if grid == "quick":
        return singles + [
            ((DOW, VOL), "any"), ((DOW, CDL), "any"), ((DOW, BRK), "any"),
            ((DOW, VOL, BRK, CDL), "any"),
            ((DOW, VOL), "confirm"), ((DOW, BRK), "confirm"), ((CDL, VOL), "confirm"),
        ]
    out = list(singles)
    names = [DOW, BRK, VOL, CDL]
    for r in (2, 3, 4):
        for combo in itertools.combinations(names, r):
            out.append((combo, "any"))
    for a, b in itertools.combinations(names, 2):
        out.append(((a, b), "all"))
    for trig in (DOW, CDL):
        for other in names:
            if other != trig:
                out.append(((trig, other), "confirm"))
    return out


def describe(conf):
    names = [STRATEGY_LABELS.get(s, s) for s in conf["active_strategies"]]
    mode = conf["combine_mode"]
    if len(names) == 1:
        strat = names[0]
    elif mode == "any":
        strat = " یا ".join(names)
    elif mode == "all":
        strat = " و ".join(names) + " (هم‌زمان)"
    else:
        strat = f"{names[0]} با تایید {' + '.join(names[1:])}"
    strict_fa = {"loose": "سبک‌گیر", "normal": "معمولی", "strict": "سخت‌گیر", "very_strict": "خیلی سخت‌گیر"}
    parts = [strat, f"سخت‌گیری: {strict_fa.get(conf['strictness'], conf['strictness'])}",
             "تریلینگ" if conf["trailing"] else "TP ثابت"]
    if conf["min_sl"]:
        parts.append("حداقل فاصله‌ی SL")
    if conf["room"]:
        parts.append("فضای تا هدف")
    if conf["btc_filter"]:
        parts.append("فیلتر BTC")
    if conf["long_only"]:
        parts.append("فقط خرید")
    return " | ".join(parts)


def _variant(cfg, min_rr, min_sl, room):
    return {"min_rr": float(min_rr),
            "min_sl_pct": float(cfg.TEST_MIN_SL_PCT) if min_sl else 0.0,
            "min_sl_atr": float(cfg.TEST_MIN_SL_ATR_MULT) if min_sl else 0.0,
            "room": bool(room)}


def evaluate(preps, order, cfg, conf, split_ms, record=False, merged_cache=None):
    """اجرای یک تنظیم و محاسبه‌ی معیارها برای IS / OOS / کل."""
    preset = cfg.STRICTNESS_PRESETS[conf["strictness"]]
    variant = _variant(cfg, preset["MIN_RISK_REWARD"], conf["min_sl"], conf["room"])
    mkey = (tuple(conf["active_strategies"]), conf["combine_mode"], tuple(sorted(variant.items())))
    merged = merged_cache.get(mkey) if merged_cache is not None else None
    if merged is None:
        cands = fast_backtest.candidates_for(preps, list(conf["active_strategies"]), conf["combine_mode"],
                                             variant, int(cfg.CONFIRM_LOOKBACK_BARS))
        merged = sim_engine.merge_candidates(cands, preps, order)
        if merged_cache is not None:
            if len(merged_cache) > 40:
                merged_cache.clear()
            merged_cache[mkey] = merged
    P = sim_engine.SimParams(cfg, preset["HTF_MIN_AGREEMENT"], conf["trailing"], allow_long=True,
                             allow_short=not conf["long_only"], btc_filter=conf["btc_filter"])
    res = sim_engine.run_portfolio(merged, preps, order, P, record=record)
    sb = P.start_balance
    return res, {
        "is": sim_engine.metrics(res["trades"], res["equity"], sb, None, split_ms),
        "oos": sim_engine.metrics(res["trades"], res["equity"], sb, split_ms, None),
        "full": sim_engine.metrics(res["trades"], res["equity"], sb),
    }


def run(preps, symbols, cfg, start_ms, end_ms, grid="quick", progress_cb=None, baselines=None,
        is_fraction=0.7):
    t0 = time.time()
    order = [s for s in symbols if s in preps]
    split_ms = int(start_ms + (end_ms - start_ms) * is_fraction)
    presets = cfg.STRICTNESS_PRESETS
    strict_names = list(presets) if grid == "full" else ["normal", "strict"]
    sets = strategy_sets(grid)

    # فهرست همه‌ی تنظیم‌ها (بدون تکرارهای بی‌اثر)
    confs = []
    for active, mode in sets:
        has_level = any(s in LEVEL_STRATEGIES for s in active)
        for st in strict_names:
            for trailing, min_sl, room, btc, long_only in itertools.product(
                    (False, True), (False, True), (False, True), (False, True), (False, True)):
                if room and not has_level:
                    continue
                confs.append({"active_strategies": list(active), "combine_mode": mode, "strictness": st,
                              "trailing": trailing, "min_sl": min_sl, "room": room, "btc_filter": btc,
                              "long_only": long_only})
    # مرتب‌سازی بر اساس «مجموعه‌ی کاندید مشترک» تا هر مجموعه فقط یک‌بار ساخته بشه و
    # کش مسیر معاملات بعد از هر گروه آزاد بشه (مصرف رم محدود می‌مونه)
    def group_key(c):
        rr = presets[c["strictness"]]["MIN_RISK_REWARD"]
        return (rr, c["min_sl"], c["room"], tuple(c["active_strategies"]), c["combine_mode"])
    confs.sort(key=lambda c: (group_key(c), c["strictness"], c["trailing"], c["btc_filter"], c["long_only"]))
    total = len(confs)
    results = []
    last_report = 0.0
    merged_cache = {}
    current_group = None
    for i, conf in enumerate(confs):
        g = group_key(conf)
        if g != current_group:
            variant_changed = current_group is None or g[:3] != current_group[:3]
            current_group = g
            merged_cache.clear()
            for p in preps.values():
                p.paths.cache.clear()
                if variant_changed:
                    p._finals.clear()
        _, m = evaluate(preps, order, cfg, conf, split_ms, merged_cache=merged_cache)
        results.append({"config": conf, **m})
        now = time.time()
        if progress_cb and (now - last_report > 0.7 or i == total - 1):
            last_report = now
            el = now - t0
            eta = el / (i + 1) * (total - i - 1)
            progress_cb(f"مقایسه {i + 1}/{total} تنظیم — باقی‌مونده حدود {int(eta)} ثانیه", (i + 1) / total)
    merged_cache.clear()
    for p in preps.values():
        p._finals.clear()

    min_is = max(30, int(40 * (split_ms - start_ms) / (365 * 86_400_000 * 0.7)))
    min_oos = max(12, min_is // 3)
    for r in results:
        r["eligible"] = r["is"]["trades"] >= min_is and r["oos"]["trades"] >= min_oos
        r["score"] = r["is"]["r_lcb"] if r["eligible"] else -99.0
        o = r["oos"]
        r["oos_ok"] = o["trades"] >= min_oos and o["avg_r"] > 0 and o["profit_factor"] > 1.05 and o["return_pct"] > 0
        r["label"] = describe(r["config"])

    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    best = ranked[0] if ranked and ranked[0]["eligible"] else None
    fallback = next((r for r in ranked[:15] if r["eligible"] and r["oos_ok"]), None) if best and not best["oos_ok"] else None

    baseline_rows = []
    for name, conf in (baselines or []):
        try:
            _, m = evaluate(preps, order, cfg, conf, split_ms)
            baseline_rows.append({"name": name, "config": conf, "label": describe(conf), **m})
        except Exception as e:
            baseline_rows.append({"name": name, "config": conf, "error": str(e)})

    report = {
        "meta": {
            "grid": grid, "n_configs": total, "symbols": order, "start_ms": start_ms, "end_ms": end_ms,
            "split_ms": split_ms, "elapsed_sec": round(time.time() - t0, 1), "min_is_trades": min_is,
            "min_oos_trades": min_oos, "eligible": sum(1 for r in results if r["eligible"]),
        },
        "recommendation": _recommendation(best, fallback, total),
        "top": [_slim(r) for r in ranked[:30]],
        "baselines": baseline_rows,
        "improvements": _paired_effects(results),
        "strategy_summary": _strategy_summary(results),
    }
    return report


def _slim(r):
    return {k: r[k] for k in ("config", "label", "is", "oos", "full", "eligible", "score", "oos_ok")}


def _recommendation(best, fallback, n_configs):
    if best is None:
        return {"status": "none", "text": "هیچ تنظیمی تعداد معامله‌ی کافی برای قضاوت آماری نداشت. "
                                           "بازه‌ی زمانی یا تعداد نمادها رو بیشتر کن."}
    rec = {"best": _slim(best), "n_configs": n_configs}
    o = best["oos"]
    if best["oos_ok"]:
        rec["status"] = "robust"
        rec["text"] = (f"برنده‌ی بخش آموزش، در بخش آزمون (دیتای دیده‌نشده) هم سودده موند: "
                       f"{o['trades']} معامله، میانگین {o['avg_r']:+.2f}R، ضریب سود {o['profit_factor']}، "
                       f"بازده {o['return_pct']:+.1f}٪. این قوی‌ترین نشونه‌ایه که یک بک‌تست می‌تونه بده؛ "
                       f"قدم بعدی: همین تنظیم رو روی ربات زنده (مجازی) حداقل ۱۰۰ معامله اجرا کن.")
    else:
        rec["status"] = "failed_oos"
        rec["text"] = (f"برنده‌ی بخش آموزش در بخش آزمون سودده نموند ({o['trades']} معامله، "
                       f"میانگین {o['avg_r']:+.2f}R). یعنی نتیجه‌ی خوبش احتمالاً شانسی/بیش‌برازش بوده. "
                       f"با {n_configs} تنظیم آزمایش‌شده، این اتفاق طبیعیه و دقیقاً دلیل وجود بخش آزمونه.")
        if fallback:
            rec["fallback"] = _slim(fallback)
            rec["text"] += " یک گزینه‌ی دیگه از ۱۵ تای اول، در هر دو بخش مثبت بوده (پایین‌تر آمده) — ولی چون با نگاه به بخش آزمون انتخاب شده، باید با احتیاط بیشتر و زمان تست زنده‌ی طولانی‌تر بررسی بشه."
        else:
            rec["text"] += " هیچ‌کدوم از ۱۵ تنظیم برتر هم در بخش آزمون سودده نبود — با این دیتا، لبه‌ی معاملاتی قابل‌اتکایی پیدا نشد. پیشنهاد: فعلاً با پول واقعی وارد نشو."
    return rec


def _paired_effects(results):
    """اثر هر بهبود با مقایسه‌ی جفتی (همه‌چیز ثابت، فقط همون یک مورد فرق داره)."""
    keyf = lambda c, drop: tuple((k, tuple(v) if isinstance(v, list) else v)
                                 for k, v in sorted(c.items()) if k != drop)
    out = []
    dims = [("trailing", "تریلینگ استاپ (در برابر TP ثابت)"),
            ("min_sl", "حداقل فاصله‌ی حد ضرر"),
            ("room", "شرط فضای کافی تا هدف"),
            ("btc_filter", "فیلتر روند بیت‌کوین"),
            ("long_only", "فقط خرید (حذف فروش)")]
    for dim, label in dims:
        on, off = {}, {}
        for r in results:
            k = keyf(r["config"], dim)
            (on if r["config"][dim] else off)[k] = r
        deltas_full, deltas_oos, pnl_deltas = [], [], []
        for k, r_on in on.items():
            r_off = off.get(k)
            if not r_off or r_on["full"]["trades"] < 10 or r_off["full"]["trades"] < 10:
                continue
            deltas_full.append(r_on["full"]["avg_r"] - r_off["full"]["avg_r"])
            deltas_oos.append(r_on["oos"]["avg_r"] - r_off["oos"]["avg_r"])
            pnl_deltas.append(r_on["full"]["return_pct"] - r_off["full"]["return_pct"])
        if not deltas_full:
            continue
        out.append({
            "dimension": dim, "label": label, "pairs": len(deltas_full),
            "improved_pct": round(sum(1 for d in deltas_full if d > 0) / len(deltas_full) * 100, 1),
            "median_delta_r": round(float(np.median(deltas_full)), 4),
            "median_delta_r_oos": round(float(np.median(deltas_oos)), 4),
            "median_delta_return_pct": round(float(np.median(pnl_deltas)), 2),
        })
    # سطح سخت‌گیری: هر سطح در برابر «سخت‌گیر»
    by_key = defaultdict(dict)
    for r in results:
        by_key[keyf(r["config"], "strictness")][r["config"]["strictness"]] = r
    levels = sorted({r["config"]["strictness"] for r in results})
    for lv in levels:
        if lv == "strict":
            continue
        d, d_oos, dp = [], [], []
        for group in by_key.values():
            a, b = group.get(lv), group.get("strict")
            if not a or not b or a["full"]["trades"] < 10 or b["full"]["trades"] < 10:
                continue
            d.append(a["full"]["avg_r"] - b["full"]["avg_r"])
            d_oos.append(a["oos"]["avg_r"] - b["oos"]["avg_r"])
            dp.append(a["full"]["return_pct"] - b["full"]["return_pct"])
        if d:
            out.append({"dimension": f"strictness:{lv}", "label": f"سخت‌گیری «{lv}» به‌جای «strict»",
                        "pairs": len(d), "improved_pct": round(sum(1 for x in d if x > 0) / len(d) * 100, 1),
                        "median_delta_r": round(float(np.median(d)), 4),
                        "median_delta_r_oos": round(float(np.median(d_oos)), 4),
                        "median_delta_return_pct": round(float(np.median(dp)), 2)})
    return out


def _strategy_summary(results):
    """بهترین نتیجه‌ی هر ترکیب استراتژی (برای دیدن این‌که کدوم استراتژی اصلاً لبه داره)."""
    groups = defaultdict(list)
    for r in results:
        c = r["config"]
        groups[(tuple(c["active_strategies"]), c["combine_mode"])].append(r)
    rows = []
    for (active, mode), rs in groups.items():
        elig = [r for r in rs if r["eligible"]]
        best = max(elig, key=lambda r: r["score"]) if elig else max(rs, key=lambda r: r["full"]["avg_r"])
        pos_share = sum(1 for r in rs if r["full"]["avg_r"] > 0 and r["full"]["trades"] >= 10) / len(rs) * 100
        rows.append({
            "active_strategies": list(active), "combine_mode": mode,
            "label": describe({**best["config"]}).split(" | ")[0],
            "configs": len(rs), "positive_share_pct": round(pos_share, 1),
            "best_label": describe(best["config"]), "best_is": best["is"], "best_oos": best["oos"],
            "best_full": best["full"], "eligible": bool(elig),
        })
    rows.sort(key=lambda x: (x["eligible"], x["best_is"]["r_lcb"]), reverse=True)
    return rows
