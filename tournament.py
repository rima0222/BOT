# -*- coding: utf-8 -*-
"""
مقایسه‌ی خودکار استراتژی‌ها («تورنمنت») روی دیتای تاریخی، در چند تایم‌فریم.

روش (سخت‌گیرانه، برای جلوگیری از خودفریبی):
  - بازه‌ی هر تایم‌فریم به دو بخش تقسیم می‌شه: ۷۰٪ اول «آموزش» (IS) و ۳۰٪ آخر «آزمون» (OOS).
    «انتخاب» فقط با بخش آموزش انجام می‌شه؛ بخش آزمون دیتاییه که انتخاب هیچ‌وقت ندیده.
  - معیار رتبه: «کران پایین ۹۵٪ میانگین R هر معامله بعد از همه‌ی هزینه‌ها» — کیفیت،
    تعداد نمونه و ثبات رو با هم می‌سنجه؛ چند معامله‌ی شانسی امتیاز بالا نمی‌گیره.
  - جستجوی دو مرحله‌ای (برای این‌که با ۱۰ استراتژی و چند تایم‌فریم هم سریع بمونه):
      مرحله‌ی ۱: همه‌ی استراتژی‌های تکی و ترکیب‌های دوتایی با چند تنظیم پایه.
      مرحله‌ی ۲: بهترین ترکیب‌های مرحله‌ی ۱ با شبکه‌ی کامل تنظیمات (سخت‌گیری، تریلینگ،
                حداقل فاصله‌ی SL، فضای تا هدف، فیلتر BTC، فقط خرید).
  - اثر هر «بهبود» به‌صورت جفتی سنجیده می‌شه (دو تنظیم که فقط در همون یک مورد فرق دارن).
"""
import csv
import itertools
import time
from collections import defaultdict

import numpy as np

import fast_backtest
import signals_engine as se
import sim_engine

LEVEL_STRATEGIES = {"dow_support_resistance", "candle_setup"}   # فیلتر «فضای تا هدف» فقط روی این‌ها اثر داره

STRATEGY_LABELS = {
    "dow_support_resistance": "داو+حمایت/مقاومت", "breakout": "بریک‌اوت", "volume_spike": "افزایش حجم",
    "candle_setup": "کندل‌ستاپ", "ema_cross": "کراس EMA", "ema_pullback": "پولبک EMA",
    "bb_reversion": "بازگشت بولینگر", "rsi_pullback": "پولبک RSI", "squeeze_breakout": "شکست Squeeze",
    "donchian_trend": "دانچیان (لاک‌پشت)",
}
TF_LABELS = {"1m": "۱ دقیقه (اسکلپ)", "5m": "۵ دقیقه (اسکلپ)", "15m": "۱۵ دقیقه", "1h": "۱ ساعته", "4h": "۴ ساعته"}
STRICT_FA = {"loose": "سبک‌گیر", "normal": "معمولی", "strict": "سخت‌گیر", "very_strict": "خیلی سخت‌گیر"}
FLAG_DIMS = ("trailing", "min_sl", "room", "btc_filter", "long_only")


def stage1_sets(grid):
    names = list(se.STRATEGY_NAMES)
    sets = [((s,), "any") for s in names]
    sets += [(pair, "any") for pair in itertools.combinations(names, 2)]
    if grid == "full":
        sets += [((a, b), "confirm") for a in names for b in names if a != b]
    return sets


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
    parts = [TF_LABELS.get(conf.get("timeframe"), conf.get("timeframe", "")), strat,
             f"سخت‌گیری: {STRICT_FA.get(conf['strictness'], conf['strictness'])}",
             "تریلینگ" if conf["trailing"] else "TP ثابت"]
    if conf["min_sl"]:
        parts.append("حداقل فاصله‌ی SL")
    if conf["room"]:
        parts.append("فضای تا هدف")
    if conf["btc_filter"]:
        parts.append("فیلتر BTC")
    if conf["long_only"]:
        parts.append("فقط خرید")
    return " | ".join(p for p in parts if p)


def _variant(cfg, min_rr, min_sl, room):
    return {"min_rr": float(min_rr),
            "min_sl_pct": float(cfg.TEST_MIN_SL_PCT) if min_sl else 0.0,
            "min_sl_atr": float(cfg.TEST_MIN_SL_ATR_MULT) if min_sl else 0.0,
            "room": bool(room)}


def evaluate(preps, order, cfg, conf, split_ms, record=False, merged_cache=None):
    """اجرای یک تنظیم (روی cfg همون تایم‌فریم) و معیارها برای IS / OOS / کل."""
    preset = cfg.STRICTNESS_PRESETS[conf["strictness"]]
    variant = _variant(cfg, preset["MIN_RISK_REWARD"], conf["min_sl"], conf["room"])
    mkey = (tuple(conf["active_strategies"]), conf["combine_mode"], tuple(sorted(variant.items())))
    merged = merged_cache.get(mkey) if merged_cache is not None else None
    if merged is None:
        cands = fast_backtest.candidates_for(preps, list(conf["active_strategies"]), conf["combine_mode"],
                                             variant, int(cfg.CONFIRM_LOOKBACK_BARS))
        merged = sim_engine.merge_candidates(cands, preps, order)
        if merged_cache is not None:
            if len(merged_cache) > 20:
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


def _conf(tf, active, mode, strictness, trailing, min_sl, room, btc, long_only):
    return {"timeframe": tf, "active_strategies": list(active), "combine_mode": mode, "strictness": strictness,
            "trailing": trailing, "min_sl": min_sl, "room": room, "btc_filter": btc, "long_only": long_only}


def _conf_key(c):
    return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(c.items()))


def _run_confs(preps, order, cfg, confs, split_ms, progress, done_offset, total_hint):
    """اجرای یک لیست تنظیم با مرتب‌سازی برای بیشترین استفاده‌ی مجدد از محاسبات و کنترل رم."""
    presets = cfg.STRICTNESS_PRESETS

    def group_key(c):
        rr = presets[c["strictness"]]["MIN_RISK_REWARD"]
        return (rr, c["min_sl"], c["room"], tuple(c["active_strategies"]), c["combine_mode"])

    confs = sorted(confs, key=lambda c: (group_key(c), c["strictness"], c["trailing"], c["btc_filter"],
                                         c["long_only"]))
    out = []
    merged_cache = {}
    current = None
    for i, conf in enumerate(confs):
        g = group_key(conf)
        if g != current:
            variant_changed = current is None or g[:3] != current[:3]
            current = g
            merged_cache.clear()
            for p in preps.values():
                p.paths.cache.clear()
                if variant_changed:
                    p._finals.clear()
        _, m = evaluate(preps, order, cfg, conf, split_ms, merged_cache=merged_cache)
        out.append({"config": conf, **m})
        if progress:
            progress(done_offset + i + 1, total_hint)
    merged_cache.clear()
    for p in preps.values():
        p._finals.clear()
        p.paths.cache.clear()
    return out


def _score(results, split_ms, start_ms):
    min_is = max(30, int(40 * (split_ms - start_ms) / (365 * 86_400_000 * 0.7)))
    min_oos = max(12, min_is // 3)
    for r in results:
        r["eligible"] = r["is"]["trades"] >= min_is and r["oos"]["trades"] >= min_oos
        r["score"] = r["is"]["r_lcb"] if r["eligible"] else -99.0
        o = r["oos"]
        r["oos_ok"] = bool(o["trades"] >= min_oos and o["avg_r"] > 0 and o["profit_factor"] > 1.05
                           and o["return_pct"] > 0)
        r["label"] = describe(r["config"])
    return min_is, min_oos


def run_timeframe(preps, symbols, cfg, tf, start_ms, end_ms, grid="quick", progress_cb=None,
                  baselines=None, is_fraction=0.7):
    """همه‌ی مراحل مقایسه برای یک تایم‌فریم. خروجی: (ردیف‌های نتیجه، متادیتا، ردیف‌های مبنا)"""
    t0 = time.time()
    order = [s for s in symbols if s in preps]
    split_ms = int(start_ms + (end_ms - start_ms) * is_fraction)
    s1_strict = ["normal", "strict"]
    s2_strict = list(cfg.STRICTNESS_PRESETS) if grid == "full" else ["normal", "strict"]
    top_k = 12 if grid == "full" else 6
    sets = stage1_sets(grid)

    s1 = []
    for active, mode in sets:
        for st in s1_strict:
            for trailing in (False, True):
                for min_sl in ((False, True) if grid == "full" else (True,)):
                    s1.append(_conf(tf, active, mode, st, trailing, min_sl, False, False, False))
    est_total = len(s1) + top_k * len(s2_strict) * 32
    last = [0.0]

    def prog(i, total):
        now = time.time()
        if progress_cb and (now - last[0] > 0.7 or i >= total):
            last[0] = now
            el = now - t0
            eta = el / max(1, i) * max(0, total - i)
            progress_cb(f"{TF_LABELS.get(tf, tf)}: تنظیم {i}/{total} — باقی‌مونده حدود {int(eta)} ثانیه",
                        min(1.0, i / max(1, total)))

    res1 = _run_confs(preps, order, cfg, s1, split_ms, prog, 0, est_total)
    min_is, min_oos = _score(res1, split_ms, start_ms)

    # بهترین ترکیب‌های استراتژی مرحله‌ی ۱
    best_by_set = {}
    for r in res1:
        k = (tuple(r["config"]["active_strategies"]), r["config"]["combine_mode"])
        val = (r["eligible"], r["score"] if r["eligible"] else r["full"]["avg_r"])
        if k not in best_by_set or val > best_by_set[k]:
            best_by_set[k] = val
    top_sets = [k for k, _ in sorted(best_by_set.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    seen = {_conf_key(r["config"]) for r in res1}
    s2 = []
    for active, mode in top_sets:
        has_level = any(x in LEVEL_STRATEGIES for x in active)
        for st in s2_strict:
            for trailing, min_sl, room, btc, long_only in itertools.product((False, True), repeat=5):
                if room and not has_level:
                    continue
                c = _conf(tf, active, mode, st, trailing, min_sl, room, btc, long_only)
                if _conf_key(c) not in seen:
                    seen.add(_conf_key(c))
                    s2.append(c)
    total = len(s1) + len(s2)
    res2 = _run_confs(preps, order, cfg, s2, split_ms, prog, len(s1), total)
    results = res1 + res2
    _score(results, split_ms, start_ms)

    base_rows = []
    for name, conf in (baselines or []):
        if conf.get("timeframe", "15m") != tf:
            continue
        try:
            _, m = evaluate(preps, order, cfg, conf, split_ms)
            row = {"name": name, "config": conf, **m}
            _score([row], split_ms, start_ms)
            base_rows.append(row)
        except Exception as e:
            base_rows.append({"name": name, "config": conf, "error": str(e)})

    meta = {"timeframe": tf, "symbols": order, "start_ms": start_ms, "end_ms": end_ms, "split_ms": split_ms,
            "n_configs": len(results), "stage1": len(s1), "stage2": len(s2),
            "top_sets": [list(a) + [m] for a, m in top_sets], "min_is_trades": min_is,
            "min_oos_trades": min_oos, "eligible": sum(1 for r in results if r["eligible"]),
            "elapsed_sec": round(time.time() - t0, 1)}
    return results, meta, base_rows


def build_report(results, tf_meta, baseline_rows, grid):
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    best = ranked[0] if ranked and ranked[0]["eligible"] else None
    fallback = None
    if best and not best["oos_ok"]:
        fallback = next((r for r in ranked[:20] if r["eligible"] and r["oos_ok"]), None)
    total = len(results)
    return {
        "meta": {"grid": grid, "n_configs": total, "timeframes": tf_meta,
                 "eligible": sum(1 for r in results if r["eligible"])},
        "recommendation": _recommendation(best, fallback, total),
        "top": [_slim(r) for r in ranked[:40]],
        "baselines": [{**_slim_base(b)} for b in baseline_rows],
        "improvements": _paired_effects(results),
        "strategy_summary": _strategy_summary(results),
        "timeframe_summary": _timeframe_summary(results, tf_meta),
    }


def write_results_csv(results, path):
    """همه‌ی نتایج (هر تنظیم یک ردیف) برای بررسی دقیق‌تر در اکسل یا ارسال برای تحلیل."""
    fields = ["timeframe", "strategies", "combine_mode", "strictness", *FLAG_DIMS, "eligible", "oos_ok", "score"]
    segs = ("is", "oos", "full")
    mets = ("trades", "win_rate", "avg_r", "r_lcb", "profit_factor", "return_pct", "max_dd_pct",
            "pos_months_pct", "fees", "long_trades", "short_trades", "avg_bars")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(fields + [f"{s}_{m}" for s in segs for m in mets])
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            c = r["config"]
            row = [c["timeframe"], "+".join(c["active_strategies"]), c["combine_mode"], c["strictness"],
                   *[int(bool(c[d])) for d in FLAG_DIMS], int(r["eligible"]), int(r["oos_ok"]),
                   round(r["score"], 4)]
            for s in segs:
                row += [r[s].get(m) for m in mets]
            w.writerow(row)


def _slim(r):
    return {k: r[k] for k in ("config", "label", "is", "oos", "full", "eligible", "score", "oos_ok")}


def _slim_base(b):
    if "error" in b:
        return {"name": b["name"], "config": b["config"], "label": describe(b["config"]), "error": b["error"]}
    return {"name": b["name"], **_slim(b)}


def _recommendation(best, fallback, n_configs):
    if best is None:
        return {"status": "none", "text": "هیچ تنظیمی تعداد معامله‌ی کافی برای قضاوت آماری نداشت. "
                                           "بازه‌ی زمانی یا تعداد نمادها رو بیشتر کن."}
    rec = {"best": _slim(best), "n_configs": n_configs}
    o = best["oos"]
    if best["oos_ok"]:
        rec["status"] = "robust"
        rec["text"] = (f"برنده‌ی بخش آموزش، در بخش آزمون (دیتای دیده‌نشده) هم سودده موند: "
                       f"{o['trades']} معامله، میانگین {o['avg_r']:+.2f}R بعد از همه‌ی هزینه‌ها، "
                       f"ضریب سود {o['profit_factor']}، بازده {o['return_pct']:+.1f}٪. این قوی‌ترین نشونه‌ایه "
                       f"که یک بک‌تست می‌تونه بده؛ قدم بعدی: همین تنظیم رو روی ربات زنده (مجازی) حداقل "
                       f"۱۰۰ معامله اجرا کن و نتیجه‌ش رو با همین عددها مقایسه کن.")
    else:
        rec["status"] = "failed_oos"
        rec["text"] = (f"برنده‌ی بخش آموزش در بخش آزمون سودده نموند ({o['trades']} معامله، "
                       f"میانگین {o['avg_r']:+.2f}R). یعنی نتیجه‌ی خوبش احتمالاً شانسی/بیش‌برازش بوده. "
                       f"با {n_configs} تنظیم آزمایش‌شده، این اتفاق طبیعیه و دقیقاً دلیل وجود بخش آزمونه.")
        if fallback:
            rec["fallback"] = _slim(fallback)
            rec["text"] += (" یک گزینه‌ی دیگه از ۲۰ تای اول، در هر دو بخش مثبت بوده (پایین‌تر آمده) — ولی چون "
                            "با نگاه به بخش آزمون انتخاب شده، باید با احتیاط بیشتر و تست زنده‌ی طولانی‌تر بررسی بشه.")
        else:
            rec["text"] += (" هیچ‌کدوم از ۲۰ تنظیم برتر هم در بخش آزمون سودده نبود — با این دیتا، لبه‌ی معاملاتی "
                            "قابل‌اتکایی پیدا نشد. پیشنهاد: فعلاً با پول واقعی وارد نشو.")
    return rec


def _paired_effects(results):
    """اثر هر بهبود با مقایسه‌ی جفتی (همه‌چیز ثابت، فقط همون یک مورد فرق داره)."""
    def keyf(c, drop):
        return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(c.items()) if k != drop)

    out = []
    dims = [("trailing", "تریلینگ استاپ (در برابر TP ثابت)"),
            ("min_sl", "حداقل فاصله‌ی حد ضرر"),
            ("room", "شرط فضای کافی تا هدف"),
            ("btc_filter", "فیلتر روند بیت‌کوین"),
            ("long_only", "فقط خرید (حذف فروش)")]

    def summarize(dim, label, pairs):
        d = [a["full"]["avg_r"] - b["full"]["avg_r"] for a, b in pairs]
        d_oos = [a["oos"]["avg_r"] - b["oos"]["avg_r"] for a, b in pairs]
        dp = [a["full"]["return_pct"] - b["full"]["return_pct"] for a, b in pairs]
        return {"dimension": dim, "label": label, "pairs": len(d),
                "improved_pct": round(sum(1 for x in d if x > 0) / len(d) * 100, 1),
                "median_delta_r": round(float(np.median(d)), 4),
                "median_delta_r_oos": round(float(np.median(d_oos)), 4),
                "median_delta_return_pct": round(float(np.median(dp)), 2)}

    for dim, label in dims:
        on, off = {}, {}
        for r in results:
            (on if r["config"][dim] else off)[keyf(r["config"], dim)] = r
        pairs = [(a, off[k]) for k, a in on.items()
                 if k in off and a["full"]["trades"] >= 10 and off[k]["full"]["trades"] >= 10]
        if pairs:
            out.append(summarize(dim, label, pairs))

    by_key = defaultdict(dict)
    for r in results:
        by_key[keyf(r["config"], "strictness")][r["config"]["strictness"]] = r
    for lv in ("loose", "normal", "very_strict"):
        pairs = []
        for g in by_key.values():
            a, b = g.get(lv), g.get("strict")
            if a and b and a["full"]["trades"] >= 10 and b["full"]["trades"] >= 10:
                pairs.append((a, b))
        if pairs:
            out.append(summarize(f"strictness:{lv}", f"سخت‌گیری «{STRICT_FA[lv]}» به‌جای «سخت‌گیر»", pairs))
    return out


def _strategy_summary(results):
    """بهترین نتیجه‌ی هر ترکیب استراتژی در هر تایم‌فریم (کدوم استراتژی اصلاً لبه داره)."""
    groups = defaultdict(list)
    for r in results:
        c = r["config"]
        groups[(c["timeframe"], tuple(c["active_strategies"]), c["combine_mode"])].append(r)
    rows = []
    for (tf, active, mode), rs in groups.items():
        elig = [r for r in rs if r["eligible"]]
        best = max(elig, key=lambda r: r["score"]) if elig else max(rs, key=lambda r: r["full"]["avg_r"])
        pos_share = sum(1 for r in rs if r["full"]["avg_r"] > 0 and r["full"]["trades"] >= 10) / len(rs) * 100
        parts = describe(best["config"]).split(" | ")
        rows.append({
            "timeframe": tf, "active_strategies": list(active), "combine_mode": mode,
            "label": f"{parts[0]} — {parts[1]}", "configs": len(rs), "positive_share_pct": round(pos_share, 1),
            "best_label": describe(best["config"]), "best_is": best["is"], "best_oos": best["oos"],
            "best_full": best["full"], "eligible": bool(elig), "best_oos_ok": best["oos_ok"],
        })
    rows.sort(key=lambda x: (x["eligible"], x["best_is"]["r_lcb"]), reverse=True)
    return rows[:60]


def _timeframe_summary(results, tf_meta):
    rows = []
    for tf, meta in tf_meta.items():
        rs = [r for r in results if r["config"]["timeframe"] == tf]
        if not rs:
            continue
        elig = [r for r in rs if r["eligible"]]
        best = max(elig, key=lambda r: r["score"]) if elig else None
        rows.append({
            "timeframe": tf, "label": TF_LABELS.get(tf, tf), "configs": len(rs), "eligible": len(elig),
            "positive_share_pct": round(sum(1 for r in rs if r["full"]["avg_r"] > 0 and r["full"]["trades"] >= 10)
                                        / len(rs) * 100, 1),
            "oos_ok_share_pct": round(sum(1 for r in elig if r["oos_ok"]) / len(elig) * 100, 1) if elig else 0.0,
            "best_label": best["label"] if best else None,
            "best_oos": best["oos"] if best else None, "best_is": best["is"] if best else None,
            "days": meta.get("days"), "symbols": len(meta.get("symbols", [])),
        })
    return rows
