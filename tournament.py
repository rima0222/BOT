# -*- coding: utf-8 -*-
"""
مقایسه‌ی خودکار استراتژی‌ها («تورنمنت») روی دیتای تاریخی، در چند تایم‌فریم.

روش (سخت‌گیرانه، برای جلوگیری از خودفریبی):
  - بازه‌ی هر تایم‌فریم به دو بخش تقسیم می‌شه: ۷۰٪ اول «آموزش» (IS) و ۳۰٪ آخر «آزمون» (OOS).
    «انتخاب» فقط با بخش آموزش انجام می‌شه؛ بخش آزمون دیتاییه که انتخاب هیچ‌وقت ندیده.
  - معیار رتبه: «کران پایین ۹۵٪ میانگین R هر معامله بعد از همه‌ی هزینه‌ها» — کیفیت،
    تعداد نمونه و ثبات رو با هم می‌سنجه؛ چند معامله‌ی شانسی امتیاز بالا نمی‌گیره.
  - دو استراتژی (شکست باکس، ترکیبی وزن‌دار) و برای هر کدوم: تایید HTF (وزن‌دار + ارز/BTC)،
    پروفایل تریلینگ (خاموش/حساس/متعادل/پلکانی)، حداقل فاصله‌ی SL، فضای تا هدف، فیلتر BTC،
    فقط خرید (و برای ترکیبی وزن‌دار: حداقل امتیاز ورود) — در هر تایم‌فریم.
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

STRATEGIES = ["box_breakout", "weighted_confluence"]
STRATEGY_LABELS = {"box_breakout": "شکست باکس", "weighted_confluence": "ترکیبی وزن‌دار"}
TRAIL_FA = {"tight": "تریلینگ حساس", "balanced": "تریلینگ متعادل", "loose": "تریلینگ پلکانی"}
TF_LABELS = {"1m": "۱ دقیقه (اسکلپ)", "5m": "۵ دقیقه (اسکلپ)", "15m": "۱۵ دقیقه", "1h": "۱ ساعته", "4h": "۴ ساعته"}
STRICT_FA = {"loose": "سبک‌گیر", "normal": "معمولی", "strict": "سخت‌گیر", "very_strict": "خیلی سخت‌گیر"}
FLAG_DIMS = ("htf", "min_sl", "room", "btc_filter", "long_only")


def _strat(conf):
    act = conf.get("active_strategies") or ["weighted_confluence"]
    return act[0]


def _trail_txt(tr):
    if not tr:
        return "بدون تریلینگ"
    if isinstance(tr, str):
        return TRAIL_FA.get(tr, tr)
    return "تریلینگ"


def describe(conf):
    if conf.get("htf", True):
        htf_txt = f"تایید HTF: {STRICT_FA.get(conf['strictness'], conf['strictness'])}"
    else:
        htf_txt = "بدون تایید HTF"
    st = _strat(conf)
    parts = [TF_LABELS.get(conf.get("timeframe"), conf.get("timeframe", "")), STRATEGY_LABELS.get(st, st)]
    if st == "weighted_confluence":
        parts.append(f"امتیاز ≥ {conf.get('min_score') or 70:g}")
    parts += [htf_txt, _trail_txt(conf["trailing"])]
    if conf["min_sl"]:
        parts.append("حداقل فاصله‌ی SL")
    if conf["room"]:
        parts.append("فضای تا هدف")
    if conf["btc_filter"]:
        parts.append("فیلتر BTC")
    if conf["long_only"]:
        parts.append("فقط خرید")
    return " | ".join(p for p in parts if p)


def _variant(cfg, min_rr, min_sl, room, min_score):
    import analysis
    return {"min_rr": float(min_rr),
            "min_sl_pct": float(cfg.TEST_MIN_SL_PCT) if min_sl else 0.0,
            "min_sl_atr": float(cfg.TEST_MIN_SL_ATR_MULT) if min_sl else 0.0,
            "room": bool(room), "min_score": float(min_score if min_score is not None else 0.0),
            "net": analysis.cfg_net_fees(cfg)}


def evaluate(preps, order, cfg, conf, split_ms, record=False, merged_cache=None):
    """اجرای یک تنظیم (روی cfg همون تایم‌فریم) و معیارها برای IS / OOS / کل."""
    preset = cfg.STRICTNESS_PRESETS[conf["strictness"]]
    ms = conf.get("min_score")
    variant = _variant(cfg, preset["MIN_RISK_REWARD"], conf["min_sl"], conf["room"],
                       cfg.WC_MIN_SCORE_PCT if ms is None else ms)
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
    P.use_htf = bool(conf.get("htf", True))
    res = sim_engine.run_portfolio(merged, preps, order, P, record=record)
    sb = P.start_balance
    return res, {
        "is": sim_engine.metrics(res["trades"], res["equity"], sb, None, split_ms),
        "oos": sim_engine.metrics(res["trades"], res["equity"], sb, split_ms, None),
        "full": sim_engine.metrics(res["trades"], res["equity"], sb),
    }


def _conf(tf, strategy, min_score, strictness, htf, trailing, min_sl, room, btc, long_only):
    """trailing: False یا اسم پروفایل تریلینگ. min_score فقط برای ترکیبی وزن‌دار (بقیه None)."""
    return {"timeframe": tf, "active_strategies": [strategy], "combine_mode": "any",
            "min_score": float(min_score) if min_score is not None else None, "strictness": strictness,
            "htf": bool(htf), "trailing": trailing,
            "min_sl": min_sl, "room": room, "btc_filter": btc, "long_only": long_only}


def _conf_key(c):
    return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(c.items()))


def _run_confs(preps, order, cfg, confs, split_ms, progress, done_offset, total_hint):
    """اجرای یک لیست تنظیم با مرتب‌سازی برای بیشترین استفاده‌ی مجدد از محاسبات و کنترل رم."""
    presets = cfg.STRICTNESS_PRESETS

    def group_key(c):
        rr = presets[c["strictness"]]["MIN_RISK_REWARD"]
        return (rr, c["min_sl"], c["room"], c.get("min_score") or 0.0)

    confs = sorted(confs, key=lambda c: (group_key(c), _strat(c), c["strictness"], str(c["trailing"]),
                                         c["btc_filter"], c["long_only"]))
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
    """همه‌ی تنظیم‌ها برای یک تایم‌فریم. خروجی: (ردیف‌های نتیجه، متادیتا، ردیف‌های مبنا)"""
    t0 = time.time()
    order = [s for s in symbols if s in preps]
    split_ms = int(start_ms + (end_ms - start_ms) * is_fraction)
    scores = [50, 60, 70, 80, 90] if grid == "full" else [70, 80]
    htf_modes = ([(True, x) for x in cfg.STRICTNESS_PRESETS] if grid == "full"
                 else [(True, "loose"), (True, "normal")]) + [(False, "normal")]
    trails = [False] + list(getattr(cfg, "TRAIL_PROFILES", {}) or [True])
    confs = []
    for strat in STRATEGIES:
        for sc in (scores if strat == "weighted_confluence" else [None]):
            for htf, st in htf_modes:
                for trailing in trails:
                    for min_sl, room, btc, long_only in itertools.product((False, True), repeat=4):
                        confs.append(_conf(tf, strat, sc, st, htf, trailing, min_sl, room, btc, long_only))
    total = len(confs)
    last = [0.0]

    def prog(i, tot):
        now = time.time()
        if progress_cb and (now - last[0] > 0.7 or i >= tot):
            last[0] = now
            eta = (now - t0) / max(1, i) * max(0, tot - i)
            progress_cb(f"{TF_LABELS.get(tf, tf)}: تنظیم {i}/{tot} — باقی‌مونده حدود {int(eta)} ثانیه",
                        min(1.0, i / max(1, tot)))

    results = _run_confs(preps, order, cfg, confs, split_ms, prog, 0, total)
    min_is, min_oos = _score(results, split_ms, start_ms)

    base_rows = []
    for name, conf in (baselines or []):
        if conf.get("timeframe", "15m") != tf:
            continue
        try:
            conf = {"min_score": cfg.WC_MIN_SCORE_PCT, **conf}
            if _strat(conf) != "weighted_confluence":
                conf["min_score"] = None
            _, m = evaluate(preps, order, cfg, conf, split_ms)
            row = {"name": name, "config": conf, **m}
            _score([row], split_ms, start_ms)
            base_rows.append(row)
        except Exception as e:
            base_rows.append({"name": name, "config": conf, "error": str(e)})

    meta = {"timeframe": tf, "symbols": order, "start_ms": start_ms, "end_ms": end_ms, "split_ms": split_ms,
            "n_configs": len(results), "min_is_trades": min_is, "min_oos_trades": min_oos,
            "eligible": sum(1 for r in results if r["eligible"]), "elapsed_sec": round(time.time() - t0, 1)}
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
    fields = ["timeframe", "strategy", "min_score", "strictness", "trailing", *FLAG_DIMS, "eligible", "oos_ok",
              "rank_score"]
    segs = ("is", "oos", "full")
    mets = ("trades", "wins", "losses", "win_rate", "trail_trades", "trail_pnl", "trail_pct", "avg_r", "r_lcb",
            "profit_factor", "return_pct", "max_dd_pct", "pos_months_pct", "fees", "long_trades", "short_trades",
            "avg_bars")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(fields + [f"{s}_{m}" for s in segs for m in mets])
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            c = r["config"]
            row = [c["timeframe"], _strat(c), c.get("min_score"), c["strictness"], c["trailing"] or "off",
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
    dims = [("htf", "تایید تایم‌فریم بالاتر (HTF) در برابر بدون آن"),
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

    # هر پروفایل تریلینگ در برابر بدون تریلینگ
    by_tr = defaultdict(dict)
    for r in results:
        by_tr[keyf(r["config"], "trailing")][r["config"]["trailing"] or False] = r
    profiles = sorted({r["config"]["trailing"] for r in results if isinstance(r["config"]["trailing"], str)})
    for prof in profiles:
        pairs = []
        for g in by_tr.values():
            a, b = g.get(prof), g.get(False)
            if a and b and a["full"]["trades"] >= 10 and b["full"]["trades"] >= 10:
                pairs.append((a, b))
        if pairs:
            out.append(summarize(f"trailing:{prof}", f"{TRAIL_FA.get(prof, prof)} در برابر بدون تریلینگ", pairs))

    # شکست باکس در برابر ترکیبی وزن‌دار (امتیاز ۷۰)، با بقیه‌ی تنظیمات یکسان
    def keys(c):
        return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(c.items())
                     if k not in ("active_strategies", "min_score"))
    brk, wc = {}, {}
    for r in results:
        c = r["config"]
        if _strat(c) == "box_breakout":
            brk[keys(c)] = r
        elif c.get("min_score") == 70.0:
            wc[keys(c)] = r
    pairs = [(a, wc[k]) for k, a in brk.items()
             if k in wc and a["full"]["trades"] >= 10 and wc[k]["full"]["trades"] >= 10]
    if pairs:
        out.append(summarize("strategy:box_breakout", "شکست باکس در برابر ترکیبی وزن‌دار (امتیاز ۷۰)", pairs))

    by_sc = defaultdict(dict)
    for r in results:
        if _strat(r["config"]) != "weighted_confluence":
            continue
        by_sc[keyf(r["config"], "min_score")][r["config"].get("min_score")] = r
    for sc in (50.0, 60.0, 80.0, 90.0):
        pairs = []
        for g in by_sc.values():
            a, b = g.get(sc), g.get(70.0)
            if a and b and a["full"]["trades"] >= 10 and b["full"]["trades"] >= 10:
                pairs.append((a, b))
        if pairs:
            out.append(summarize(f"min_score:{sc:g}", f"حداقل امتیاز {sc:g} به‌جای ۷۰", pairs))

    by_key = defaultdict(dict)
    for r in results:
        if not r["config"].get("htf", True):
            continue   # سطح سخت‌گیری بیشتر به تایید HTF مربوطه؛ فقط وقتی روشنه مقایسه می‌شه
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
    """بهترین نتیجه برای هر (تایم‌فریم، حداقل امتیاز) — ببینیم امتیاز بالاتر واقعاً بهتره یا نه."""
    groups = defaultdict(list)
    for r in results:
        c = r["config"]
        groups[(c["timeframe"], _strat(c), c.get("min_score"))].append(r)
    rows = []
    for (tf, strat, sc), rs in groups.items():
        elig = [r for r in rs if r["eligible"]]
        best = max(elig, key=lambda r: r["score"]) if elig else max(rs, key=lambda r: r["full"]["avg_r"])
        pos_share = sum(1 for r in rs if r["full"]["avg_r"] > 0 and r["full"]["trades"] >= 10) / len(rs) * 100
        rows.append({
            "timeframe": tf, "strategy": strat, "min_score": sc,
            "label": f"{TF_LABELS.get(tf, tf)} — {STRATEGY_LABELS.get(strat, strat)}"
                     + (f" — امتیاز ≥ {sc:g}" if sc is not None else ""),
            "configs": len(rs), "positive_share_pct": round(pos_share, 1),
            "best_label": describe(best["config"]), "best_is": best["is"], "best_oos": best["oos"],
            "best_full": best["full"], "eligible": bool(elig), "best_oos_ok": best["oos_ok"],
        })
    rows.sort(key=lambda x: (x["eligible"], x["best_is"]["r_lcb"]), reverse=True)
    return rows


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
