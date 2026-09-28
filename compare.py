# -*- coding: utf-8 -*-
"""
مقایسه‌ی خودکار استراتژی‌ها — هم از پنل صدا زده می‌شه (در یک پروسه‌ی جدا، با اولویت
پایین تا ربات زنده کند نشه)، هم مستقیم از ترمینال:

    cd ~/tradingbot && venv/bin/python3 compare.py --days 730 --top 20 --grid quick

خروجی: فایل reports/compare_<شناسه>.json (پنل همین رو نشون می‌ده) + خلاصه در ترمینال.
"""
import argparse
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import config  # noqa: E402
import backtest  # noqa: E402
import data_fetcher  # noqa: E402
import fast_backtest  # noqa: E402
import market_data  # noqa: E402
import tournament  # noqa: E402


class Progress:
    def __init__(self, path, job_id, params):
        self.path = path
        self.state = {"job_id": job_id, "state": "running", "progress": 0.0, "message": "شروع",
                      "started_at": datetime.utcnow().isoformat(), "pid": os.getpid(), "params": params}
        self.write()

    def write(self):
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def update(self, msg, frac=None, lo=0.0, hi=1.0):
        self.state["message"] = msg
        if frac is not None:
            self.state["progress"] = round(lo + (hi - lo) * max(0.0, min(1.0, frac)), 4)
        self.write()
        print(f"[{self.state['progress'] * 100:5.1f}%] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="مقایسه‌ی خودکار استراتژی‌ها روی دیتای تاریخی")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--top", type=int, default=20, help="تعداد نمادهای پرحجم")
    ap.add_argument("--symbols", type=str, default="", help="لیست نماد با کاما (به‌جای --top)")
    ap.add_argument("--timeframes", type=str, default="15m,1h,4h",
                    help="تایم‌فریم‌ها با کاما: 1m,5m (اسکلپ) 15m,1h,4h (نوسان‌گیری)")
    ap.add_argument("--grid", choices=["quick", "full"], default="quick")
    ap.add_argument("--job-id", type=str, default="")
    ap.add_argument("--progress-file", type=str, default="")
    ap.add_argument("--baseline-file", type=str, default="")
    ap.add_argument("--offline", action="store_true", help="فقط از دیتای کش‌شده (بدون اینترنت)")
    ap.add_argument("--data-only", action="store_true",
                    help="فقط دانلود/به‌روزرسانی دیتای تاریخی و بررسی اعتبارش (بدون بک‌تست)")
    args = ap.parse_args()

    job_id = args.job_id or uuid.uuid4().hex[:10]
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    tfs = [t.strip() for t in args.timeframes.split(",") if t.strip() in config.TIMEFRAME_PROFILES] or ["15m"]
    params = {"days": args.days, "top": args.top, "grid": args.grid, "timeframes": tfs}
    prog = Progress(args.progress_file, job_id, params)
    t0 = time.time()
    try:
        base_cfg = backtest.build_config(config)
        cache = market_data.MarketDataCache(config.DATA_CACHE_DIR, config.EXCHANGE_TRY_ORDER,
                                            config.DATA_FETCH_MAX_REQ_PER_SEC, config.DATA_FETCH_THREADS)
        if args.symbols:
            all_symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        else:
            prog.update("دریافت لیست پرحجم‌ترین نمادها", 0.0)
            try:
                all_symbols, _ = data_fetcher.get_top_symbols(config.EXCHANGE_TRY_ORDER, quote=config.QUOTE_CURRENCY,
                                                              top_n=args.top + 10,
                                                              exclude_keywords=config.EXCLUDE_KEYWORDS, cfg=config)
            except Exception:
                all_symbols = sorted({k.split("|")[0] for k in cache.index})
            all_symbols = all_symbols[:args.top]
        all_symbols = [s for s in all_symbols if data_fetcher.is_symbol_allowed(s, config)]
        if not all_symbols:
            raise RuntimeError("هیچ نمادی برای تست پیدا نشد")
        params["symbols"] = all_symbols

        baselines = []
        if args.baseline_file and os.path.exists(args.baseline_file):
            with open(args.baseline_file, "r", encoding="utf-8") as f:
                baselines = [(b["name"], {"timeframe": "15m", "htf": True, **b["config"]}) for b in json.load(f)]

        if args.data_only:
            run_data_only(args, tfs, all_symbols, base_cfg, cache, prog, job_id, t0)
            return

        all_results, tf_meta, base_rows = [], {}, []
        n_tf = len(tfs)
        for i, tf in enumerate(tfs):
            prof = config.TIMEFRAME_PROFILES[tf]
            cfg = fast_backtest.profile_cfg(base_cfg, tf)
            cfg._NEED_HTF = True   # مقایسه هر دو حالت (با و بدون تایید HTF) رو تست می‌کنه
            days = min(args.days, int(prof["MAX_DAYS"]))
            symbols = all_symbols[:min(len(all_symbols), int(prof["SCAN_SYMBOLS"]))]
            lo, span = i / n_tf, 1.0 / n_tf
            label = tournament.TF_LABELS.get(tf, tf)
            plan, data = fast_backtest.load_market_data(
                cache, symbols, days, cfg, lambda m, f: prog.update(f"{label} — {m}", f, lo, lo + span * 0.4),
                offline=args.offline)
            preps, meta = fast_backtest.prepare_all(plan, data, symbols, cfg,
                                                    lambda m, f: prog.update(f"{label} — {m}", f, lo + span * 0.4,
                                                                             lo + span * 0.5))
            del data
            if not preps:
                tf_meta[tf] = {"timeframe": tf, "days": days, "symbols": [], "error": "دیتای قابل‌استفاده نبود",
                               "data": meta}
                continue
            results, m, brows = tournament.run_timeframe(
                preps, list(preps), cfg, tf, plan["start_ms"], plan["now_ms"], grid=args.grid,
                progress_cb=lambda msg, f: prog.update(msg, f, lo + span * 0.5, lo + span),
                baselines=baselines)
            m.update({"days": days, "data": meta})
            tf_meta[tf] = m
            all_results += results
            base_rows += brows
            del preps

        if not all_results:
            raise RuntimeError("برای هیچ تایم‌فریمی نتیجه‌ای تولید نشد: " + json.dumps(tf_meta, ensure_ascii=False)[:500])

        report = tournament.build_report(all_results, tf_meta, base_rows, args.grid)
        csv_path = os.path.join(config.REPORTS_DIR, f"results_{job_id}.csv")
        tournament.write_results_csv(all_results, csv_path)
        report["meta"].update({
            "job_id": job_id, "days": args.days, "created_at": datetime.utcnow().isoformat(),
            "symbols": all_symbols, "total_elapsed_sec": round(time.time() - t0, 1),
            "cache_mb": cache.disk_usage_mb(), "results_csv": os.path.basename(csv_path),
            "fees": {"maker": base_cfg.MAKER_FEE_PCT, "taker": base_cfg.TAKER_FEE_PCT,
                     "slippage": base_cfg.TAKER_SLIPPAGE_PCT, "funding_8h": base_cfg.FUNDING_PCT_PER_8H},
            "entry_mode": base_cfg.ENTRY_MODE, "risk_pct": base_cfg.RISK_PER_TRADE_PCT,
            "test_min_sl_pct": base_cfg.TEST_MIN_SL_PCT, "test_min_sl_atr": base_cfg.TEST_MIN_SL_ATR_MULT,
        })
        out_path = os.path.join(config.REPORTS_DIR, f"compare_{job_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False)
        prog.state.update({"state": "done", "progress": 1.0, "message": "تمام شد", "report": out_path})
        prog.write()
        print_summary(report, os.path.join(config.REPORTS_DIR, f"summary_{job_id}.txt"))
    except Exception as e:
        prog.state.update({"state": "error", "message": f"خطا: {e}", "error": traceback.format_exc()[-2000:]})
        prog.write()
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


def run_data_only(args, tfs, all_symbols, base_cfg, cache, prog, job_id, t0):
    """فقط دیتا: دانلود/تکمیل کش برای همه‌ی تایم‌فریم‌ها + گزارش اعتبار هر نماد."""
    report = {"kind": "data", "job_id": job_id, "created_at": datetime.utcnow().isoformat(),
              "days": args.days, "timeframes": {}}
    n_tf = len(tfs)
    min_cov = float(getattr(base_cfg, "MIN_DATA_COVERAGE_PCT", 0) or 0)
    for i, tf in enumerate(tfs):
        prof = config.TIMEFRAME_PROFILES[tf]
        cfg = fast_backtest.profile_cfg(base_cfg, tf)
        cfg._NEED_HTF = True   # دیتای کامل، برای هر دو حالت
        days = min(args.days, int(prof["MAX_DAYS"]))
        symbols = all_symbols[:min(len(all_symbols), int(prof["SCAN_SYMBOLS"]))]
        label = tournament.TF_LABELS.get(tf, tf)
        plan, data = fast_backtest.load_market_data(
            cache, symbols, days, cfg, lambda m, f: prog.update(f"{label} — {m}", f, i / n_tf, (i + 1) / n_tf),
            offline=args.offline)
        rows = []
        for sym in symbols:
            arr = data.get((sym, tf))
            if arr is None or isinstance(arr, Exception):
                rows.append({"symbol": sym, "ok": False, "error": str(arr) if arr is not None else "دیتا نیست"})
                continue
            q = market_data.data_quality(arr, tf, plan["start_ms"], plan["now_ms"])
            htf_bad = [t for t in plan["fetched"] if isinstance(data.get((sym, t)), Exception) or data.get((sym, t)) is None]
            ok = (not min_cov or q["coverage_pct"] >= min_cov) and not htf_bad
            rows.append({"symbol": sym, "ok": ok, "quality": q, "missing_htf": htf_bad})
        report["timeframes"][tf] = {"days": days, "start_ms": plan["start_ms"], "end_ms": plan["now_ms"],
                                    "valid": sum(1 for r in rows if r["ok"]), "total": len(rows), "rows": rows}
        del data
    report["elapsed_sec"] = round(time.time() - t0, 1)
    report["cache_mb"] = cache.disk_usage_mb()
    report["min_coverage_pct"] = min_cov
    out_path = os.path.join(config.REPORTS_DIR, f"data_{job_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    prog.state.update({"state": "done", "progress": 1.0, "message": "دیتا آماده‌ست", "report": out_path,
                       "kind": "data"})
    prog.write()
    for tf, t in report["timeframes"].items():
        print(f"{tf}: {t['valid']}/{t['total']} نماد با دیتای کامل ({t['days']} روز)")
        for r in t["rows"]:
            if not r["ok"]:
                q = r.get("quality") or {}
                print(f"   ✗ {r['symbol']}: پوشش {q.get('coverage_pct', 0)}٪ {r.get('error', '')}")


def print_summary(report, path=None):
    lines = []

    def out(*a):
        lines.append(" ".join(str(x) for x in a))

    m = report["meta"]
    out()
    out(f"=== {m['n_configs']} تنظیم، تایم‌فریم‌ها: {', '.join(m['timeframes'])} — {m['total_elapsed_sec']} ثانیه ===")
    rec = report["recommendation"]
    out("\nنتیجه:", rec.get("text"))
    if rec.get("best"):
        b = rec["best"]
        out("\nپیشنهاد:", b["label"])
        for seg, name in (("is", "آموزش"), ("oos", "آزمون"), ("full", "کل")):
            x = b[seg]
            out(f"  {name:6s}: {x['trades']:4d} معامله | وین‌ریت {x['win_rate']:5.1f}% | میانگین {x['avg_r']:+.3f}R "
                  f"| ضریب سود {x['profit_factor']:.2f} | بازده {x['return_pct']:+.1f}% | افت {x['max_dd_pct']:.1f}%")
    out("\nخلاصه‌ی تایم‌فریم‌ها:")
    for t in report.get("timeframe_summary", []):
        o = t["best_oos"] or {}
        out(f"  {t['label']}: {t['configs']} تنظیم، {t['positive_share_pct']}% سودده، "
              f"بهترین در آزمون: {o.get('avg_r', 0):+.3f}R ({o.get('trades', 0)} معامله)")
    out("\n۱۰ تنظیم برتر (بر اساس بخش آموزش):")
    for i, r in enumerate(report["top"][:10], 1):
        o = r["oos"]
        out(f" {i:2d}. {r['label']}\n     آموزش {r['is']['avg_r']:+.3f}R ({r['is']['trades']}) | "
              f"آزمون {o['avg_r']:+.3f}R ({o['trades']}) {'✓' if r['oos_ok'] else '✗'}")
    out("\nاثر هر بهبود (مقایسه‌ی جفتی):")
    for e in report["improvements"]:
        out(f"  {e['label']}: در {e['improved_pct']}% موارد بهتر | تغییر میانه {e['median_delta_r']:+.3f}R "
              f"(آزمون {e['median_delta_r_oos']:+.3f}R)")
    for b in report.get("baselines", []):
        if "full" in b:
            out(f"\n[{b['name']}] {b['label']}: کل {b['full']['trades']} معامله، {b['full']['avg_r']:+.3f}R، "
                f"بازده {b['full']['return_pct']:+.1f}% | آزمون {b['oos']['avg_r']:+.3f}R")
    out("\nروش اجرا/هزینه‌ها:", json.dumps({k: m.get(k) for k in ("fees", "entry_mode", "risk_pct")}, ensure_ascii=False))
    for tf, tm in (m.get("timeframes") or {}).items():
        bad = [k for k, v in (tm.get("data") or {}).items() if not v.get("ok")]
        out(f"  {tf}: {tm.get('days')} روز، {len(tm.get('symbols', []))} نماد، {tm.get('n_configs')} تنظیم"
            + (f"، بدون دیتا: {', '.join(bad)}" if bad else ""))
    out(f"\nفایل کامل همه‌ی نتایج: reports/{m.get('results_csv')}")
    text = "\n".join(lines)
    print(text)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return text


if __name__ == "__main__":
    main()
