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
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np  # noqa: E402
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
        self.state["updated_at"] = datetime.utcnow().isoformat()   # ضربان: پنل می‌فهمه هنوز زنده‌ست
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
    ap.add_argument("--grid", choices=["quick", "full", "focus", "all", "fib", "pattern", "contrarian", "wave2"], default="quick",
                    help="focus = فقط استراتژی شکست باکس و گزینه‌های جدیدش (سریع)")
    ap.add_argument("--end", type=str, default="",
                    help="تاریخ پایان بازه (YYYY-MM-DD، UTC) — برای تست روی گذشته؛ خالی = الان")
    ap.add_argument("--min-coverage", type=float, default=None,
                    help="حداقل پوشش دیتای هر نماد در بازه (٪)؛ پیش‌فرض config.MIN_DATA_COVERAGE_PCT. برای تست گذشته "
                         "کمترش کن تا ارزهایی که وسط بازه لیست شدن هم (فقط از زمان لیست شدن) حساب بشن")
    ap.add_argument("--job-id", type=str, default="")
    ap.add_argument("--progress-file", type=str, default="")
    ap.add_argument("--baseline-file", type=str, default="")
    ap.add_argument("--offline", action="store_true", help="فقط از دیتای کش‌شده (بدون اینترنت)")
    ap.add_argument("--signals", choices=["core", "patterns", "funding", "all"], default="core",
                    help="سنجش ورود: core = استراتژی‌ها و فرضیه‌ها، patterns = الگوهای کندلی و کلاسیک، all = همه")
    ap.add_argument("--resume", action="store_true",
                    help="ادامه‌ی یک سنجش ورود نیمه‌کاره با همون --job-id (از جایی که قطع شده)")
    ap.add_argument("--entry-study", action="store_true",
                    help="سنجش کیفیت ورود: هر سیگنال در برابر ورود شانسی (بدون تریلینگ/مدیریت)")
    ap.add_argument("--scalp-study", action="store_true",
                    help="سنجش اسکلپ: روند ۱۵ دقیقه و ۱ ساعته، ورود روی ۱ دقیقه، ریسک ۱$ (کندل‌های ۱ دقیقه‌ای واقعی)")
    ap.add_argument("--move-study", action="store_true",
                    help="سنجش فاصله‌ی حد ضرر × کارمزد × زمان روی کندل‌های ۵ دقیقه‌ای واقعی (ورود شانسی، ریسک ۱$)")
    ap.add_argument("--data-only", action="store_true",
                    help="فقط دانلود/به‌روزرسانی دیتای تاریخی و بررسی اعتبارش (بدون بک‌تست)")
    args = ap.parse_args()

    job_id = args.job_id or uuid.uuid4().hex[:10]
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    tfs = [t.strip() for t in args.timeframes.split(",") if t.strip() in config.TIMEFRAME_PROFILES] or ["15m"]
    end_ms = None
    if args.end:
        end_ms = int(datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    params = {"days": args.days, "top": args.top, "grid": args.grid, "timeframes": tfs, "end": args.end or None,
              "baseline_file": args.baseline_file or None, "min_coverage": args.min_coverage}
    prog = Progress(args.progress_file, job_id, params)
    if args.entry_study:
        prog.state["kind"] = "entry"
        prog.write()
    if args.move_study or args.scalp_study:
        prog.state["kind"] = "move" if args.move_study else "scalp"
        prog.write()
    t0 = time.time()
    try:
        base_cfg = backtest.build_config(config, {} if args.min_coverage is None else
                                         {"MIN_DATA_COVERAGE_PCT": float(args.min_coverage)})
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
        if args.entry_study:
            run_entry_study(args, tfs, all_symbols, base_cfg, cache, prog, job_id, t0)
            return
        if args.move_study:
            run_move_study(args, all_symbols, base_cfg, cache, prog, job_id, t0)
            return
        if args.scalp_study:
            run_scalp_study(args, all_symbols, base_cfg, cache, prog, job_id, t0)
            return

        # ذخیره‌ی مرحله‌به‌مرحله: هر تایم‌فریمِ تمام‌شده جدا ذخیره می‌شه؛ اگه کار وسط راه قطع بشه (ری‌استارت ربات/
        # سرور، کمبود رم)، «ادامه» با همون شناسه از اولین تایم‌فریمِ ناتموم شروع می‌کنه (دیتای دانلودشده هم در کشه).
        work = os.path.join(config.REPORTS_DIR, f"compare_work_{job_id}")
        os.makedirs(work, exist_ok=True)
        wp = os.path.join(work, "params.json")
        if os.path.exists(wp):
            with open(wp, "r", encoding="utf-8") as f:
                end_ms = int(json.load(f)["end_ms"])
        else:
            end_ms = end_ms or int(time.time() * 1000)
            with open(wp, "w", encoding="utf-8") as f:
                json.dump({"end_ms": end_ms, **params}, f, ensure_ascii=False)

        all_results, tf_meta, base_rows = [], {}, []
        n_tf = len(tfs)
        for i, tf in enumerate(tfs):
            tf_path = os.path.join(work, f"tf_{tf}.json")
            if os.path.exists(tf_path):
                try:
                    with open(tf_path, "r", encoding="utf-8") as f:
                        saved = json.load(f)
                    all_results += saved["results"]
                    base_rows += saved["base_rows"]
                    tf_meta[tf] = saved["meta"]
                    prog.update(f"{tournament.TF_LABELS.get(tf, tf)}: از قبل انجام شده (ذخیره)", (i + 1) / n_tf)
                    continue
                except Exception:
                    pass
            prof = config.TIMEFRAME_PROFILES[tf]
            cfg = fast_backtest.profile_cfg(base_cfg, tf)
            cfg._NEED_HTF = True   # مقایسه هر دو حالت (با و بدون تایید HTF) رو تست می‌کنه
            cfg._NEED_PAIR = True  # دیتای ارز÷BTC و شاخص آلت‌ها (برای فیلترهای نسخه‌ی ۱۹)
            days = min(args.days, int(prof["MAX_DAYS"]))
            symbols = all_symbols[:min(len(all_symbols), int(prof["SCAN_SYMBOLS"]))]
            lo, span = i / n_tf, 1.0 / n_tf
            label = tournament.TF_LABELS.get(tf, tf)
            plan, data = fast_backtest.load_market_data(
                cache, symbols, days, cfg, lambda m, f: prog.update(f"{label} — {m}", f, lo, lo + span * 0.4),
                offline=args.offline, now_ms=end_ms)
            needed = tournament.needed_strategies(cfg, tf, args.grid, baselines)
            preps, meta = fast_backtest.prepare_all(plan, data, symbols, cfg,
                                                    lambda m, f: prog.update(f"{label} — {m}", f, lo + span * 0.4,
                                                                             lo + span * 0.5),
                                                    strategies_needed=needed)
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
            tmp = tf_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"results": results, "base_rows": brows, "meta": m}, f, ensure_ascii=False,
                          default=lambda o: o.item() if hasattr(o, "item") else str(o))
            os.replace(tmp, tf_path)
            del preps

        if not all_results:
            raise RuntimeError("برای هیچ تایم‌فریمی نتیجه‌ای تولید نشد: " + json.dumps(tf_meta, ensure_ascii=False)[:500])

        report = tournament.build_report(all_results, tf_meta, base_rows, args.grid)
        csv_path = os.path.join(config.REPORTS_DIR, f"results_{job_id}.csv")
        tournament.write_results_csv(all_results, csv_path)
        report["meta"].update({
            "job_id": job_id, "days": args.days, "created_at": datetime.utcnow().isoformat(), "end": args.end or None,
            "symbols": all_symbols, "total_elapsed_sec": round(time.time() - t0, 1),
            "cache_mb": cache.disk_usage_mb(), "results_csv": os.path.basename(csv_path),
            "fees": {"maker": base_cfg.MAKER_FEE_PCT, "taker": base_cfg.TAKER_FEE_PCT,
                     "slippage": base_cfg.TAKER_SLIPPAGE_PCT, "funding_8h": base_cfg.FUNDING_PCT_PER_8H},
            "entry_mode": base_cfg.ENTRY_MODE, "risk_pct": base_cfg.RISK_PER_TRADE_PCT,
            "risk_mode": base_cfg.RISK_MODE, "risk_usd": base_cfg.RISK_USD,
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


def run_scalp_study(args, all_symbols, base_cfg, cache, prog, job_id, t0):
    import scalp_study as ss
    days = max(14, min(int(args.days), 90))
    now = int(time.time() * 1000)
    since = now - days * 86_400_000
    prog.update("دانلود کندل‌های ۱ دقیقه‌ای", 0.0)
    if args.offline:
        data = {}
        for s in all_symbols:
            a = cache.load(s, "1m")
            data[(s, "1m")] = a if a is not None else RuntimeError("در کش نیست")
    else:
        data = cache.ensure_many([(s, "1m", since - 2 * 86_400_000) for s in all_symbols],
                                 lambda d, tot, sym, tf, res: prog.update(f"دیتا {d}/{tot}: {sym}", d / tot, 0, 0.5))
    fees = ss.fee_model(base_cfg)
    split_ms = int(since + (now - since) * ss.IS_FRACTION)
    per, ok_syms = {}, []
    for i, s in enumerate(all_symbols):
        a = data.get((s, "1m"))
        if a is None or isinstance(a, Exception):
            continue
        a = np.asarray(a)
        a = a[a[:, 0] >= since - 2 * 86_400_000]     # ۲ روز گرم‌کردن برای SMA99 ساعتی
        if len(a) < 7 * 1440:
            continue
        res = ss.run_symbol(a, fees)
        per[s] = {k: [x for x in v if x[0] >= since] for k, v in res.items()}
        ok_syms.append(s)
        prog.update(f"شبیه‌سازی {s} ({i + 1}/{len(all_symbols)})", (i + 1) / len(all_symbols), 0.5, 1.0)
    if not per:
        raise RuntimeError("دیتای ۱ دقیقه‌ای قابل‌استفاده نبود")
    rows = ss.summarize(per, days, split_ms)
    report = {"kind": "scalp", "job_id": job_id, "created_at": datetime.utcnow().isoformat(), "days": days,
              "symbols": ok_syms, "rows": rows, "split_ms": split_ms,
              "fees_pct": {"maker": base_cfg.MAKER_FEE_PCT, "taker": base_cfg.TAKER_FEE_PCT,
                           "slippage": base_cfg.TAKER_SLIPPAGE_PCT, "funding_8h": base_cfg.FUNDING_PCT_PER_8H},
              "elapsed_sec": round(time.time() - t0, 1)}
    out = os.path.join(config.REPORTS_DIR, f"scalp_{job_id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    prog.state.update({"state": "done", "progress": 1.0, "message": "تمام شد", "report": out})
    prog.write()
    for r in rows:
        print(r)


def run_move_study(args, all_symbols, base_cfg, cache, prog, job_id, t0):
    import move_study as ms
    days = max(14, min(int(args.days), 180))
    now = int(time.time() * 1000)
    since = now - days * 86_400_000
    jobs = [(s, "5m", since) for s in all_symbols]
    prog.update("دانلود کندل‌های ۵ دقیقه‌ای", 0.0)
    data = {}
    if args.offline:
        for s in all_symbols:
            a = cache.load(s, "5m")
            data[(s, "5m")] = a if a is not None else RuntimeError("در کش نیست")
    else:
        data = cache.ensure_many(jobs, lambda d, tot, sym, tf, res: prog.update(f"دیتا {d}/{tot}: {sym}", d / tot, 0, 0.4))
    fees = ms.fee_model(base_cfg)
    per, atrs_all = {}, {"5m": [], "15m": [], "1h": [], "4h": [], "1d": []}
    ok_syms = []
    for i, s in enumerate(all_symbols):
        a = data.get((s, "5m"))
        if a is None or isinstance(a, Exception) or len(a) < 2000:
            continue
        a = np.asarray(a)
        a = a[a[:, 0] >= since]
        if len(a) < 2000:
            continue
        ok_syms.append(s)
        per[s] = ms.simulate_symbol(a, fees)
        atrs_all["5m"].append(ms.atr_pct(a))
        for tf in ("15m", "1h", "4h", "1d"):
            r = fast_backtest._resample(a, "5m", tf)
            atrs_all[tf].append(ms.atr_pct(r))
        prog.update(f"شبیه‌سازی {s} ({i + 1}/{len(all_symbols)})", (i + 1) / len(all_symbols), 0.4, 1.0)
    if not per:
        raise RuntimeError("دیتای ۵ دقیقه‌ای قابل‌استفاده نبود")
    atrs = {tf: round(float(np.median([v for v in vals if v])), 3) for tf, vals in atrs_all.items()
            if any(v for v in vals)}
    rows = ms.summarize(per, fees)
    text = ms.text_summary(rows, atrs)
    report = {"kind": "move", "job_id": job_id, "created_at": datetime.utcnow().isoformat(), "days": days,
              "symbols": ok_syms, "atr_pct": atrs, "rows": rows, "text": text,
              "fees_pct": {"maker": base_cfg.MAKER_FEE_PCT, "taker": base_cfg.TAKER_FEE_PCT,
                           "slippage": base_cfg.TAKER_SLIPPAGE_PCT, "funding_8h": base_cfg.FUNDING_PCT_PER_8H},
              "elapsed_sec": round(time.time() - t0, 1)}
    out = os.path.join(config.REPORTS_DIR, f"move_{job_id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    prog.state.update({"state": "done", "progress": 1.0, "message": "تمام شد", "report": out})
    prog.write()
    print(text)


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


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def entry_work_dir(job_id):
    return os.path.join(config.REPORTS_DIR, f"entry_work_{job_id}")


def run_entry_study(args, tfs, all_symbols, base_cfg, cache, prog, job_id, t0):
    """
    سنجش کیفیت ورود برای همه‌ی تایم‌فریم‌ها → reports/entry_<id>.json + CSVها + خلاصه.

    ذخیره‌ی مرحله‌ای (قابل ادامه): ورودهای هر نماد بعد از محاسبه در reports/entry_work_<id>/<tf>/ ذخیره
    می‌شن و نتیجه‌ی هر تایم‌فریم تمام‌شده هم جدا. اگه کار وسط راه قطع بشه (ری‌استارت ربات، کمبود رم،
    خاموشی سرور)، با --resume همون شناسه از همون‌جا ادامه پیدا می‌کنه — و نتیجه دقیقاً مثل اجرای
    بدون وقفه‌ست (زمان پایان دیتای هر تایم‌فریم هم ذخیره می‌شه).
    """
    import shutil
    import entry_study as es
    rd = config.REPORTS_DIR
    work = entry_work_dir(job_id)
    os.makedirs(work, exist_ok=True)
    ppath = os.path.join(work, "params.json")
    params = _read_json(ppath) if args.resume else None
    if not params:
        params = {"days": args.days, "timeframes": tfs, "symbols": all_symbols, "signals": args.signals,
                  "created_at": datetime.utcnow().isoformat(), "now_ms": {}, "runs": 0}
    params["runs"] = int(params.get("runs", 0)) + 1
    _write_json(ppath, params)
    tfs, all_symbols = params["timeframes"], params["symbols"]
    signal_set = params.get("signals", "core")
    days_req = int(params["days"])
    prog.state["params"].update({"timeframes": tfs, "symbols": all_symbols, "days": days_req})
    if params["runs"] > 1:
        prog.update(f"ادامه‌ی سنجش (بار {params['runs']})", 0.0)

    all_rows, feat_rows, tf_meta = [], [], {}
    n_tf = len(tfs)
    btc = getattr(base_cfg, "BTC_REGIME_SYMBOL", "BTC/USDT")
    rows_written = 0
    per_tf_cap = max(20000, int(base_cfg.ENTRY_STUDY_MAX_CSV_ROWS) // max(1, n_tf))

    def safe(sym):
        return sym.replace("/", "_").replace(":", "_")

    # فاندینگ و اوپن اینترست (یک‌بار برای همه‌ی تایم‌فریم‌ها؛ روی دیسک کش می‌شه)
    derivs, deriv_cov = {}, {}
    if signal_set in ("funding", "all"):
        import market_derivs
        dc = market_derivs.DerivCache(config.DATA_CACHE_DIR, getattr(base_cfg, "DERIV_EXCHANGES", []),
                                      log=lambda m: print("[derivs]", m, flush=True))
        end_all = int(time.time() * 1000)
        start_all = end_all - int(days_req) * 86_400_000 - 100 * 86_400_000   # + گرم‌کردن رتبه‌ی ۹۰ روزه
        for j, sym in enumerate(all_symbols):
            prog.update(f"دیتای فاندینگ و اوپن اینترست: {sym} ({j + 1}/{len(all_symbols)})",
                        j / max(1, len(all_symbols)), 0.0, 0.02)
            fa, fm = dc.ensure(sym, "funding", start_all, end_all, offline=args.offline)
            oa, om = dc.ensure(sym, "oi", start_all, end_all, offline=args.offline)
            derivs[sym] = {"funding": fa, "oi": oa}
            deriv_cov[sym] = {"funding_exchange": (fm or {}).get("exchange"),
                              "funding_cov_pct": market_derivs.coverage(fa, end_all - int(days_req) * 86_400_000,
                                                                        end_all),
                              "oi_exchange": (om or {}).get("exchange"),
                              "oi_cov_pct": market_derivs.coverage(oa, end_all - int(days_req) * 86_400_000, end_all)}

    for i, tf in enumerate(tfs):
        lo, span = i / n_tf, 1.0 / n_tf
        label = tournament.TF_LABELS.get(tf, tf)
        tdir = os.path.join(work, tf)
        os.makedirs(tdir, exist_ok=True)
        done_path = os.path.join(tdir, "done.json")
        done = _read_json(done_path)
        if done:
            all_rows += done["rows"]
            feat_rows += done["frows"]
            tf_meta[tf] = done["meta"]
            rows_written += int(done.get("events_rows", 0))
            prog.update(f"{label} — قبلاً تمام شده بود (از ذخیره)", 1.0, lo, lo + span)
            continue
        prof = config.TIMEFRAME_PROFILES[tf]
        cfg = fast_backtest.profile_cfg(base_cfg, tf)
        cfg._NEED_HTF = True
        days = min(days_req, int(prof["MAX_DAYS"]))
        symbols = all_symbols[:min(len(all_symbols), int(prof["SCAN_SYMBOLS"]))]
        now_ms = params["now_ms"].get(tf)
        if not now_ms:
            now_ms = int(time.time() * 1000)
            params["now_ms"][tf] = now_ms
            _write_json(ppath, params)
        plan0 = fast_backtest.plan_jobs(symbols[:1], days, cfg, now_ms)
        start_ms, end_ms = plan0["start_ms"], plan0["now_ms"]
        split_ms = int(start_ms + (end_ms - start_ms) * float(cfg.ENTRY_STUDY_IS_FRACTION))
        study = es.Study(cfg, tf, split_ms, symbols, xs=(tf == "1d"), signal_set=signal_set)
        variant = fast_backtest.variant_from_cfg(cfg)
        # فقط الگوها: استراتژی‌ها لازم نیستن (سریع‌تر)
        needed = list(es.STRATEGY_SIGNALS) if signal_set != "patterns" else ["none"]
        base_n = int(cfg.ENTRY_STUDY_BASE_SAMPLES)

        def sym_file(sym):
            return os.path.join(tdir, f"sym_{safe(sym)}.npz")

        todo = [s_ for s_ in symbols if not os.path.exists(sym_file(s_))]
        if len(todo) < len(symbols):
            prog.update(f"{label} — {len(symbols) - len(todo)} نماد از قبل ذخیره شده؛ ادامه از نماد بعدی",
                        0.0, lo, lo + span * 0.05)
        if todo:
            prog.update(f"{label} — دیتای BTC", 0.0, lo, lo + span * 0.05)
            bplan, bdata = fast_backtest.load_market_data(cache, [btc], days, cfg, None, args.offline, now_ms)
            bp, _ = fast_backtest.prepare_all(bplan, bdata, [btc], cfg, strategies_needed=["trend_follow"])
            btc_ctx = es.btc_context(bp[btc]) if btc in bp else None
            del bp, bdata
            if tf == "1d" and signal_set != "patterns":
                # مومنتوم هفتگی به همه‌ی نمادها با هم نیاز داره (روزانه سبکه)
                plan, data = fast_backtest.load_market_data(
                    cache, symbols, days, cfg, lambda m, f: prog.update(f"{label} — {m}", f, lo, lo + span * 0.3),
                    offline=args.offline, now_ms=now_ms)
                preps, meta = fast_backtest.prepare_all(plan, data, symbols, cfg, strategies_needed=needed,
                                                        keep_volume=True)
                del data
                xs = fast_backtest.xs_finals(preps, variant)
                for j, sym in enumerate(todo):
                    if sym in preps:
                        study.add_symbol(preps[sym], variant, btc_ctx, xs.get(sym), is_btc=(sym == btc),
                                         base_samples=base_n, keep_part=False, deriv=derivs.get(sym))
                        es.save_symbol(sym_file(sym), sym, study.last_part, study._cur, meta.get(sym))
                    else:
                        es.save_symbol(sym_file(sym), sym, None, {}, meta.get(sym))
                    prog.update(f"{label} — ورودهای {sym} ({j + 1}/{len(todo)})", (j + 1) / len(todo),
                                lo + span * 0.3, lo + span * 0.8)
                del preps, xs
            else:
                for j, sym in enumerate(todo):
                    # دیتای هر نماد جدا خونده می‌شه (رم کم)
                    plan, data = fast_backtest.load_market_data(cache, [sym], days, cfg, None, args.offline, now_ms)
                    p1, m1 = fast_backtest.prepare_all(plan, data, [sym], cfg, strategies_needed=needed,
                                                       keep_volume=True)
                    del data
                    if sym in p1:
                        study.add_symbol(p1[sym], variant, btc_ctx, None, is_btc=(sym == btc),
                                         base_samples=base_n, keep_part=False, deriv=derivs.get(sym))
                        es.save_symbol(sym_file(sym), sym, study.last_part, study._cur, m1.get(sym))
                    else:
                        es.save_symbol(sym_file(sym), sym, None, {}, m1.get(sym))
                    study.last_part = None
                    del p1
                    prog.update(f"{label} — ورودهای {sym} ({symbols.index(sym) + 1}/{len(symbols)})",
                                (j + 1) / len(todo), lo + span * 0.05, lo + span * 0.8)
        prog.update(f"{label} — خوندن ورودهای ذخیره‌شده", 0.0, lo + span * 0.8, lo + span)
        meta, n_ev = study.assemble([sym_file(s_) for s_ in symbols])
        rows = es.analyze(study, cb=lambda f, nm: prog.update(f"{label} — تحلیل آماری: {es.label_of(nm)}", f,
                                                            lo + span * 0.8, lo + span * 0.88))
        frows = es.analyze_features(study, cb=lambda f, nm: prog.update(
            f"{label} — تحلیل فیلترها: {es.label_of(nm)}", f, lo + span * 0.88, lo + span * 0.97))
        ev_path = os.path.join(tdir, "events.csv.gz")
        n_rows = es.write_events(study, ev_path, per_tf_cap, symbols, header=False)
        ok_syms = [s_ for s_ in symbols if (meta.get(s_) or {}).get("ok")]
        m_tf = {"timeframe": tf, "label": label, "days": days, "symbols": ok_syms,
                "bad_symbols": {s_: (meta.get(s_) or {}).get("error", "دیتا نیست") for s_ in symbols
                                if not (meta.get(s_) or {}).get("ok")},
                "start_ms": start_ms, "end_ms": end_ms, "split_ms": split_ms,
                "horizon_bars": study.H, "horizons": study.hz, "r_atr": study.r_atr,
                "cooldown_bars": study.gap, "cluster_days": round(study.cluster_ms / 86_400_000, 1),
                "events": int(n_ev), "htf_required_pct": list(study.htf_req),
                "cost_market_pct": round(study.cost_market * 100, 4),
                "cost_limit_pct": round(study.cost_limit * 100, 4)}
        rows_json = [es.row_to_json(r) for r in rows]
        _write_json(done_path, {"rows": rows_json, "frows": frows, "meta": m_tf, "events_rows": n_rows,
                                "events_header": es.events_header(study.hz)})
        for s_ in symbols:   # ورودهای خام دیگه لازم نیست (جای دیسک)
            try:
                os.remove(sym_file(s_))
            except OSError:
                pass
        all_rows += rows_json
        feat_rows += frows
        tf_meta[tf] = m_tf
        rows_written += n_rows
        del study

    if not all_rows:
        raise RuntimeError("هیچ ورودی برای سنجش پیدا نشد: " + json.dumps(tf_meta, ensure_ascii=False)[:500])
    report = es_build_report(all_rows, feat_rows, tf_meta)
    report["meta"].update({
        "job_id": job_id, "days": days_req, "created_at": datetime.utcnow().isoformat(), "symbols": all_symbols,
        "total_elapsed_sec": round(time.time() - t0, 1), "bot_version": getattr(config, "BOT_VERSION", ""),
        "events_rows": rows_written, "runs": params["runs"], "started_at": params["created_at"],
        "signal_set": signal_set, "deriv_coverage": deriv_cov,
        "fees": {"maker": base_cfg.MAKER_FEE_PCT, "taker": base_cfg.TAKER_FEE_PCT,
                 "slippage": base_cfg.TAKER_SLIPPAGE_PCT}})
    # فایل همه‌ی ورودها: یک سرتیتر + ورودهای هر تایم‌فریم (چند بخش gzip پشت‌سرهم = یک فایل gzip معتبر)
    import csv
    import gzip
    import io
    events_path = os.path.join(rd, f"entry_events_{job_id}.csv.gz")
    first = _read_json(os.path.join(work, tfs[0], "done.json")) or {}
    buf = io.StringIO()
    csv.writer(buf).writerow(first.get("events_header") or [])
    with open(events_path + ".tmp", "wb") as out:
        out.write(gzip.compress(buf.getvalue().encode("utf-8")))
        for tf in tfs:
            pth = os.path.join(work, tf, "events.csv.gz")
            if os.path.exists(pth):
                with open(pth, "rb") as f:
                    shutil.copyfileobj(f, out)
    os.replace(events_path + ".tmp", events_path)
    with open(os.path.join(rd, f"entry_signals_{job_id}.csv"), "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(es.signal_csv_rows(all_rows, _hz_union(tf_meta)))
    fcols = ["tf", "signal", "label", "feature", "feature_label", "bucket", "lo", "hi", "n_is", "n_oos", "p1_is",
             "p1_oos", "lift1_is", "lift1_oos", "p2_is", "p2_oos", "ev2_is", "ev2_oos", "d_ev2_is", "d_ev2_oos", "z_is",
             "z_oos", "tfs_agree", "bucket_idx", "sig_p1_is", "sig_p1_oos", "sig_ev2_is", "sig_ev2_oos", "flag"]
    with open(os.path.join(rd, f"entry_features_{job_id}.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fcols)
        w.writeheader()
        for r in feat_rows:
            w.writerow({k: r.get(k) for k in fcols})
    out_path = os.path.join(rd, f"entry_{job_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    print_entry_summary(report, os.path.join(rd, f"entry_summary_{job_id}.txt"))
    shutil.rmtree(work, ignore_errors=True)
    prog.state.update({"state": "done", "progress": 1.0, "message": "تمام شد", "report": out_path, "kind": "entry"})
    prog.write()


def _hz_union(tf_meta):
    # ستون‌های بازده آینده در CSV: برای چند تایم‌فریم، به ترتیب «۱، ۳، ۶، ۱۲، ۲۴، H» (H هر تایم‌فریم فرق داره)
    return ["1", "3", "6", "12", "24", "H"]


def es_build_report(rows, feat_rows, tf_meta):
    import entry_study as es
    sig_rows = [es.row_to_json(r) for r in rows]
    cand = [r for r in sig_rows if r.get("verdict") not in ("base", None)]

    def oos_best(r):
        b = r.get("best_b")
        ev = (r.get("oos") or {}).get("ev") or []
        return ev[b] if b is not None and b < len(ev) and ev[b] is not None else -9

    cand.sort(key=lambda r: (es.VERDICT_RANK.get(r["verdict"], 9), -oos_best(r)))
    robust = [r for r in cand if r["verdict"] == "robust"]
    near = [r for r in cand if r["verdict"] in ("edge_weak_oos", "edge_costs", "weak")]
    # فیلتری که در چند تایم‌فریم تکرار می‌شه قابل‌اعتمادتره (همون سیگنال، همون ویژگی، همون سطل)
    agree = {}
    for f in feat_rows:
        if f["flag"]:
            agree.setdefault((f["signal"], f["feature"], f["bucket_idx"], f["flag"]), set()).add(f["tf"])
    for f in feat_rows:
        if f["flag"]:
            f["tfs_agree"] = len(agree[(f["signal"], f["feature"], f["bucket_idx"], f["flag"])])
    good = sorted([f for f in feat_rows if f["flag"] == "good"],
                  key=lambda f: (-f.get("tfs_agree", 1), -min(f.get("d_ev2_is") or 0, f.get("d_ev2_oos") or 0)))
    bad = sorted([f for f in feat_rows if f["flag"] == "bad"],
                 key=lambda f: max(f.get("d_ev2_is") or 0, f.get("d_ev2_oos") or 0))
    if robust:
        text = (f"{len(robust)} ورود مزیت پایدار نشون دادن (در آموزش معنادار، در آزمون هم مثبت، بعد از کارمزد). "
                "این‌ها کاندیدای ساختن استراتژی نهایی‌ان؛ قدم بعد: ترکیب با فیلترهای مفید و چند ماه پیپر ترید.")
        status = "robust"
    elif near:
        text = ("هیچ ورودی بعد از کارمزد در هر دو بخش سودده نبود؛ ولی چندتا از شانس بهترن "
                "(جدول «نزدیک‌ترین‌ها»). با فیلترهای مفید یا تایم‌فریم بالاتر شاید به سود برسن.")
        status = "near"
    else:
        text = "هیچ ورودی از ورود شانسی بهتر نبود. یعنی این قانون‌ها روی این ارزها و این دوره مزیتی ندارن."
        status = "none"
    return {"kind": "entry", "meta": {"timeframes": tf_meta, "brackets": es.BRACKET_FA,
                                      "bracket_keys": es.BRACKET_KEYS, "main_bracket": es.MAIN_B},
            "recommendation": {"status": status, "text": text, "robust": robust[:15], "near": near[:15]},
            "signals": sig_rows, "top": cand[:40], "good_filters": good[:60], "bad_filters": bad[:30],
            "features": feat_rows}


def print_entry_summary(report, path=None):
    lines = []

    def out(*a):
        lines.append(" ".join(str(x) for x in a))

    def pct(x):
        return "—" if x is None else f"{x * 100:.1f}%"

    def r3(x):
        return "—" if x is None else f"{x:+.3f}R"

    m = report["meta"]
    rec = report["recommendation"]
    out(f"=== سنجش کیفیت ورود — {', '.join(m['timeframes'])} — {m.get('total_elapsed_sec')} ثانیه ===")
    out("براکت‌ها:", " | ".join(m["brackets"]), "— واحد R = ATR×", next(iter(m["timeframes"].values()))["r_atr"])
    out("\nنتیجه:", rec["text"])
    for tf, tm in m["timeframes"].items():
        out(f"\n--- {tm['label']} ({tm['days']} روز، {len(tm['symbols'])} نماد، {tm['events']} ورود، "
            f"افق {tm['horizon_bars']} کندل، هزینه‌ی ورود و خروج بازار {tm['cost_market_pct']}٪) ---")
        base = [r for r in report["signals"] if r["tf"] == tf and r["signal"] == "base" and r["side"] != "both"]
        for b in base:
            a = b["all"]
            out(f"  پایه ({'خرید' if b['side'] == 'long' else 'فروش'}): +1R قبل از −1R = {pct(a['p'][0])}، "
                f"+2R قبل از −1R = {pct(a['p'][1])}، ارزش خالص RR2 = {r3(a['ev'][1])} (هزینه {a['cost_r']:.3f}R)")
        rows = [r for r in report["top"] if r["tf"] == tf][:12]
        for r in rows:
            b = r.get("best_b", 1)
            i_, o_ = r["is"], r["oos"]
            out(f"  {r['verdict_text'][:2]} {r['label']} [{r['side']}] — آموزش {i_.get('n')} ورود، آزمون {o_.get('n')}")
            out(f"      +1R/−1R: آموزش {pct((i_.get('p') or [None])[0])} (پایه {pct((i_.get('base_p') or [None])[0])})"
                f" | آزمون {pct((o_.get('p') or [None])[0])} (پایه {pct((o_.get('base_p') or [None])[0])})")
            out(f"      بهترین براکت «{m['brackets'][b]}»: ارزش خالص آموزش {r3((i_.get('ev') or [None]*4)[b])} "
                f"(z={(i_.get('z_ev') or [None]*4)[b]}) | آزمون {r3((o_.get('ev') or [None]*4)[b])}"
                f" | بهتر از شانس در {r.get('sym_better_pct')}٪ نمادها")
    good = report.get("good_filters") or []
    out("\nفیلترهای مفید (در آموزش و آزمون، هر دو بهتر):")
    for f in good[:20]:
        out(f"  [{f['tf']}] {f['label']} ← {f['feature_label']}: {f['bucket']} — +1R/−1R از {pct(f['sig_p1_is'])} به "
            f"{pct(f['p1_is'])} (آزمون {pct(f['sig_p1_oos'])}→{pct(f['p1_oos'])})، ارزش RR2 آزمون "
            f"{r3(f['sig_ev2_oos'])}→{r3(f['ev2_oos'])} ({f['n_is']}/{f['n_oos']} ورود"
            + (f"، تکرار در {f['tfs_agree']} تایم‌فریم" if f.get("tfs_agree", 1) > 1 else "") + ")")
    if not good:
        out("  هیچ فیلتری در هر دو بخش به‌طور معنادار کمک نکرد.")
    text = "\n".join(lines)
    print(text)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return text


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
            out(f"  {name:6s}: {x['trades']:4d} معامله | سودده {x.get('pos_rate', 0):5.1f}% | میانگین {x['avg_r']:+.3f}R "
                f"| هفته‌ای {x.get('avg_week_usd', 0):+.2f}$ ({x.get('pos_weeks_pct', 0):.0f}% هفته‌ها مثبت) "
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
    out("\nروش اجرا/هزینه‌ها:", json.dumps({k: m.get(k) for k in ("fees", "entry_mode", "risk_mode", "risk_usd")}, ensure_ascii=False))
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
