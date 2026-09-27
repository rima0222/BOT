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
    ap.add_argument("--grid", choices=["quick", "full"], default="quick")
    ap.add_argument("--job-id", type=str, default="")
    ap.add_argument("--progress-file", type=str, default="")
    ap.add_argument("--baseline-file", type=str, default="")
    ap.add_argument("--offline", action="store_true", help="فقط از دیتای کش‌شده (بدون اینترنت)")
    args = ap.parse_args()

    job_id = args.job_id or uuid.uuid4().hex[:10]
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    params = {"days": args.days, "top": args.top, "grid": args.grid}
    prog = Progress(args.progress_file, job_id, params)
    t0 = time.time()
    try:
        cfg = backtest.build_config(config)
        cache = market_data.MarketDataCache(config.DATA_CACHE_DIR, config.EXCHANGE_TRY_ORDER,
                                            config.DATA_FETCH_MAX_REQ_PER_SEC, config.DATA_FETCH_THREADS)
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        else:
            prog.update("دریافت لیست پرحجم‌ترین نمادها", 0.0)
            try:
                symbols, _ = data_fetcher.get_top_symbols(config.EXCHANGE_TRY_ORDER, quote=config.QUOTE_CURRENCY,
                                                          top_n=args.top + 10,
                                                          exclude_keywords=config.EXCLUDE_KEYWORDS, cfg=config)
            except Exception:
                symbols = sorted({k.split("|")[0] for k in cache.index})
            symbols = symbols[:args.top]
        symbols = [s for s in symbols if data_fetcher.is_symbol_allowed(s, config)]
        if not symbols:
            raise RuntimeError("هیچ نمادی برای تست پیدا نشد")
        params["symbols"] = symbols

        plan, data = fast_backtest.load_market_data(
            cache, symbols, args.days, cfg, lambda m, f: prog.update(m, f, 0.0, 0.45), offline=args.offline)
        preps, meta = fast_backtest.prepare_all(plan, data, symbols, cfg,
                                                lambda m, f: prog.update(m, f, 0.45, 0.55))
        if not preps:
            raise RuntimeError("برای هیچ نمادی دیتای قابل‌استفاده نبود: " +
                               "; ".join(f"{k}: {v.get('error')}" for k, v in meta.items()))

        baselines = []
        if args.baseline_file and os.path.exists(args.baseline_file):
            with open(args.baseline_file, "r", encoding="utf-8") as f:
                baselines = [(b["name"], b["config"]) for b in json.load(f)]

        report = tournament.run(preps, list(preps), cfg, plan["start_ms"], plan["now_ms"], grid=args.grid,
                                progress_cb=lambda m, f: prog.update(m, f, 0.55, 1.0), baselines=baselines)
        report["meta"].update({
            "job_id": job_id, "days": args.days, "created_at": datetime.utcnow().isoformat(),
            "data": meta, "total_elapsed_sec": round(time.time() - t0, 1),
            "cache_mb": cache.disk_usage_mb(),
            "fees": {"maker": cfg.MAKER_FEE_PCT, "taker": cfg.TAKER_FEE_PCT, "slippage": cfg.TAKER_SLIPPAGE_PCT},
            "risk_pct": cfg.RISK_PER_TRADE_PCT, "test_min_sl_pct": cfg.TEST_MIN_SL_PCT,
            "test_min_sl_atr": cfg.TEST_MIN_SL_ATR_MULT,
        })
        out_path = os.path.join(config.REPORTS_DIR, f"compare_{job_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False)
        prog.state.update({"state": "done", "progress": 1.0, "message": "تمام شد", "report": out_path})
        prog.write()
        print_summary(report)
    except Exception as e:
        prog.state.update({"state": "error", "message": f"خطا: {e}", "error": traceback.format_exc()[-2000:]})
        prog.write()
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


def print_summary(report):
    m = report["meta"]
    print()
    print(f"=== {m['n_configs']} تنظیم روی {len(m['symbols'])} نماد، {m['days']} روز — {m['total_elapsed_sec']} ثانیه ===")
    rec = report["recommendation"]
    print("\nنتیجه:", rec.get("text"))
    if rec.get("best"):
        b = rec["best"]
        print("\nپیشنهاد:", b["label"])
        for seg, name in (("is", "آموزش"), ("oos", "آزمون"), ("full", "کل")):
            x = b[seg]
            print(f"  {name:6s}: {x['trades']:4d} معامله | وین‌ریت {x['win_rate']:5.1f}% | میانگین {x['avg_r']:+.3f}R "
                  f"| ضریب سود {x['profit_factor']:.2f} | بازده {x['return_pct']:+.1f}% | افت {x['max_dd_pct']:.1f}%")
    print("\n۱۰ تنظیم برتر (بر اساس بخش آموزش):")
    for i, r in enumerate(report["top"][:10], 1):
        o = r["oos"]
        print(f" {i:2d}. {r['label']}\n     آموزش {r['is']['avg_r']:+.3f}R ({r['is']['trades']}) | "
              f"آزمون {o['avg_r']:+.3f}R ({o['trades']}) {'✓' if r['oos_ok'] else '✗'}")
    print("\nاثر هر بهبود (مقایسه‌ی جفتی):")
    for e in report["improvements"]:
        print(f"  {e['label']}: در {e['improved_pct']}% موارد بهتر | تغییر میانه {e['median_delta_r']:+.3f}R "
              f"(آزمون {e['median_delta_r_oos']:+.3f}R)")
    for b in report.get("baselines", []):
        if "full" in b:
            print(f"\n[{b['name']}] {b['label']}: کل {b['full']['trades']} معامله، {b['full']['avg_r']:+.3f}R، "
                  f"بازده {b['full']['return_pct']:+.1f}%")


if __name__ == "__main__":
    main()
