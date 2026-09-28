# -*- coding: utf-8 -*-
"""
تحلیل‌گر بک‌تست ("ایجنت" تحلیل رفتار).

عمداً به‌جای یک مدل زبانی مبهم، این ماژول یک خط‌لوله‌ی تحلیل قانون‌محور و
قابل‌ردیابیه: هر ایراد دقیقاً بر اساس یک آستانه‌ی آماری/ریسکی مشخص تشخیص داده
می‌شه و پیشنهادش هم دقیقاً معلومه از کجا اومده. برای چیزی که قراره پایه‌ی تصمیم
برای پول واقعی باشه، این شفافیت مهم‌تر از "هوشمندتر به‌نظر رسیدن" یک جعبه‌سیاهه.
"""
import math
from collections import defaultdict
from datetime import datetime

import paper_trader


def _safe_div(a, b, default=0.0):
    return a / b if b else default


def _confidence_interval_pp(win_rate_pct, n, z=1.96):
    """نیم‌عرض بازه‌ی اطمینان ۹۵٪ برای یک نسبت دوجمله‌ای، بر حسب واحد درصد."""
    if n == 0:
        return None
    p = win_rate_pct / 100
    return z * math.sqrt(p * (1 - p) / n) * 100


def _max_drawdown(equity_curve):
    """بیشینه‌ی افت از قله، به درصد، + طول دوره‌ی افت (روز)."""
    if len(equity_curve) < 2:
        return 0.0, 0
    peak = equity_curve[0][1]
    peak_time = equity_curve[0][0]
    max_dd = 0.0
    max_dd_days = 0
    for t, bal in equity_curve:
        if bal > peak:
            peak = bal
            peak_time = t
        dd = (peak - bal) / peak * 100 if peak else 0
        if dd > max_dd:
            max_dd = dd
            try:
                days = (datetime.fromisoformat(t) - datetime.fromisoformat(peak_time)).total_seconds() / 86400
                max_dd_days = max(max_dd_days, round(days, 1))
            except Exception:
                pass
    return round(max_dd, 2), max_dd_days


def _streaks(results):
    """طولانی‌ترین رشته‌ی برد و باخت پشت‌سرهم."""
    longest_win = longest_loss = cur_win = cur_loss = 0
    for r in results:
        if r == "WIN":
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        longest_win = max(longest_win, cur_win)
        longest_loss = max(longest_loss, cur_loss)
    return longest_win, longest_loss


def _sharpe_like(equity_curve, freq_per_year=365):
    """یک تخمین ساده‌شده‌ی نسبت شارپ از روی تغییرات equity (نه بازده روزانه‌ی دقیق مالی)."""
    if len(equity_curve) < 3:
        return None
    balances = [b for _, b in equity_curve]
    returns = [(balances[i] - balances[i - 1]) / balances[i - 1] for i in range(1, len(balances)) if balances[i - 1]]
    if len(returns) < 2:
        return None
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return None
    return round((mean_r / std) * math.sqrt(freq_per_year), 2)


def analyze(conn, cfg, meta):
    """
    خروجی: دیکشنری کامل شامل معیارهای عملکرد، شکست به تفکیک‌های مختلف،
    و لیست ایرادات با شدت و پیشنهاد مشخص.
    """
    has_score = "score" in {r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()}
    closed = conn.execute(
        "SELECT symbol, side, result, pnl, fee_cost, margin, leverage, rr_planned, htf_agree, "
        "exit_type, strategy_name, open_time, close_time" + (", score" if has_score else ", NULL") +
        " FROM trades WHERE status='CLOSED' ORDER BY id"
    ).fetchall()
    cols = ["symbol", "side", "result", "pnl", "fee_cost", "margin", "leverage", "rr_planned",
            "htf_agree", "exit_type", "strategy_name", "open_time", "close_time", "score"]
    trades = [dict(zip(cols, row)) for row in closed]
    equity_curve = paper_trader.get_equity_curve(conn, limit=100000)

    total = len(trades)
    # برد = رسیدن به هدف R:R، باخت = حد ضرر اولیه. خروج تریلینگ و بقیه جدا شمرده می‌شن.
    for t in trades:
        t["result"] = paper_trader.result_of(t["exit_type"])
    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    trails = [t for t in trades if t["result"] == "TRAIL"]
    others = [t for t in trades if t["result"] == "OTHER"]
    rr_total = len(wins) + len(losses)
    win_rate = round(len(wins) / rr_total * 100, 2) if rr_total else 0.0

    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    total_pnl = round(sum(t["pnl"] for t in trades), 2)
    total_fees = round(sum(t["fee_cost"] or 0 for t in trades), 2)
    profit_factor = round(_safe_div(gross_profit, gross_loss, default=float("inf") if gross_profit > 0 else 0), 2)

    avg_win = _safe_div(sum(t["pnl"] for t in wins), len(wins))
    avg_loss = _safe_div(abs(sum(t["pnl"] for t in losses)), len(losses))
    expectancy_per_trade = round(_safe_div(total_pnl, total), 4)

    balance = paper_trader.get_balance(conn, cfg.VIRTUAL_BALANCE_START)
    total_return_pct = round((balance - cfg.VIRTUAL_BALANCE_START) / cfg.VIRTUAL_BALANCE_START * 100, 2) \
        if cfg.VIRTUAL_BALANCE_START else 0.0

    max_dd_pct, max_dd_days = _max_drawdown(equity_curve)
    longest_win_streak, longest_loss_streak = _streaks([t["result"] for t in trades if t["result"] in ("WIN", "LOSS")])
    sharpe = _sharpe_like(equity_curve)

    ci = _confidence_interval_pp(win_rate, rr_total)

    # --- شکست به تفکیک نماد ---
    by_symbol = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        s = by_symbol[t["symbol"]]
        s["trades"] += 1
        s["wins"] += 1 if t["result"] == "WIN" else 0
        s["rr"] = s.get("rr", 0) + (1 if t["result"] in ("WIN", "LOSS") else 0)
        s["pnl"] += t["pnl"]
    by_symbol_list = []
    for sym, s in by_symbol.items():
        by_symbol_list.append({
            "symbol": sym, "trades": s["trades"],
            "win_rate": round(s["wins"] / s["rr"] * 100, 1) if s.get("rr") else 0,
            "pnl": round(s["pnl"], 2),
        })
    by_symbol_list.sort(key=lambda x: x["pnl"])

    # --- شکست به تفکیک جهت (خرید/فروش) ---
    by_side = {}
    for side in ("LONG", "SHORT"):
        side_trades = [t for t in trades if t["side"] == side]
        n = len(side_trades)
        w = sum(1 for t in side_trades if t["result"] == "WIN")
        nr = sum(1 for t in side_trades if t["result"] in ("WIN", "LOSS"))
        by_side[side] = {
            "trades": n, "win_rate": round(w / nr * 100, 1) if nr else 0,
            "pnl": round(sum(t["pnl"] for t in side_trades), 2),
        }

    # --- شکست به تفکیک تعداد تاییدهای چند-تایم‌فریمی ---
    by_htf = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    htf_weighted = bool(getattr(cfg, "HTF_WEIGHTED", False))
    for t in trades:
        if t["htf_agree"] is None:
            continue
        # حالت وزن‌دار: درصد وزن موافق، در دسته‌های ۱۰ تایی
        k = min(int(t["htf_agree"] // 10 * 10), 90) if htf_weighted else t["htf_agree"]
        by_htf[k]["trades"] += 1
        by_htf[k]["wins"] += 1 if t["result"] == "WIN" else 0
        by_htf[k]["rr"] = by_htf[k].get("rr", 0) + (1 if t["result"] in ("WIN", "LOSS") else 0)
        by_htf[k]["pnl"] += t["pnl"]
    by_htf_list = sorted([
        {"agree_count": k, "label": (f"{k}-{k + 10}٪" if htf_weighted else f"{k}/5"), "trades": v["trades"],
         "win_rate": round(v["wins"] / v["rr"] * 100, 1) if v.get("rr") else 0,
         "pnl": round(v["pnl"], 2)}
        for k, v in by_htf.items()
    ], key=lambda x: x["agree_count"])

    # --- شکست ماهانه (برای دیدن ثبات در رژیم‌های مختلف بازار) ---
    by_month = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        if not t["close_time"]:
            continue
        month_key = t["close_time"][:7]  # YYYY-MM
        by_month[month_key]["trades"] += 1
        by_month[month_key]["wins"] += 1 if t["result"] == "WIN" else 0
        by_month[month_key]["rr"] = by_month[month_key].get("rr", 0) + (1 if t["result"] in ("WIN", "LOSS") else 0)
        by_month[month_key]["pnl"] += t["pnl"]
    by_month_list = sorted([
        {"month": k, "trades": v["trades"],
         "win_rate": round(v["wins"] / v["rr"] * 100, 1) if v.get("rr") else 0,
         "pnl": round(v["pnl"], 2)}
        for k, v in by_month.items()
    ], key=lambda x: x["month"])

    # --- شکست به تفکیک نوع خروج (TP معمولی / SL اولیه / تریلینگ / سربه‌سر) ---
    # این دقیقاً همون چیزیه که برای سنجش "آیا تریلینگ استاپ ارزش داره یا نه" لازمه:
    # اگه TRAIL_SL و BREAKEVEN رو جدا از TP/SL عادی بشماریم، می‌بینیم این مکانیزم
    # واقعاً به نتیجه کمک می‌کنه یا نه.
    by_exit_type = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        et = t.get("exit_type") or "UNKNOWN"
        by_exit_type[et]["trades"] += 1
        by_exit_type[et]["wins"] += 1 if t["result"] == "WIN" else 0
        by_exit_type[et]["rr"] = by_exit_type[et].get("rr", 0) + (1 if t["result"] in ("WIN", "LOSS") else 0)
        by_exit_type[et]["pnl"] += t["pnl"]
    by_exit_type_list = sorted([
        {"exit_type": k, "trades": v["trades"],
         "win_rate": round(v["wins"] / v["rr"] * 100, 1) if v.get("rr") else 0,
         "pnl": round(v["pnl"], 2)}
        for k, v in by_exit_type.items()
    ], key=lambda x: -x["trades"])

    # --- شکست به تفکیک استراتژی ---
    by_strategy = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        sn = t.get("strategy_name") or "نامشخص"
        by_strategy[sn]["trades"] += 1
        by_strategy[sn]["wins"] += 1 if t["result"] == "WIN" else 0
        by_strategy[sn]["rr"] = by_strategy[sn].get("rr", 0) + (1 if t["result"] in ("WIN", "LOSS") else 0)
        by_strategy[sn]["pnl"] += t["pnl"]
    by_strategy_list = sorted([
        {"strategy": k, "trades": v["trades"],
         "win_rate": round(v["wins"] / v["rr"] * 100, 1) if v.get("rr") else 0,
         "pnl": round(v["pnl"], 2)}
        for k, v in by_strategy.items()
    ], key=lambda x: -x["trades"])

    # --- شکست به تفکیک امتیاز ورود (آیا امتیاز بالاتر واقعاً بهتره؟) ---
    by_score = defaultdict(lambda: {"trades": 0, "wins": 0, "rr": 0, "pnl": 0.0})
    for t in trades:
        sc = t.get("score")
        if sc is None:
            continue
        b = min(int(sc // 10 * 10), 90)
        d = by_score[b]
        d["trades"] += 1
        d["wins"] += 1 if t["result"] == "WIN" else 0
        d["rr"] += 1 if t["result"] in ("WIN", "LOSS") else 0
        d["pnl"] += t["pnl"]
    by_score_list = [{"bucket": f"{k}-{min(k + 10, 100)}", "trades": v["trades"],
                      "win_rate": round(v["wins"] / v["rr"] * 100, 1) if v["rr"] else 0,
                      "pnl": round(v["pnl"], 2)} for k, v in sorted(by_score.items())]

    # --- خلاصه‌ی لاگ سیگنال (چرا سیگنال‌ها اجرا نشدن) ---
    signal_summary = _summarize_signal_log(conn)

    # --- آمار لوریج/مارجین ---
    leverages = [t["leverage"] for t in trades if t["leverage"]]
    avg_leverage = round(sum(leverages) / len(leverages), 2) if leverages else 0

    overview = {
        "total_trades": total, "rr_trades": rr_total, "wins": len(wins), "losses": len(losses), "win_rate": win_rate,
        "trail_trades": len(trails), "trail_pnl": round(sum(t["pnl"] for t in trails), 2),
        "trail_pct": round(sum(t["pnl"] for t in trails) / cfg.VIRTUAL_BALANCE_START * 100, 2)
        if cfg.VIRTUAL_BALANCE_START else 0.0,
        "trail_positive": sum(1 for t in trails if t["pnl"] > 0),
        "other_trades": len(others), "other_pnl": round(sum(t["pnl"] for t in others), 2),
        "win_rate_95ci_pp": round(ci, 1) if ci else None,
        "profit_factor": profit_factor, "expectancy_per_trade": expectancy_per_trade,
        "avg_win": round(avg_win, 4), "avg_loss": round(avg_loss, 4),
        "total_pnl": total_pnl, "total_fees": total_fees, "total_return_pct": total_return_pct,
        "final_balance": round(balance, 2),
        "max_drawdown_pct": max_dd_pct, "max_drawdown_days": max_dd_days,
        "longest_win_streak": longest_win_streak, "longest_loss_streak": longest_loss_streak,
        "sharpe_like": sharpe, "avg_leverage": avg_leverage,
    }

    issues = _detect_issues(overview, by_symbol_list, by_side, by_htf_list, by_month_list, cfg, meta,
                             by_exit_type_list, signal_summary)
    verdict = _verdict(overview, issues)

    return {
        "overview": overview,
        "by_symbol": by_symbol_list,
        "by_side": by_side,
        "by_htf_agreement": by_htf_list,
        "by_month": by_month_list,
        "by_exit_type": by_exit_type_list,
        "by_strategy": by_strategy_list,
        "by_score": by_score_list,
        "signal_summary": signal_summary,
        "issues": issues,
        "verdict": verdict,
        "symbol_meta": meta.get("symbols", {}),
    }


def _summarize_signal_log(conn):
    try:
        rows = conn.execute("SELECT opened, rejection_reason FROM signal_log").fetchall()
    except Exception:
        return None
    total = len(rows)
    opened = sum(1 for o, _ in rows if o)
    reasons = defaultdict(int)
    for opened_flag, reason in rows:
        if not opened_flag and reason:
            reasons[reason] += 1
    return {
        "total_signals": total, "opened": opened, "rejected": total - opened,
        "rejection_reasons": dict(reasons),
    }


def _detect_issues(ov, by_symbol, by_side, by_htf, by_month, cfg, meta, by_exit_type=None, signal_summary=None):
    issues = []
    by_exit_type = by_exit_type or []

    # ۱) حجم نمونه
    if ov["total_trades"] < 30:
        issues.append({"severity": "critical", "code": "SAMPLE_TOO_SMALL",
                        "message": f"فقط {ov['total_trades']} معامله بسته شده — عملاً هیچ نتیجه‌ای از این قابل استخراج نیست.",
                        "suggestion": "بازه‌ی بک‌تست رو زیاد کن (مثلاً به ۲ سال) یا تعداد نمادها رو بیشتر کن."})
    elif ov["total_trades"] < 100:
        issues.append({"severity": "warning", "code": "SAMPLE_SMALL",
                        "message": f"{ov['total_trades']} معامله بسته شده — کمتر از حداقل قابل‌قبول آماری (۱۰۰ معامله). "
                                   f"بازه‌ی اطمینان ۹۵٪ وین ریت خیلی گسترده‌ست (±{ov['win_rate_95ci_pp']} واحد درصد).",
                        "suggestion": "قبل از هر تصمیمی، بازه یا تعداد نماد رو بیشتر کن تا حداقل ۱۰۰-۱۵۰ معامله جمع بشه."})

    # ۲) سودآوری خام
    if ov["profit_factor"] < 1.0:
        issues.append({"severity": "critical", "code": "NEGATIVE_EXPECTANCY",
                        "message": f"Profit Factor = {ov['profit_factor']} (زیر ۱ یعنی در مجموع ضررده، حتی قبل از احتساب سختی اجرای واقعی).",
                        "suggestion": "این ترکیب پارامتر آماده‌ی پول واقعی نیست. یا MIN_RISK_REWARD رو بالاتر ببر، یا PROXIMITY_PCT رو سخت‌گیرتر کن، یا نمادهای ضررده رو (از جدول تفکیک نماد) حذف کن."})
    elif ov["profit_factor"] < 1.3:
        issues.append({"severity": "warning", "code": "THIN_MARGIN",
                        "message": f"Profit Factor = {ov['profit_factor']} — حاشیه‌ی سود نازکه، با کوچیک‌ترین بدشانسی یا اختلاف کارمزد واقعی می‌تونه منفی بشه.",
                        "suggestion": "قبل از پول واقعی، حاشیه‌ی امن‌تری (حداقل ۱.۵) هدف بگیر؛ می‌تونی MIN_RISK_REWARD رو کمی بالاتر ببری."})

    # ۳) افت سرمایه
    if ov["max_drawdown_pct"] > 35:
        issues.append({"severity": "critical", "code": "DRAWDOWN_TOO_HIGH",
                        "message": f"بیشینه افت سرمایه {ov['max_drawdown_pct']}٪ بوده (در {ov['max_drawdown_days']} روز) — برای سرمایه‌ی واقعی خیلی خطرناکه.",
                        "suggestion": "درصد ریسک هر معامله رو کم کن (مثلاً از ۰.۵٪ به ۰.۲۵٪)، یا MAX_LEVERAGE رو پایین بیار، یا MAX_OPEN_POSITIONS رو محدودتر کن."})
    elif ov["max_drawdown_pct"] > 20:
        issues.append({"severity": "warning", "code": "DRAWDOWN_ELEVATED",
                        "message": f"بیشینه افت سرمایه {ov['max_drawdown_pct']}٪ بوده — قابل‌تحمله ولی مرزیه.",
                        "suggestion": "اگه با پول واقعی راحت نیستی این‌قدر افت رو تحمل کنی، ریسک هر معامله یا لوریج رو یه‌کم کمتر کن."})

    # ۴) رشته‌ی باخت طولانی نسبت به ریسک فعلی
    theoretical_dd_from_streak = ov["longest_loss_streak"] * cfg.RISK_PER_TRADE_PCT
    if theoretical_dd_from_streak > 15:
        issues.append({"severity": "warning", "code": "LOSING_STREAK_RISK",
                        "message": f"طولانی‌ترین رشته‌ی باخت پشت‌سرهم {ov['longest_loss_streak']} معامله بوده؛ با ریسک فعلی "
                                   f"({cfg.RISK_PER_TRADE_PCT}٪ هر معامله)، یعنی تقریباً {round(theoretical_dd_from_streak,1)}٪ افت پشت‌سرهم ممکنه.",
                        "suggestion": "اگه این رشته دوباره تکرار بشه و روانی نتونی تحملش کنی، ریسک هر معامله رو کمتر انتخاب کن."})

    # ۵) هزینه‌ی کارمزد
    if ov["total_pnl"] > 0 and ov["total_fees"] > 0:
        fee_drag = ov["total_fees"] / (ov["total_pnl"] + ov["total_fees"]) * 100
        if fee_drag > 30:
            issues.append({"severity": "warning", "code": "FEE_DRAG_HIGH",
                            "message": f"کارمزد معادل {round(fee_drag,1)}٪ از سود ناخالص رو خورده.",
                            "suggestion": "می‌تونی COOLDOWN_HOURS رو بیشتر کنی یا فیلترها رو سخت‌گیرتر کنی تا معاملات کم‌تر ولی باکیفیت‌تر بشن."})

    # ۶) تمرکز روی چند نماد خاص
    if by_symbol and ov["total_pnl"] != 0:
        top_contrib = max(abs(s["pnl"]) for s in by_symbol)
        if top_contrib / max(abs(ov["total_pnl"]), 1) > 0.4 and len(by_symbol) > 3:
            worst_or_best = max(by_symbol, key=lambda s: abs(s["pnl"]))
            issues.append({"severity": "warning", "code": "SYMBOL_CONCENTRATION",
                            "message": f"نماد {worst_or_best['symbol']} به تنهایی {round(abs(worst_or_best['pnl'])/max(abs(ov['total_pnl']),1)*100,1)}٪ "
                                       f"از کل سود/زیان رو ساخته — نتیجه‌ی کلی خیلی به یک نماد وابسته‌ست.",
                            "suggestion": "این یعنی نتیجه ممکنه تعمیم‌پذیر نباشه؛ با تعداد نماد بیشتر دوباره تست کن تا مطمئن بشی شانسی نبوده."})

    # ۷) نمادهای واضحاً ضررده (نمونه‌ی کافی داشته باشن)
    bad_symbols = [s for s in by_symbol if s["trades"] >= 8 and s["win_rate"] < 30 and s["pnl"] < 0]
    if bad_symbols:
        names = "، ".join(s["symbol"] for s in bad_symbols[:5])
        issues.append({"severity": "info", "code": "CONSISTENTLY_BAD_SYMBOLS",
                        "message": f"این نمادها به‌طور مشخص ضررده بودن: {names}",
                        "suggestion": "می‌تونی این‌ها رو از EXCLUDE_KEYWORDS یا لیست دستی نمادها کنار بذاری."})

    # ۸) عدم تقارن جهت (خرید در برابر فروش)
    l, s = by_side.get("LONG", {}), by_side.get("SHORT", {})
    if l.get("trades", 0) >= 20 and s.get("trades", 0) >= 20:
        diff = abs(l["win_rate"] - s["win_rate"])
        if diff > 15:
            worse = "SHORT" if s["win_rate"] < l["win_rate"] else "LONG"
            issues.append({"severity": "warning", "code": "SIDE_ASYMMETRY",
                            "message": f"وین ریت خرید {l['win_rate']}٪ در برابر فروش {s['win_rate']}٪ — اختلاف {round(diff,1)} واحد درصد.",
                            "suggestion": f"سمت {worse} رو جدا بررسی کن؛ ممکنه یه بایاس یا باگ ظریف توی منطق تشخیص روند/سطح اون سمت باشه، یا صرفاً بازار توی این بازه بیشتر یک‌طرفه بوده."})

    # ۹) ارزش تایید چند-تایم‌فریمی
    if len(by_htf) >= 2:
        low = min(by_htf, key=lambda x: x["agree_count"])
        high = max(by_htf, key=lambda x: x["agree_count"])
        if low["trades"] >= 10 and high["trades"] >= 10:
            if high["win_rate"] - low["win_rate"] < 3:
                issues.append({"severity": "info", "code": "HTF_NOT_ADDING_VALUE",
                                "message": f"درصد برد با تایید HTF «{low['label']}» ({low['win_rate']}٪) و با "
                                           f"«{high['label']}» ({high['win_rate']}٪) تقریباً فرقی نداره.",
                                "suggestion": "شاید HTF_MIN_AGREEMENT فعلی داره فقط تعداد معامله رو کم می‌کنه بدون بهبود واقعی کیفیت؛ می‌تونی امتحان کنی با HTF_MIN_AGREEMENT پایین‌تر همون کیفیت با معاملات بیشتر بگیری."})
            else:
                issues.append({"severity": "info", "code": "HTF_ADDS_VALUE",
                                "message": f"تایید بیشتر تایم‌فریم بالاتر واقعاً کیفیت رو بالا می‌بره: «{low['label']}» = {low['win_rate']}٪ "
                                           f"در برابر «{high['label']}» = {high['win_rate']}٪.",
                                "suggestion": "می‌تونی حتی HTF_MIN_AGREEMENT رو سخت‌گیرتر هم بکنی (مثلاً بالاتر ببری) اگه تعداد معامله برات مهم‌تر از کیفیته."})

    # ۱۰) ثبات در طول زمان (رژیم‌های مختلف بازار)
    weak_months = [m for m in by_month if m["trades"] >= 5 and m["win_rate"] < 25]
    if weak_months and len(by_month) >= 3:
        names = "، ".join(m["month"] for m in weak_months[:4])
        issues.append({"severity": "info", "code": "REGIME_WEAKNESS",
                        "message": f"توی این ماه‌ها عملکرد خیلی ضعیف بوده: {names}",
                        "suggestion": "ببین توی بازار اون ماه‌ها چه اتفاقی افتاده (رنج شدید؟ نوسان خیلی زیاد؟)؛ شاید نیاز به فیلتر نوسان بازار (ATR نسبی) داشته باشی."})

    # ۱۱) نمادهایی که دیتاشون گرفته نشد
    failed_symbols = [sym for sym, m in meta.get("symbols", {}).items() if not m.get("ok")]
    if failed_symbols:
        issues.append({"severity": "info", "code": "DATA_FETCH_FAILURES",
                        "message": f"دیتای {len(failed_symbols)} نماد قابل‌دریافت نبود: {', '.join(failed_symbols[:5])}"
                                   f"{' و ...' if len(failed_symbols) > 5 else ''}",
                        "suggestion": "این نمادها از نتیجه‌ی بک‌تست کنار موندن؛ اگه مهمن، دوباره امتحان کن یا صرافی دیگه رو اول لیست EXCHANGE_TRY_ORDER بذار."})

    # ۱۲) ارزش تریلینگ استاپ (اگه فعال بوده): مقایسه‌ی TRAIL_SL با TP معمولی
    trail = next((e for e in by_exit_type if e["exit_type"] == "TRAIL_SL"), None)
    tp_normal = next((e for e in by_exit_type if e["exit_type"] == "TP"), None)
    sl_normal = next((e for e in by_exit_type if e["exit_type"] == "SL"), None)
    breakeven = next((e for e in by_exit_type if e["exit_type"] == "BREAKEVEN"), None)
    if trail and trail["trades"] >= 10:
        avg_pnl_trail = trail["pnl"] / trail["trades"]
        # حد سود ثابت ۲R همیشه فعاله و تریلینگ فقط وقتی می‌بنده که قیمت به ۲R نرسیده و برگشته؛
        # پس مقایسه‌ی میانگین تریلینگ با TP بی‌معنیه. فقط اگه تریلینگ در مجموع ضرر داده، هشدار بده.
        if trail["pnl"] < 0:
            issues.append({"severity": "warning", "code": "TRAILING_NEGATIVE",
                            "message": f"معاملاتی که با تریلینگ بسته شدن در مجموع ${round(trail['pnl'],2)} ضرر دادن (بعد از کارمزد).",
                            "suggestion": "پله‌ی اول نردبان (TRAILING_SL_LADDER) رو بالاتر ببر یا تریلینگ رو خاموش کن."})
        else:
            issues.append({"severity": "info", "code": "TRAILING_INFO",
                            "message": f"{trail['trades']} معامله با تریلینگ و در مجموع ${round(trail['pnl'],2)} بسته شدن "
                                       f"(میانگین ${round(avg_pnl_trail,2)}).",
                            "suggestion": "برای اینکه بفهمی تریلینگ واقعاً بهتره یا نه، همین تست رو با تریلینگ خاموش هم بگیر "
                                          "(یا از «مقایسه‌ی خودکار» استفاده کن) و سود کل رو مقایسه کن."})
        if breakeven and breakeven["trades"] >= max(5, trail["trades"] * 0.3):
            issues.append({"severity": "info", "code": "TRAILING_MANY_BREAKEVENS",
                            "message": f"{breakeven['trades']} معامله دقیقاً روی سربه‌سر (بدون سود/زیان قابل‌توجه) با تریلینگ بسته شدن.",
                            "suggestion": "این طبیعیه (تریلینگ محافظه‌کارانه عمل کرده)، ولی اگه تعدادش زیاده، شاید پله‌ی اول نردبان (TRAILING_SL_LADDER) رو دیرتر (R بالاتر) تنظیم کنی بهتر باشه."})

    # ۱۳) خلاصه‌ی رد شدن سیگنال‌ها (اگه لاگ سیگنال موجود بود)
    if signal_summary and signal_summary["total_signals"] >= 20:
        rejected_pct = round(signal_summary["rejected"] / signal_summary["total_signals"] * 100, 1)
        if rejected_pct > 70:
            top_reason = max(signal_summary["rejection_reasons"].items(), key=lambda x: x[1], default=(None, 0))
            issues.append({"severity": "info", "code": "HIGH_SIGNAL_REJECTION",
                            "message": f"{rejected_pct}٪ سیگنال‌های تولیدشده هیچ‌وقت اجرا نشدن. بیشترین دلیل رد شدن: "
                                       f"«{top_reason[0]}» ({top_reason[1]} بار).",
                            "suggestion": "اگه دلیل غالب htf_disagreement بود، شاید سطح سخت‌گیری خیلی بالاست؛ اگه max_positions_reached "
                                          "یا insufficient_capital بود، شاید MAX_OPEN_POSITIONS یا MAX_POSITION_PCT_OF_CAPITAL محدودکننده‌ست."})

    return issues


def _verdict(ov, issues):
    critical = [i for i in issues if i["severity"] == "critical"]
    warnings = [i for i in issues if i["severity"] == "warning"]

    if critical:
        return {"level": "NOT_READY", "label": "❌ آماده نیست",
                "reason": "حداقل یک ایراد بحرانی وجود داره که باید قبل از هر قدم بعدی برطرف بشه."}
    if ov["total_trades"] < 100:
        return {"level": "NEED_MORE_DATA", "label": "⏳ نیاز به نمونه‌ی بیشتر",
                "reason": "نمونه برای نتیجه‌گیری قابل‌اعتماد کافی نیست، حتی اگه فعلاً بد به‌نظر نمی‌رسه."}
    if len(warnings) >= 3:
        return {"level": "CAUTION", "label": "⚠️ با احتیاط",
                "reason": "چند هشدار هم‌زمان وجود داره؛ قبل از پول واقعی باید بررسی و رفعشون کرد."}
    if ov["profit_factor"] >= 1.3 and ov["max_drawdown_pct"] <= 20 and ov["win_rate"] >= 40:
        return {"level": "PROMISING", "label": "✅ امیدوارکننده",
                "reason": "معیارهای اصلی (سودآوری، افت سرمایه، وین ریت) در محدوده‌ی قابل‌قبول هستن. همچنان با سرمایه‌ی کوچیک و مانیتورینگ نزدیک شروع کن."}
    return {"level": "MIXED", "label": "🔶 نتیجه‌ی مبهم",
            "reason": "نه واضحاً خرابه نه واضحاً خوب؛ به هشدارهای بالا نگاه کن و پارامترها رو اصلاح کن."}
