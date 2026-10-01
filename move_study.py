# -*- coding: utf-8 -*-
"""
سنجش «فاصله‌ی حد ضرر × کارمزد × زمان» روی کندل‌های واقعی (بدون هیچ استراتژی).

سؤال: اگه هر معامله دقیقاً ۱ دلار ضرر خالص (با همه‌ی کارمزدها) و ۲ یا ۲.۵ دلار سود خالص داشته باشه، با چه فاصله‌ی
حد ضرری (به درصد قیمت):
  - اندازه‌ی پوزیشن (و لوریج لازم برای یک مارجین مشخص) چقدره،
  - کارمزد چند درصد از اون ۱ دلار رو می‌خوره،
  - معامله معمولاً چقدر طول می‌کشه تا تکلیفش روشن بشه (و چند درصد ظرف ۳ ساعت)،
  - و ورود شانسی (بدون هیچ مزیتی) به‌طور میانگین چند دلار می‌ده؟ (یعنی «مانعی» که ورود باید ازش رد بشه)

ورود: بسته‌شدن یک کندل ۵ دقیقه‌ای (هر WALK_STEP کندل یک‌بار، هم خرید هم فروش). مسیر: کندل‌های ۵ دقیقه‌ای بعدی
با سقف/کف؛ اگه حد ضرر و حد سود در یک کندل خوردن، ضرر حساب می‌شه (بدبینانه). تا MAX_HOURS ساعت؛ بعدش با قیمت
بازار بسته می‌شه.

کارمزد (از config): ورود لیمیت = میکر، حد سود لیمیت = میکر، حد ضرر = تیکر + اسلیپیج، فاندینگ هر ۸ ساعت.
"""
import numpy as np

STOP_PCTS = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
RR_LIST = [2.0, 2.5, 3.0, 4.0]
WALK_STEP = 6          # هر ۳۰ دقیقه یک ورود نمونه
MAX_HOURS = 48
BAR_MIN = 5
RISK_USD = 1.0


def fee_model(cfg):
    fe = float(cfg.MAKER_FEE_PCT) / 100.0                                        # ورود
    f_sl = (float(cfg.TAKER_FEE_PCT) + float(cfg.TAKER_SLIPPAGE_PCT)) / 100.0      # خروج با حد ضرر
    f_tp = float(cfg.MAKER_FEE_PCT) / 100.0                                       # خروج با حد سود (لیمیت)
    f_mk = (float(cfg.TAKER_FEE_PCT) + float(cfg.TAKER_SLIPPAGE_PCT)) / 100.0      # خروج بازار (پایان مهلت)
    fund_h = float(getattr(cfg, "FUNDING_PCT_PER_8H", 0.0) or 0.0) / 100.0 / 8.0
    return fe, f_sl, f_tp, f_mk, fund_h


def levels(d, rr, fees):
    """
    d: فاصله‌ی حد ضرر (کسر قیمت). خروجی: اندازه‌ی پوزیشن N (دلار) و فاصله‌ی حد سود u (کسر قیمت) طوری که
    ضرر خالص با کارمزد = RISK_USD و سود خالص با کارمزد = rr × RISK_USD (فاندینگ جدا حساب می‌شه).
    """
    fe, f_sl, f_tp, _, _ = fees
    n = RISK_USD / (d + fe + f_sl)
    u = rr * RISK_USD / n + fe + f_tp
    return n, u


def simulate_symbol(arr5, fees, step=WALK_STEP, max_hours=MAX_HOURS):
    """arr5: کندل‌های ۵ دقیقه‌ای [ts, o, h, l, c, v]. خروجی: dict (d, rr) → لیست (pnl, minutes, resolved, side)."""
    fe, f_sl, f_tp, f_mk, fund_h = fees
    h, l, c = arr5[:, 2], arr5[:, 3], arr5[:, 4]
    n = len(c)
    W = int(max_hours * 60 / BAR_MIN)
    out = {(d, rr): [] for d in STOP_PCTS for rr in RR_LIST}
    lv = {(d, rr): levels(d / 100.0, rr, fees) for d in STOP_PCTS for rr in RR_LIST}
    for t in range(0, n - W - 1, step):
        e = float(c[t])
        if not e > 0:
            continue
        hi = h[t + 1:t + 1 + W] / e - 1.0
        lo = l[t + 1:t + 1 + W] / e - 1.0
        last = float(c[t + W]) / e - 1.0
        for (d, rr), (N, u) in lv.items():
            dd = d / 100.0
            for side in (1, -1):
                if side == 1:
                    sl_hit = lo <= -dd
                    tp_hit = hi >= u
                else:
                    sl_hit = hi >= dd
                    tp_hit = lo <= -u
                i_sl = int(np.argmax(sl_hit)) if sl_hit.any() else W
                i_tp = int(np.argmax(tp_hit)) if tp_hit.any() else W
                if i_sl == W and i_tp == W:
                    k = W
                    pnl = N * side * last - N * (fe + f_mk)
                    resolved = False
                elif i_sl <= i_tp:
                    k = i_sl + 1
                    pnl = -RISK_USD
                    resolved = True
                else:
                    k = i_tp + 1
                    pnl = rr * RISK_USD
                    resolved = True
                minutes = k * BAR_MIN
                pnl -= N * fund_h * minutes / 60.0
                out[(d, rr)].append((pnl, minutes, resolved))
    return out


def atr_pct(arr, period=14):
    """میانه‌ی ATR/قیمت (٪) — برای این‌که بدونیم هر فاصله‌ی حد ضرر در هر تایم‌فریم چند ATR می‌شه."""
    if arr is None or len(arr) < period + 5:
        return None
    h, l, c = arr[:, 2], arr[:, 3], arr[:, 4]
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    atr = np.convolve(tr, np.ones(period) / period, mode="valid")
    return float(np.median(atr / c[period - 1:] * 100.0))


def summarize(per_symbol, fees, margin_usd=20.0):
    rows = []
    for d in STOP_PCTS:
        for rr in RR_LIST:
            allv = [x for s in per_symbol.values() for x in s.get((d, rr), [])]
            if not allv:
                continue
            pnl = np.array([x[0] for x in allv])
            mins = np.array([x[1] for x in allv], dtype=float)
            res = np.array([x[2] for x in allv])
            N, u = levels(d / 100.0, rr, fees)
            fe, f_sl, f_tp, _, _ = fees
            fee_loss = N * (fe + f_sl)
            wins = (pnl > 0) & res
            # سربه‌سر: p × rr = (1 − p) × 1  →  p = 1 / (1 + rr)
            rows.append({
                "stop_pct": d, "rr": rr, "notional_usd": round(N, 1), "tp_pct": round(u * 100, 3),
                "leverage_for_margin": round(N / margin_usd, 1),
                "fee_on_loss_usd": round(fee_loss, 3), "fee_share_of_risk_pct": round(fee_loss / RISK_USD * 100, 1),
                "random_win_pct": round(wins.mean() * 100, 1),
                "breakeven_win_pct": round(100.0 / (1.0 + rr), 1),
                "random_avg_usd": round(float(pnl.mean()), 3),
                "median_minutes": round(float(np.median(mins)), 0),
                "resolved_3h_pct": round(float(((mins <= 180) & res).mean() * 100), 1),
                "resolved_24h_pct": round(float(((mins <= 1440) & res).mean() * 100), 1),
                "n": int(len(allv)),
            })
    return rows


def text_summary(rows, atrs, margin_usd=20.0):
    lines = ["سنجش «فاصله‌ی حد ضرر × کارمزد × زمان» — ورود شانسی، ضرر خالص ۱$، سود خالص ۲ یا ۲.۵$ (با همه‌ی کارمزدها)", ""]
    if atrs:
        lines.append("میانه‌ی ATR هر تایم‌فریم (٪ قیمت): " + "، ".join(f"{tf}: {v:.2f}٪" for tf, v in atrs.items() if v))
        lines.append("")
    lines.append(f"{'SL٪':>5} {'RR':>4} {'پوزیشن$':>8} {'لوریج(مارجین ' + str(int(margin_usd)) + '$)':>18} "
                 f"{'کارمزد از ۱$':>12} {'برد شانسی':>9} {'برد لازم':>8} {'میانگین$':>9} {'میانه‌ی زمان':>12} {'تموم≤۳ساعت':>10}")
    for r in rows:
        lines.append(f"{r['stop_pct']:>5} {r['rr']:>4} {r['notional_usd']:>8} {r['leverage_for_margin']:>18}x "
                     f"{r['fee_on_loss_usd']:>11}$ {r['random_win_pct']:>8}% {r['breakeven_win_pct']:>7}% "
                     f"{r['random_avg_usd']:>+9.3f} {int(r['median_minutes']):>9} دقیقه {r['resolved_3h_pct']:>9}%")
    return "\n".join(lines)
