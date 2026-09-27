# -*- coding: utf-8 -*-
"""
موتور شبیه‌سازی سریع بک‌تست: مسیر هر معامله + مدیریت سرمایه‌ی کل سبد.

دو اصلاح مهم دقت نسبت به موتور قبلی:
  ۱) همه‌ی نمادها «هم‌زمان و به ترتیب زمان واقعی» شبیه‌سازی می‌شن (قبلاً نماد به نماد
     از اول تا آخر اجرا می‌شد؛ یعنی سرمایه و سقف پوزیشن‌ها در یک نماد تحت تاثیر نتایجِ
     آینده‌ی نماد قبلی بود، و آخرین پوزیشنِ باز هر نماد تا ابد مارجین قفل می‌کرد).
  ۲) حد ضرر/حد سود با سایه‌ی کندل (high/low) فعال می‌شه، نه فقط با قیمت بسته‌شدن —
     دقیقاً مثل سفارش استاپ واقعی صرافی. اگه یک کندل هم SL و هم TP رو لمس کنه، فرض
     بدبینانه (اول SL) گرفته می‌شه. گپ قیمتی هم با قیمت واقعی باز شدن حساب می‌شه.

قانون مدیریت هر کندل همون paper_trader.step_bar ربات زنده‌ست (نسخه‌ی وکتوریزه‌ی
دقیقاً معادل؛ تست‌ها برابری رو تایید می‌کنن). حجم/لوریج/مارجین هم از همون
paper_trader.compute_position_size میاد. کارمزد میکر/تیکر و اسلیپیج هم همون تابع زنده.
"""
import heapq
import math
from datetime import datetime

import numpy as np

import paper_trader
import signals_engine as se


# ==================== مسیر یک معامله (وکتوریزه) ====================

def _trail_vec(P, E, SL0, risk, trig_adj, locks, last_trigger, beyond):
    """نسخه‌ی آرایه‌ای paper_trader.compute_trailing_sl برای خرید (فروش با قرینه‌سازی)."""
    peak_r = (P - E) / risk
    idx = np.searchsorted(trig_adj, peak_r, "right") - 1
    has = idx >= 0
    locked = np.where(has, locks[np.clip(idx, 0, len(locks) - 1)], 0.0)
    beyond_mask = has & (peak_r > last_trigger + 1e-9)
    locked = np.where(beyond_mask, np.maximum(locked, peak_r - beyond), locked)
    new_sl = E + locked * risk
    return np.where(has, np.maximum(new_sl, SL0), SL0)


class PathSim:
    """شبیه‌ساز مسیر معاملات یک نماد، با کش نتایج (هر کاندید فقط یک‌بار شبیه‌سازی می‌شه)."""

    def __init__(self, series, ladder, beyond):
        self.s = series
        lad = sorted(ladder, key=lambda x: x[0])
        self.has_ladder = len(lad) > 0
        self.trig_adj = np.array([t - 1e-9 for t, _ in lad], dtype=np.float64)
        self.locks = np.array([lk for _, lk in lad], dtype=np.float64)
        self.last_trigger = lad[-1][0] if lad else 0.0
        self.beyond = beyond
        self.cache = {}

    def _slices(self, side, k, k1):
        """برش کندل‌ها؛ برای فروش قرینه می‌شن تا همون منطق خرید استفاده بشه
        (منفی کردن قیمت در IEEE دقیق و بدون خطاست، پس نتیجه دقیقاً یکیه)."""
        s = self.s
        if side == se.LONG:
            return s.o[k:k1], s.h[k:k1], s.l[k:k1], s.c[k:k1]
        return -s.o[k:k1], -s.l[k:k1], -s.h[k:k1], -s.c[k:k1]

    def run(self, t0, side, entry, sl0, tp, trailing, max_bars=0):
        key = (t0, side, entry, sl0, tp, trailing, max_bars)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        res = self._run(t0, side, entry, sl0, tp, trailing, max_bars)
        self.cache[key] = res
        return res

    def _run(self, t0, side, entry, sl0, tp, trailing, max_bars=0):
        """مدیریت از کندل t0+1. max_bars>0 یعنی حد زمانی: اگه تا اون تعداد کندل بسته نشد،
        در قیمت بسته شدن آخرین کندل مجاز با سفارش بازار بسته می‌شه (TIME)."""
        sgn = 1.0 if side == se.LONG else -1.0
        E, SL0, TP = sgn * entry, sgn * sl0, sgn * tp
        risk = abs(entry - sl0)
        n_all = self.s.n
        n = min(n_all, t0 + 1 + max_bars) if max_bars and max_bars > 0 else n_all
        k = t0 + 1
        cur_sl, peak = SL0, E
        trailing = trailing and self.has_ladder and risk > 0
        chunk = 48
        while k < n:
            k1 = min(n, k + chunk)
            oo, hh, ll, cc = self._slices(side, k, k1)
            if not trailing:
                stop = ll <= cur_sl
                tph = hh >= TP
                hitm = stop | tph
                if hitm.any():
                    i = int(np.argmax(hitm))
                    if stop[i]:
                        price = oo[i] if oo[i] < cur_sl else cur_sl
                        return (k + i, sgn * price, "STOP", sgn * cur_sl, sgn * peak)
                    price = oo[i] if oo[i] > TP else TP
                    return (k + i, sgn * price, "TP", sgn * TP, sgn * peak)
            else:
                P = np.maximum(np.maximum.accumulate(hh), peak)
                SLk = _trail_vec(P, E, SL0, risk, self.trig_adj, self.locks, self.last_trigger, self.beyond)
                SLk = np.maximum.accumulate(np.maximum(SLk, cur_sl))
                SLprev = np.empty_like(SLk)
                SLprev[0] = cur_sl
                SLprev[1:] = SLk[:-1]
                c1 = ll <= SLprev
                c2 = (SLk > SLprev) & (cc <= SLk)
                hitm = c1 | c2
                if hitm.any():
                    i = int(np.argmax(hitm))
                    if c1[i]:
                        lvl = SLprev[i]
                        price = oo[i] if oo[i] < lvl else lvl
                        return (k + i, sgn * price, "STOP", sgn * lvl, sgn * (P[i - 1] if i > 0 else peak))
                    return (k + i, sgn * SLk[i], "STOP", sgn * SLk[i], sgn * P[i])
                peak, cur_sl = P[-1], SLk[-1]
            k = k1
            chunk = min(chunk * 4, 20000)
        if n < n_all:
            return (n - 1, float(self.s.c[n - 1]), "TIME", None, sgn * peak)
        # تا آخر دیتا باز مونده: با آخرین قیمت (به‌صورت سفارش بازار) بسته حساب می‌شه
        return (n - 1, float(self.s.c[n - 1]), "END", None, sgn * peak)


# ==================== سبد (پورتفو) ====================

class SimParams:
    """پارامترهای مدیریت سرمایه/فیلترها که روی هر اجرای سبد اثر دارن."""

    def __init__(self, cfg, htf_min_agreement, trailing, allow_long=True, allow_short=True,
                 btc_filter=False, risk_pct=None):
        """htf_min_agreement: مقیاس «از ۵» (مثل پیش‌تنظیم‌های سخت‌گیری)؛ برای پروفایل‌هایی
        که تعداد تایم‌فریم تایید متفاوتی دارن، متناسب تبدیل می‌شه."""
        self.start_balance = float(cfg.VIRTUAL_BALANCE_START)
        self.risk_pct = float(risk_pct if risk_pct is not None else cfg.RISK_PER_TRADE_PCT)
        self.min_notional = cfg.MIN_NOTIONAL_USD
        self.max_open = cfg.MAX_OPEN_POSITIONS
        self.max_leverage = cfg.MAX_LEVERAGE
        self.lev_mult = cfg.LEVERAGE_SAFETY_MULTIPLIER
        self.pos_cap = cfg.MAX_POSITION_PCT_OF_CAPITAL
        self.cooldown_ms = int(cfg.COOLDOWN_HOURS * 3_600_000)
        self.maker = getattr(cfg, "MAKER_FEE_PCT", 0.0)
        self.taker = getattr(cfg, "TAKER_FEE_PCT", 0.0)
        self.slip = getattr(cfg, "TAKER_SLIPPAGE_PCT", 0.0)
        self.use_htf = bool(getattr(cfg, "USE_HTF_CONFIRMATION", True))
        n_tf = len(getattr(cfg, "HTF_TIMEFRAMES", []))
        self.htf_min_long, self.htf_min_short = htf_required(htf_min_agreement, n_tf,
                                                             getattr(cfg, "SHORT_EXTRA_HTF_AGREEMENT", 0))
        self.trailing = bool(trailing)
        self.allow_long = bool(allow_long)
        self.allow_short = bool(allow_short)
        self.btc_filter = bool(btc_filter)
        # اجرای سفارش و هزینه‌ها
        self.entry_mode = getattr(cfg, "ENTRY_MODE", "limit")
        tf_ms = _tf_ms(getattr(cfg, "TIMEFRAME", "15m"))
        self.tf_ms = tf_ms
        self.limit_wait_bars = max(1, int(getattr(cfg, "LIMIT_WAIT_BARS", 1)))
        hold_min = float(getattr(cfg, "MAX_HOLD_MINUTES", 0) or 0)
        self.max_hold_bars = int(math.ceil(hold_min * 60_000 / tf_ms)) if hold_min > 0 else 0
        self.funding_8h = float(getattr(cfg, "FUNDING_PCT_PER_8H", 0.0) or 0.0)


def _tf_ms(tf):
    import market_data
    return market_data.TF_MS.get(tf, 900_000)


def htf_required(preset_of_5, n_tf, short_extra=0):
    """حداقل تعداد تایید برای خرید/فروش. پیش‌تنظیم‌ها بر اساس ۵ تایم‌فریم تعریف شدن
    (۲، ۳، ۴، ۵ از ۵)؛ برای n تایم‌فریم به همون نسبت گرد به بالا تبدیل می‌شن."""
    n_tf = int(n_tf)
    if n_tf <= 0:
        return 0, 0
    req = int(preset_of_5) if n_tf == 5 else int(math.ceil(int(preset_of_5) * n_tf / 5.0 - 1e-9))
    req = max(1, min(n_tf, req))
    return req, min(n_tf, req + int(short_extra))


def merge_candidates(per_symbol, preps, symbol_order):
    """
    ادغام کاندیدهای همه‌ی نمادها به ترتیب زمان بسته‌شدن کندل (و بعد ترتیب نمادها، مثل
    ترتیب اسکن ربات زنده). per_symbol: dict symbol -> خروجی signals_engine.combine
    """
    cols = {k: [] for k in ("time", "rank", "sym", "idx", "side", "sl", "tp", "rr", "entry",
                            "htf_l", "htf_s", "btc")}
    labels = []
    for rank, sym in enumerate(symbol_order):
        cand = per_symbol.get(sym)
        if cand is None or len(cand["idx"]) == 0:
            continue
        p = preps[sym]
        idx = cand["idx"]
        cols["time"].append(p.series.close_ts[idx])
        cols["rank"].append(np.full(len(idx), rank, dtype=np.int32))
        cols["sym"].append(np.full(len(idx), rank, dtype=np.int32))
        cols["idx"].append(idx)
        cols["side"].append(cand["side"])
        cols["sl"].append(cand["sl"])
        cols["tp"].append(cand["tp"])
        cols["rr"].append(cand["rr"])
        cols["entry"].append(p.series.c[idx])
        cols["htf_l"].append(p.htf_long[idx])
        cols["htf_s"].append(p.htf_short[idx])
        cols["btc"].append(p.btc_trend[idx])
        labels.extend(cand["label"])
    if not cols["time"]:
        return None
    m = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((m["rank"], m["time"]))
    m = {k: v[order] for k, v in m.items()}
    m["label"] = [labels[i] for i in order.tolist()]
    return m


def run_portfolio(merged, preps, symbol_order, P, record=False):
    """
    اجرای ترتیبی سبد، دقیقاً با ترتیب چک‌های ربات زنده:
    پوزیشن باز روی همین نماد → کول‌داون → جهت مجاز → رژیم BTC → تایید HTF → باز کردن
    (سقف پوزیشن‌ها، سرمایه‌ی آزاد، حداقل حجم).
    خروجی: dict شامل trades (لیست)، equity (لیست (زمان، موجودی))، signals (اگه record)
    """
    trades, equity, signals = [], [], []
    balance = P.start_balance
    locked = 0.0
    open_by_sym = {}
    last_close = {}
    heap = []
    seq = 0
    start_t = int(merged["time"][0]) if merged is not None and len(merged["time"]) else 0
    equity.append((start_t, balance))

    if merged is None:
        return {"trades": trades, "equity": equity, "signals": signals, "final_balance": balance}

    side_arr, htf_l, htf_s, btc = merged["side"], merged["htf_l"], merged["htf_s"], merged["btc"]
    n = len(side_arr)
    reason = np.zeros(n, dtype=np.int8)   # ۰ = قبول فیلترهای ایستا
    if not P.allow_long:
        reason[(side_arr == se.LONG) & (reason == 0)] = 1
    if not P.allow_short:
        reason[(side_arr == se.SHORT) & (reason == 0)] = 1
    if P.btc_filter:
        reason[(side_arr == se.LONG) & (btc == -1) & (reason == 0)] = 2
        reason[(side_arr == se.SHORT) & (btc == 1) & (reason == 0)] = 2
    if P.use_htf:
        bad_l = (side_arr == se.LONG) & (htf_l < P.htf_min_long)
        bad_s = (side_arr == se.SHORT) & (htf_s < P.htf_min_short)
        reason[(bad_l | bad_s) & (reason == 0)] = 3
    static_reason = {1: "side_disabled", 2: "btc_regime", 3: "htf_disagreement"}

    iter_idx = np.arange(n) if record else np.flatnonzero(reason == 0)
    times = merged["time"]
    syms = merged["sym"]
    t_idx = merged["idx"]
    entries, sls, tps, rrs = merged["entry"], merged["sl"], merged["tp"], merged["rr"]
    labels = merged["label"]

    def flush(until):
        nonlocal balance, locked
        while heap and heap[0][0] <= until:
            exit_t, _, sym_rank, margin, pnl, real_trade = heapq.heappop(heap)
            locked -= margin
            open_by_sym.pop(sym_rank, None)
            if real_trade:
                balance += pnl
                last_close[sym_rank] = exit_t
                equity.append((exit_t, balance))

    for j in iter_idx.tolist():
        T = int(times[j])
        flush(T)
        sr = int(syms[j])
        if sr in open_by_sym:
            continue
        side = int(side_arr[j])
        htf_agree = int(htf_l[j] if side == se.LONG else htf_s[j])
        sig = None
        if record:
            sig = {"time": T, "symbol": symbol_order[sr], "side": "LONG" if side == se.LONG else "SHORT",
                   "strategy": labels[j], "entry": float(entries[j]), "sl": float(sls[j]), "tp": float(tps[j]),
                   "rr": float(rrs[j]), "htf_agree": htf_agree if P.use_htf else None,
                   "opened": 0, "reason": None}
        lc = last_close.get(sr)
        if lc is not None and T - lc < P.cooldown_ms:
            if record:
                sig["reason"] = "cooldown"
                sig["htf_agree"] = None
                signals.append(sig)
            continue
        if reason[j] != 0:
            if record:
                sig["reason"] = static_reason[int(reason[j])]
                signals.append(sig)
            continue

        side_str = "LONG" if side == se.LONG else "SHORT"
        sl, tp = float(sls[j]), float(tps[j])
        prep = preps[symbol_order[sr]]
        ser = prep.series
        t = int(t_idx[j])
        if t + 1 >= ser.n:
            continue   # سیگنال روی آخرین کندل دیتا؛ کندلی برای اجرا نمونده
        entry_taker = P.entry_mode == "market"
        if entry_taker:
            # ورود با سفارش بازار در قیمت باز شدن کندل بعد + اسلیپیج
            o_next = float(ser.o[t + 1])
            entry = o_next * (1 + P.slip / 100) if side == se.LONG else o_next * (1 - P.slip / 100)
            bad = (entry <= sl or entry >= tp) if side == se.LONG else (entry >= sl or entry <= tp)
            if bad:
                if record:
                    sig["reason"] = "gap_past_level"
                    signals.append(sig)
                continue
        else:
            entry = float(entries[j])

        pos = paper_trader.compute_position_size(
            side_str, entry, sl, P.risk_pct, balance, locked, len(open_by_sym),
            min_notional=P.min_notional, max_open_positions=P.max_open, max_leverage=P.max_leverage,
            leverage_safety_mult=P.lev_mult, position_pct_cap=P.pos_cap,
        )
        if not pos["ok"]:
            if record:
                sig["reason"] = pos["reason"]
                signals.append(sig)
            continue

        if entry_taker:
            t0, fill_t = t, int(ser.close_ts[t])
        else:
            # سفارش لیمیت: فقط اگه قیمت در چند کندل بعد واقعاً از قیمت ورود رد بشه، پر می‌شه
            k_end = min(ser.n - 1, t + P.limit_wait_bars)
            window = ser.l[t + 1:k_end + 1] < entry if side == se.LONG else ser.h[t + 1:k_end + 1] > entry
            if not window.any():
                # پر نشد: مارجین و جای پوزیشن تا انقضای سفارش رزرو بود، بعد آزاد (بدون کول‌داون)
                seq += 1
                heapq.heappush(heap, (int(ser.close_ts[k_end]), seq, sr, pos["margin"], 0.0, False))
                open_by_sym[sr] = True
                locked += pos["margin"]
                if record:
                    sig["reason"] = "limit_not_filled"
                    signals.append(sig)
                continue
            k_fill = t + 1 + int(np.argmax(window))
            t0, fill_t = k_fill - 1, int(ser.close_ts[k_fill - 1])

        exit_i, exit_price, kind, level, peak = prep.paths.run(t0, side, entry, sl, tp, P.trailing,
                                                               P.max_hold_bars)
        exit_type = paper_trader.classify_exit_level(kind, level, sl, entry, P.trailing)
        size = pos["size"]
        gross = (exit_price - entry) * size if side == se.LONG else (entry - exit_price) * size
        fee, _, _, _ = paper_trader._fee_for_exit(entry, exit_price, size, exit_type, P.maker, P.taker, P.slip,
                                                  entry_taker=entry_taker)
        exit_t = int(ser.close_ts[exit_i])
        funding = paper_trader.funding_cost(pos["notional"], fill_t, exit_t, P.funding_8h)
        fee += funding
        pnl = gross - fee
        risk_usd = pos["per_unit_risk"] * size
        seq += 1
        heapq.heappush(heap, (exit_t, seq, sr, pos["margin"], pnl, True))
        open_by_sym[sr] = True
        locked += pos["margin"]
        trades.append({
            "symbol": symbol_order[sr], "side": side_str, "entry": entry, "sl": sl, "tp": tp,
            "size": size, "notional": pos["notional"], "margin": pos["margin"], "leverage": pos["leverage"],
            "liquidation_price": pos["liquidation_price"], "strategy": labels[j],
            "open_time": T, "close_time": exit_t, "close_price": float(exit_price), "pnl": pnl, "fee": fee,
            "funding": funding, "exit_type": exit_type, "rr": float(rrs[j]),
            "htf_agree": htf_agree if P.use_htf else None,
            "R": pnl / risk_usd if risk_usd > 0 else 0.0, "peak": float(peak), "bars": int(exit_i - t),
        })
        if record:
            sig["opened"] = 1
            signals.append(sig)

    flush(float("inf"))
    return {"trades": trades, "equity": equity, "signals": signals, "final_balance": balance}


# ==================== معیارهای عملکرد ====================

def metrics(trades, equity, start_balance, t_from=None, t_to=None):
    """معیارها برای معاملاتی که در بازه‌ی [t_from, t_to) باز شدن."""
    sel = [t for t in trades if (t_from is None or t["open_time"] >= t_from) and (t_to is None or t["open_time"] < t_to)]
    n = len(sel)
    out = {"trades": n}
    if n == 0:
        out.update({"win_rate": 0.0, "avg_r": 0.0, "r_lcb": -9.0, "profit_factor": 0.0, "pnl": 0.0,
                    "return_pct": 0.0, "max_dd_pct": 0.0, "pos_months_pct": 0.0, "fees": 0.0,
                    "long_trades": 0, "short_trades": 0, "avg_bars": 0.0})
        return out
    rs = np.array([t["R"] for t in sel])
    pnls = np.array([t["pnl"] for t in sel])
    wins = int((pnls >= 0).sum())
    gp = float(pnls[pnls > 0].sum())
    gl = float(-pnls[pnls < 0].sum())
    avg_r = float(rs.mean())
    sd = float(rs.std(ddof=1)) if n > 1 else 0.0
    lcb = avg_r - 1.96 * sd / math.sqrt(n) if n > 1 else avg_r - 1.0

    # موجودی تحقق‌یافته در بازه (برای بازده و افت سرمایه)
    eq_t = np.array([e[0] for e in equity], dtype=np.float64)
    eq_b = np.array([e[1] for e in equity], dtype=np.float64)
    lo = -np.inf if t_from is None else t_from
    hi = np.inf if t_to is None else t_to
    before = eq_b[eq_t < lo]
    start_bal = before[-1] if len(before) else start_balance
    seg = eq_b[(eq_t >= lo) & (eq_t < hi)]
    seg = np.concatenate([[start_bal], seg]) if len(seg) else np.array([start_bal])
    peak = np.maximum.accumulate(seg)
    dd = float(((peak - seg) / peak).max() * 100) if len(seg) else 0.0
    ret = float((seg[-1] - start_bal) / start_bal * 100) if start_bal else 0.0

    months = {}
    for t in sel:
        key = datetime.utcfromtimestamp(t["close_time"] / 1000).strftime("%Y-%m")
        months[key] = months.get(key, 0.0) + t["pnl"]
    pos_months = sum(1 for v in months.values() if v > 0) / len(months) * 100 if months else 0.0

    out.update({
        "win_rate": round(wins / n * 100, 2),
        "avg_r": round(avg_r, 4),
        "r_lcb": round(lcb, 4),
        "profit_factor": round(gp / gl, 3) if gl > 0 else (99.0 if gp > 0 else 0.0),
        "pnl": round(float(pnls.sum()), 2),
        "return_pct": round(ret, 2),
        "max_dd_pct": round(dd, 2),
        "pos_months_pct": round(pos_months, 1),
        "fees": round(float(sum(t["fee"] for t in sel)), 2),
        "long_trades": sum(1 for t in sel if t["side"] == "LONG"),
        "short_trades": sum(1 for t in sel if t["side"] == "SHORT"),
        "avg_bars": round(float(np.mean([t["bars"] for t in sel])), 1),
    })
    return out


# ==================== خروجی SQLite (سازگار با تحلیلگر و پنل) ====================

def _iso(ms):
    return datetime.utcfromtimestamp(ms / 1000).isoformat()


def to_sqlite(result, start_balance, trailing):
    conn = paper_trader.get_conn(":memory:")
    for col, coltype in (("rr_planned", "REAL"), ("htf_agree", "INTEGER"), ("r_multiple", "REAL")):
        try:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {coltype}")
        except Exception:
            pass
    eq = result["equity"]
    conn.executemany("INSERT INTO equity (time, balance) VALUES (?, ?)",
                     [(_iso(t), b) for t, b in eq] if eq else [(_iso(0), start_balance)])
    rows = []
    for t in sorted(result["trades"], key=lambda x: x["open_time"]):
        rows.append((t["symbol"], t["side"], t["entry"], t["sl"], t["tp"], t["sl"], t["peak"], t["size"],
                     t["notional"], t["margin"], t["leverage"], t["liquidation_price"], 1 if trailing else 0,
                     t["strategy"], "CLOSED", _iso(t["open_time"]), _iso(t["close_time"]), t["close_price"],
                     t["pnl"], t["fee"], t["exit_type"], "WIN" if t["pnl"] >= 0 else "LOSS",
                     t["rr"], t["htf_agree"], t["R"]))
    conn.executemany("""
        INSERT INTO trades (symbol, side, entry, sl, tp, initial_sl, peak_price, size, notional, margin,
                            leverage, liquidation_price, trailing_enabled, strategy_name, status, open_time,
                            close_time, close_price, pnl, fee_cost, exit_type, result, rr_planned, htf_agree,
                            r_multiple)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.executemany("""
        INSERT INTO signal_log (time, symbol, side, strategy_name, entry, sl, tp, rr, htf_agree, opened,
                                rejection_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, [(_iso(s["time"]), s["symbol"], s["side"], s["strategy"], s["entry"], s["sl"], s["tp"], s["rr"],
           s["htf_agree"], s["opened"], s["reason"]) for s in result["signals"]])
    conn.commit()
    return conn
