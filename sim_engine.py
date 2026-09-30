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

import money
import paper_trader
import signals_engine as se


# ==================== مسیر یک معامله (وکتوریزه) ====================

def _trail_vec(P, E, SL0, risk, trig_adj, locks, last_trigger, beyond, F=None, floor_r=None):
    """نسخه‌ی آرایه‌ای paper_trader.compute_trailing_sl برای خرید (فروش با قرینه‌سازی).
    F: کف سربه‌سر بعد از کارمزد (قرینه‌شده برای فروش) یا None. floor_r: کف فقط از این R سود به بعد."""
    peak_r = (P - E) / risk
    idx = np.searchsorted(trig_adj, peak_r, "right") - 1
    has = idx >= 0
    locked = np.where(has, locks[np.clip(idx, 0, len(locks) - 1)], 0.0)
    beyond_mask = has & (peak_r > last_trigger + 1e-9)
    locked = np.where(beyond_mask, np.maximum(locked, peak_r - beyond), locked)
    new_sl = E + locked * risk
    if F is not None:
        fmask = has if floor_r is None else (peak_r >= floor_r - 1e-9)
        new_sl = np.where(fmask, np.maximum(new_sl, np.minimum(F, P)), new_sl)
    return np.where(has, np.maximum(new_sl, SL0), SL0)


class PathSim:
    """شبیه‌ساز مسیر معاملات یک نماد، با کش نتایج (هر کاندید فقط یک‌بار شبیه‌سازی می‌شه)."""

    def __init__(self, series, ladder, beyond, profiles=None):
        """ladder/beyond: تریلینگ پیش‌فرض (trailing=True). profiles: {نام: {"ladder", "beyond"}}
        برای وقتی که trailing اسم یک پروفایل باشه."""
        self.s = series
        self.prof = {True: self._compile(ladder, beyond)}
        for name, pr in (profiles or {}).items():
            self.prof[name] = self._compile(pr["ladder"], pr["beyond"], pr.get("floor_r"))
        self.cache = {}

    @staticmethod
    def _compile(ladder, beyond, floor_r=None):
        lad = sorted(ladder, key=lambda x: x[0])
        return {"has": len(lad) > 0,
                "trig_adj": np.array([t - 1e-9 for t, _ in lad], dtype=np.float64),
                "locks": np.array([lk for _, lk in lad], dtype=np.float64),
                "last_trigger": lad[-1][0] if lad else 0.0, "beyond": beyond, "floor_r": floor_r}

    def _slices(self, side, k, k1):
        """برش کندل‌ها؛ برای فروش قرینه می‌شن تا همون منطق خرید استفاده بشه
        (منفی کردن قیمت در IEEE دقیق و بدون خطاست، پس نتیجه دقیقاً یکیه)."""
        s = self.s
        if side == se.LONG:
            return s.o[k:k1], s.h[k:k1], s.l[k:k1], s.c[k:k1]
        return -s.o[k:k1], -s.l[k:k1], -s.h[k:k1], -s.c[k:k1]

    def run(self, t0, side, entry, sl0, tp, trailing, max_bars=0, floor=None, early=None, fill_bar=False, cut=None):
        key = (t0, side, entry, sl0, tp, trailing, max_bars, floor, early, fill_bar, cut)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        res = self._run(t0, side, entry, sl0, tp, trailing, max_bars, floor, early, fill_bar, cut)
        self.cache[key] = res
        return res

    def _run(self, t0, side, entry, sl0, tp, trailing, max_bars=0, floor=None, early=None, fill_bar=False, cut=None):
        """مدیریت از کندل t0+1. max_bars>0 یعنی حد زمانی: اگه تا اون تعداد کندل بسته نشد،
        در قیمت بسته شدن آخرین کندل مجاز با سفارش بازار بسته می‌شه (TIME).
        floor: حد ضرر تریلینگ هیچ‌وقت عقب‌تر از این قیمت (سربه‌سر بعد از کارمزد) نمی‌ره.
        early: (N, min_r) — خروج زودهنگام: اگه بعد از N کندل بیشترین سود کمتر از min_r بوده،
               با قیمت بسته‌شدن کندل Nام بسته می‌شه (EARLY)."""
        sgn = 1.0 if side == se.LONG else -1.0
        E, SL0, TP = sgn * entry, sgn * sl0, sgn * tp
        F = sgn * floor if floor is not None else None
        risk = abs(entry - sl0)
        if cut and cut > 0 and risk > 0:
            # بستن در ‎-cut×R: حد ضرر مؤثر بالاتر، ولی R (برای تریلینگ/خروج زودهنگام) همون ریسک اولیه
            SL0 = max(SL0, E - cut * risk)
        n_all = self.s.n
        n = min(n_all, t0 + 1 + max_bars) if max_bars and max_bars > 0 else n_all
        pr = self.prof.get(trailing if isinstance(trailing, str) else True) if trailing else None
        trailing = bool(pr) and pr["has"] and risk > 0
        st = {"sl": SL0, "peak": E}
        k = t0 + 1
        if fill_bar and k < n:
            # کندل پر شدن سفارش لیمیت: فقط حد ضرر (سقف/کفش ممکنه قبل از پر شدن بوده) — مثل process_bars
            oo, hh, ll, cc = self._slices(side, k, k + 1)
            if ll[0] <= SL0:
                price = oo[0] if oo[0] < SL0 else SL0
                return (k, sgn * price, "STOP", sgn * SL0, sgn * E)
            k += 1
        k_first = k
        if early and early[0] > 0 and risk > 0:
            k_mid = min(n, t0 + 1 + int(early[0]))
            r = self._scan(k, k_mid, side, sgn, E, SL0, TP, risk, pr, F, trailing, st) if k_mid > k else None
            if r is not None:
                return r
            if k_mid == t0 + 1 + int(early[0]) and k_mid > t0 + 1:
                best = E   # مثل ربات زنده: بیشترین سود از قیمت ورود شروع می‌شه (کندل پر شدن حساب نمی‌شه)
                if k_mid > k_first:
                    _, hh, _, _ = self._slices(side, k_first, k_mid)
                    best = max(float(hh.max()), E)
                if (best - E) / risk < early[1]:
                    return (k_mid - 1, float(self.s.c[k_mid - 1]), "EARLY", None, sgn * max(best, st["peak"]))
            k = k_mid
        r = self._scan(k, n, side, sgn, E, SL0, TP, risk, pr, F, trailing, st)
        if r is not None:
            return r
        peak = st["peak"]
        if n < n_all:
            return (n - 1, float(self.s.c[n - 1]), "TIME", None, sgn * peak)
        # تا آخر دیتا باز مونده: با آخرین قیمت (به‌صورت سفارش بازار) بسته حساب می‌شه
        return (n - 1, float(self.s.c[n - 1]), "END", None, sgn * peak)

    def _scan(self, k, n, side, sgn, E, SL0, TP, risk, pr, F, trailing, st):
        """کندل‌های [k, n) رو اعمال می‌کنه؛ اگه بسته شد نتیجه، وگرنه None (و وضعیت st آپدیت می‌شه)."""
        cur_sl, peak = st["sl"], st["peak"]
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
                SLk = _trail_vec(P, E, SL0, risk, pr["trig_adj"], pr["locks"], pr["last_trigger"], pr["beyond"], F,
                                 pr.get("floor_r"))
                SLk = np.maximum.accumulate(np.maximum(SLk, cur_sl))
                SLprev = np.empty_like(SLk)
                SLprev[0] = cur_sl
                SLprev[1:] = SLk[:-1]
                c1 = ll <= SLprev
                tph = hh >= TP          # حد سود R:R با تریلینگ هم فعاله
                c2 = (SLk > SLprev) & (ll <= SLk)   # بدبینانه: کف بعد از سقف (مثل step_bar)
                hitm = c1 | tph | c2
                if hitm.any():
                    i = int(np.argmax(hitm))
                    if c1[i]:
                        lvl = SLprev[i]
                        price = oo[i] if oo[i] < lvl else lvl
                        return (k + i, sgn * price, "STOP", sgn * lvl, sgn * (P[i - 1] if i > 0 else peak))
                    if tph[i]:
                        price = oo[i] if oo[i] > TP else TP
                        return (k + i, sgn * price, "TP", sgn * TP, sgn * P[i])
                    return (k + i, sgn * SLk[i], "STOP", sgn * SLk[i], sgn * P[i])
                peak, cur_sl = P[-1], SLk[-1]
            k = k1
            chunk = min(chunk * 4, 20000)
        st["sl"], st["peak"] = cur_sl, peak
        return None


# ==================== سبد (پورتفو) ====================

class SimParams:
    """پارامترهای مدیریت سرمایه/فیلترها که روی هر اجرای سبد اثر دارن."""

    def __init__(self, cfg, htf_min_agreement, trailing, allow_long=True, allow_short=True,
                 btc_filter=False, risk_pct=None):
        """htf_min_agreement: مقیاس «از ۵» (مثل پیش‌تنظیم‌های سخت‌گیری)؛ برای پروفایل‌هایی
        که تعداد تایم‌فریم تایید متفاوتی دارن، متناسب تبدیل می‌شه."""
        self.start_balance = float(cfg.VIRTUAL_BALANCE_START)
        self.risk_pct = float(risk_pct if risk_pct is not None else cfg.RISK_PER_TRADE_PCT)
        # مدیریت سرمایه‌ی یکسان: ضرر خالص ثابت دلاری (با کارمزد) یا درصدی از موجودی
        self.risk_usd = float(cfg.RISK_USD) if getattr(cfg, "RISK_MODE", "pct") == "usd" else None
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
        short_extra = getattr(cfg, "SHORT_EXTRA_HTF_AGREEMENT", 0)
        self.htf_min_long, self.htf_min_short = htf_required(htf_min_agreement, n_tf, short_extra)
        self.htf_weighted = bool(getattr(cfg, "HTF_WEIGHTED", False))
        self.htf_min_long_pct, self.htf_min_short_pct = htf_required_pct(htf_min_agreement, short_extra)
        # trailing: False، True (تریلینگ پیش‌فرض) یا اسم یکی از TRAIL_PROFILES
        self.trailing = trailing if isinstance(trailing, str) else bool(trailing)
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
        self.fi, self.fs, self.fm = money.fee_fracs(self.maker, self.taker, self.slip, self.entry_mode == "market")
        self.fi_limit = money.fee_fracs(self.maker, self.taker, self.slip, False)[0]
        self.fi_market = money.fee_fracs(self.maker, self.taker, self.slip, True)[0]
        self.trail_floor = bool(getattr(cfg, "TRAIL_BREAKEVEN_FLOOR", False))
        # ورود با پولبک (شکست باکس): همیشه لیمیت، با مهلت خودش
        self.retest_wait = max(1, int(getattr(cfg, "BRK_RETEST_WAIT_BARS", 1)))
        # خروج زودهنگام و حد ضرر روزانه
        self.early = ((int(cfg.EARLY_EXIT_BARS), float(cfg.EARLY_EXIT_MIN_R))
                      if getattr(cfg, "EARLY_EXIT", False) and int(getattr(cfg, "EARLY_EXIT_BARS", 0)) > 0 else None)
        self.daily_loss = float(getattr(cfg, "DAILY_LOSS_LIMIT_USD", 0.0) or 0.0)
        self.cut = float(getattr(cfg, "CUT_LOSS_R", 0.0) or 0.0)
        self.xs_hold_bars = max(1, int(round(float(getattr(cfg, "XS_HOLD_DAYS", 7)) * 86_400_000 / tf_ms)))


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


def trade_mae_r(ser, t0, exit_i, side, entry, sl, exit_type):
    """بیشترین حرکت خلاف جهت (بر حسب R اولیه) از کندل پر شدن تا خروج. کندل خروجِ تریلینگ حساب
    نمی‌شه (حد ضرر تریلینگ بالای ‎-xR بوده و زودتر خورده). یعنی: یک حد ضرر در ‎-xR دقیقاً وقتی
    زودتر خورده بود که mae_r ≥ x."""
    k0 = t0 + 1
    k_end = exit_i if exit_type in ("TRAIL_SL", "BREAKEVEN") else exit_i + 1
    risk = abs(entry - sl)
    if k_end <= k0 or risk <= 0:
        return 0.0
    if side == se.LONG:
        return max(0.0, (entry - float(ser.l[k0:k_end].min())) / risk)
    return max(0.0, (float(ser.h[k0:k_end].max()) - entry) / risk)


def htf_weights(tf_list):
    """وزن هر تایم‌فریم تایید: به ترتیب لیست (از پایین به بالا) ۱، ۲، ۳، ... — بالاتر = وزن بیشتر."""
    return {tf: float(i + 1) for i, tf in enumerate(tf_list)}


def htf_weighted_pct(trends, tf_list, side, pair_trend=None, pair_weight=0.0):
    """ربات زنده: درصد وزن تایم‌فریم‌های هم‌جهت. trends: {tf: "uptrend"/"downtrend"/...} فقط برای
    تایم‌فریم‌هایی که بررسی شدن. pair_trend: روند نمودار ارز÷BTC با وزن pair_weight (۰ = حساب نشه).
    همون ترتیب جمع موتور بک‌تست (پس نتیجه دقیقاً برابره)."""
    weights = htf_weights(tf_list)
    want = "uptrend" if side == "LONG" else "downtrend"
    acc = 0.0
    for tf in tf_list:
        acc = acc + weights[tf] * float(trends.get(tf) == want)
    total = sum(weights.values())
    if pair_weight and pair_weight > 0:
        acc = acc + float(pair_weight) * float(pair_trend == want)
        total = total + float(pair_weight)
    return acc / total * 100 if total > 0 else 0.0


def pair_weight_for(symbol, cfg):
    """وزن نمودار ارز÷BTC برای این نماد (برای خود BTC صفر)."""
    if not getattr(cfg, "HTF_WEIGHTED", False):
        return 0.0
    if symbol == getattr(cfg, "BTC_REGIME_SYMBOL", "BTC/USDT"):
        return 0.0
    return float(getattr(cfg, "BTC_PAIR_WEIGHT", 0.0) or 0.0)


def htf_required_pct(preset_of_5, short_extra=0):
    """حداقل درصد وزن موافق در حالت وزن‌دار: «n از ۵» = n×۲۰٪ (فروش: n+اضافه)."""
    lo = min(100.0, max(0.0, float(preset_of_5) * 20.0))
    return lo, min(100.0, max(0.0, float(int(preset_of_5) + int(short_extra)) * 20.0))


def merge_candidates(per_symbol, preps, symbol_order):
    """
    ادغام کاندیدهای همه‌ی نمادها به ترتیب زمان بسته‌شدن کندل (و بعد ترتیب نمادها، مثل
    ترتیب اسکن ربات زنده). per_symbol: dict symbol -> خروجی signals_engine.combine
    """
    cols = {k: [] for k in ("time", "rank", "sym", "idx", "side", "sl", "tp", "rr", "entry",
                            "htf_l", "htf_s", "htf_wl", "htf_ws", "btc", "score")}
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
        ent = cand.get("entry")
        cols["entry"].append(p.series.c[idx] if ent is None else ent)
        cols["htf_l"].append(p.htf_long[idx])
        cols["htf_s"].append(p.htf_short[idx])
        cols["htf_wl"].append(p.htf_wlong[idx])
        cols["htf_ws"].append(p.htf_wshort[idx])
        cols["btc"].append(p.btc_trend[idx])
        cols["score"].append(cand.get("score", np.zeros(len(idx))))
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
        if P.htf_weighted:
            bad_l = (side_arr == se.LONG) & (merged["htf_wl"] < P.htf_min_long_pct - 1e-9)
            bad_s = (side_arr == se.SHORT) & (merged["htf_ws"] < P.htf_min_short_pct - 1e-9)
        else:
            bad_l = (side_arr == se.LONG) & (htf_l < P.htf_min_long)
            bad_s = (side_arr == se.SHORT) & (htf_s < P.htf_min_short)
        reason[(bad_l | bad_s) & (reason == 0)] = 3
    static_reason = {1: "side_disabled", 2: "btc_regime", 3: "htf_disagreement"}

    iter_idx = np.arange(n) if record else np.flatnonzero(reason == 0)
    times = merged["time"]
    syms = merged["sym"]
    t_idx = merged["idx"]
    entries, sls, tps, rrs = merged["entry"], merged["sl"], merged["tp"], merged["rr"]
    scores = merged["score"]
    labels = merged["label"]

    day_pnl = {}   # سود/زیان تحقق‌یافته‌ی هر روز (UTC) برای حد ضرر روزانه

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
                d = int(exit_t // 86_400_000)
                day_pnl[d] = day_pnl.get(d, 0.0) + pnl

    for j in iter_idx.tolist():
        T = int(times[j])
        flush(T)
        sr = int(syms[j])
        if sr in open_by_sym:
            continue
        side = int(side_arr[j])
        if P.htf_weighted:
            htf_agree = int(round(float(merged["htf_wl"][j] if side == se.LONG else merged["htf_ws"][j])))
        else:
            htf_agree = int(htf_l[j] if side == se.LONG else htf_s[j])
        sig = None
        if record:
            sig = {"time": T, "symbol": symbol_order[sr], "side": "LONG" if side == se.LONG else "SHORT",
                   "strategy": labels[j], "entry": float(entries[j]), "sl": float(sls[j]), "tp": float(tps[j]),
                   "rr": float(rrs[j]), "htf_agree": htf_agree if P.use_htf else None,
                   "opened": 0, "reason": None, "score": round(float(scores[j]), 1)}
        is_xs = labels[j] == "xs_momentum"
        lc = last_close.get(sr)
        if lc is not None and T - lc < P.cooldown_ms and not is_xs:
            if record:
                sig["reason"] = "cooldown"
                sig["htf_agree"] = None
                signals.append(sig)
            continue
        if P.daily_loss > 0 and day_pnl.get(int(T // 86_400_000), 0.0) <= -P.daily_loss:
            if record:
                sig["reason"] = "daily_loss_limit"
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
        retest = labels[j].endswith("@retest")
        is_trend = labels[j].startswith("trend_follow")
        # روندگیر و مومنتوم: همیشه ورود بازار؛ پولبک: همیشه لیمیت
        entry_taker = (P.entry_mode == "market" or is_trend or is_xs) and not retest
        fi = P.fi_market if entry_taker else P.fi_limit
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
            risk_usd=P.risk_usd, fees=(fi, P.fs),
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
            k_end = min(ser.n - 1, t + (P.retest_wait if retest else P.limit_wait_bars))
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

        # روندگیر همیشه با تریلینگ شاندلیر خودش مدیریت می‌شه (حد سود ثابت نداره)
        # مومنتوم: بدون تریلینگ و خروج زودهنگام؛ فقط حد ضرر محافظ و بستن در پایان هفته
        trail = "trend" if is_trend else (False if is_xs else P.trailing)
        floor = money.breakeven_stop(side == se.LONG, entry, fi, P.fs) if (trail and P.trail_floor) else None
        exit_i, exit_price, kind, level, peak = prep.paths.run(t0, side, entry, sl, tp, trail,
                                                               P.xs_hold_bars if is_xs else P.max_hold_bars, floor,
                                                               None if is_xs else P.early,
                                                               fill_bar=not entry_taker, cut=P.cut or None)
        cut_px = None
        if P.cut:
            rk = abs(entry - sl)
            cut_px = entry - P.cut * rk if side == se.LONG else entry + P.cut * rk
        if cut_px is not None and kind == "STOP" and level == cut_px:
            exit_type = "CUT"
        else:
            exit_type = paper_trader.classify_exit_level(kind, level, sl, entry, bool(trail))
        size = pos["size"]
        gross = (exit_price - entry) * size if side == se.LONG else (entry - exit_price) * size
        fee, _, _, _ = paper_trader._fee_for_exit(entry, exit_price, size, exit_type, P.maker, P.taker, P.slip,
                                                  entry_taker=entry_taker)
        exit_t = int(ser.close_ts[exit_i])
        funding = paper_trader.funding_cost(pos["notional"], fill_t, exit_t, P.funding_8h)
        fee += funding
        pnl = gross - fee
        risk_usd = pos["per_unit_loss"] * size   # ضرر خالص (با کارمزد) اگه حد ضرر اولیه می‌خورد = ۱R
        # بیشترین حرکت خلاف جهت قبل از خروج (MAE) بر حسب R — برای تحلیل «اگه در ‎-xR می‌بستیم»
        mae_r = trade_mae_r(ser, t0, exit_i, side, entry, sl, exit_type)
        seq += 1
        heapq.heappush(heap, (exit_t, seq, sr, pos["margin"], pnl, True))
        open_by_sym[sr] = True
        locked += pos["margin"]
        trades.append({
            "symbol": symbol_order[sr], "side": side_str, "entry": entry, "sl": sl, "tp": tp,
            "size": size, "notional": pos["notional"], "margin": pos["margin"], "leverage": pos["leverage"],
            "liquidation_price": pos["liquidation_price"], "strategy": labels[j],
            "open_time": T, "close_time": exit_t, "close_price": float(exit_price), "pnl": pnl, "fee": fee,
            "funding": funding, "exit_type": exit_type, "rr": float(rrs[j]), "score": round(float(scores[j]), 1),
            "htf_agree": htf_agree if P.use_htf else None,
            "R": pnl / risk_usd if risk_usd > 0 else 0.0, "peak": float(peak), "bars": int(exit_i - t),
            "mae_r": mae_r, "entry_fee_frac": fi,
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
                    "long_trades": 0, "short_trades": 0, "avg_bars": 0.0, "wins": 0, "losses": 0,
                    "trail_trades": 0, "trail_pnl": 0.0, "trail_pct": 0.0, "early_trades": 0, "early_pnl": 0.0,
                    "pos_rate": 0.0, "avg_week_usd": 0.0, "pos_weeks_pct": 0.0, "worst_week_usd": 0.0,
                    "avg_month_usd": 0.0, "pos_months_all_pct": 0.0, "worst_month_usd": 0.0,
                    "best_month_usd": 0.0, "trades_per_week": 0.0})
        return out
    rs = np.array([t["R"] for t in sel])
    pnls = np.array([t["pnl"] for t in sel])
    # برد/باخت فقط از معاملاتی که به هدف R:R یا حد ضرر اولیه رسیدن؛ تریلینگ جدا
    res_kind = [paper_trader.result_of(t["exit_type"]) for t in sel]
    wins = res_kind.count("WIN")
    # بستن در ‎-xR (CUT) هم باخته (حد ضرر زودتر)؛ قبلاً حساب نمی‌شد و وین‌ریت الکی ۱۰۰٪ نشون می‌داد
    losses = res_kind.count("LOSS") + sum(1 for t in sel if t["exit_type"] == "CUT")
    trail_pnl = float(sum(t["pnl"] for t, k in zip(sel, res_kind) if k == "TRAIL"))
    early_pnl = float(sum(t["pnl"] for t, k in zip(sel, res_kind) if k == "EARLY"))
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
    # درآمد هفتگی و ماهانه (دلار): همه‌ی هفته‌ها/ماه‌های بازه حساب می‌شن، حتی بدون معامله (= صفر)
    span_lo = t_from if t_from is not None else min(t["open_time"] for t in sel)
    span_hi = t_to if t_to is not None else max(t["close_time"] for t in sel)
    if len(eq_t):
        span_lo = min(span_lo, float(eq_t[0])) if t_from is None else span_lo
        span_hi = max(span_hi, float(eq_t[-1])) if t_to is None else span_hi
    WEEK = 7 * 86_400_000
    wk0 = int(span_lo // WEEK)
    n_weeks = max(1, int(span_hi // WEEK) - wk0 + 1)
    weeks = np.zeros(n_weeks)
    for t in sel:
        k = min(n_weeks - 1, max(0, int(t["close_time"] // WEEK) - wk0))
        weeks[k] += t["pnl"]
    m0 = datetime.utcfromtimestamp(span_lo / 1000)
    m1 = datetime.utcfromtimestamp(span_hi / 1000)
    n_months = max(1, (m1.year - m0.year) * 12 + m1.month - m0.month + 1)
    mvals = np.zeros(n_months)
    for key, val in months.items():
        y, mo = int(key[:4]), int(key[5:7])
        k = (y - m0.year) * 12 + mo - m0.month
        mvals[min(n_months - 1, max(0, k))] += val

    out.update({
        "win_rate": round(wins / (wins + losses) * 100, 2) if (wins + losses) else 0.0,
        "pos_rate": round(float((pnls > 0).mean()) * 100, 1),   # ٪ معاملات سودده (هر نوع خروج)
        "wins": wins, "losses": losses,
        "trail_trades": res_kind.count("TRAIL"), "trail_pnl": round(trail_pnl, 2),
        "early_trades": res_kind.count("EARLY"), "early_pnl": round(early_pnl, 2),
        "trail_pct": round(trail_pnl / start_balance * 100, 2) if start_balance else 0.0,
        "avg_r": round(avg_r, 4),
        "r_lcb": round(lcb, 4),
        "profit_factor": round(gp / gl, 3) if gl > 0 else (99.0 if gp > 0 else 0.0),
        "pnl": round(float(pnls.sum()), 2),
        "return_pct": round(ret, 2),
        "max_dd_pct": round(dd, 2),
        "pos_months_pct": round(pos_months, 1),
        "avg_week_usd": round(float(weeks.mean()), 2),
        "pos_weeks_pct": round(float((weeks > 0).mean()) * 100, 1),
        "worst_week_usd": round(float(weeks.min()), 2),
        "avg_month_usd": round(float(mvals.mean()), 2),
        "pos_months_all_pct": round(float((mvals > 0).mean()) * 100, 1),
        "worst_month_usd": round(float(mvals.min()), 2),
        "best_month_usd": round(float(mvals.max()), 2),
        "trades_per_week": round(n / n_weeks, 2),
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
    for col, coltype in (("rr_planned", "REAL"), ("htf_agree", "INTEGER"), ("r_multiple", "REAL"), ("score", "REAL"),
                         ("mae_r", "REAL"), ("entry_fee_frac", "REAL")):
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
                     t["pnl"], t["fee"], t["exit_type"], paper_trader.result_of(t["exit_type"]),
                     t["rr"], t["htf_agree"], t["R"], t.get("score"), t.get("mae_r"), t.get("entry_fee_frac")))
    conn.executemany("""
        INSERT INTO trades (symbol, side, entry, sl, tp, initial_sl, peak_price, size, notional, margin,
                            leverage, liquidation_price, trailing_enabled, strategy_name, status, open_time,
                            close_time, close_price, pnl, fee_cost, exit_type, result, rr_planned, htf_agree,
                            r_multiple, score, mae_r, entry_fee_frac)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.executemany("""
        INSERT INTO signal_log (time, symbol, side, strategy_name, entry, sl, tp, rr, htf_agree, opened,
                                rejection_reason, score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, [(_iso(s["time"]), s["symbol"], s["side"], s["strategy"], s["entry"], s["sl"], s["tp"], s["rr"],
           s["htf_agree"], s["opened"], s["reason"], s.get("score")) for s in result["signals"]])
    conn.commit()
    return conn
