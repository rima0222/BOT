# -*- coding: utf-8 -*-
"""
سنجش کیفیت ورود — «این ورود واقعاً از ورود شانسی بهتره یا نه؟»

هر سیگنال ورود (استراتژی‌های ربات + چند فرضیه‌ی ساده‌ی شناخته‌شده) جدا از مدیریت معامله
سنجیده می‌شه؛ هیچ تریلینگ، خروج زودهنگام یا فیلتر سبدی دخالت نداره. برای هر ورود:

  • براکت‌ها: آیا قیمت اول به +xR رسید یا اول به −yR؟ (۱:۱، ۲:۱، ۳:۱ و ۲:۰.۵)
    اگه هر دو در یک کندل لمس شدن = ضرر (محافظه‌کارانه). اگه تا H کندل به هیچ‌کدوم نرسید،
    نتیجه = بازده تا بسته‌شدن کندل H‌ام.
  • بازده بعد از ۱، ۳، ۶، ۱۲، ۲۴ و H کندل؛ بیشترین سود/ضرر شناور (MFE/MAE) در H کندل.
  • کارمزد واقعی به واحد R کم می‌شه (ارزش مورد انتظار خالص).

واحد R برای همه یکسانه: ENTRY_STUDY_R_ATR × ATR(14) کندل سیگنال. برای استراتژی‌های ربات،
نتیجه با حد ضرر خود استراتژی (RR2) هم جدا حساب می‌شه.

پایه‌ی مقایسه: «ورود شانسی» روی نمونه‌ی منظمی از همه‌ی کندل‌های همون نمادها و همون دوره
(خرید و فروش جدا) — پس روند کلی بازار در اون دوره از نتیجه حذف می‌شه.

اعتبارسنجی:
  • ۷۰٪ اول زمان = آموزش، ۳۰٪ آخر = آزمون (دیتای دیده‌نشده).
  • خطای استاندارد «خوشه‌ای» (خوشه = هفته، یا H کندل اگه طولانی‌تر باشه) — چون ورودهای
    هم‌زمان در ارزهای مختلف و ورودهای پشت‌سرهم مستقل نیستن. بدون این، اطمینان الکی بالا می‌ره.
  • درصد نمادهایی که سیگنال در اون‌ها از ورود شانسی بهتر بوده.

خروجی: تحلیل هر سیگنال، تحلیل فیلترها (کدوم شرط، ورود رو در هر دو بخش آموزش و آزمون
بهتر کرده) و فایل همه‌ی ورودها برای بررسی دقیق‌تر.
"""
import csv
import gzip
import json
import os

import numpy as np

import analysis
import fast_backtest
import indicators as ind
import sim_engine
import signals_engine as se

DAY_MS = 86_400_000
BRACKETS = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (2.0, 0.5)]
BRACKET_KEYS = ["1_1", "2_1", "3_1", "2_05"]
BRACKET_FA = ["+1R قبل از −1R", "+2R قبل از −1R", "+3R قبل از −1R", "+2R قبل از −0.5R"]
MAIN_B = 1   # براکت اصلی برای تحلیل فیلترها: RR2 (هدف ربات)

STRATEGY_SIGNALS = ["weighted_confluence", "box_breakout", "box_breakout@retest", "trend_follow", "fib_phase"]
RAW_SIGNALS = ["donch20", "donch55", "ma_cross", "dow_flip", "pullback_trend", "rsi_revert", "bb_revert",
               "squeeze_break", "vol_spike", "big_candle", "btc_lead"]

SIG_FA = {
    "base": "ورود شانسی (پایه‌ی مقایسه)",
    "weighted_confluence": "ترکیبی وزن‌دار (استراتژی ربات)",
    "box_breakout": "شکست باکس (استراتژی ربات)",
    "box_breakout@retest": "شکست باکس + پولبک (استراتژی ربات)",
    "trend_follow": "روندگیر دونچیان (استراتژی ربات)",
    "fib_phase": "فیبوناچی حرکت دوم (همه‌ی امتیازها)",
    "xs_momentum": "مومنتوم هفتگی (استراتژی ربات)",
    "donch20": "شکست سقف/کف ۲۰ کندل",
    "donch55": "شکست سقف/کف ۵۵ کندل",
    "ma_cross": "کراس میانگین ۲۰ و ۱۰۰",
    "dow_flip": "شروع روند داو (سقف و کف بالاتر/پایین‌تر)",
    "pullback_trend": "پولبک در روند (برگشت RSI از ۴۰/۶۰)",
    "rsi_revert": "برگشت از اشباع RSI (۳۰/۷۰)",
    "bb_revert": "برگشت به داخل باند بولینگر",
    "squeeze_break": "شکست بعد از فشردگی نوسان",
    "vol_spike": "کندل هم‌جهت با حجم ۳ برابر",
    "big_candle": "ادامه‌ی کندل بزرگ (بیش از ۲ ATR)",
    "btc_lead": "حرکت تند BTC و جاماندن ارز",
}

# ویژگی‌های لحظه‌ی ورود (همه «هم‌جهت» شدن: برای فروش علامت برعکس، تا بشه خرید و فروش رو با هم تحلیل کرد)
FEATURES = [
    ("side", "جهت معامله", "cat", {1: "خرید", -1: "فروش"}),
    ("trend", "روند داو همین تایم‌فریم", "cat", {1: "هم‌جهت", 0: "خنثی", -1: "خلاف جهت"}),
    ("htf", "وزن تایید تایم‌فریم‌های بالاتر (٪)", "bins", [0, 40, 60, 80, 100.01]),
    ("btc", "روند BTC", "cat", {1: "هم‌جهت", 0: "خنثی", -1: "خلاف جهت"}),
    ("vol_ratio", "حجم نسبت به میانگین ۲۰ کندل", "q", None),
    ("atr_pct", "نوسان (ATR به درصد قیمت)", "q", None),
    ("ma_dist", "فاصله از میانگین ۱۰۰ (ATR، هم‌جهت)", "q", None),
    ("mom20", "حرکت ۲۰ کندل اخیر (ATR، هم‌جهت)", "q", None),
    ("rsi", "RSI هم‌جهت (فروش: ۱۰۰−RSI)", "q", None),
    ("body", "بدنه‌ی کندل هم‌جهت (−۱ تا +۱)", "q", None),
    ("range_atr", "اندازه‌ی کندل سیگنال (ATR)", "q", None),
    ("bb_ratio", "پهنای بولینگر نسبت به میانگین (فشردگی)", "q", None),
    ("btc_bar", "حرکت آخرین کندل BTC (ATR، هم‌جهت)", "q", None),
    ("hour", "ساعت بسته‌شدن کندل (UTC)", "bins", [0, 4, 8, 12, 16, 20, 24]),
    ("weekend", "آخر هفته", "cat", {1: "شنبه/یکشنبه", 0: "روزهای کاری"}),
    ("score", "امتیاز کیفیت استراتژی (فیبوناچی/ترکیبی)", "bins", [0, 34, 50, 67, 84, 100.01]),
]
SIDED = {"trend", "btc", "ma_dist", "mom20", "body", "btc_bar"}


# ==================== آمار ====================

def cluster_mean_se(x, g):
    """میانگین + خطای استاندارد خوشه‌ای (CR0 با اصلاح G/(G-1))."""
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    m = float(np.mean(x))
    if n < 2:
        return m, float("nan")
    _, gi = np.unique(g, return_inverse=True)
    G = int(gi.max()) + 1
    if G < 2:
        return m, float("nan")
    sums = np.bincount(gi, weights=np.asarray(x, dtype=np.float64) - m)
    se = float(np.sqrt((sums ** 2).sum()) / n * np.sqrt(G / (G - 1.0)))
    return m, se


def _f(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, nd)


# ==================== ویژگی‌ها و سیگنال‌های ساده ====================

def _shift(a, k):
    out = np.full(len(a), np.nan)
    if k < len(a):
        out[k:] = a[:-k]
    return out


def btc_context(prep):
    """سری BTC در همون تایم‌فریم (برای «حرکت BTC» و سیگنال جاماندن ارز)."""
    s = prep.series
    atr = fast_backtest._prep_atr(prep)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = (s.c - _shift(s.c, 1)) / atr
    return {"close_ts": s.close_ts, "ret": ret}


def compute_features(prep, cfg, btc=None):
    """همه‌ی مقادیر فقط از کندل‌های بسته‌شده تا همون کندل (بدون نگاه به آینده)."""
    s = prep.series
    n = s.n
    o, h, l, c = s.o, s.h, s.l, s.c
    v = s.v if s.v is not None else np.ones(n)
    atr = fast_backtest._prep_atr(prep)
    order = cfg.SWING_ORDER
    W = int(getattr(cfg, "CANDLE_LIMIT", 300))
    sh, slp = se.swing_positions(h, l, order)
    a_h, b_h, a_l, b_l, _ = se.window_swing_bounds(n, W, order, sh, slp)
    trend = se.dow_trend_array(h, l, sh, slp, a_h, b_h, a_l, b_l).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        sma20 = ind.sma(c, 20)
        sma100 = ind.sma(c, 100)
        rsi = ind.rsi_sma(c, 14)
        vol_ratio = v / ind.prev_mean(v, 20)
        mid, up, lo = ind.bollinger(c, 20, 2.0)
        bbw = (up - lo) / mid
        bb_ratio = bbw / ind.prev_mean(bbw, 120)
        rng = h - l
        body = np.where(rng > 0, (c - o) / np.where(rng > 0, rng, 1.0), 0.0)
        F = {
            "atr": atr, "trend": trend, "sma20": sma20, "sma100": sma100, "rsi": rsi,
            "vol_ratio": vol_ratio, "bb_up": up, "bb_lo": lo, "bb_ratio": bb_ratio,
            "mom20": (c - _shift(c, 20)) / atr, "ma_dist": (c - sma100) / atr,
            "body": body, "range_atr": rng / atr, "range_atr_prev": rng / _shift(atr, 1),
            "atr_pct": atr / c * 100.0, "own_ret": (c - _shift(c, 1)) / atr,
            "htf_l": np.asarray(prep.htf_wlong, dtype=np.float64),
            "htf_s": np.asarray(prep.htf_wshort, dtype=np.float64),
            "btc": np.asarray(prep.btc_trend, dtype=np.float64),
        }
    F["hour"] = ((s.close_ts // 3_600_000) % 24).astype(np.float64)
    F["weekend"] = ((((s.close_ts - 1) // DAY_MS) + 3) % 7 >= 5).astype(np.float64)
    btc_bar = np.full(n, np.nan)
    if btc is not None:
        pos = np.searchsorted(btc["close_ts"], s.close_ts)
        ok = pos < len(btc["close_ts"])
        ok[ok] = btc["close_ts"][pos[ok]] == s.close_ts[ok]
        btc_bar[ok] = btc["ret"][pos[ok]]
    F["btc_bar"] = btc_bar
    return F


def raw_signals(prep, F, is_btc=False):
    """فرضیه‌های ساده‌ی شناخته‌شده؛ همه روی بسته‌شدن کندل و فقط اولین کندل هر وضعیت (تقاطع)."""
    s = prep.series
    o, h, l, c = s.o, s.h, s.l, s.c
    cp = _shift(c, 1)
    out = {}
    with np.errstate(invalid="ignore"):
        for name, N in (("donch20", 20), ("donch55", 55)):
            top, bot = ind.prev_max(h, N), ind.prev_min(l, N)
            out[name] = ((c > top) & (cp <= _shift(top, 1)), (c < bot) & (cp >= _shift(bot, 1)))
        s20, s100 = F["sma20"], F["sma100"]
        out["ma_cross"] = ((s20 > s100) & (_shift(s20, 1) <= _shift(s100, 1)),
                           (s20 < s100) & (_shift(s20, 1) >= _shift(s100, 1)))
        tr, trp = F["trend"], _shift(F["trend"], 1)
        out["dow_flip"] = ((tr == 1) & (trp != 1) & ~np.isnan(trp), (tr == -1) & (trp != -1) & ~np.isnan(trp))
        rsi, rp = F["rsi"], _shift(F["rsi"], 1)
        s100p = _shift(s100, 10)
        out["pullback_trend"] = ((c > s100) & (s100 > s100p) & (rp < 40) & (rsi >= 40),
                                 (c < s100) & (s100 < s100p) & (rp > 60) & (rsi <= 60))
        out["rsi_revert"] = ((rp < 30) & (rsi >= 30), (rp > 70) & (rsi <= 70))
        up, lo = F["bb_up"], F["bb_lo"]
        out["bb_revert"] = ((cp < _shift(lo, 1)) & (c > lo), (cp > _shift(up, 1)) & (c < up))
        sq = _shift(F["bb_ratio"], 1) < 0.6
        out["squeeze_break"] = (sq & (c > up) & (cp <= _shift(up, 1)), sq & (c < lo) & (cp >= _shift(lo, 1)))
        vr, body = F["vol_ratio"], F["body"]
        out["vol_spike"] = ((vr >= 3.0) & (body >= 0.5), (vr >= 3.0) & (body <= -0.5))
        big = F["range_atr_prev"] >= 2.0
        out["big_candle"] = (big & (body >= 0.6), big & (body <= -0.6))
        if not is_btc and np.isfinite(F["btc_bar"]).any():
            bb, own = F["btc_bar"], F["own_ret"]
            out["btc_lead"] = ((bb >= 1.0) & (own < 0.3), (bb <= -1.0) & (own > -0.3))
    return out


def dedupe(idx, gap):
    """بین دو ورود پشت‌سرهم یک سیگنال (در یک نماد و یک جهت) حداقل gap کندل فاصله (مثل کول‌داون ربات)."""
    if len(idx) == 0 or gap <= 1:
        return idx
    keep = []
    last = -10 ** 12
    for i in idx.tolist():
        if i - last >= gap:
            keep.append(i)
            last = i
    return np.array(keep, dtype=np.int64)


# ==================== مسیر قیمت بعد از ورود ====================

def horizons_for(H):
    return sorted({k for k in (1, 3, 6, 12, 24, H) if k <= H})


def path_outcomes(h, l, c, p0, entry, side, rdist, H, brackets, stop_only0=None, chunk=5000):
    """
    برای هر ورود: کندل‌های p0..p0+H-1 بعد از ورود.
    stop_only0: کندل اول فقط برای حد ضرر چک بشه (کندل پر شدن سفارش لیمیت — مثل موتور بک‌تست).
    خروجی: out (m×nb: ۱ سود، −۱ ضرر، ۰ هیچ‌کدوم)، res (نتیجه به R)، fin (بازده کندل آخر)، mfe، mae
    """
    m = len(p0)
    nb = len(brackets)
    out = np.zeros((m, nb), dtype=np.int8)
    res = np.zeros((m, nb))
    mfe = np.zeros(m)
    mae = np.zeros(m)
    ar = np.arange(H)
    for a in range(0, m, chunk):
        b = min(m, a + chunk)
        J = p0[a:b, None] + ar[None, :]
        hi, lo = h[J], l[J]
        e = entry[a:b, None]
        r = rdist[a:b, None]
        lg = side[a:b, None] > 0
        up = np.where(lg, (hi - e) / r, (e - lo) / r)
        dn = np.where(lg, (e - lo) / r, (hi - e) / r)
        if stop_only0 is not None:
            up[stop_only0[a:b], 0] = -np.inf
        mfe[a:b] = np.maximum(up.max(axis=1), 0.0)
        mae[a:b] = np.maximum(dn.max(axis=1), 0.0)
        fin = (c[J[:, -1]] - e[:, 0]) * np.where(lg[:, 0], 1.0, -1.0) / r[:, 0]
        for k, (tp, sl) in enumerate(brackets):
            wu = up >= tp - 1e-12
            ld = dn >= sl - 1e-12
            fw = np.where(wu.any(axis=1), wu.argmax(axis=1), H)
            fl = np.where(ld.any(axis=1), ld.argmax(axis=1), H)
            loss = (fl < H) & (fl <= fw)
            win = (fw < H) & (fw < fl)
            out[a:b, k] = np.where(win, 1, np.where(loss, -1, 0))
            res[a:b, k] = np.where(win, tp, np.where(loss, -sl, fin))
    return out, res, mfe, mae


def retest_fill(h, l, idx, side, entry, wait):
    """سفارش لیمیت پولبک: اولین کندل از idx+1 تا idx+wait که قیمت «از» قیمت ورود رد بشه (مثل موتور بک‌تست)."""
    n = len(h)
    J = idx[:, None] + np.arange(1, wait + 1)[None, :]
    inside = J <= n - 1
    J = np.minimum(J, n - 1)
    lg = side[:, None] > 0
    en = entry[:, None]
    hit = np.where(lg, l[J] < en, h[J] > en) & inside
    any_hit = hit.any(axis=1)
    return any_hit, idx + 1 + hit.argmax(axis=1)


def forward_returns(c, p0, entry, side, rdist, horizons):
    sgn = np.where(side > 0, 1.0, -1.0)
    return np.column_stack([(c[p0 + k - 1] - entry) * sgn / rdist for k in horizons]) if len(p0) else \
        np.zeros((0, len(horizons)))


# ==================== ورودهای یک نماد ====================

class Study:
    """تنظیمات ثابت یک تایم‌فریم + ثبت همه‌ی ورودها."""

    def __init__(self, cfg, tf, split_ms, symbols, xs=False):
        self.cfg = cfg
        self.tf = tf
        self.split_ms = int(split_ms)
        self.H = int(cfg.ENTRY_STUDY_HORIZON.get(tf, 48))
        self.hz = horizons_for(self.H)
        self.r_atr = float(cfg.ENTRY_STUDY_R_ATR)
        tf_ms = int(fast_backtest.market_data.TF_MS[tf])
        self.tf_ms = tf_ms
        self.gap = max(1, int(round(float(cfg.COOLDOWN_HOURS) * 3_600_000 / tf_ms)))
        self.cluster_ms = max(7 * DAY_MS, self.H * tf_ms)
        taker = float(cfg.TAKER_FEE_PCT) / 100.0 + float(cfg.TAKER_SLIPPAGE_PCT) / 100.0
        self.cost_market = 2.0 * taker
        self.cost_limit = float(cfg.MAKER_FEE_PCT) / 100.0 + taker
        self.retest_wait = int(getattr(cfg, "BRK_RETEST_WAIT_BARS", 6))
        lo, hi = sim_engine.htf_required_pct(getattr(cfg, "HTF_MIN_AGREEMENT", 4),
                                             getattr(cfg, "SHORT_EXTRA_HTF_AGREEMENT", 0))
        self.htf_req = (lo, hi)
        names = ["base"]
        for sname in STRATEGY_SIGNALS:
            names += [sname, sname + "|htf"]
        if xs:
            names += ["xs_momentum"]
        names += RAW_SIGNALS
        self.names = names
        self.code = {nm: i for i, nm in enumerate(names)}
        self.symbols = list(symbols)
        self.sym_code = {s: i for i, s in enumerate(self.symbols)}
        self.parts = []
        self.counts = {}   # (name, side) -> {"signals": n, "filled": n}
        self._cur = {}     # شمارش‌های آخرین نماد (برای ذخیره‌ی مرحله‌ای)
        self.last_part = None
        self.E = None

    def _count(self, name, side, key, k):
        # فقط شمارش همین نماد؛ در انتها (یا موقع خوندن از فایل ذخیره) به کل اضافه می‌شه — نه دوبار
        d = self._cur.setdefault((name, int(side)), {"signals": 0, "filled": 0, "no_room": 0})
        d[key] += int(k)

    def merge_counts(self, counts):
        for key, d in counts.items():
            dst = self.counts.setdefault(key, {"signals": 0, "filled": 0, "no_room": 0})
            for k, v in d.items():
                dst[k] = dst.get(k, 0) + int(v)

    def add_symbol(self, prep, variant, btc=None, xs_final=None, is_btc=False, base_samples=3000, keep_part=True):
        """ورودهای یک نماد. keep_part=False: فقط در last_part نگه‌داشته می‌شه (برای ذخیره روی دیسک)."""
        self._cur = {}
        self.last_part = None
        cfg = self.cfg
        s = prep.series
        n = s.n
        H = self.H
        F = compute_features(prep, cfg, btc)
        atr = F["atr"]
        ar = np.arange(n)
        bar_ok = (ar >= prep.first_idx) & np.isfinite(atr) & (atr > 0)
        groups = []   # (name, side, idx, entry, sl(None), retest)

        for sname in STRATEGY_SIGNALS:
            if sname not in prep.structural:
                continue
            # فیبوناچی: همه‌ی امتیازها (حداقل امتیاز صفر)؛ اثر امتیاز در تحلیل فیلترها («امتیاز کیفیت») دیده می‌شه
            f = prep.finals(sname, dict(variant, min_score=0.0) if sname == "fib_phase" else variant)
            if len(f.idx) == 0:
                continue
            for sd in (se.LONG, se.SHORT):
                m = f.side == sd
                if not m.any():
                    continue
                idx, ent, sl, sc = f.idx[m].astype(np.int64), f.entry[m], f.sl[m], f.score[m]
                groups.append((sname, sd, idx, ent, sl, sname.endswith("@retest"), sc))
                hw = F["htf_l"][idx] if sd == se.LONG else F["htf_s"][idx]
                req = self.htf_req[0] if sd == se.LONG else self.htf_req[1]
                k = hw >= req - 1e-9
                groups.append((sname + "|htf", sd, idx[k], ent[k], sl[k], sname.endswith("@retest"), sc[k]))
        if xs_final is not None and len(xs_final.idx):
            for sd in (se.LONG, se.SHORT):
                m = xs_final.side == sd
                if m.any():
                    groups.append(("xs_momentum", sd, xs_final.idx[m].astype(np.int64), xs_final.entry[m],
                                   xs_final.sl[m], False, None))
        for name, (Lm, Sm) in raw_signals(prep, F, is_btc).items():
            for sd, mask in ((se.LONG, Lm), (se.SHORT, Sm)):
                idx = np.flatnonzero(mask & bar_ok)
                groups.append((name, sd, idx, s.c[idx], None, False, None))
        # پایه: نمونه‌ی منظم از همه‌ی کندل‌ها
        cand = np.flatnonzero(bar_ok & (ar + H + 1 <= n - 1))
        if len(cand):
            step = max(1, len(cand) // max(1, base_samples // 2))
            bidx = cand[::step]
            for sd in (se.LONG, se.SHORT):
                groups.append(("base", sd, bidx, s.c[bidx], None, False, None))

        cols = {k: [] for k in ("sig", "side", "idx", "entry", "sl", "retest", "score")}
        for name, sd, idx, ent, sl, retest, sc in groups:
            if name not in self.code:
                continue
            if len(idx):
                ok = bar_ok[idx]
                idx, ent = idx[ok], ent[ok]
                sl = None if sl is None else sl[ok]
                sc = None if sc is None else sc[ok]
            if name != "base" and len(idx):
                keep = dedupe(idx, self.gap)
                sel = np.searchsorted(idx, keep)
                idx, ent = idx[sel], ent[sel]
                sl = None if sl is None else sl[sel]
                sc = None if sc is None else sc[sel]
            self._count(name, sd, "signals", len(idx))
            if not len(idx):
                continue
            cols["sig"].append(np.full(len(idx), self.code[name], dtype=np.int16))
            cols["side"].append(np.full(len(idx), sd, dtype=np.int8))
            cols["idx"].append(idx)
            cols["entry"].append(np.asarray(ent, dtype=np.float64))
            cols["sl"].append(np.full(len(idx), np.nan) if sl is None else np.asarray(sl, dtype=np.float64))
            cols["retest"].append(np.full(len(idx), bool(retest)))
            cols["score"].append(np.full(len(idx), np.nan) if sc is None else np.asarray(sc, dtype=np.float64))
        if not cols["sig"]:
            if keep_part:
                self.merge_counts(self._cur)
            return 0
        E = {k: np.concatenate(v) for k, v in cols.items()}

        # نقطه‌ی شروع مسیر: ورود بازار = کندل بعد؛ پولبک = کندل پر شدن لیمیت (اگه در مهلت پر شد)
        p0 = E["idx"] + 1
        stop_only = np.zeros(len(p0), dtype=bool)
        filled = np.ones(len(p0), dtype=bool)
        rt = np.flatnonzero(E["retest"])
        if len(rt):
            any_hit, jf = retest_fill(s.h, s.l, E["idx"][rt], E["side"][rt], E["entry"][rt], self.retest_wait)
            filled[rt] = any_hit
            p0[rt] = np.where(any_hit, jf, p0[rt])
            stop_only[rt] = True
        room = p0 + H - 1 <= n - 1
        rdist = self.r_atr * atr[E["idx"]]
        keep = filled & room & np.isfinite(rdist) & (rdist > 0) & np.isfinite(E["entry"]) & (E["entry"] > 0)
        # شمارش پر شدن/جا نداشتن
        for code in np.unique(E["sig"]).tolist():
            for sd in (se.LONG, se.SHORT):
                m = (E["sig"] == code) & (E["side"] == sd)
                if m.any():
                    nm = self.names[code]
                    self._count(nm, sd, "filled", int((m & filled).sum()))
                    self._count(nm, sd, "no_room", int((m & filled & ~room).sum()))
        E = {k: v[keep] for k, v in E.items()}
        p0, stop_only, rdist = p0[keep], stop_only[keep], rdist[keep]
        m = len(p0)
        if keep_part:
            self.merge_counts(self._cur)
        if m == 0:
            return 0
        out, res, mfe, mae = path_outcomes(s.h, s.l, s.c, p0, E["entry"], E["side"], rdist, H, BRACKETS, stop_only)
        fwd = forward_returns(s.c, p0, E["entry"], E["side"], rdist, self.hz)
        cost_frac = np.where(E["retest"], self.cost_limit, self.cost_market)
        cost = cost_frac * E["entry"] / rdist
        # با حد ضرر خود استراتژی: +2R قبل از −1R
        own_d = np.abs(E["entry"] - E["sl"])
        has_own = np.isfinite(own_d) & (own_d > 0)
        own_out = np.zeros(m, dtype=np.int8)
        own_res = np.full(m, np.nan)
        own_cost = np.full(m, np.nan)
        if has_own.any():
            w = np.flatnonzero(has_own)
            o2, r2, _, _ = path_outcomes(s.h, s.l, s.c, p0[w], E["entry"][w], E["side"][w], own_d[w], H,
                                         [(2.0, 1.0)], stop_only[w])
            own_out[w] = o2[:, 0]
            own_res[w] = r2[:, 0]
            own_cost[w] = cost_frac[w] * E["entry"][w] / own_d[w]

        t_ms = s.close_ts[E["idx"]]
        sgn = E["side"].astype(np.float64)
        idx = E["idx"]
        feat = {
            "side": sgn,
            "trend": F["trend"][idx] * sgn,
            "htf": np.where(sgn > 0, F["htf_l"][idx], F["htf_s"][idx]),
            "btc": F["btc"][idx] * sgn,
            "vol_ratio": F["vol_ratio"][idx],
            "atr_pct": F["atr_pct"][idx],
            "ma_dist": F["ma_dist"][idx] * sgn,
            "mom20": F["mom20"][idx] * sgn,
            "rsi": np.where(sgn > 0, F["rsi"][idx], 100.0 - F["rsi"][idx]),
            "body": F["body"][idx] * sgn,
            "range_atr": F["range_atr"][idx],
            "bb_ratio": F["bb_ratio"][idx],
            "btc_bar": F["btc_bar"][idx] * sgn,
            "hour": F["hour"][idx] if self.tf != "1d" else np.full(m, np.nan),
            "weekend": F["weekend"][idx],
            "score": E["score"],
        }
        part = {
            "sig": E["sig"], "side": E["side"], "sym": np.full(m, self.sym_code.get(prep.symbol, -1), dtype=np.int16),
            "t_ms": t_ms.astype(np.int64), "seg": (t_ms >= self.split_ms).astype(np.int8),
            "clus": (t_ms // self.cluster_ms).astype(np.int32),
            "entry": E["entry"], "rdist_pct": (rdist / E["entry"] * 100.0).astype(np.float32),
            "out": out, "res": res.astype(np.float32), "cost": cost.astype(np.float32),
            "fwd": fwd.astype(np.float32), "mfe": mfe.astype(np.float32), "mae": mae.astype(np.float32),
            "own_out": own_out, "own_res": own_res.astype(np.float32), "own_cost": own_cost.astype(np.float32),
            "feat": {k: np.asarray(v, dtype=np.float32) for k, v in feat.items()},
        }
        self.last_part = part
        if keep_part:
            self.parts.append(part)
        return m

    def assemble(self, paths):
        """
        ورودهای ذخیره‌شده‌ی همه‌ی نمادها (به ترتیب نمادها) → جدول نهایی. آرایه‌ها از قبل با اندازه‌ی نهایی
        ساخته و فایل‌به‌فایل پر می‌شن (رم کمتر از چسباندن همه با هم). خروجی: {نماد: meta}، تعداد ورود
        """
        infos = []
        for pth in paths:
            with np.load(pth) as z:
                infos.append(json.loads(bytes(z["__info"]).decode("utf-8")))
        total = sum(int(i["m"]) for i in infos)
        metas = {}
        E = None
        pos = 0
        for pth, info in zip(paths, infos):
            self.merge_counts({(c[0], int(c[1])): c[2] for c in info["counts"]})
            metas[info["symbol"]] = info.get("meta") or {}
            m = int(info["m"])
            if m == 0:
                continue
            with np.load(pth) as z:
                if E is None:
                    E = {"feat": {}}
                    for k in z.files:
                        if k == "__info":
                            continue
                        a = z[k]
                        dst = np.empty((total,) + a.shape[1:], dtype=a.dtype)
                        if k.startswith("feat__"):
                            E["feat"][k[6:]] = dst
                        else:
                            E[k] = dst
                for k in z.files:
                    if k == "__info":
                        continue
                    dst = E["feat"][k[6:]] if k.startswith("feat__") else E[k]
                    dst[pos:pos + m] = z[k]
            pos += m
        self.E = E
        return metas, total

    def finish(self):
        """چسباندن ورودهای همه‌ی نمادها."""
        if not self.parts:
            self.E = None
            return None
        E = {}
        for k in self.parts[0]:
            if k == "feat":
                E["feat"] = {f: np.concatenate([p["feat"][f] for p in self.parts]) for f in self.parts[0]["feat"]}
            else:
                E[k] = np.concatenate([p[k] for p in self.parts])
        self.parts = []
        self.E = E
        return E


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def save_symbol(path, symbol, part, counts, meta):
    """ذخیره‌ی ورودهای یک نماد روی دیسک (برای ادامه‌ی کار بعد از قطع شدن)."""
    arrays = {}
    m = 0
    if part is not None:
        m = len(part["sig"])
        for k, v in part.items():
            if k == "feat":
                for f, a in v.items():
                    arrays["feat__" + f] = a
            else:
                arrays[k] = v
    info = {"symbol": symbol, "m": m, "meta": meta or {},
            "counts": [[k[0], int(k[1]), v] for k, v in (counts or {}).items()]}
    arrays["__info"] = np.frombuffer(json.dumps(info, ensure_ascii=False, default=_json_default).encode("utf-8"),
                                     dtype=np.uint8)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ==================== تحلیل ====================

def _seg_mask(E, seg):
    if seg == "is":
        return E["seg"] == 0
    if seg == "oos":
        return E["seg"] == 1
    return np.ones(len(E["seg"]), dtype=bool)


def group_stats(E, m):
    """آمار یک گروه ورود (m = ماسک بولی)."""
    n = int(m.sum())
    st = {"n": n}
    if n == 0:
        return st
    clus = E["clus"][m]
    st["clusters"] = int(len(np.unique(clus)))
    out = E["out"][m]
    res = E["res"][m].astype(np.float64)
    cost = E["cost"][m].astype(np.float64)
    st["p"], st["p_se"], st["ev"], st["ev_se"], st["timeout"] = [], [], [], [], []
    for k in range(len(BRACKETS)):
        rs = out[:, k] != 0
        pm, pse = cluster_mean_se((out[rs, k] == 1).astype(np.float64), clus[rs]) if rs.any() else (np.nan, np.nan)
        em, ese = cluster_mean_se(res[:, k] - cost, clus)
        st["p"].append(pm)
        st["p_se"].append(pse)
        st["ev"].append(em)
        st["ev_se"].append(ese)
        st["timeout"].append(float((~rs).mean()))
    st["gross_ev"] = [float(res[:, k].mean()) for k in range(len(BRACKETS))]
    st["cost_r"] = float(np.median(cost))
    st["fwd"] = [float(np.mean(E["fwd"][m][:, j])) for j in range(E["fwd"].shape[1])]
    fm, fse = cluster_mean_se(E["fwd"][m][:, -1].astype(np.float64), clus)
    st["fwd_h_se"] = fse
    st["mfe_med"] = float(np.median(E["mfe"][m]))
    st["mae_med"] = float(np.median(E["mae"][m]))
    st["rdist_pct"] = float(np.median(E["rdist_pct"][m]))
    own = np.isfinite(E["own_res"][m])
    if own.any():
        oo = E["own_out"][m][own]
        rs = oo != 0
        st["own_n"] = int(own.sum())
        st["own_p2"] = float((oo[rs] == 1).mean()) if rs.any() else float("nan")
        st["own_ev2"] = float(np.mean(E["own_res"][m][own].astype(np.float64) - E["own_cost"][m][own]))
        st["own_cost"] = float(np.median(E["own_cost"][m][own]))
    return st


def _combine_base(bases, weights):
    """پایه‌ی ترکیبی خرید+فروش با همون نسبت ورودهای سیگنال."""
    wt = sum(weights)
    if wt <= 0:
        return None
    out = {"p": [], "p_se": [], "ev": [], "ev_se": []}
    for key, sek in (("p", "p_se"), ("ev", "ev_se")):
        for k in range(len(BRACKETS)):
            v = sum(w / wt * b[key][k] for b, w in zip(bases, weights) if w > 0)
            se_ = np.sqrt(sum((w / wt) ** 2 * b[sek][k] ** 2 for b, w in zip(bases, weights) if w > 0))
            out[key].append(v)
            out[sek].append(se_)
    return out


def _edges(st, base):
    """فاصله با ورود شانسی + آماره‌ی z (خطای خوشه‌ای هر دو)."""
    if not base or st.get("n", 0) == 0:
        return
    st["base_p"], st["base_ev"] = list(base["p"]), list(base["ev"])
    st["edge_p"], st["z_p"], st["edge_ev"], st["z_ev"] = [], [], [], []
    for k in range(len(BRACKETS)):
        for key, sek, ek, zk in (("p", "p_se", "edge_p", "z_p"), ("ev", "ev_se", "edge_ev", "z_ev")):
            d = st[key][k] - base[key][k]
            se_ = np.sqrt(np.nan_to_num(st[sek][k]) ** 2 + np.nan_to_num(base[sek][k]) ** 2)
            st[ek].append(d)
            st[zk].append(d / se_ if se_ > 0 else float("nan"))


def verdict(row):
    """حکم نهایی: فقط چیزی «پایدار»ه که در آموزش معنادار و در آزمون (دیتای دیده‌نشده) هم مثبت باشه."""
    si, so = row["is"], row["oos"]
    if si.get("n", 0) < 30 or so.get("n", 0) < 15 or "ev" not in si or "ev" not in so:
        return "few", "کم‌داده"
    b = int(np.nanargmax(np.array(si["ev"], dtype=np.float64)))
    row["best_b"] = b
    z_is = si.get("z_ev", [np.nan] * 4)[b]
    z_oos = so.get("z_ev", [np.nan] * 4)[b]
    e_oos = so.get("edge_ev", [np.nan] * 4)[b]
    net_is, net_oos = si["ev"][b], so["ev"][b]
    if z_is >= 2 and e_oos > 0 and z_oos >= 1 and net_is > 0 and net_oos > 0:
        return "robust", "✓ مزیت پایدار (بعد از کارمزد، در هر دو بخش)"
    if z_is >= 2 and e_oos > 0:
        if net_is > 0 or net_oos > 0:
            return "edge_weak_oos", "◐ بهتر از شانس، ولی سود خالص فقط در یک بخش"
        return "edge_costs", "◐ بهتر از شانس، ولی کارمزد سودش رو می‌خوره"
    zp_is = si.get("z_p", [np.nan])[0]
    ep_oos = so.get("edge_p", [np.nan])[0]
    if zp_is <= -2 and ep_oos < 0:
        return "reverse", "↺ بدتر از شانس (جهت برعکسش ارزش بررسی داره)"
    if si.get("edge_ev", [0] * 4)[b] > 0 and e_oos > 0:
        return "weak", "~ کمی بهتر از شانس، ولی معنادار نیست"
    return "none", "✗ بدون مزیت"


VERDICT_RANK = {"robust": 0, "edge_weak_oos": 1, "edge_costs": 2, "weak": 3, "none": 4, "reverse": 5, "few": 6}


def _subset(E, idx):
    S = {k: v[idx] for k, v in E.items() if k != "feat"}
    S["feat"] = {f: v[idx] for f, v in E["feat"].items()}
    return S


def _by_signal(E):
    """اندیس ورودهای هر سیگنال (ترتیب اصلی حفظ می‌شه) — به‌جای ماسک روی کل جدول (خیلی سریع‌تر)."""
    order = np.argsort(E["sig"], kind="stable")
    srt = E["sig"][order]
    out = {}
    for c in np.unique(srt).tolist():
        lo = np.searchsorted(srt, c, "left")
        hi = np.searchsorted(srt, c, "right")
        out[int(c)] = order[lo:hi]
    return out


def analyze(study, cb=None):
    """جدول سیگنال‌ها (هر سیگنال × جهت × بخش) با مقایسه‌ی پایه."""
    E = study.E
    rows = []
    if E is None:
        return rows
    groups = _by_signal(E)
    base_code = study.code["base"]
    base = {}
    B = _subset(E, groups[base_code]) if base_code in groups else None
    for seg in ("is", "oos", "all"):
        for sd in (1, -1):
            base[(seg, sd)] = group_stats(B, _seg_mask(B, seg) & (B["side"] == sd)) if B is not None else {"n": 0}
    # پایه‌ی هر نماد (برای «در چند درصد نمادها بهتر بوده»)
    sym_base = {}
    if B is not None:
        for sym in np.unique(B["sym"]).tolist():
            for sd in (1, -1):
                mm = (B["sym"] == sym) & (B["side"] == sd)
                o = B["out"][mm][:, 0]
                rs = o != 0
                sym_base[(sym, sd)] = float((o[rs] == 1).mean()) if rs.any() else np.nan

    todo = [(c, nm) for c, nm in enumerate(study.names) if c in groups]
    for i, (code, name) in enumerate(todo):
        S = B if code == base_code else _subset(E, groups[code])
        for sd, sd_name in ((1, "long"), (-1, "short"), (0, "both")):
            mside = np.ones(len(S["sig"]), dtype=bool) if sd == 0 else (S["side"] == sd)
            if not mside.any():
                continue
            row = {"tf": study.tf, "signal": name, "label": label_of(name), "side": sd_name}
            for seg in ("is", "oos", "all"):
                m = mside & _seg_mask(S, seg)
                st = group_stats(S, m)
                if name != "base" and st["n"]:
                    if sd == 0:
                        nl = int((m & (S["side"] == 1)).sum())
                        ns = int((m & (S["side"] == -1)).sum())
                        b = _combine_base([base[(seg, 1)], base[(seg, -1)]], [nl, ns]) \
                            if base[(seg, 1)].get("n") and base[(seg, -1)].get("n") else None
                    else:
                        b = base[(seg, sd)] if base[(seg, sd)].get("n") else None
                    _edges(st, b)
                row[seg] = st
            if name != "base":
                # سازگاری بین نمادها (کل دوره، براکت ۱:۱)
                better = tot = 0
                for sym in np.unique(S["sym"][mside]).tolist():
                    mm = mside & (S["sym"] == sym)
                    o = S["out"][mm][:, 0]
                    rs = o != 0
                    if rs.sum() < 10:
                        continue
                    p = float((o[rs] == 1).mean())
                    sides = S["side"][mm][rs]
                    ref = [sym_base.get((sym, int(x)), np.nan) for x in (1, -1)]
                    bref = (np.mean(sides == 1) * np.nan_to_num(ref[0]) + np.mean(sides == -1) * np.nan_to_num(ref[1]))
                    tot += 1
                    better += int(p > bref)
                row["sym_better_pct"] = round(100.0 * better / tot, 1) if tot else None
                row["sym_tested"] = tot
                code_v, text = verdict(row)
                row["verdict"], row["verdict_text"] = code_v, text
            else:
                row["verdict"], row["verdict_text"] = "base", "پایه"
            cnt = [study.counts.get((name, s_), {}) for s_ in ((1, -1) if sd == 0 else (sd,))]
            sig_n = sum(c.get("signals", 0) for c in cnt)
            fill_n = sum(c.get("filled", 0) for c in cnt)
            row["signals_raw"] = sig_n
            row["fill_pct"] = round(100.0 * fill_n / sig_n, 1) if (name.endswith("@retest") or "@retest|" in name) \
                and sig_n else None
            rows.append(row)
        if cb:
            cb((i + 1) / len(todo), name)
    return rows


def label_of(name):
    if name.endswith("|htf"):
        return SIG_FA.get(name[:-4], name[:-4]) + " + تایید HTF"
    return SIG_FA.get(name, name)


def _bucketize(vals, kind, spec, is_mask):
    """برچسب سطل هر ورود + فهرست سطل‌ها. مرز صدک‌ها فقط از بخش آموزش (بدون نگاه به آزمون)."""
    fin = np.isfinite(vals)
    if kind == "cat":
        keys = [k for k in spec if (fin & (vals == k)).any()]
        lab = np.full(len(vals), -1, dtype=np.int16)
        for i, k in enumerate(keys):
            lab[fin & (vals == k)] = i
        return lab, [(spec[k], k, k) for k in keys]
    if kind == "bins":
        edges = np.array(spec, dtype=np.float64)
    else:
        v_is = vals[fin & is_mask]
        if len(v_is) < 50:
            return None, []
        edges = np.unique(np.quantile(v_is, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        if len(edges) < 3:
            return None, []
        edges[0], edges[-1] = -np.inf, np.inf
    lab = np.full(len(vals), -1, dtype=np.int16)
    b = np.searchsorted(edges, vals, side="right") - 1
    ok = fin & (b >= 0) & (b < len(edges) - 1)
    lab[ok] = b[ok]
    buckets = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if kind == "q":
            if i == 0:
                txt = f"کمتر از {hi:.3g}"
            elif i == len(edges) - 2:
                txt = f"{lo:.3g} و بیشتر"
            else:
                txt = f"{lo:.3g} تا {hi:.3g}"
        else:
            txt = f"{lo:g} تا {min(hi, 100):g}" if kind == "bins" and spec[-1] > 99 else f"{lo:g} تا {hi:g}"
        buckets.append((txt, float(lo), float(hi)))
    return lab, buckets


def _hit_ev(E, m, k):
    o = E["out"][m][:, k]
    rs = o != 0
    p = float((o[rs] == 1).mean()) if rs.any() else np.nan
    ev = float(np.mean(E["res"][m][:, k].astype(np.float64) - E["cost"][m])) if m.any() else np.nan
    return p, ev


def _z_bucket(E, m, ref):
    """معناداری فاصله‌ی ارزش مورد انتظار RR2 یک سطل با کل سیگنال (خطای خوشه‌ای سطل)."""
    x = E["res"][m][:, MAIN_B].astype(np.float64) - E["cost"][m]
    mean, se_b = cluster_mean_se(x, E["clus"][m])
    return (mean - ref) / se_b if se_b and np.isfinite(se_b) and se_b > 0 else np.nan


def analyze_features(study, min_is=80, cb=None):
    """
    کدوم شرط (فیلتر) کیفیت ورود رو بهتر می‌کنه؟ برای هر سیگنال (خرید و فروش با هم، ویژگی‌ها
    هم‌جهت)، ورودها بر اساس هر ویژگی به سطل‌ها تقسیم می‌شن و درصد «+1R قبل از −1R» و ارزش
    مورد انتظار خالص RR2 هر سطل با کل سیگنال مقایسه می‌شه — جدا در آموزش و آزمون.
    «فیلتر مفید» = در هر دو بخش بهتر (نه فقط در یکی).
    """
    E = study.E
    rows = []
    if E is None:
        return rows
    groups = _by_signal(E)
    todo = [(c, nm) for c, nm in enumerate(study.names) if c in groups]
    for i, (code, name) in enumerate(todo):
        if cb:
            cb(i / len(todo), name)
        S = _subset(E, groups[code])
        is_all = S["seg"] == 0
        oos_all = S["seg"] == 1
        n_is = int(is_all.sum())
        if n_is < min_is or int(oos_all.sum()) < 20:
            continue
        p_is, ev_is = _hit_ev(S, is_all, MAIN_B)
        p_oos, ev_oos = _hit_ev(S, oos_all, MAIN_B)
        p1_is, _ = _hit_ev(S, is_all, 0)
        p1_oos, _ = _hit_ev(S, oos_all, 0)
        for fname, flabel, kind, spec in FEATURES:
            vals = S["feat"][fname].astype(np.float64)
            lab, buckets = _bucketize(vals, kind, spec, is_all)
            if lab is None or len(buckets) < 2:
                continue
            for bi, (btxt, lo, hi) in enumerate(buckets):
                bm = lab == bi
                mi, mo = bm & is_all, bm & oos_all
                ni, no = int(mi.sum()), int(mo.sum())
                if ni < 15:
                    continue
                bp1_is, bev_is = _hit_ev(S, mi, MAIN_B)
                bp1o_is, _ = _hit_ev(S, mi, 0)
                bp_oos, bev_oos = _hit_ev(S, mo, MAIN_B) if no else (np.nan, np.nan)
                bp1o_oos, _ = _hit_ev(S, mo, 0) if no else (np.nan, np.nan)
                # معناداری بهبود در آموزش (خوشه‌ای): ارزش مورد انتظار سطل منهای کل سیگنال
                z = _z_bucket(S, mi, ev_is)
                z_o = _z_bucket(S, mo, ev_oos) if no >= 2 else np.nan
                lift1_is, lift1_oos = bp1o_is - p1_is, bp1o_oos - p1_oos
                d_is, d_oos = bev_is - ev_is, bev_oos - ev_oos
                # سخت‌گیرانه (هزاران سطل آزمایش می‌شه؛ با z≥۲ حدود ۱ از ۱۵۰ سطل شانسی «مفید» درمیومد):
                # بهبود در آموزش خیلی معنادار (z≥۲.۵) و در آزمون هم معنادار (z≥۱.۵) + درصد برد هم در هر دو بالاتر
                flag = ""
                if ni >= 40 and no >= 15 and np.isfinite(z) and np.isfinite(z_o):
                    if z >= 2.5 and z_o >= 1.5 and lift1_is > 0 and lift1_oos > 0:
                        flag = "good"
                    elif z <= -2.5 and z_o <= -1.5 and lift1_is < 0 and lift1_oos < 0:
                        flag = "bad"
                rows.append({"tf": study.tf, "signal": name, "label": label_of(name), "feature": fname,
                             "feature_label": flabel, "bucket": btxt, "bucket_idx": bi, "n_buckets": len(buckets), "lo": _f(lo), "hi": _f(hi),
                             "n_is": ni, "n_oos": no,
                             "p1_is": _f(bp1o_is), "p1_oos": _f(bp1o_oos), "lift1_is": _f(lift1_is),
                             "lift1_oos": _f(lift1_oos),
                             "p2_is": _f(bp1_is), "p2_oos": _f(bp_oos),
                             "ev2_is": _f(bev_is), "ev2_oos": _f(bev_oos), "d_ev2_is": _f(d_is),
                             "d_ev2_oos": _f(d_oos), "z_is": _f(z, 2), "z_oos": _f(z_o, 2),
                             "sig_p1_is": _f(p1_is), "sig_p1_oos": _f(p1_oos),
                             "sig_ev2_is": _f(ev_is), "sig_ev2_oos": _f(ev_oos), "flag": flag})
    return rows


# ==================== خروجی ====================

def row_to_json(row):
    out = {k: v for k, v in row.items() if k not in ("is", "oos", "all")}
    for seg in ("is", "oos", "all"):
        st = row.get(seg) or {}
        d = {}
        for k, v in st.items():
            if isinstance(v, list):
                d[k] = [_f(x) for x in v]
            elif isinstance(v, (int, np.integer)):
                d[k] = int(v)
            else:
                d[k] = _f(v)
        out[seg] = d
    return out


def signal_csv_rows(rows, hz):
    head = ["tf", "signal", "label", "side", "verdict", "verdict_text", "signals_raw", "fill_pct", "sym_better_pct",
            "sym_tested", "best_bracket"]
    segcols = ["n", "clusters"]
    for key in BRACKET_KEYS:
        segcols += [f"p_{key}", f"base_p_{key}", f"edge_p_{key}", f"z_p_{key}", f"ev_net_{key}", f"base_ev_{key}",
                    f"edge_ev_{key}", f"z_ev_{key}", f"timeout_{key}"]
    segcols += [f"fwd_{k}" for k in hz] + ["mfe_med", "mae_med", "rdist_pct", "cost_r", "own_n", "own_p2", "own_ev2",
                                           "own_cost"]
    header = head + [f"{seg}_{c}" for seg in ("is", "oos", "all") for c in segcols]
    out = [header]
    for r in rows:
        line = [r.get("tf"), r.get("signal"), r.get("label"), r.get("side"), r.get("verdict"), r.get("verdict_text"),
                r.get("signals_raw"), r.get("fill_pct"), r.get("sym_better_pct"), r.get("sym_tested"),
                BRACKET_KEYS[r["best_b"]] if "best_b" in r else ""]
        for seg in ("is", "oos", "all"):
            st = r.get(seg) or {}
            vals = [st.get("n"), st.get("clusters")]
            for k in range(len(BRACKETS)):
                for key in ("p", "base_p", "edge_p", "z_p", "ev", "base_ev", "edge_ev", "z_ev", "timeout"):
                    arr = st.get(key)
                    vals.append(_f(arr[k]) if arr is not None and k < len(arr) else None)
            fw = st.get("fwd") or []
            vals += [_f(x) for x in fw] + [None] * (len(hz) - len(fw))
            vals += [_f(st.get(k)) for k in ("mfe_med", "mae_med", "rdist_pct", "cost_r")]
            vals += [st.get("own_n"), _f(st.get("own_p2")), _f(st.get("own_ev2")), _f(st.get("own_cost"))]
            line += vals
        out.append(line)
    return out


def fwd_labels(hz):
    return [str(k) for k in hz[:-1]] + ["H"]


def events_header(hz):
    return (["tf", "symbol", "signal", "side", "time_utc", "segment", "entry", "r_pct", "cost_r"]
            + [f"out_{k}" for k in BRACKET_KEYS] + [f"res_{k}" for k in BRACKET_KEYS]
            + [f"fwd_{k}" for k in fwd_labels(hz)] + ["mfe", "mae", "own_out_2_1", "own_res_2_1", "own_cost"]
            + [f[0] for f in FEATURES])


def write_events(study, path, max_rows, symbols, append=False, header=True):
    """همه‌ی ورودها (با سقف) در یک CSV فشرده — برای بررسی دقیق‌تر و ساختن فیلترهای ترکیبی."""
    E = study.E
    if E is None:
        return 0
    rng = np.random.default_rng(7)
    base_code = study.code["base"]
    idx_sig = np.flatnonzero(E["sig"] != base_code)
    idx_base = np.flatnonzero(E["sig"] == base_code)
    cap_sig = int(max_rows * 0.75)
    if len(idx_sig) > cap_sig:
        idx_sig = np.sort(rng.choice(idx_sig, cap_sig, replace=False))
    cap_base = max_rows - len(idx_sig)
    if len(idx_base) > cap_base:
        idx_base = np.sort(rng.choice(idx_base, max(0, cap_base), replace=False))
    sel = np.concatenate([idx_sig, idx_base])
    fnames = [f[0] for f in FEATURES]
    mode = "at" if append else "wt"
    with gzip.open(path, mode, encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if header and not append:
            w.writerow(events_header(study.hz))
        for i in sel.tolist():
            t = np.datetime64(int(E["t_ms"][i]), "ms").astype("datetime64[m]")
            sym_i = int(E["sym"][i])
            line = [study.tf, symbols[sym_i] if 0 <= sym_i < len(symbols) else "", study.names[int(E["sig"][i])],
                    "LONG" if E["side"][i] > 0 else "SHORT", str(t), "IS" if E["seg"][i] == 0 else "OOS",
                    f"{E['entry'][i]:.8g}", f"{E['rdist_pct'][i]:.4f}", f"{E['cost'][i]:.4f}"]
            line += [int(x) for x in E["out"][i]] + [f"{x:.4f}" for x in E["res"][i]]
            line += [f"{x:.4f}" for x in E["fwd"][i]]
            line += [f"{E['mfe'][i]:.4f}", f"{E['mae'][i]:.4f}", int(E["own_out"][i]),
                     "" if not np.isfinite(E["own_res"][i]) else f"{E['own_res'][i]:.4f}",
                     "" if not np.isfinite(E["own_cost"][i]) else f"{E['own_cost'][i]:.4f}"]
            line += ["" if not np.isfinite(E["feat"][f][i]) else f"{E['feat'][f][i]:.4g}" for f in fnames]
            w.writerow(line)
    return len(sel)
