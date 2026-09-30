# -*- coding: utf-8 -*-
"""
کتابخونه‌ی الگوها برای «سنجش کیفیت ورود»: الگوهای کندلی + الگوهای کلاسیک چارتی.

همه‌ی الگوها فقط با کندل‌های بسته‌شده تا همون لحظه شناسایی می‌شن (بدون نگاه به آینده):
  - الگوی کندلی روی بسته‌شدن آخرین کندل الگو.
  - الگوی چارتی روی «اولین بسته‌شدن» بیرون خط گردن/ضلع الگو. سقف/کف‌های سوینگ فقط وقتی استفاده
    می‌شن که تایید شده باشن (SWING_ORDER کندل بعدشون بسته شده).

خروجی هر الگو یک جفت‌الگوست: نسخه‌ی خرید و نسخه‌ی فروش (مثلاً «کف دوقلو» / «سقف دوقلو»)، به‌همراه
حد ضرر کتابی الگو (خرید: زیر کف الگو، فروش: بالای سقف الگو، با بافر کوچک ATR).
"""
import numpy as np

import signals_engine as se

try:   # کتابخونه‌ی استاندارد TA-Lib (تعریف‌های مرجع الگوهای کندلی) — اختیاری
    import talib
except Exception:   # نصب نشده: فقط الگوهای خود ربات
    talib = None

# الگوهای جهت‌دار TA-Lib (خروجی مثبت = صعودی، منفی = نزولی). الگوهای بی‌جهت (دوجی ساده، فرفره، کندل
# بلند/کوتاه، ...) که علامتشون فقط رنگ کندله حذف شدن. سنگ قبر در TA-Lib همیشه مثبته ولی نزولیه → برعکس.
TALIB_PATTERNS = [
    ("CDL2CROWS", "دو کلاغ"), ("CDL3BLACKCROWS", "سه کلاغ سیاه"), ("CDL3INSIDE", "سه درونی"),
    ("CDL3LINESTRIKE", "ضربه‌ی سه‌خطی"), ("CDL3OUTSIDE", "سه بیرونی"), ("CDL3STARSINSOUTH", "سه ستاره در جنوب"),
    ("CDL3WHITESOLDIERS", "سه سرباز سفید"), ("CDLABANDONEDBABY", "بچه‌ی رهاشده"), ("CDLADVANCEBLOCK", "بلوک پیشروی"),
    ("CDLBELTHOLD", "کمربند"), ("CDLBREAKAWAY", "جدایی"), ("CDLCLOSINGMARUBOZU", "ماروبوزوی بسته‌شدن"),
    ("CDLCONCEALBABYSWALL", "پرستوی پنهان"), ("CDLCOUNTERATTACK", "ضدحمله"), ("CDLDARKCLOUDCOVER", "ابر سیاه"),
    ("CDLDOJISTAR", "ستاره‌ی دوجی"), ("CDLDRAGONFLYDOJI", "دوجی سنجاقک"), ("CDLENGULFING", "پوشا"),
    ("CDLEVENINGDOJISTAR", "ستاره‌ی دوجی شامگاهی"), ("CDLEVENINGSTAR", "ستاره‌ی شامگاهی"),
    ("CDLGAPSIDESIDEWHITE", "گپ با دو کندل سفید"), ("CDLGRAVESTONEDOJI", "دوجی سنگ قبر"), ("CDLHAMMER", "چکش"),
    ("CDLHANGINGMAN", "مرد آویزان"), ("CDLHARAMI", "هارامی"), ("CDLHARAMICROSS", "هارامی صلیبی"),
    ("CDLHIKKAKE", "هیکاکه"), ("CDLHIKKAKEMOD", "هیکاکه‌ی اصلاح‌شده"), ("CDLHOMINGPIGEON", "کبوتر خانگی"),
    ("CDLIDENTICAL3CROWS", "سه کلاغ یکسان"), ("CDLINNECK", "درون گردن"), ("CDLINVERTEDHAMMER", "چکش وارونه"),
    ("CDLKICKING", "لگد"), ("CDLKICKINGBYLENGTH", "لگد (بر اساس طول)"), ("CDLLADDERBOTTOM", "کف نردبانی"),
    ("CDLMATCHINGLOW", "کف هم‌سطح"), ("CDLMATHOLD", "نگه‌داشت"), ("CDLMORNINGDOJISTAR", "ستاره‌ی دوجی صبحگاهی"),
    ("CDLMORNINGSTAR", "ستاره‌ی صبحگاهی"), ("CDLONNECK", "روی گردن"), ("CDLPIERCING", "نفوذی"),
    ("CDLRISEFALL3METHODS", "سه روش صعودی/نزولی"), ("CDLSEPARATINGLINES", "خطوط جداشونده"),
    ("CDLSHOOTINGSTAR", "ستاره‌ی دنباله‌دار"), ("CDLSTALLEDPATTERN", "توقف"), ("CDLSTICKSANDWICH", "ساندویچ"),
    ("CDLTAKURI", "تاکوری"), ("CDLTASUKIGAP", "گپ تاسوکی"), ("CDLTHRUSTING", "رانشی"), ("CDLTRISTAR", "سه‌ستاره"),
    ("CDLUNIQUE3RIVER", "سه رود یکتا"), ("CDLUPSIDEGAP2CROWS", "گپ صعودی دو کلاغ"),
    ("CDLXSIDEGAP3METHODS", "سه روش گپ"),
]
TALIB_FLIP = {"CDLGRAVESTONEDOJI"}
TALIB_NAMES = ["tal_" + k[3:].lower() for k, _ in TALIB_PATTERNS] if talib is not None else []
TALIB_LABELS = {"tal_" + k[3:].lower(): v for k, v in TALIB_PATTERNS}

# (کلید، برچسب خرید، برچسب فروش)
CANDLE_PATTERNS = [
    ("cdl_hammer", "چکش (بعد از ریزش)", "مرد آویزان (بعد از رشد)"),
    ("cdl_shooting", "چکش وارونه (بعد از ریزش)", "ستاره‌ی دنباله‌دار (بعد از رشد)"),
    ("cdl_engulfing", "پوشای صعودی", "پوشای نزولی"),
    ("cdl_piercing", "الگوی نفوذی", "ابر سیاه"),
    ("cdl_star", "ستاره‌ی صبحگاهی", "ستاره‌ی شامگاهی"),
    ("cdl_3soldiers", "سه سرباز سفید", "سه کلاغ سیاه"),
    ("cdl_harami", "هارامی صعودی", "هارامی نزولی"),
    ("cdl_tweezer", "کف انبرکی", "سقف انبرکی"),
    ("cdl_doji", "دوجی سنجاقک", "دوجی سنگ قبر"),
    ("cdl_3inside", "سه درونی صعودی", "سه درونی نزولی"),
    ("cdl_marubozu", "ماروبوزو سبز", "ماروبوزو قرمز"),
    ("cdl_inside_break", "شکست بالای کندل درونی", "شکست پایین کندل درونی"),
]
CHART_PATTERNS = [
    ("pat_double", "کف دوقلو", "سقف دوقلو"),
    ("pat_triple", "کف سه‌قلو", "سقف سه‌قلو"),
    ("pat_hs", "سر و شونه‌ی معکوس", "سر و شونه"),
    ("pat_triangle_flat", "مثلث افزایشی", "مثلث کاهشی"),
    ("pat_triangle_sym", "مثلث متقارن — شکست بالا", "مثلث متقارن — شکست پایین"),
    ("pat_wedge", "کنج نزولی (شکست بالا)", "کنج صعودی (شکست پایین)"),
    ("pat_flag", "پرچم صعودی", "پرچم نزولی"),
    ("pat_cup", "فنجان و دسته", "فنجان وارونه"),
]
ALL_PATTERNS = [p[0] for p in CANDLE_PATTERNS + CHART_PATTERNS] + TALIB_NAMES
LABELS = {k: (a, b) for k, a, b in CANDLE_PATTERNS + CHART_PATTERNS}


def label(name):
    if name in TALIB_LABELS:
        return f"🕯 TA-Lib: {TALIB_LABELS[name]}"
    a, b = LABELS.get(name, (name, name))
    icon = "🕯" if name.startswith("cdl_") else "📐"
    return f"{icon} {a} / {b}"


def _sh(a, k):
    out = np.full(len(a), np.nan)
    if k < len(a):
        out[k:] = a[:-k]
    return out


# ==================== الگوهای کندلی (وکتوریزه) ====================

def candle_patterns(o, h, l, c, atr):
    """{name: (mask_long, mask_short, sl_long, sl_short)} — آرایه‌های هم‌طول سری."""
    n = len(c)
    rng = h - l
    body = np.abs(c - o)
    top = np.maximum(o, c)
    bot = np.minimum(o, c)
    up_sh = h - top
    lo_sh = bot - l
    green = c > o
    red = c < o
    big = rng >= 0.8 * atr

    def p(a, k):
        return _sh(a, k)

    o1, h1, l1, c1, rng1, body1 = p(o, 1), p(h, 1), p(l, 1), p(c, 1), p(rng, 1), p(body, 1)
    o2, h2, l2, c2, rng2, body2 = p(o, 2), p(h, 2), p(l, 2), p(c, 2), p(rng, 2), p(body, 2)
    green1, red1 = p(green.astype(float), 1) == 1, p(red.astype(float), 1) == 1
    green2, red2 = p(green.astype(float), 2) == 1, p(red.astype(float), 2) == 1
    atr1 = p(atr, 1)

    def ctx(k, down):
        # روند قبل از الگو (الگو k کندله): ۵ کندل قبل از شروع الگو حداقل ۱ ATR حرکت
        a_ = p(c, k)          # بسته‌شدن کندل قبل از شروع الگو
        b_ = p(c, k + 5)
        return (b_ - a_ >= atr) if down else (a_ - b_ >= atr)

    down1, up1 = ctx(1, True), ctx(1, False)
    down2, up2 = ctx(2, True), ctx(2, False)
    down3, up3 = ctx(3, True), ctx(3, False)
    buf = 0.1 * atr
    low2 = np.fmin(l, l1)
    high2 = np.fmax(h, h1)
    low3 = np.fmin(low2, l2)
    high3 = np.fmax(high2, h2)
    out = {}
    with np.errstate(invalid="ignore"):
        ham = big & (lo_sh >= 2.0 * body) & (up_sh <= 0.15 * rng)
        out["cdl_hammer"] = (ham & down1, ham & up1, l - buf, h + buf)
        inv = big & (up_sh >= 2.0 * body) & (lo_sh <= 0.15 * rng)
        out["cdl_shooting"] = (inv & down1, inv & up1, l - buf, h + buf)
        eng_l = red1 & green & (o <= c1) & (c >= o1) & (body > body1) & down2
        eng_s = green1 & red & (o >= c1) & (c <= o1) & (body > body1) & up2
        out["cdl_engulfing"] = (eng_l, eng_s, low2 - buf, high2 + buf)
        long1 = (body1 >= 0.6 * rng1) & (rng1 >= 0.8 * atr1)
        mid1 = (o1 + c1) / 2.0
        pierce = red1 & long1 & green & (o <= c1) & (c > mid1) & (c < o1) & down2
        cloud = green1 & long1 & red & (o >= c1) & (c < mid1) & (c > o1) & up2
        out["cdl_piercing"] = (pierce, cloud, low2 - buf, high2 + buf)
        long2 = (body2 >= 0.6 * rng2) & (rng2 >= 0.8 * p(atr, 2))
        mid2 = (o2 + c2) / 2.0
        small1 = body1 <= 0.3 * body2
        morning = red2 & long2 & small1 & green & (c > mid2) & down3
        evening = green2 & long2 & small1 & red & (c < mid2) & up3
        out["cdl_star"] = (morning, evening, low3 - buf, high3 + buf)
        strong = body >= 0.5 * rng
        strong1, strong2 = p(strong.astype(float), 1) == 1, p(strong.astype(float), 2) == 1
        soldiers = (green2 & green1 & green & (c1 > c2) & (c > c1) & (o1 >= o2) & (o1 <= c2) & (o >= o1) & (o <= c1)
                    & strong & strong1 & strong2 & (up_sh <= 0.3 * rng))
        crows = (red2 & red1 & red & (c1 < c2) & (c < c1) & (o1 <= o2) & (o1 >= c2) & (o <= o1) & (o >= c1)
                 & strong & strong1 & strong2 & (lo_sh <= 0.3 * rng))
        out["cdl_3soldiers"] = (soldiers, crows, low3 - buf, high3 + buf)
        har_l = red1 & long1 & green & (o >= c1) & (c <= o1) & (body <= 0.5 * body1) & down2
        har_s = green1 & long1 & red & (o <= c1) & (c >= o1) & (body <= 0.5 * body1) & up2
        out["cdl_harami"] = (har_l, har_s, low2 - buf, high2 + buf)
        tw_l = red1 & green & (np.abs(l - l1) <= 0.1 * atr) & down2
        tw_s = green1 & red & (np.abs(h - h1) <= 0.1 * atr) & up2
        out["cdl_tweezer"] = (tw_l, tw_s, low2 - buf, high2 + buf)
        doji = big & (body <= 0.1 * rng)
        dragon = doji & (up_sh <= 0.1 * rng) & (lo_sh >= 0.7 * rng) & down1
        grave = doji & (lo_sh <= 0.1 * rng) & (up_sh >= 0.7 * rng) & up1
        out["cdl_doji"] = (dragon, grave, l - buf, h + buf)
        long2b = (body2 >= 0.6 * rng2) & (rng2 >= 0.8 * p(atr, 2))
        in_up = red2 & long2b & green1 & (o1 >= c2) & (c1 <= o2) & (c > o2) & down3
        in_dn = green2 & long2b & red1 & (o1 <= c2) & (c1 >= o2) & (c < o2) & up3
        out["cdl_3inside"] = (in_up, in_dn, low3 - buf, high3 + buf)
        maru = (body >= 0.9 * rng) & (rng >= 1.2 * atr)
        out["cdl_marubozu"] = (maru & green, maru & red, l - buf, h + buf)
        inside1 = (h1 <= h2) & (l1 >= l2)
        out["cdl_inside_break"] = (inside1 & (c > h2), inside1 & (c < l2), low3 - buf, high3 + buf)
    for k, (a, b, s1, s2) in out.items():
        ok = np.isfinite(atr) & (atr > 0)
        out[k] = (a & ok, b & ok, s1, s2)
    return out


# ==================== الگوهای کلاسیک چارتی (زیگزاگ سوینگ‌های تاییدشده) ====================

def chart_patterns(o, h, l, c, atr, order=3, max_age=30, tol_atr=0.5):
    """
    زیگزاگ از سقف/کف‌های سوینگ تاییدشده (سوینگ i از کندل i+order به بعد شناخته می‌شه)، بعد در هر کندل
    الگوها روی آخرین نقاط زیگزاگ + شکست با بسته‌شدن کندل. خروجی: {name: {"L": ([idx], [sl]), "S": (...)}}
    """
    n = len(c)
    sh_pos, sl_pos = se.swing_positions(h, l, order)
    ev = sorted([(int(i) + order, 0, int(i)) for i in sl_pos] + [(int(i) + order, 1, int(i)) for i in sh_pos])
    out = {k: {"L": ([], []), "S": ([], [])} for k, _, _ in CHART_PATTERNS}
    zig = []   # [type(0=کف،1=سقف), idx, price]
    ptr = 0

    def add(kind, i):
        price = float(h[i]) if kind == 1 else float(l[i])
        if zig and zig[-1][0] == kind:
            if (kind == 1 and price > zig[-1][2]) or (kind == 0 and price < zig[-1][2]):
                zig[-1] = [kind, i, price]
            return
        zig.append([kind, i, price])
        if len(zig) > 8:
            del zig[0]

    def emit(name, side, t, sl):
        lst = out[name][side]
        lst[0].append(t)
        lst[1].append(sl)

    def line(p1, p2, x):
        return p1[2] + (p2[2] - p1[2]) * (x - p1[1]) / (p2[1] - p1[1])

    for t in range(1, n):
        while ptr < len(ev) and ev[ptr][0] <= t:
            add(ev[ptr][1], ev[ptr][2])
            ptr += 1
        A = float(atr[t])
        if not (A > 0) or len(zig) < 3:
            continue
        tol = tol_atr * A
        ct, cp = float(c[t]), float(c[t - 1])
        Z = zig
        last = Z[-1]
        fresh = t - last[1] <= max_age

        def flat_break_up(level, since):
            return ct > level and cp <= level and (since + 1 >= t or float(np.max(c[since + 1:t])) <= level)

        def flat_break_dn(level, since):
            return ct < level and cp >= level and (since + 1 >= t or float(np.min(c[since + 1:t])) >= level)

        def slope_break(p1, p2, since, up):
            """اولین بسته‌شدن بیرون خط شیب‌دار (خط گردن/ضلع) از بعد از آخرین نقطه‌ی الگو — نه عبور دوباره."""
            lt = line(p1, p2, t)
            if up:
                if not (ct > lt and cp <= line(p1, p2, t - 1)):
                    return False
            elif not (ct < lt and cp >= line(p1, p2, t - 1)):
                return False
            ks = np.arange(since + 1, t)
            if len(ks):
                lv = p1[2] + (p2[2] - p1[2]) * (ks - p1[1]) / (p2[1] - p1[1])
                if (up and (c[ks] > lv).any()) or (not up and (c[ks] < lv).any()):
                    return False
            return True

        # --- کف/سقف دوقلو ---
        a, b, d = Z[-3], Z[-2], Z[-1]
        if fresh and a[0] == 0 and b[0] == 1 and d[0] == 0 and abs(d[2] - a[2]) <= tol \
                and b[2] - max(a[2], d[2]) >= 1.5 * A and flat_break_up(b[2], d[1]):
            emit("pat_double", "L", t, min(a[2], d[2]) - 0.2 * A)
        if fresh and a[0] == 1 and b[0] == 0 and d[0] == 1 and abs(d[2] - a[2]) <= tol \
                and min(a[2], d[2]) - b[2] >= 1.5 * A and flat_break_dn(b[2], d[1]):
            emit("pat_double", "S", t, max(a[2], d[2]) + 0.2 * A)
        if len(Z) >= 5:
            p1, p2, p3, p4, p5 = Z[-5:]
            # --- سه‌قلو ---
            if fresh and [p[0] for p in (p1, p2, p3, p4, p5)] == [0, 1, 0, 1, 0]:
                lows = [p1[2], p3[2], p5[2]]
                neck = max(p2[2], p4[2])
                if max(lows) - min(lows) <= tol and neck - max(lows) >= 1.5 * A and flat_break_up(neck, p5[1]):
                    emit("pat_triple", "L", t, min(lows) - 0.2 * A)
                # سر و شونه‌ی معکوس
                if p3[2] < min(p1[2], p5[2]) - 0.5 * A and abs(p1[2] - p5[2]) <= 2 * tol:
                    nl_t = line(p2, p4, t)
                    if slope_break(p2, p4, p5[1], True) and nl_t - p3[2] >= 1.5 * A:
                        emit("pat_hs", "L", t, p5[2] - 0.2 * A)
            if fresh and [p[0] for p in (p1, p2, p3, p4, p5)] == [1, 0, 1, 0, 1]:
                highs = [p1[2], p3[2], p5[2]]
                neck = min(p2[2], p4[2])
                if max(highs) - min(highs) <= tol and min(highs) - neck >= 1.5 * A and flat_break_dn(neck, p5[1]):
                    emit("pat_triple", "S", t, max(highs) + 0.2 * A)
                if p3[2] > max(p1[2], p5[2]) + 0.5 * A and abs(p1[2] - p5[2]) <= 2 * tol:
                    nl_t = line(p2, p4, t)
                    if slope_break(p2, p4, p5[1], False) and p3[2] - nl_t >= 1.5 * A:
                        emit("pat_hs", "S", t, p5[2] + 0.2 * A)
        if len(Z) >= 4 and fresh:
            q = Z[-4:]
            highs = [p for p in q if p[0] == 1]
            lows = [p for p in q if p[0] == 0]
            if len(highs) == 2 and len(lows) == 2:
                H1, H2 = highs
                L1, L2 = lows
                # --- مثلث افزایشی / کاهشی ---
                if abs(H1[2] - H2[2]) <= tol and L2[2] > L1[2] + 0.3 * A and last is L2:
                    res = max(H1[2], H2[2])
                    if flat_break_up(res, L2[1]):
                        emit("pat_triangle_flat", "L", t, L2[2] - 0.2 * A)
                if abs(L1[2] - L2[2]) <= tol and H2[2] < H1[2] - 0.3 * A and last is H2:
                    sup = min(L1[2], L2[2])
                    if flat_break_dn(sup, H2[1]):
                        emit("pat_triangle_flat", "S", t, H2[2] + 0.2 * A)
                up_t = line(H1, H2, t)
                lo_t = line(L1, L2, t)
                if up_t > lo_t:
                    # --- مثلث متقارن ---
                    if H2[2] < H1[2] - 0.3 * A and L2[2] > L1[2] + 0.3 * A:
                        if slope_break(H1, H2, last[1], True):
                            emit("pat_triangle_sym", "L", t, L2[2] - 0.2 * A)
                        elif slope_break(L1, L2, last[1], False):
                            emit("pat_triangle_sym", "S", t, H2[2] + 0.2 * A)
                    # --- کنج‌ها ---
                    sh_ = (H2[2] - H1[2]) / (H2[1] - H1[1])
                    sl_ = (L2[2] - L1[2]) / (L2[1] - L1[1])
                    # حد ضرر کنج: آخرین کف (خرید) / سقف (فروش) الگو؛ باید سمت درست ورود باشه (پایین‌تر چک می‌شه)
                    if H2[2] < H1[2] and L2[2] < L1[2] and sh_ < sl_ < 0 and slope_break(H1, H2, last[1], True):
                        emit("pat_wedge", "L", t, L2[2] - 0.2 * A)
                    if H2[2] > H1[2] and L2[2] > L1[2] and sl_ > sh_ > 0 and slope_break(L1, L2, last[1], False):
                        emit("pat_wedge", "S", t, H2[2] + 0.2 * A)
                # --- فنجان و دسته ---
                if [p[0] for p in q] == [1, 0, 1, 0]:
                    R1, B, R2, Hd = q
                    rim = max(R1[2], R2[2])
                    depth = min(R1[2], R2[2]) - B[2]
                    if abs(R1[2] - R2[2]) <= 1.5 * tol and depth >= 3 * A and R2[1] - R1[1] >= 15 \
                            and R2[2] - Hd[2] <= 0.4 * depth and Hd[2] > B[2] and t - Hd[1] <= 20 \
                            and flat_break_up(rim, Hd[1]):
                        emit("pat_cup", "L", t, Hd[2] - 0.2 * A)
                if [p[0] for p in q] == [0, 1, 0, 1]:
                    R1, B, R2, Hd = q
                    rim = min(R1[2], R2[2])
                    depth = B[2] - max(R1[2], R2[2])
                    if abs(R1[2] - R2[2]) <= 1.5 * tol and depth >= 3 * A and R2[1] - R1[1] >= 15 \
                            and Hd[2] - R2[2] <= 0.4 * depth and Hd[2] < B[2] and t - Hd[1] <= 20 \
                            and flat_break_dn(rim, Hd[1]):
                        emit("pat_cup", "S", t, Hd[2] + 0.2 * A)
        # --- پرچم: دیرک سریع (≥۳ ATR در ≤۱۵ کندل)، اصلاح ≤ ۵۰٪، شکست سقف/کف دیرک ---
        for kind, side in ((1, "L"), (0, "S")):
            k = len(Z) - 1
            while k >= 0 and Z[k][0] != kind:
                k -= 1
            if k < 1:
                continue
            top_p, start = Z[k], Z[k - 1]
            age = t - top_p[1]
            if not (3 <= age <= 20) or top_p[1] - start[1] > 15:
                continue
            pole = (top_p[2] - start[2]) if kind == 1 else (start[2] - top_p[2])
            if pole < 3 * A:
                continue
            if kind == 1:
                lo_c = float(np.min(l[top_p[1] + 1:t + 1]))
                if (top_p[2] - lo_c) <= 0.5 * pole and flat_break_up(top_p[2], top_p[1]):
                    emit("pat_flag", "L", t, lo_c - 0.2 * A)
            else:
                hi_c = float(np.max(h[top_p[1] + 1:t + 1]))
                if (hi_c - top_p[2]) <= 0.5 * pole and flat_break_dn(top_p[2], top_p[1]):
                    emit("pat_flag", "S", t, hi_c + 0.2 * A)
    res = {}
    for k, d in out.items():
        res[k] = {s: (np.array(d[s][0], dtype=np.int64), np.array(d[s][1], dtype=np.float64)) for s in ("L", "S")}
    return res


def talib_patterns(o, h, l, c, atr):
    """تعریف‌های استاندارد TA-Lib (کد مرجع: github.com/TA-Lib/ta-lib). هر الگو فقط از کندل‌های تا همون لحظه
    استفاده می‌کنه. حد ضرر: زیر کف/بالای سقف ۵ کندل آخر (+۰.۱ ATR؛ الگوهای تا ۵ کندلی رو کامل می‌پوشونه)."""
    if talib is None:
        return {}
    o, h, l, c = [np.ascontiguousarray(x, dtype=np.float64) for x in (o, h, l, c)]
    buf = 0.1 * atr
    lo3 = l.copy()
    hi3 = h.copy()
    for k in range(1, 5):
        lo3 = np.fmin(lo3, _sh(l, k))
        hi3 = np.fmax(hi3, _sh(h, k))
    ok = np.isfinite(atr) & (atr > 0)
    out = {}
    for fn, _ in TALIB_PATTERNS:
        r = getattr(talib, fn)(o, h, l, c).astype(np.float64)
        if fn in TALIB_FLIP:
            r = -r
        out["tal_" + fn[3:].lower()] = ((r > 0) & ok, (r < 0) & ok, lo3 - buf, hi3 + buf)
    return out


def _valid_stop(out, c):
    """فقط ورودهایی که حد ضررشون سمت درسته (خرید: زیر قیمت ورود، فروش: بالای قیمت ورود)."""
    res = {}
    for k, lst in out.items():
        new = []
        for sd, idx, slv in lst:
            ok = (slv < c[idx]) if sd > 0 else (slv > c[idx])
            ok &= np.isfinite(slv)
            new.append((sd, idx[ok], slv[ok]))
        res[k] = new
    return res


def all_patterns(o, h, l, c, atr, order=3):
    """{name: [(side(+1/-1), idx, sl), ...]} برای همه‌ی الگوها (حد ضرر سمت اشتباه = حذف)."""
    return _valid_stop(_all_patterns(o, h, l, c, atr, order), c)


def _all_patterns(o, h, l, c, atr, order=3):
    out = {}
    for k, (ml, ms, sl_l, sl_s) in talib_patterns(o, h, l, c, atr).items():
        il, is_ = np.flatnonzero(ml), np.flatnonzero(ms)
        out[k] = [(1, il, sl_l[il]), (-1, is_, sl_s[is_])]
    for k, (ml, ms, sl_l, sl_s) in candle_patterns(o, h, l, c, atr).items():
        il, is_ = np.flatnonzero(ml), np.flatnonzero(ms)
        out[k] = [(1, il, sl_l[il]), (-1, is_, sl_s[is_])]
    for k, d in chart_patterns(o, h, l, c, atr, order).items():
        out[k] = [(1, d["L"][0], d["L"][1]), (-1, d["S"][0], d["S"][1])]
    return out


# ==================== سیگنال استراتژی «الگو + ساختار بازار» ====================
# الگوهای برگشتی کلاسیک (برای استراتژی خلاف‌جهت «جمعیت دیررس»): پوشا، چکش/چکش وارونه/ستاره‌ی دنباله‌دار/
# مرد آویزان، ستاره‌ی صبحگاهی/شامگاهی، نفوذی/ابر سیاه، کف/سقف دوقلو — نسخه‌ی خودمون اول، بعد TA-Lib (اگه نصبه).
REVERSAL_PATTERNS = [
    "cdl_engulfing", "cdl_hammer", "cdl_shooting", "cdl_star", "cdl_piercing", "pat_double",
    "tal_engulfing", "tal_hammer", "tal_invertedhammer", "tal_shootingstar", "tal_hangingman",
    "tal_morningstar", "tal_eveningstar", "tal_morningdojistar", "tal_eveningdojistar",
    "tal_piercing", "tal_darkcloudcover",
]
PATTERN_SETS = {
    "all": lambda: list(ALL_PATTERNS),
    "own": lambda: [k for k, _, _ in CANDLE_PATTERNS],
    "chart": lambda: [k for k, _, _ in CHART_PATTERNS],
    "talib": lambda: list(TALIB_NAMES),
    "own+chart": lambda: [k for k, _, _ in CANDLE_PATTERNS + CHART_PATTERNS],
    "reversal": lambda: [k for k in REVERSAL_PATTERNS if k in ALL_PATTERNS],
}


def pattern_names(pset):
    return PATTERN_SETS.get(pset, PATTERN_SETS["all"])()


def pattern_signals(o, h, l, c, atr, order, names):
    """
    برای هر کندل: اولین الگوی خرید/فروش (به ترتیب names) و حد ضرر خودش.
    خروجی: (sl_long, name_long, sl_short, name_short) — NaN / -1 یعنی الگویی نیست.
    """
    n = len(c)
    allp = all_patterns(o, h, l, c, atr, order)
    out = []
    for side in (1, -1):
        sl = np.full(n, np.nan)
        nm = np.full(n, -1, dtype=np.int64)
        for j in range(len(names) - 1, -1, -1):     # از آخر به اول: اولویت با اولین اسم
            lst = allp.get(names[j])
            if not lst:
                continue
            for sd, idx, slv in lst:
                if sd == side and len(idx):
                    sl[idx] = slv
                    nm[idx] = j
        out += [sl, nm]
    return tuple(out)
