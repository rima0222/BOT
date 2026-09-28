# -*- coding: utf-8 -*-
"""
موتور معامله مجازی. هیچ سفارش واقعی به هیچ صرافی ارسال نمی‌شه؛
همه‌چیز شبیه‌سازی و در SQLite ثبت می‌شه (سبک، بدون نیاز به دیتابیس سنگین).

نکات مهم منطقی:
- هر پوزیشن با یک "لوریج ایمن" باز می‌شه: لوریج طوری محاسبه می‌شه که قیمت لیکویید
  همیشه از حد ضرر دورتر باشه (یعنی همیشه SL قبل از لیکویید فعال می‌شه، نه برعکس).
- فقط "مارجین" واقعی هر پوزیشن (نه کل ارزش پوزیشن) از سرمایه قفل می‌شه — دقیقاً
  مثل حساب فیوچرز واقعی. همین باعث می‌شه سرمایه برای چند پوزیشن هم‌زمان کافی بمونه.
- سقفی روی سهم هر پوزیشن از *سرمایه‌ی آزادِ لحظه‌ای* گذاشته شده تا افت سرمایه‌ی
  آزاد به‌آرومی و بدون پرش ناگهانی اتفاق بیفته.
- کارمزد به‌صورت میکر/تیکر واقعی حساب می‌شه: ورود و برخورد به TP فرض می‌شه با
  سفارش لیمیتی (میکر، ارزون‌تر)، برخورد به حد ضرر (اولیه یا تریلینگ) با سفارش
  استاپ فوری‌الاجرا (تیکر، گرون‌تر + کمی اسلیپیج تخمینی).
- تریلینگ استاپ اختیاریه: اگه روی یک پوزیشن فعال باشه (به انتخاب لحظه‌ی باز شدنش،
  نه با تغییر بعدی تنظیمات)، به‌جای TP ثابت، با نردبان R مدیریت می‌شه و سقف مشخصی
  نداره.
- تغییر سرمایه اولیه («ریست سرمایه») یک عملیات جدا و آگاهانه‌ست، هیچ‌وقت به‌صورت
  جانبی از ذخیره‌ی تنظیمات دیگه (مثل درصد ریسک) اجرا نمی‌شه.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

import money


def utc_ms(dt):
    """datetime بدون منطقه‌ی زمانی (که همیشه UTC ذخیره می‌شه) → میلی‌ثانیه؛ مستقل از ساعت سرور."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _now(as_of=None):
    """زمان مرجع: در حالت زنده همیشه None (یعنی ساعت واقعی)؛ در بک‌تست، زمان
    شبیه‌سازی‌شده‌ی همون کندل تاریخی پاس داده می‌شه تا equity curve و کول‌داون
    و تاریخچه‌ی معاملات همه بر اساس زمان واقعیِ تاریخی ثبت بشن، نه ساعت اجرای اسکریپت."""
    return as_of if as_of is not None else datetime.utcnow()


def get_conn(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, side TEXT,
            entry REAL, sl REAL, tp REAL, initial_sl REAL, peak_price REAL,
            size REAL, notional REAL, margin REAL, leverage REAL, liquidation_price REAL,
            trailing_enabled INTEGER DEFAULT 0, strategy_name TEXT,
            status TEXT,
            open_time TEXT, close_time TEXT,
            close_price REAL, pnl REAL, fee_cost REAL, exit_type TEXT, result TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT, balance REAL
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER, time TEXT, price REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT, symbol TEXT, side TEXT, strategy_name TEXT,
            entry REAL, sl REAL, tp REAL, rr REAL, htf_agree INTEGER,
            opened INTEGER, rejection_reason TEXT
        )
    """)
    for col, coltype in (("score", "REAL"), ("reasons", "TEXT")):
        try:
            conn.execute(f"ALTER TABLE signal_log ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass
    conn.commit()

    # مهاجرت نرم برای دیتابیس‌های قدیمی‌تر که این ستون‌ها رو نداشتن
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
    new_cols = (
        ("notional", "REAL"), ("fee_cost", "REAL"), ("margin", "REAL"),
        ("leverage", "REAL"), ("liquidation_price", "REAL"),
        ("initial_sl", "REAL"), ("peak_price", "REAL"), ("trailing_enabled", "INTEGER DEFAULT 0"),
        ("strategy_name", "TEXT"), ("exit_type", "TEXT"), ("last_bar_ts", "INTEGER"),
        ("entry_taker", "INTEGER DEFAULT 0"), ("expire_ts", "INTEGER"), ("max_hold_min", "REAL DEFAULT 0"),
        ("fill_ts", "INTEGER"), ("timeframe", "TEXT"), ("score", "REAL"),
    )
    for col, coltype in new_cols:
        if col not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {coltype}")
                conn.commit()
            except sqlite3.OperationalError:
                pass

    conn.execute("UPDATE trades SET margin = notional WHERE margin IS NULL AND notional IS NOT NULL")
    conn.execute("UPDATE trades SET leverage = 1 WHERE leverage IS NULL")
    conn.execute("UPDATE trades SET initial_sl = sl WHERE initial_sl IS NULL")
    conn.commit()
    return conn


# ==================== موجودی / سرمایه ====================

def get_balance(conn, start_balance, as_of=None):
    row = conn.execute("SELECT balance FROM equity ORDER BY id DESC LIMIT 1").fetchone()
    if row:
        return row[0]
    conn.execute("INSERT INTO equity (time, balance) VALUES (?, ?)",
                 (_now(as_of).isoformat(), start_balance))
    conn.commit()
    return start_balance


def record_equity(conn, balance, as_of=None):
    conn.execute("INSERT INTO equity (time, balance) VALUES (?, ?)",
                 (_now(as_of).isoformat(), balance))
    conn.commit()


def get_locked_capital(conn):
    """مجموع *مارجین* (نه کل ارزش پوزیشن) که الان توی پوزیشن‌های باز قفل شده."""
    row = conn.execute("SELECT COALESCE(SUM(margin), 0) FROM trades WHERE status IN ('OPEN','PENDING')").fetchone()
    return row[0] or 0.0


def get_available_capital(conn, start_balance):
    return get_balance(conn, start_balance) - get_locked_capital(conn)


def reset_capital(conn, amount, as_of=None):
    """
    عملیات آگاهانه‌ی «ریست سرمایه» — فقط باید از یک اکشن جدا و صریح صدا زده بشه
    (نه به‌صورت جانبی همراه ذخیره‌ی تنظیمات دیگه)، چون خط پایه‌ی محاسبه‌ی بازده رو
    عوض می‌کنه. تاریخچه‌ی معاملات پاک نمی‌شه، فقط baseline موجودی/بازده ریست می‌شه.
    """
    record_equity(conn, amount, as_of=as_of)
    set_setting(conn, "initial_capital", amount)
    set_setting(conn, "capital_reset_time", _now(as_of).isoformat())


# ==================== وضعیت پوزیشن‌ها ====================

def get_open_symbols(conn):
    rows = conn.execute("SELECT DISTINCT symbol FROM trades WHERE status IN ('OPEN','PENDING')").fetchall()
    return [r[0] for r in rows]


def get_open_position_count(conn):
    row = conn.execute("SELECT COUNT(*) FROM trades WHERE status IN ('OPEN','PENDING')").fetchone()
    return row[0] or 0


def get_last_close_time(conn, symbol):
    row = conn.execute(
        "SELECT close_time FROM trades WHERE symbol=? AND status='CLOSED' ORDER BY id DESC LIMIT 1",
        (symbol,)
    ).fetchone()
    if not row or not row[0]:
        return None
    return datetime.fromisoformat(row[0])


def is_in_cooldown(conn, symbol, cooldown_hours, as_of=None):
    last_close = get_last_close_time(conn, symbol)
    if not last_close:
        return False
    return (_now(as_of) - last_close) < timedelta(hours=cooldown_hours)


def has_open_trade(conn, symbol):
    return conn.execute(
        "SELECT id FROM trades WHERE symbol=? AND status IN ('OPEN','PENDING')", (symbol,)
    ).fetchone() is not None


# ==================== تنظیمات کلید-مقدار ====================

def get_setting(conn, key, default=None):
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_setting(conn, key, value):
    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
    conn.commit()


# ==================== باز کردن پوزیشن ====================

def compute_position_size(side, entry, sl, risk_pct, balance, locked_margin, open_count,
                          min_notional=5.0, max_open_positions=None, max_leverage=5,
                          leverage_safety_mult=1.6, position_pct_cap=20.0, risk_usd=None, fees=None):
    """
    محاسبه‌ی خالص (بدون دیتابیس) حجم/لوریج/مارجین یک پوزیشن جدید. هم ربات زنده
    (open_trade) و هم موتور بک‌تست دقیقاً از همین تابع استفاده می‌کنن.
    risk_usd: اگه داده بشه، ضرر خالص در حد ضرر = همین دلار (مدیریت سرمایه‌ی یکسان)؛
              وگرنه risk_pct درصد از موجودی.
    fees: (fi, fs) از money.fee_fracs — ضرر هر واحد شامل کارمزد ورود و خروج با استاپ حساب می‌شه.
    """
    if max_open_positions is not None and open_count >= max_open_positions:
        return {"ok": False, "reason": "max_positions_reached"}
    if entry <= 0:
        return {"ok": False, "reason": "invalid_entry"}
    per_unit_risk = abs(entry - sl)
    if per_unit_risk <= 0:
        return {"ok": False, "reason": "invalid_stop"}

    available_margin = balance - locked_margin
    if available_margin < min_notional:
        return {"ok": False, "reason": "insufficient_capital"}

    sl_distance_pct = per_unit_risk / entry * 100
    safe_leverage = 100 / (sl_distance_pct * leverage_safety_mult) if sl_distance_pct > 0 else max_leverage
    leverage = max(1.0, min(max_leverage, safe_leverage))
    leverage = round(leverage, 2)

    per_unit_loss = money.loss_per_unit(side == "LONG", entry, sl, fees[0], fees[1]) if fees else per_unit_risk
    risk_amount = float(risk_usd) if risk_usd else balance * (risk_pct / 100)
    raw_size = risk_amount / per_unit_loss
    raw_notional = entry * raw_size
    raw_margin = raw_notional / leverage

    margin_cap = available_margin * (position_pct_cap / 100)

    capped = False
    if raw_margin > margin_cap:
        capped = True
        margin = margin_cap
        notional = margin * leverage
        size = notional / entry
    else:
        margin, notional, size = raw_margin, raw_notional, raw_size

    if margin < min_notional:
        return {"ok": False, "reason": "notional_too_small"}

    liquidation_price = entry * (1 - 1 / leverage) if side == "LONG" else entry * (1 + 1 / leverage)
    return {"ok": True, "size": size, "notional": notional, "margin": margin, "leverage": leverage,
            "liquidation_price": liquidation_price, "capped": capped, "per_unit_risk": per_unit_risk,
            "per_unit_loss": per_unit_loss, "risk_amount": per_unit_loss * size}


def open_trade(conn, symbol, side, entry, sl, tp, risk_pct, start_balance,
                min_notional=5.0, max_open_positions=None,
                max_leverage=5, leverage_safety_mult=1.6, position_pct_cap=20.0,
                trailing_enabled=False, strategy_name=None, as_of=None,
                pending=False, expire_ts=None, entry_taker=False, max_hold_min=0, timeframe=None, score=None,
                risk_usd=None, fees=None):
    """
    باز کردن پوزیشن مجازی با لوریج و مدیریت سرمایه‌ی واقعی.
    pending=True: سفارش لیمیت ورود ثبت می‌شه (وضعیت PENDING، مارجین و جای پوزیشن رزرو)؛
    فقط اگه قیمت تا expire_ts از قیمت ورود رد بشه پر می‌شه، وگرنه لغو (CANCELLED).
    خروجی: dict شامل opened (True/False) و در صورت باز شدن، جزئیات کامل پوزیشن.
    """
    balance = get_balance(conn, start_balance, as_of=as_of)
    pos = compute_position_size(
        side, entry, sl, risk_pct, balance, get_locked_capital(conn), get_open_position_count(conn),
        min_notional=min_notional, max_open_positions=max_open_positions, max_leverage=max_leverage,
        leverage_safety_mult=leverage_safety_mult, position_pct_cap=position_pct_cap,
        risk_usd=risk_usd, fees=fees,
    )
    if not pos["ok"]:
        return {"opened": False, "reason": pos["reason"]}
    size, notional, margin = pos["size"], pos["notional"], pos["margin"]
    leverage, liquidation_price, capped = pos["leverage"], pos["liquidation_price"], pos["capped"]

    now_ms = utc_ms(_now(as_of))
    conn.execute("""
        INSERT INTO trades (symbol, side, entry, sl, tp, initial_sl, peak_price,
                             size, notional, margin, leverage, liquidation_price,
                             trailing_enabled, strategy_name, status, open_time,
                             entry_taker, expire_ts, max_hold_min, fill_ts, timeframe, score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (symbol, side, entry, sl, tp, sl, entry,
          size, notional, margin, leverage, liquidation_price,
          1 if trailing_enabled else 0, strategy_name, "PENDING" if pending else "OPEN", _now(as_of).isoformat(),
          1 if entry_taker else 0, expire_ts, float(max_hold_min or 0), None if pending else now_ms, timeframe,
          score))
    conn.commit()

    return {
        "opened": True, "size": size, "notional": notional, "margin": margin,
        "leverage": leverage, "liquidation_price": liquidation_price, "capped": capped,
        "risk_amount": pos["risk_amount"],
    }


# ==================== تریلینگ استاپ ====================

def compute_trailing_sl(entry, initial_sl, side, peak_price, ladder, beyond_distance_r, floor=None):
    """
    بر اساس بیشترین سود دیده‌شده تا الان (peak_price)، حد ضرر پویا رو حساب می‌کنه.
    ladder: لیستی از (trigger_R, lock_R) که باید بر اساس trigger_R مرتب باشه.
    بعد از آخرین پله، با فاصله‌ی ثابت (beyond_distance_r) پشت بیشینه ادامه پیدا می‌کنه
    — یعنی سقفی نداره و پوزیشن‌های قوی می‌تونن فراتر از R:R برنامه‌ریزی‌شده هم برن.
    """
    risk_per_unit = abs(entry - initial_sl)
    if risk_per_unit <= 0:
        return initial_sl

    peak_r = (peak_price - entry) / risk_per_unit if side == "LONG" else (entry - peak_price) / risk_per_unit

    ladder = sorted(ladder, key=lambda x: x[0])
    locked_r = None
    for trigger_r, lock_r in ladder:
        if peak_r >= trigger_r - 1e-9:
            locked_r = lock_r

    if locked_r is None:
        return initial_sl

    if ladder:
        last_trigger_r = ladder[-1][0]
        if peak_r > last_trigger_r + 1e-9:
            locked_r = max(locked_r, peak_r - beyond_distance_r)

    new_sl = entry + locked_r * risk_per_unit if side == "LONG" else entry - locked_r * risk_per_unit
    # کف سربه‌سر بعد از کارمزد (floor): وقتی تریلینگ فعال شد، دیگه با ضرر بسته نمی‌شه.
    # (هیچ‌وقت بالاتر از بیشترین قیمت دیده‌شده نمی‌ره — قیمتی که بازار بهش نرسیده قفل نمی‌شه)
    if floor is not None:
        new_sl = max(new_sl, min(floor, peak_price)) if side == "LONG" else min(new_sl, max(floor, peak_price))
    # حد ضرر هیچ‌وقت نباید عقب‌تر از حد ضرر اولیه بره (فقط جلو می‌ره، هیچ‌وقت بدتر نمی‌شه)
    if side == "LONG":
        return max(new_sl, initial_sl)
    return min(new_sl, initial_sl)


def update_trailing_stops(conn, symbol, current_price, ladder, beyond_distance_r, as_of=None):
    """برای همه‌ی پوزیشن‌های باز این نماد که تریلینگ روشن دارن، peak_price و sl رو آپدیت می‌کنه."""
    rows = conn.execute(
        "SELECT id, side, entry, initial_sl, peak_price FROM trades "
        "WHERE symbol=? AND status='OPEN' AND trailing_enabled=1", (symbol,)
    ).fetchall()
    for trade_id, side, entry, initial_sl, peak_price in rows:
        peak_price = peak_price if peak_price is not None else entry
        new_peak = max(peak_price, current_price) if side == "LONG" else min(peak_price, current_price)
        new_sl = compute_trailing_sl(entry, initial_sl, side, new_peak, ladder, beyond_distance_r)
        conn.execute("UPDATE trades SET peak_price=?, sl=? WHERE id=?", (new_peak, new_sl, trade_id))
    conn.commit()


# ==================== مدیریت کندل‌به‌کندل (مشترک بین زنده و بک‌تست) ====================

def step_bar(side, entry, initial_sl, sl, tp, peak, trailing, o, h, l, c, ladder, beyond_distance_r, floor=None):
    """
    یک کندل (مثلاً ۱ دقیقه‌ای در زنده، ۱۵ دقیقه‌ای در بک‌تست) رو روی یک پوزیشن باز
    اعمال می‌کنه — با سایه‌ها (high/low)، نه فقط قیمت بسته‌شدن؛ چون حد ضرر واقعی صرافی
    با سایه هم فعال می‌شه. قواعد (محافظه‌کارانه، به ضرر خودمون):
      ۱) اول حد ضرری که از قبل فعال بوده چک می‌شه. اگه قیمت باز شدن کندل از خودِ حد ضرر
         هم رد شده باشه (گپ)، خروج با قیمت باز شدن (بدتر از SL) حساب می‌شه.
      ۲) بدون تریلینگ: اگه همون کندل هم SL و هم TP رو لمس کرده، فرض می‌شه SL اول خورده.
      ۳) با تریلینگ: بیشینه‌ی سود با سایه‌ی کندل آپدیت می‌شه؛ اگه SL جدید بالا رفت و
         کندل پایین‌تر از اون بسته شد، یعنی قیمت بعد از سقف برگشته و به SL جدید خورده.
    خروجی: (بسته شد؟, قیمت خروج, نوع خروج 'STOP'/'TP', سطح فعال‌شده, sl جدید, peak جدید)
    """
    if side == "LONG":
        if l <= sl:
            return True, (o if o < sl else sl), "STOP", sl, sl, peak
        # حد سود (R:R) همیشه فعاله — حتی با تریلینگ: اگه قیمت به هدف رسید، همون‌جا بسته می‌شه
        if h >= tp:
            return True, (o if o > tp else tp), "TP", tp, sl, peak
        if not trailing:
            return False, None, None, None, sl, peak
        new_peak = max(peak, h)
        new_sl = max(compute_trailing_sl(entry, initial_sl, side, new_peak, ladder, beyond_distance_r, floor), sl)
        if new_sl > sl and c <= new_sl:
            return True, new_sl, "STOP", new_sl, new_sl, new_peak
        return False, None, None, None, new_sl, new_peak
    else:
        if h >= sl:
            return True, (o if o > sl else sl), "STOP", sl, sl, peak
        if l <= tp:
            return True, (o if o < tp else tp), "TP", tp, sl, peak
        if not trailing:
            return False, None, None, None, sl, peak
        new_peak = min(peak, l)
        new_sl = min(compute_trailing_sl(entry, initial_sl, side, new_peak, ladder, beyond_distance_r, floor), sl)
        if new_sl < sl and c >= new_sl:
            return True, new_sl, "STOP", new_sl, new_sl, new_peak
        return False, None, None, None, new_sl, new_peak


def result_of(exit_type):
    """
    برد و باخت فقط برای معاملاتی که به هدف R:R یا حد ضرر اولیه رسیدن حساب می‌شه:
      TP → WIN، SL → LOSS
      خروج با تریلینگ (قبل از رسیدن به هدف) → TRAIL (جدا شمرده می‌شه، نه برد نه باخت)
      حد زمانی / دستی / پایان دیتا → OTHER
    """
    if exit_type == "TP":
        return "WIN"
    if exit_type == "SL":
        return "LOSS"
    if exit_type in ("TRAIL_SL", "BREAKEVEN"):
        return "TRAIL"
    return "OTHER"


def classify_exit_level(kind, level, initial_sl, entry, trailing_enabled):
    """نوع خروج بر اساس «سطحی که فعال شد» (نه قیمت پرشده، که با گپ ممکنه فرق کنه)."""
    if kind == "TP":
        return "TP"
    if kind in ("END", "TIME"):
        return kind
    if not trailing_enabled:
        return "SL"
    if level == initial_sl:
        return "SL"
    if abs(level - entry) < max(entry * 0.0005, 1e-12):
        return "BREAKEVEN"
    return "TRAIL_SL"


def process_bars(conn, symbol, bars, start_balance, ladder, beyond_distance_r,
                 maker_fee_pct=0.0, taker_fee_pct=0.0, taker_slippage_pct=0.0, funding_pct_8h=0.0,
                 now_ms=None, trail_floor=False):
    """
    ربات زنده: کندل‌های ۱ دقیقه‌ایِ بسته‌شده‌ی جدید رو روی سفارش‌ها/پوزیشن‌های این نماد
    اعمال می‌کنه — دقیقاً با همون قواعد موتور بک‌تست:
      - سفارش لیمیت در انتظار (PENDING): اگه قیمت از قیمت ورود رد بشه پر می‌شه (و همون
        کندل هم محافظه‌کارانه روی پوزیشن اعمال می‌شه)؛ اگه تا زمان انقضا پر نشد، لغو.
      - پوزیشن باز: step_bar (SL/TP/تریلینگ با سایه‌ی کندل) + حد زمانی (اگه تعیین شده).
    bars: لیست (open_ms, o, h, l, c) مرتب. آخرین کندل پردازش‌شده ذخیره می‌شه، پس بعد از
    خاموش/روشن شدن ربات هم هیچ کندلی جا نمی‌افته.
    """
    now_ms = now_ms if now_ms is not None else utc_ms(datetime.utcnow())
    rows = conn.execute(
        "SELECT id, side, entry, sl, tp, initial_sl, peak_price, size, trailing_enabled, open_time, last_bar_ts, "
        "status, expire_ts, max_hold_min, fill_ts, entry_taker, notional "
        "FROM trades WHERE symbol=? AND status IN ('OPEN','PENDING')", (symbol,)
    ).fetchall()
    if not rows:
        return
    for (trade_id, side, entry, sl, tp, initial_sl, peak, size, trailing, open_time, last_bar_ts,
         status, expire_ts, max_hold_min, fill_ts, entry_taker, notional) in rows:
        trailing = bool(trailing)
        peak = peak if peak is not None else entry
        initial_sl = initial_sl if initial_sl is not None else sl
        open_ms = utc_ms(datetime.fromisoformat(open_time)) if open_time else 0
        # کندلی که قبل از لحظه‌ی ثبت سفارش شروع شده حساب نمی‌شه
        min_start = max(open_ms - (open_ms % 60_000), (last_bar_ts or -1) + 1)
        closed = False
        floor = None
        if trailing and trail_floor:
            fi, fs, _ = money.fee_fracs(maker_fee_pct, taker_fee_pct, taker_slippage_pct, bool(entry_taker))
            floor = money.breakeven_stop(side == "LONG", entry, fi, fs)
        for (bar_open_ms, o, h, l, c) in bars:
            if bar_open_ms < min_start:
                continue
            bar_close_ms = bar_open_ms + 60_000
            if status == "PENDING":
                if expire_ts and bar_open_ms >= expire_ts:
                    break
                filled = (l < entry) if side == "LONG" else (h > entry)
                last_bar_ts = bar_open_ms
                if not filled:
                    continue
                status, fill_ts = "OPEN", bar_open_ms
                conn.execute("UPDATE trades SET status='OPEN', fill_ts=? WHERE id=?", (fill_ts, trade_id))
            hit, price, kind, level, sl, peak = step_bar(side, entry, initial_sl, sl, tp, peak, trailing,
                                                         o, h, l, c, ladder, beyond_distance_r, floor)
            last_bar_ts = bar_open_ms
            if not hit and max_hold_min and fill_ts and bar_close_ms >= fill_ts + max_hold_min * 60_000:
                hit, price, kind, level = True, c, "TIME", None
            if hit:
                exit_type = classify_exit_level(kind, level, initial_sl, entry, trailing)
                _close_trade_row(conn, trade_id, side, entry, size, price, exit_type, start_balance,
                                 maker_fee_pct, taker_fee_pct, taker_slippage_pct,
                                 as_of=datetime.utcfromtimestamp(bar_close_ms / 1000),
                                 entry_taker=bool(entry_taker), notional=notional, fill_ts=fill_ts,
                                 funding_pct_8h=funding_pct_8h)
                closed = True
                break
        if closed:
            continue
        if status == "PENDING" and expire_ts and now_ms >= expire_ts:
            cancel_pending(conn, trade_id)
            continue
        conn.execute("UPDATE trades SET sl=?, peak_price=?, last_bar_ts=? WHERE id=?",
                     (sl, peak, last_bar_ts, trade_id))
    conn.commit()


def cancel_pending(conn, trade_id, reason="CANCELLED"):
    """لغو سفارش ورودی که پر نشده — مارجین آزاد می‌شه، سود/زیانی نداره، کول‌داون هم نمی‌خوره."""
    conn.execute("UPDATE trades SET status='CANCELLED', close_time=?, pnl=0, fee_cost=0, exit_type=? "
                 "WHERE id=? AND status='PENDING'", (_now().isoformat(), reason, trade_id))
    conn.commit()


def _close_trade_row(conn, trade_id, side, entry, size, close_price, exit_type, start_balance,
                     maker_fee_pct, taker_fee_pct, taker_slippage_pct, as_of=None,
                     entry_taker=False, notional=None, fill_ts=None, funding_pct_8h=0.0):
    balance = get_balance(conn, start_balance, as_of=as_of)
    pnl_gross = (close_price - entry) * size if side == "LONG" else (entry - close_price) * size
    fee_cost, _, _, _ = _fee_for_exit(entry, close_price, size, exit_type,
                                       maker_fee_pct, taker_fee_pct, taker_slippage_pct, entry_taker=entry_taker)
    if funding_pct_8h and fill_ts:
        close_ms = utc_ms(_now(as_of))
        fee_cost += funding_cost(notional if notional is not None else entry * size, fill_ts, close_ms,
                                 funding_pct_8h)
    pnl_net = pnl_gross - fee_cost
    balance += pnl_net
    result = result_of(exit_type)
    conn.execute("""
        UPDATE trades SET status='CLOSED', close_time=?, close_price=?, pnl=?, fee_cost=?,
                           exit_type=?, result=?
        WHERE id=?
    """, (_now(as_of).isoformat(), close_price, pnl_net, fee_cost, exit_type, result, trade_id))
    conn.commit()
    record_equity(conn, balance, as_of=as_of)
    return pnl_net


# ==================== بستن پوزیشن (خودکار یا دستی) ====================

def funding_cost(notional, from_ms, to_ms, pct_per_8h):
    """هزینه‌ی فاندینگ فیوچرز برای مدت نگه‌داری (محافظه‌کارانه همیشه پرداختی)."""
    if not pct_per_8h or to_ms <= from_ms:
        return 0.0
    hours = (to_ms - from_ms) / 3_600_000
    return notional * (pct_per_8h / 100) * hours / 8


def _fee_for_exit(entry, close_price, size, exit_type, maker_fee_pct, taker_fee_pct, taker_slippage_pct,
                  entry_taker=False):
    """
    کارمزد واقعی: ورود با سفارش لیمیت میکره (با سفارش بازار تیکر؛ اسلیپیج ورود بازار
    از قبل توی قیمت ورود اعمال شده). خروج با TP میکر (سفارش لیمیتی منتظرمونده)؛ خروج با
    SL/تریلینگ/حد زمانی همیشه تیکر (سفارش فوری) + کمی اسلیپیج.
    """
    entry_fee = entry * size * ((taker_fee_pct if entry_taker else maker_fee_pct) / 100)
    if exit_type == "TP":
        exit_fee = close_price * size * (maker_fee_pct / 100)
        slippage_cost = 0.0
    else:  # SL, TRAIL_SL, BREAKEVEN همگی از طریق سفارش استاپ (تیکر) اجرا می‌شن
        exit_fee = close_price * size * (taker_fee_pct / 100)
        slippage_cost = close_price * size * (taker_slippage_pct / 100)
    return entry_fee + exit_fee + slippage_cost, entry_fee, exit_fee, slippage_cost


def _classify_exit(side, close_price, initial_sl, tp, entry, trailing_enabled):
    """تشخیص نوع خروج برای آمار جدا (آیا این معامله واقعاً به TP/SL اولیه خورده یا کار تریلینگ بوده)."""
    eps = 1e-9
    if not trailing_enabled:
        if abs(close_price - tp) < eps or (side == "LONG" and close_price >= tp) or (side == "SHORT" and close_price <= tp):
            return "TP"
        return "SL"
    # حالت تریلینگ: مقایسه با سطح اولیه‌ی حد ضرر
    if abs(close_price - initial_sl) < eps:
        return "SL"
    if abs(close_price - entry) < max(entry * 0.0005, eps):
        return "BREAKEVEN"
    return "TRAIL_SL"


def check_and_close_trades(conn, symbol, current_price, start_balance,
                            maker_fee_pct=0.0, taker_fee_pct=0.0, taker_slippage_pct=0.0, as_of=None):
    rows = conn.execute(
        "SELECT id, side, entry, sl, tp, initial_sl, size, trailing_enabled FROM trades "
        "WHERE symbol=? AND status='OPEN'", (symbol,)
    ).fetchall()
    if not rows:
        return
    balance = get_balance(conn, start_balance, as_of=as_of)
    for trade_id, side, entry, sl, tp, initial_sl, size, trailing_enabled in rows:
        closed, close_price = False, None
        if side == "LONG":
            if current_price <= sl:
                closed, close_price = True, sl
            elif (not trailing_enabled) and current_price >= tp:
                closed, close_price = True, tp
        else:  # SHORT
            if current_price >= sl:
                closed, close_price = True, sl
            elif (not trailing_enabled) and current_price <= tp:
                closed, close_price = True, tp

        if closed:
            exit_type = _classify_exit(side, close_price, initial_sl, tp, entry, bool(trailing_enabled))
            pnl_gross = (close_price - entry) * size if side == "LONG" else (entry - close_price) * size
            fee_cost, _, _, _ = _fee_for_exit(entry, close_price, size, exit_type,
                                               maker_fee_pct, taker_fee_pct, taker_slippage_pct)
            pnl_net = pnl_gross - fee_cost
            balance += pnl_net
            result = result_of(exit_type)
            conn.execute("""
                UPDATE trades SET status='CLOSED', close_time=?, close_price=?, pnl=?, fee_cost=?,
                                   exit_type=?, result=?
                WHERE id=?
            """, (_now(as_of).isoformat(), close_price, pnl_net, fee_cost, exit_type, result, trade_id))
            conn.commit()
            record_equity(conn, balance, as_of=as_of)


def close_trade_manually(conn, trade_id, current_price, start_balance,
                          maker_fee_pct=0.0, taker_fee_pct=0.0, taker_slippage_pct=0.0, as_of=None):
    """بستن دستی یک پوزیشن از پنل — چون تصمیم دستیه، مثل یک سفارش بازار (تیکر) حساب می‌شه."""
    pend = conn.execute("SELECT symbol FROM trades WHERE id=? AND status='PENDING'", (trade_id,)).fetchone()
    if pend:
        cancel_pending(conn, trade_id, "MANUAL_CANCEL")
        return {"ok": True, "symbol": pend[0], "pnl": 0.0, "cancelled": True}
    row = conn.execute(
        "SELECT symbol, side, entry, size, initial_sl, tp, entry_taker FROM trades WHERE id=? AND status='OPEN'",
        (trade_id,)
    ).fetchone()
    if not row:
        return {"ok": False, "error": "پوزیشن باز با این شناسه پیدا نشد"}
    symbol, side, entry, size, initial_sl, tp, entry_taker = row
    balance = get_balance(conn, start_balance, as_of=as_of)

    pnl_gross = (current_price - entry) * size if side == "LONG" else (entry - current_price) * size
    fee_cost, _, _, _ = _fee_for_exit(entry, current_price, size, "MANUAL",
                                       maker_fee_pct, taker_fee_pct, taker_slippage_pct,
                                       entry_taker=bool(entry_taker))
    pnl_net = pnl_gross - fee_cost
    balance += pnl_net
    result = "OTHER"

    conn.execute("""
        UPDATE trades SET status='CLOSED', close_time=?, close_price=?, pnl=?, fee_cost=?,
                           exit_type='MANUAL', result=?
        WHERE id=?
    """, (_now(as_of).isoformat(), current_price, pnl_net, fee_cost, result, trade_id))
    conn.commit()
    record_equity(conn, balance, as_of=as_of)
    return {"ok": True, "symbol": symbol, "pnl": round(pnl_net, 4)}


# ==================== ثبت لحظه‌ای قیمت پوزیشن‌های باز ====================

def log_price_snapshot(conn, trade_id, price, as_of=None):
    conn.execute("INSERT INTO price_snapshots (trade_id, time, price) VALUES (?,?,?)",
                 (trade_id, _now(as_of).isoformat(), price))


def commit(conn):
    conn.commit()


def get_price_history(conn, trade_id):
    rows = conn.execute(
        "SELECT time, price FROM price_snapshots WHERE trade_id=? ORDER BY id", (trade_id,)
    ).fetchall()
    return [{"time": t, "price": p} for t, p in rows]


def get_open_trade_ids(conn, symbol=None):
    if symbol:
        rows = conn.execute("SELECT id FROM trades WHERE status='OPEN' AND symbol=?", (symbol,)).fetchall()
    else:
        rows = conn.execute("SELECT id FROM trades WHERE status='OPEN'").fetchall()
    return [r[0] for r in rows]


# ==================== لاگ سیگنال (چه اجرا شده چه رد شده) ====================

def log_signal(conn, symbol, side, strategy_name, entry, sl, tp, rr, htf_agree, opened, rejection_reason=None,
               as_of=None, score=None, reasons=None):
    conn.execute("""
        INSERT INTO signal_log (time, symbol, side, strategy_name, entry, sl, tp, rr, htf_agree,
                                 opened, rejection_reason, score, reasons)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (_now(as_of).isoformat(), symbol, side, strategy_name, entry, sl, tp, rr, htf_agree,
          1 if opened else 0, rejection_reason, score, reasons))
    conn.commit()


def get_signal_log(conn, limit=100, offset=0, only_rejected=False):
    cols = ["id", "time", "symbol", "side", "strategy_name", "entry", "sl", "tp", "rr",
            "htf_agree", "opened", "rejection_reason", "score", "reasons"]
    where = "WHERE opened=0" if only_rejected else ""
    total = conn.execute(f"SELECT COUNT(*) FROM signal_log {where}").fetchone()[0]
    rows = conn.execute(
        f"SELECT {','.join(cols)} FROM signal_log {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows], total


# ==================== آمار ====================

def get_stats(conn, initial_capital=None):
    """
    آمار: برد و باخت فقط از معاملاتی که به هدف R:R (برد) یا حد ضرر اولیه (باخت) رسیدن.
    خروج‌های تریلینگ (قبل از رسیدن به هدف) جدا: تعداد، سود/زیان دلاری و درصدی از سرمایه.
    (بر اساس نوع خروج حساب می‌شه، پس معاملات قدیمی‌تر هم درست دسته‌بندی می‌شن.)
    """
    closed = conn.execute("SELECT pnl, exit_type FROM trades WHERE status='CLOSED'").fetchall()
    total = len(closed)
    groups = {"WIN": [], "LOSS": [], "TRAIL": [], "OTHER": []}
    exit_type_counts = {}
    for pnl, et in closed:
        groups[result_of(et)].append(pnl or 0.0)
        et = et or "UNKNOWN"
        exit_type_counts[et] = exit_type_counts.get(et, 0) + 1
    wins, losses = len(groups["WIN"]), len(groups["LOSS"])
    rr_total = wins + losses
    trail_pnl = sum(groups["TRAIL"])
    cap = float(initial_capital) if initial_capital else None
    return {
        "total_trades": total, "rr_trades": rr_total, "wins": wins, "losses": losses,
        "win_rate": round(wins / rr_total * 100, 2) if rr_total else 0.0,
        "total_pnl": round(sum(p or 0 for p, _ in closed), 2),
        "rr_pnl": round(sum(groups["WIN"]) + sum(groups["LOSS"]), 2),
        "trail_trades": len(groups["TRAIL"]), "trail_pnl": round(trail_pnl, 2),
        "trail_pct": round(trail_pnl / cap * 100, 2) if cap else None,
        "trail_positive": sum(1 for p in groups["TRAIL"] if p > 0),
        "other_trades": len(groups["OTHER"]), "other_pnl": round(sum(groups["OTHER"]), 2),
        "exit_type_counts": exit_type_counts,
    }


def get_open_trades(conn):
    cols = ["id", "symbol", "side", "entry", "sl", "tp", "initial_sl", "peak_price", "size", "notional",
            "margin", "leverage", "liquidation_price", "trailing_enabled", "strategy_name", "open_time",
            "status", "timeframe", "expire_ts", "max_hold_min", "fill_ts"]
    rows = conn.execute(f"SELECT {','.join(cols)} FROM trades WHERE status IN ('OPEN','PENDING') ORDER BY id DESC").fetchall()
    return [dict(zip(cols, r)) for r in rows]


def get_closed_trades(conn, limit=50, offset=0):
    cols = ["id", "symbol", "side", "entry", "sl", "tp", "close_price", "pnl", "fee_cost",
            "margin", "leverage", "notional", "exit_type", "strategy_name", "result", "open_time", "close_time",
            "score"]
    total = conn.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED'").fetchone()[0]
    rows = conn.execute(
        f"SELECT {','.join(cols)} FROM trades WHERE status='CLOSED' ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows], total


def get_equity_curve(conn, limit=200):
    rows = conn.execute(
        "SELECT time, balance FROM equity ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    rows.reverse()
    return rows
