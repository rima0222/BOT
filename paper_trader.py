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
from datetime import datetime, timedelta


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
    conn.commit()

    # مهاجرت نرم برای دیتابیس‌های قدیمی‌تر که این ستون‌ها رو نداشتن
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
    new_cols = (
        ("notional", "REAL"), ("fee_cost", "REAL"), ("margin", "REAL"),
        ("leverage", "REAL"), ("liquidation_price", "REAL"),
        ("initial_sl", "REAL"), ("peak_price", "REAL"), ("trailing_enabled", "INTEGER DEFAULT 0"),
        ("strategy_name", "TEXT"), ("exit_type", "TEXT"),
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
    row = conn.execute("SELECT COALESCE(SUM(margin), 0) FROM trades WHERE status='OPEN'").fetchone()
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
    rows = conn.execute("SELECT DISTINCT symbol FROM trades WHERE status='OPEN'").fetchall()
    return [r[0] for r in rows]


def get_open_position_count(conn):
    row = conn.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN'").fetchone()
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
        "SELECT id FROM trades WHERE symbol=? AND status='OPEN'", (symbol,)
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

def open_trade(conn, symbol, side, entry, sl, tp, risk_pct, start_balance,
                min_notional=5.0, max_open_positions=None,
                max_leverage=5, leverage_safety_mult=1.6, position_pct_cap=20.0,
                trailing_enabled=False, strategy_name=None, as_of=None):
    """
    باز کردن پوزیشن مجازی با لوریج و مدیریت سرمایه‌ی واقعی.
    خروجی: dict شامل opened (True/False) و در صورت باز شدن، جزئیات کامل پوزیشن.
    """
    if max_open_positions is not None and get_open_position_count(conn) >= max_open_positions:
        return {"opened": False, "reason": "max_positions_reached"}

    if entry <= 0:
        return {"opened": False, "reason": "invalid_entry"}

    per_unit_risk = abs(entry - sl)
    if per_unit_risk <= 0:
        return {"opened": False, "reason": "invalid_stop"}

    balance = get_balance(conn, start_balance, as_of=as_of)
    available_margin = balance - get_locked_capital(conn)
    if available_margin < min_notional:
        return {"opened": False, "reason": "insufficient_capital"}

    sl_distance_pct = per_unit_risk / entry * 100
    safe_leverage = 100 / (sl_distance_pct * leverage_safety_mult) if sl_distance_pct > 0 else max_leverage
    leverage = max(1.0, min(max_leverage, safe_leverage))
    leverage = round(leverage, 2)

    risk_amount = balance * (risk_pct / 100)
    raw_size = risk_amount / per_unit_risk
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
        return {"opened": False, "reason": "notional_too_small"}

    liquidation_price = entry * (1 - 1 / leverage) if side == "LONG" else entry * (1 + 1 / leverage)

    conn.execute("""
        INSERT INTO trades (symbol, side, entry, sl, tp, initial_sl, peak_price,
                             size, notional, margin, leverage, liquidation_price,
                             trailing_enabled, strategy_name, status, open_time)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'OPEN', ?)
    """, (symbol, side, entry, sl, tp, sl, entry,
          size, notional, margin, leverage, liquidation_price,
          1 if trailing_enabled else 0, strategy_name, _now(as_of).isoformat()))
    conn.commit()

    return {
        "opened": True, "size": size, "notional": notional, "margin": margin,
        "leverage": leverage, "liquidation_price": liquidation_price, "capped": capped,
    }


# ==================== تریلینگ استاپ ====================

def compute_trailing_sl(entry, initial_sl, side, peak_price, ladder, beyond_distance_r):
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

    locked_r = None
    for trigger_r, lock_r in ladder:
        if peak_r >= trigger_r:
            locked_r = lock_r

    if locked_r is None:
        return initial_sl

    if ladder:
        last_trigger_r = ladder[-1][0]
        if peak_r > last_trigger_r:
            locked_r = max(locked_r, peak_r - beyond_distance_r)

    new_sl = entry + locked_r * risk_per_unit if side == "LONG" else entry - locked_r * risk_per_unit
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


# ==================== بستن پوزیشن (خودکار یا دستی) ====================

def _fee_for_exit(entry, close_price, size, exit_type, maker_fee_pct, taker_fee_pct, taker_slippage_pct):
    """
    کارمزد واقعی: ورود همیشه میکر فرض می‌شه. خروج با TP هم میکر (سفارش لیمیتی
    منتظرمونده)؛ خروج با SL/تریلینگ همیشه تیکر (سفارش استاپ فوری) + کمی اسلیپیج.
    """
    entry_fee = entry * size * (maker_fee_pct / 100)
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
            result = "WIN" if pnl_net >= 0 else "LOSS"
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
    row = conn.execute(
        "SELECT symbol, side, entry, size, initial_sl, tp FROM trades WHERE id=? AND status='OPEN'", (trade_id,)
    ).fetchone()
    if not row:
        return {"ok": False, "error": "پوزیشن باز با این شناسه پیدا نشد"}
    symbol, side, entry, size, initial_sl, tp = row
    balance = get_balance(conn, start_balance, as_of=as_of)

    pnl_gross = (current_price - entry) * size if side == "LONG" else (entry - current_price) * size
    fee_cost, _, _, _ = _fee_for_exit(entry, current_price, size, "MANUAL",
                                       maker_fee_pct, taker_fee_pct, taker_slippage_pct)
    pnl_net = pnl_gross - fee_cost
    balance += pnl_net
    result = "WIN" if pnl_net >= 0 else "LOSS"

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

def log_signal(conn, symbol, side, strategy_name, entry, sl, tp, rr, htf_agree, opened, rejection_reason=None, as_of=None):
    conn.execute("""
        INSERT INTO signal_log (time, symbol, side, strategy_name, entry, sl, tp, rr, htf_agree,
                                 opened, rejection_reason)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (_now(as_of).isoformat(), symbol, side, strategy_name, entry, sl, tp, rr, htf_agree,
          1 if opened else 0, rejection_reason))
    conn.commit()


def get_signal_log(conn, limit=100):
    cols = ["id", "time", "symbol", "side", "strategy_name", "entry", "sl", "tp", "rr",
            "htf_agree", "opened", "rejection_reason"]
    rows = conn.execute(
        f"SELECT {','.join(cols)} FROM signal_log ORDER BY id DESC LIMIT {limit}"
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


# ==================== آمار ====================

def get_stats(conn):
    closed = conn.execute("SELECT result, pnl, exit_type FROM trades WHERE status='CLOSED'").fetchall()
    total = len(closed)
    wins = sum(1 for r, _, _ in closed if r == "WIN")
    losses = total - wins
    win_rate = round(wins / total * 100, 2) if total else 0.0
    total_pnl = round(sum(p or 0 for _, p, _ in closed), 2)

    exit_type_counts = {}
    for _, _, et in closed:
        et = et or "UNKNOWN"
        exit_type_counts[et] = exit_type_counts.get(et, 0) + 1

    return {"total_trades": total, "wins": wins, "losses": losses,
            "win_rate": win_rate, "total_pnl": total_pnl, "exit_type_counts": exit_type_counts}


def get_open_trades(conn):
    cols = ["id", "symbol", "side", "entry", "sl", "tp", "initial_sl", "peak_price", "size", "notional",
            "margin", "leverage", "liquidation_price", "trailing_enabled", "strategy_name", "open_time"]
    rows = conn.execute(f"SELECT {','.join(cols)} FROM trades WHERE status='OPEN' ORDER BY id DESC").fetchall()
    return [dict(zip(cols, r)) for r in rows]


def get_closed_trades(conn, limit=50):
    cols = ["id", "symbol", "side", "entry", "sl", "tp", "close_price", "pnl", "fee_cost",
            "margin", "leverage", "notional", "exit_type", "strategy_name", "result", "open_time", "close_time"]
    rows = conn.execute(
        f"SELECT {','.join(cols)} FROM trades WHERE status='CLOSED' ORDER BY id DESC LIMIT {limit}"
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def get_equity_curve(conn, limit=200):
    rows = conn.execute(
        "SELECT time, balance FROM equity ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    rows.reverse()
    return rows
