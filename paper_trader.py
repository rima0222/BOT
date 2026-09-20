# -*- coding: utf-8 -*-
"""
موتور معامله مجازی. هیچ سفارش واقعی به هیچ صرافی ارسال نمی‌شه؛
همه‌چیز شبیه‌سازی و در SQLite ثبت می‌شه (سبک، بدون نیاز به دیتابیس سنگین).

نکات مهم منطقی:
- سرمایه‌ی هر پوزیشن باز واقعاً از موجودی در دسترس "قفل" می‌شه (notional).
  پوزیشن جدید فقط وقتی باز می‌شه که سرمایه‌ی کافی واقعاً آزاد باشه؛ در غیر
  این صورت یا حجمش کوچیک‌تر می‌شه یا اصلاً باز نمی‌شه.
- کارمزد و اسلیپیج تخمینی از سود هر معامله کم می‌شه تا آمار واقعی‌تر باشه.
"""
import sqlite3
from datetime import datetime, timedelta


def get_conn(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, side TEXT,
            entry REAL, sl REAL, tp REAL, size REAL, notional REAL,
            status TEXT,
            open_time TEXT, close_time TEXT,
            close_price REAL, pnl REAL, fee_cost REAL, result TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT, balance REAL
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()

    # مهاجرت نرم برای دیتابیس‌های قدیمی‌تر که این ستون‌ها رو نداشتن
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
    for col, coltype in (("notional", "REAL"), ("fee_cost", "REAL")):
        if col not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {coltype}")
                conn.commit()
            except sqlite3.OperationalError:
                pass
    return conn


def get_balance(conn, start_balance):
    row = conn.execute("SELECT balance FROM equity ORDER BY id DESC LIMIT 1").fetchone()
    if row:
        return row[0]
    conn.execute("INSERT INTO equity (time, balance) VALUES (?, ?)",
                 (datetime.utcnow().isoformat(), start_balance))
    conn.commit()
    return start_balance


def record_equity(conn, balance):
    conn.execute("INSERT INTO equity (time, balance) VALUES (?, ?)",
                 (datetime.utcnow().isoformat(), balance))
    conn.commit()


def get_locked_capital(conn):
    """مجموع سرمایه‌ای که الان توی پوزیشن‌های باز قفل شده."""
    row = conn.execute("SELECT COALESCE(SUM(notional), 0) FROM trades WHERE status='OPEN'").fetchone()
    return row[0] or 0.0


def get_available_capital(conn, start_balance):
    """سرمایه‌ای که واقعاً برای پوزیشن جدید آزاده (کل موجودی منهای قفل‌شده‌ها)."""
    return get_balance(conn, start_balance) - get_locked_capital(conn)


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


def is_in_cooldown(conn, symbol, cooldown_hours):
    last_close = get_last_close_time(conn, symbol)
    if not last_close:
        return False
    return (datetime.utcnow() - last_close) < timedelta(hours=cooldown_hours)


def get_setting(conn, key, default=None):
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_setting(conn, key, value):
    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
    conn.commit()


def has_open_trade(conn, symbol):
    return conn.execute(
        "SELECT id FROM trades WHERE symbol=? AND status='OPEN'", (symbol,)
    ).fetchone() is not None


def open_trade(conn, symbol, side, entry, sl, tp, risk_pct, start_balance,
                min_notional=5.0, max_open_positions=None):
    """
    باز کردن پوزیشن مجازی با رعایت واقعی سرمایه:
    - ریسک بر اساس درصدی از کل موجودی محاسبه می‌شه (نه فقط سرمایه‌ی آزاد)
    - ولی حجم واقعی پوزیشن هیچ‌وقت از سرمایه‌ی *در دسترس* (آزاد) بیشتر نمی‌شه
    - اگه سرمایه‌ی کافی نباشه یا سقف تعداد پوزیشن پر باشه، پوزیشن باز نمی‌شه

    خروجی: dict شامل opened (True/False) و reason (در صورت رد شدن)
    """
    if max_open_positions is not None and get_open_position_count(conn) >= max_open_positions:
        return {"opened": False, "reason": "max_positions_reached"}

    balance = get_balance(conn, start_balance)
    available = balance - get_locked_capital(conn)

    if available < min_notional:
        return {"opened": False, "reason": "insufficient_capital"}

    per_unit_risk = abs(entry - sl)
    if per_unit_risk <= 0:
        return {"opened": False, "reason": "invalid_stop"}

    risk_amount = balance * (risk_pct / 100)
    size = risk_amount / per_unit_risk
    notional = entry * size

    # اگه حجم محاسبه‌شده از سرمایه‌ی آزاد بیشتر بود، محدودش کن به همون سرمایه‌ی آزاد
    # (چون بدون اهرم/لوریج، نمی‌شه بیشتر از پول موجود پوزیشن باز کرد)
    capped = False
    if notional > available:
        size = available / entry
        notional = available
        capped = True

    if notional < min_notional:
        return {"opened": False, "reason": "notional_too_small"}

    conn.execute("""
        INSERT INTO trades (symbol, side, entry, sl, tp, size, notional, status, open_time)
        VALUES (?,?,?,?,?,?,?, 'OPEN', ?)
    """, (symbol, side, entry, sl, tp, size, notional, datetime.utcnow().isoformat()))
    conn.commit()

    return {"opened": True, "size": size, "notional": notional, "capped": capped}


def check_and_close_trades(conn, symbol, current_price, start_balance, cost_pct_per_side=0.0):
    rows = conn.execute(
        "SELECT id, side, entry, sl, tp, size FROM trades WHERE symbol=? AND status='OPEN'",
        (symbol,)
    ).fetchall()
    if not rows:
        return
    balance = get_balance(conn, start_balance)
    for trade_id, side, entry, sl, tp, size in rows:
        closed, result, close_price = False, None, None
        if side == "LONG":
            if current_price <= sl:
                closed, result, close_price = True, "LOSS", sl
            elif current_price >= tp:
                closed, result, close_price = True, "WIN", tp
        else:  # SHORT
            if current_price >= sl:
                closed, result, close_price = True, "LOSS", sl
            elif current_price <= tp:
                closed, result, close_price = True, "WIN", tp

        if closed:
            pnl_gross = (close_price - entry) * size if side == "LONG" else (entry - close_price) * size
            # کارمزد + اسلیپیج تخمینی روی هر دو پای معامله (ورود و خروج)
            fee_cost = (entry * size + close_price * size) * (cost_pct_per_side / 100)
            pnl_net = pnl_gross - fee_cost
            balance += pnl_net
            conn.execute("""
                UPDATE trades SET status='CLOSED', close_time=?, close_price=?, pnl=?, fee_cost=?, result=?
                WHERE id=?
            """, (datetime.utcnow().isoformat(), close_price, pnl_net, fee_cost, result, trade_id))
            conn.commit()
            record_equity(conn, balance)


def close_trade_manually(conn, trade_id, current_price, start_balance, cost_pct_per_side=0.0):
    """بستن دستی یک پوزیشن از پنل (نه با برخورد به SL/TP، بلکه به قیمت لحظه‌ای فعلی)."""
    row = conn.execute(
        "SELECT symbol, side, entry, size FROM trades WHERE id=? AND status='OPEN'", (trade_id,)
    ).fetchone()
    if not row:
        return {"ok": False, "error": "پوزیشن باز با این شناسه پیدا نشد"}
    symbol, side, entry, size = row
    balance = get_balance(conn, start_balance)

    pnl_gross = (current_price - entry) * size if side == "LONG" else (entry - current_price) * size
    fee_cost = (entry * size + current_price * size) * (cost_pct_per_side / 100)
    pnl_net = pnl_gross - fee_cost
    balance += pnl_net
    result = "WIN" if pnl_net >= 0 else "LOSS"

    conn.execute("""
        UPDATE trades SET status='CLOSED', close_time=?, close_price=?, pnl=?, fee_cost=?, result=?
        WHERE id=?
    """, (datetime.utcnow().isoformat(), current_price, pnl_net, fee_cost, result, trade_id))
    conn.commit()
    record_equity(conn, balance)
    return {"ok": True, "symbol": symbol, "pnl": round(pnl_net, 4)}


def get_stats(conn):
    closed = conn.execute("SELECT result, pnl FROM trades WHERE status='CLOSED'").fetchall()
    total = len(closed)
    wins = sum(1 for r, _ in closed if r == "WIN")
    losses = total - wins
    win_rate = round(wins / total * 100, 2) if total else 0.0
    total_pnl = round(sum(p or 0 for _, p in closed), 2)
    return {"total_trades": total, "wins": wins, "losses": losses,
            "win_rate": win_rate, "total_pnl": total_pnl}


def get_open_trades(conn):
    cols = ["id", "symbol", "side", "entry", "sl", "tp", "size", "notional", "open_time"]
    rows = conn.execute(f"SELECT {','.join(cols)} FROM trades WHERE status='OPEN' ORDER BY id DESC").fetchall()
    return [dict(zip(cols, r)) for r in rows]


def get_closed_trades(conn, limit=50):
    cols = ["id", "symbol", "side", "entry", "sl", "tp", "close_price", "pnl", "fee_cost",
            "result", "open_time", "close_time"]
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


def reset_capital(conn, amount):
    """ست‌کردن دستی سرمایه (مثلاً وقتی کاربر مبلغ اولیه رو از پنل تغییر می‌ده)."""
    record_equity(conn, amount)
    set_setting(conn, "initial_capital", amount)
