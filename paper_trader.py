# -*- coding: utf-8 -*-
"""
موتور معامله مجازی. هیچ سفارش واقعی به هیچ صرافی ارسال نمی‌شه؛
همه‌چیز شبیه‌سازی و در SQLite ثبت می‌شه (سبک، بدون نیاز به دیتابیس سنگین).
"""
import sqlite3
from datetime import datetime


def get_conn(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, side TEXT,
            entry REAL, sl REAL, tp REAL, size REAL,
            status TEXT,
            open_time TEXT, close_time TEXT,
            close_price REAL, pnl REAL, result TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT, balance REAL
        )
    """)
    conn.commit()
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


def get_open_symbols(conn):
    rows = conn.execute("SELECT DISTINCT symbol FROM trades WHERE status='OPEN'").fetchall()
    return [r[0] for r in rows]


def has_open_trade(conn, symbol):
    return conn.execute(
        "SELECT id FROM trades WHERE symbol=? AND status='OPEN'", (symbol,)
    ).fetchone() is not None


def open_trade(conn, symbol, side, entry, sl, tp, risk_pct, start_balance):
    balance = get_balance(conn, start_balance)
    risk_amount = balance * (risk_pct / 100)
    per_unit_risk = abs(entry - sl)
    size = risk_amount / per_unit_risk if per_unit_risk > 0 else 0
    conn.execute("""
        INSERT INTO trades (symbol, side, entry, sl, tp, size, status, open_time)
        VALUES (?,?,?,?,?,?, 'OPEN', ?)
    """, (symbol, side, entry, sl, tp, size, datetime.utcnow().isoformat()))
    conn.commit()


def check_and_close_trades(conn, symbol, current_price, start_balance):
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
            pnl = (close_price - entry) * size if side == "LONG" else (entry - close_price) * size
            balance += pnl
            conn.execute("""
                UPDATE trades SET status='CLOSED', close_time=?, close_price=?, pnl=?, result=?
                WHERE id=?
            """, (datetime.utcnow().isoformat(), close_price, pnl, result, trade_id))
            conn.commit()
            record_equity(conn, balance)


def get_stats(conn):
    closed = conn.execute("SELECT result, pnl FROM trades WHERE status='CLOSED'").fetchall()
    total = len(closed)
    wins = sum(1 for r, _ in closed if r == "WIN")
    losses = total - wins
    win_rate = round(wins / total * 100, 2) if total else 0.0
    total_pnl = round(sum(p for _, p in closed), 2)
    return {"total_trades": total, "wins": wins, "losses": losses,
            "win_rate": win_rate, "total_pnl": total_pnl}


def get_open_trades(conn):
    cols = ["id", "symbol", "side", "entry", "sl", "tp", "size", "open_time"]
    rows = conn.execute(f"SELECT {','.join(cols)} FROM trades WHERE status='OPEN'").fetchall()
    return [dict(zip(cols, r)) for r in rows]


def get_closed_trades(conn, limit=50):
    cols = ["id", "symbol", "side", "entry", "sl", "tp", "close_price", "pnl", "result", "open_time", "close_time"]
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
