# -*- coding: utf-8 -*-
"""
دیتای تاریخی بازار فیوچرز که توی چارت نیست: نرخ فاندینگ و اوپن اینترست (OI).

  - فاندینگ: هر ۸ ساعت (یا کمتر) معامله‌گرهای طرف شلوغ‌تر به طرف دیگه پول می‌دن. فاندینگ خیلی مثبت یعنی
    خیلی‌ها با اهرم لانگ گرفتن (بازار شلوغ از خرید)، خیلی منفی یعنی شلوغ از فروش.
  - اوپن اینترست: حجم کل پوزیشن‌های باز. افزایش تند = پول اهرمی جدید وارد شده.

دانلود با ccxt از اولین صرافی‌ای که تاریخچه‌ی کافی بده (config.DERIV_EXCHANGES به ترتیب)، ذخیره روی دیسک
(data_cache/derivs) و فقط تکمیل کم‌وکسرها در دفعات بعد. هم‌ترازی با کندل‌ها بدون نگاه به آینده: در لحظه‌ی
بسته‌شدن هر کندل فقط آخرین فاندینگ تسویه‌شده / آخرین OI ثبت‌شده تا همون لحظه استفاده می‌شه.
"""
import json
import os
import time

import numpy as np

DAY_MS = 86_400_000


def perp_symbol(sym):
    base, quote = sym.split("/")[:2]
    quote = quote.split(":")[0]
    return f"{base}/{quote}:{quote}"


class DerivCache:
    def __init__(self, cache_dir, exchanges, log=None):
        self.dir = os.path.join(cache_dir, "derivs")
        os.makedirs(self.dir, exist_ok=True)
        self.exchanges = list(exchanges)
        self.log = log or (lambda m: None)
        self.index_path = os.path.join(self.dir, "index.json")
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.index = json.load(f)
        except Exception:
            self.index = {}
        self._ex = {}
        self._dead = set()     # صرافی‌هایی که اصلاً وصل نشدن (مثلاً مسدود برای این سرور) — دوباره امتحان نمی‌شن

    def _save_index(self):
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.index, f)
        os.replace(tmp, self.index_path)

    def _path(self, sym, kind):
        return os.path.join(self.dir, f"{sym.replace('/', '_').replace(':', '_')}__{kind}.npy")

    def load(self, sym, kind):
        meta = self.index.get(f"{sym}|{kind}")
        p = self._path(sym, kind)
        if not meta or not os.path.exists(p):
            return None, None
        return np.load(p), meta

    def save(self, sym, kind, arr, exchange):
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr):
            arr = arr[np.argsort(arr[:, 0], kind="stable")]
            keep = np.r_[True, np.diff(arr[:, 0]) > 0]
            arr = arr[keep]
        np.save(self._path(sym, kind), arr)
        self.index[f"{sym}|{kind}"] = {"exchange": exchange, "first": int(arr[0, 0]) if len(arr) else None,
                                       "last": int(arr[-1, 0]) if len(arr) else None, "n": int(len(arr)),
                                       "updated": int(time.time() * 1000)}
        self._save_index()

    def _exchange(self, name):
        if name in self._dead:
            raise RuntimeError("در دسترس نیست")
        if name not in self._ex:
            try:
                import ccxt
                ex = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 20000,
                                          "options": {"defaultType": "swap"}})
                ex.load_markets()
            except Exception:
                self._dead.add(name)
                raise
            self._ex[name] = ex
        return self._ex[name]

    # ---------- دانلود ----------
    def _fetch(self, ex_name, sym, kind, since, until):
        ex = self._exchange(ex_name)
        psym = perp_symbol(sym)
        if psym not in ex.markets:
            return []
        rows = []
        cursor = int(since)
        guard = 0
        while cursor < until and guard < 400:
            guard += 1
            if kind == "funding":
                batch = ex.fetch_funding_rate_history(psym, since=cursor, limit=200)
                got = [(int(r["timestamp"]), float(r["fundingRate"])) for r in batch
                       if r.get("timestamp") is not None and r.get("fundingRate") is not None]
            else:
                batch = ex.fetch_open_interest_history(psym, "4h", since=cursor, limit=200)
                got = []
                for r in batch:
                    v = r.get("openInterestValue")
                    if v is None:
                        v = r.get("openInterestAmount")
                    if r.get("timestamp") is not None and v is not None:
                        got.append((int(r["timestamp"]), float(v)))
            got = [g for g in got if g[0] >= cursor]
            if not got:
                break
            rows += got
            last = max(g[0] for g in got)
            if last <= cursor:
                break
            cursor = last + 1
        if not rows or min(r[0] for r in rows) > since + 7 * DAY_MS:
            # بعضی صرافی‌ها با since از جدیدترین برمی‌گردونن؛ صفحه‌بندی خود ccxt
            try:
                if kind == "funding":
                    batch = ex.fetch_funding_rate_history(psym, since=int(since), params={"paginate": True,
                                                                                         "paginationCalls": 80})
                    rows2 = [(int(r["timestamp"]), float(r["fundingRate"])) for r in batch
                             if r.get("timestamp") is not None and r.get("fundingRate") is not None]
                else:
                    batch = ex.fetch_open_interest_history(psym, "4h", since=int(since),
                                                           params={"paginate": True, "paginationCalls": 80})
                    rows2 = [(int(r["timestamp"]), float(r.get("openInterestValue") or r.get("openInterestAmount")))
                             for r in batch if r.get("timestamp") is not None and
                             (r.get("openInterestValue") or r.get("openInterestAmount")) is not None]
                if rows2 and (not rows or min(r[0] for r in rows2) < min(r[0] for r in rows)):
                    rows = rows2
            except Exception:
                pass
        return [r for r in rows if since <= r[0] < until]

    def ensure(self, sym, kind, since, until, offline=False):
        """دیتای [since, until) — از کش، و اگه لازم باشه دانلود (بهترین صرافی از نظر پوشش)."""
        arr, meta = self.load(sym, kind)
        if offline:
            return arr, meta
        need_old = arr is None or not len(arr) or arr[0, 0] > since + 7 * DAY_MS
        need_new = arr is None or not len(arr) or arr[-1, 0] < until - DAY_MS
        if not need_old and not need_new:
            return arr, meta
        order = self.exchanges
        if meta and meta.get("exchange") in order and not need_old:
            order = [meta["exchange"]]      # فقط تکمیل انتهای همون صرافی
        best, best_ex = arr, (meta or {}).get("exchange")
        for ex_name in order:
            try:
                start = since if need_old or best is None or not len(best) else int(best[-1, 0]) + 1
                rows = self._fetch(ex_name, sym, kind, start, until)
            except Exception as e:
                self.log(f"{sym} {kind} از {ex_name}: {str(e)[:120]}")
                continue
            if not rows:
                continue
            new = np.array(rows, dtype=np.float64)
            if best is not None and len(best) and best_ex == ex_name and not need_old:
                new = np.vstack([best, new])
            if best is None or not len(best) or new[0, 0] < best[0, 0] - DAY_MS or \
                    (best_ex == ex_name and len(new) >= len(best)):
                best, best_ex = new, ex_name
            if best[0, 0] <= since + 7 * DAY_MS:
                break
        if best is not None and len(best) and best_ex:
            self.save(sym, kind, best, best_ex)
            return self.load(sym, kind)
        return arr, meta


# ==================== هم‌ترازی با کندل‌ها (بدون نگاه به آینده) ====================

def funding_interval_ms(ts):
    if ts is None or len(ts) < 3:
        return 8 * 3_600_000
    return float(np.median(np.diff(ts)))


def align_last(close_ts, ts, vals):
    """برای هر کندل: آخرین مقدار با زمان ≤ بسته‌شدن کندل (NaN اگه نیست)."""
    out = np.full(len(close_ts), np.nan)
    if ts is None or not len(ts):
        return out
    pos = np.searchsorted(ts, close_ts, side="right") - 1
    ok = pos >= 0
    out[ok] = vals[pos[ok]]
    return out


def trailing_pct_rank(vals, ts, window_ms):
    """رتبه‌ی صدکی هر مقدار نسبت به مقادیر window_ms قبلش (شامل خودش) — فقط گذشته."""
    n = len(vals)
    out = np.full(n, np.nan)
    lo = np.searchsorted(ts, ts - window_ms, side="left")
    for i in range(n):
        w = vals[lo[i]:i + 1]
        if len(w) >= 20:
            out[i] = (w < vals[i]).mean() * 100.0 + (w == vals[i]).mean() * 50.0
    return out


def deriv_features(close_ts, deriv):
    """
    deriv: {"funding": arr[ts, rate] یا None, "oi": arr[ts, value] یا None}
    خروجی (هم‌طول کندل‌ها): fund (٪ در ۸ ساعت)، fund_pct (رتبه در ۹۰ روز)، oi_chg (٪ تغییر ۲۴ ساعت)
    """
    n = len(close_ts)
    F = {"fund": np.full(n, np.nan), "fund_pct": np.full(n, np.nan), "oi_chg": np.full(n, np.nan)}
    if not deriv:
        return F
    fa = deriv.get("funding")
    if fa is not None and len(fa) >= 30:
        ts, r = fa[:, 0].astype(np.int64), fa[:, 1]
        per8 = r * (8 * 3_600_000 / funding_interval_ms(ts)) * 100.0
        F["fund"] = align_last(close_ts, ts, per8)
        F["fund_pct"] = align_last(close_ts, ts, trailing_pct_rank(per8, ts, 90 * DAY_MS))
    oa = deriv.get("oi")
    if oa is not None and len(oa) >= 30:
        ts, v = oa[:, 0].astype(np.int64), oa[:, 1]
        now = align_last(close_ts, ts, v)
        prev = align_last(close_ts - DAY_MS, ts, v)
        with np.errstate(invalid="ignore", divide="ignore"):
            F["oi_chg"] = np.where(prev > 0, (now / prev - 1.0) * 100.0, np.nan)
    return F


def coverage(arr, start_ms, end_ms):
    if arr is None or not len(arr):
        return 0.0
    a = max(float(arr[0, 0]), start_ms)
    return round(max(0.0, min(end_ms, float(arr[-1, 0])) - a) / max(1.0, end_ms - start_ms) * 100.0, 1)
