# -*- coding: utf-8 -*-
"""
کش محلی دیتای تاریخی (OHLCV) برای بک‌تست و مقایسه‌ی استراتژی‌ها.

چرا: قبلاً هر بک‌تست کل ۱-۲ سال دیتا رو از اول از صرافی دانلود می‌کرد (برای هر نماد
ده‌ها درخواست، برای هر تایم‌فریم جدا) — همین بک‌تست رو کند و عملاً غیرقابل‌استفاده
می‌کرد. الان:
  - هر (صرافی، نماد، تایم‌فریم) فقط یک‌بار دانلود و روی دیسک ذخیره می‌شه.
  - دفعه‌های بعد فقط کندل‌های جدید (از آخرین کندل ذخیره‌شده تا الان) گرفته می‌شن.
  - اگه بازه‌ی قدیمی‌تری خواسته بشه، فقط همون تیکه‌ی قبلی اضافه دانلود می‌شه.
  - دانلود چند نماد موازی انجام می‌شه، با یک سقف سراسری درخواست در ثانیه تا
    محدودیت صرافی (و ربات زنده که هم‌زمان داره کار می‌کنه) اذیت نشه.
  - فقط کندل‌های «بسته‌شده» ذخیره می‌شن؛ کندل در حال تشکیل هیچ‌وقت وارد کش نمی‌شه.

فرمت ذخیره: فایل .npy با آرایه‌ی (n, 6) از float64:
  [زمان باز شدن کندل (میلی‌ثانیه), open, high, low, close, volume]
با mmap خونده می‌شه، پس حتی دیتای بزرگ هم رم زیادی اشغال نمی‌کنه.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000, "12h": 43_200_000,
    "1d": 86_400_000, "1w": 604_800_000,
}
MAX_LIMIT = {"binance": 1000, "kucoin": 1500, "okx": 300, "bybit": 1000}


def close_times_ms(open_ts_ms, timeframe):
    """زمان بسته‌شدن هر کندل (میلی‌ثانیه). برای ماهانه، شروع ماه بعد."""
    open_ts_ms = np.asarray(open_ts_ms, dtype=np.int64)
    if timeframe == "1M":
        months = open_ts_ms.astype("datetime64[ms]").astype("datetime64[M]")
        return (months + 1).astype("datetime64[ms]").astype(np.int64)
    return open_ts_ms + TF_MS[timeframe]


def approx_tf_ms(timeframe):
    return TF_MS.get(timeframe, 31 * 86_400_000)


def drop_unclosed(rows, timeframe, now_ms=None):
    """حذف کندل(های) هنوز بسته‌نشده از انتهای لیست."""
    if not rows:
        return rows
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    ts = np.array([r[0] for r in rows], dtype=np.int64)
    ct = close_times_ms(ts, timeframe)
    keep = ct <= now_ms
    return [r for r, k in zip(rows, keep) if k]


class _RateLimiter:
    """سقف سراسری درخواست در ثانیه، مشترک بین همه‌ی threadها."""

    def __init__(self, per_sec):
        self.min_interval = 1.0 / max(0.5, per_sec)
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_time:
                time.sleep(self.next_time - now)
                now = time.monotonic()
            self.next_time = now + self.min_interval


class MarketDataCache:
    def __init__(self, cache_dir, exchange_order, max_req_per_sec=6, threads=3):
        self.cache_dir = cache_dir
        self.exchange_order = list(exchange_order)
        self.limiter = _RateLimiter(max_req_per_sec)
        self.threads = max(1, int(threads))
        self._local = threading.local()
        self._index_lock = threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)
        self.index_path = os.path.join(cache_dir, "index.json")
        self.index = self._load_index()

    # ---------- index (کدوم صرافی برای کدوم نماد، از کِی دیتا داریم) ----------
    def _load_index(self):
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self):
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.index, f)
        os.replace(tmp, self.index_path)

    def _key(self, symbol, timeframe):
        return f"{symbol}|{timeframe}"

    def _path(self, exchange, symbol, timeframe):
        safe = symbol.replace("/", "_").replace(":", "_")
        d = os.path.join(self.cache_dir, exchange)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{safe}__{timeframe}.npy")

    # ---------- صرافی ----------
    def _exchange(self, name):
        cache = getattr(self._local, "ex", None)
        if cache is None:
            cache = {}
            self._local.ex = cache
        if name not in cache:
            import ccxt
            cache[name] = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 20000})
        return cache[name]

    def _fetch_range(self, exchange, symbol, timeframe, since_ms, until_ms):
        """دریافت کندل‌ها از since_ms تا until_ms (یا تا الان) با صفحه‌بندی."""
        ex = self._exchange(exchange)
        limit = MAX_LIMIT.get(exchange, 500)
        rows = []
        cursor = int(since_ms)
        guard = 0
        while cursor < until_ms and guard < 20000:
            guard += 1
            self.limiter.wait()
            batch = None
            for attempt in range(4):
                try:
                    batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(1.5 * (attempt + 1))
            if not batch:
                break
            rows.extend(batch)
            last_ts = batch[-1][0]
            if last_ts < cursor:
                break
            cursor = last_ts + 1
            if len(batch) < 2 and last_ts + approx_tf_ms(timeframe) >= until_ms:
                break
        return [r for r in rows if r[0] < until_ms]

    # ---------- API اصلی ----------
    def load(self, symbol, timeframe):
        """دیتای کش‌شده (بدون شبکه). None اگه هنوز کش نشده."""
        meta = self.index.get(self._key(symbol, timeframe))
        if not meta:
            return None
        path = self._path(meta["exchange"], symbol, timeframe)
        if not os.path.exists(path):
            return None
        return np.load(path, mmap_mode="r")

    def ensure(self, symbol, timeframe, since_ms, now_ms=None):
        """
        مطمئن می‌شه دیتای [since_ms, الان] برای این نماد/تایم‌فریم توی کش هست و
        آرایه‌ی کامل رو برمی‌گردونه. فقط تیکه‌های کم‌وکسر از شبکه گرفته می‌شن.
        """
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        key = self._key(symbol, timeframe)
        with self._index_lock:
            meta = dict(self.index.get(key) or {})

        exchanges = [meta["exchange"]] if meta.get("exchange") else self.exchange_order
        last_err = None
        for exchange in exchanges:
            try:
                path = self._path(exchange, symbol, timeframe)
                existing = None
                if meta.get("exchange") == exchange and os.path.exists(path):
                    existing = np.load(path)
                pieces = []
                earliest_req = meta.get("earliest_requested")

                if existing is None or len(existing) == 0:
                    rows = self._fetch_range(exchange, symbol, timeframe, since_ms, now_ms)
                    pieces.append(rows)
                    earliest_req = since_ms
                else:
                    first_ts, last_ts = int(existing[0, 0]), int(existing[-1, 0])
                    if since_ms < first_ts and (earliest_req is None or since_ms < earliest_req):
                        pieces.append(self._fetch_range(exchange, symbol, timeframe, since_ms, first_ts))
                        earliest_req = since_ms
                    pieces.append(self._fetch_range(exchange, symbol, timeframe, last_ts + 1, now_ms))

                new_rows = drop_unclosed([r for p in pieces for r in p], timeframe, now_ms)
                if new_rows:
                    new_arr = np.array([[float(x) if x is not None else np.nan for x in r[:6]] for r in new_rows],
                                       dtype=np.float64)
                    arr = new_arr if existing is None else np.vstack([existing, new_arr])
                    _, uniq_idx = np.unique(arr[:, 0], return_index=True)
                    arr = arr[np.sort(uniq_idx)]
                    arr = arr[np.argsort(arr[:, 0], kind="stable")]
                    # حجم ناموجود = صفر؛ کندلی که قیمت نداره حذف می‌شه
                    arr[:, 5] = np.nan_to_num(arr[:, 5], nan=0.0)
                    arr = arr[~np.isnan(arr[:, 1:5]).any(axis=1)]
                    tmp = path + ".tmp.npy"
                    np.save(tmp, arr)
                    os.replace(tmp, path)
                elif existing is None:
                    raise RuntimeError("صرافی هیچ کندلی برنگردوند")

                new_earliest = since_ms if earliest_req is None else min(int(earliest_req), int(since_ms))
                with self._index_lock:
                    self.index[key] = {"exchange": exchange, "earliest_requested": new_earliest,
                                       "updated": now_ms}
                    self._save_index()
                return np.load(path, mmap_mode="r")
            except Exception as e:
                last_err = e
                # اگه شبکه/صرافی موقتاً جواب نداد ولی دیتای قبلی توی کش هست، با همون ادامه بده
                path = self._path(exchange, symbol, timeframe)
                if meta.get("exchange") == exchange and os.path.exists(path):
                    return np.load(path, mmap_mode="r")
                continue
        raise RuntimeError(f"دریافت دیتای {symbol} ({timeframe}) ناموفق بود: {last_err}")

    def ensure_many(self, jobs, progress_cb=None):
        """
        jobs: لیست (symbol, timeframe, since_ms). دانلود موازی.
        خروجی: dict از (symbol, timeframe) -> آرایه یا Exception
        """
        results = {}
        total = len(jobs)
        done = [0]
        lock = threading.Lock()

        def run(job):
            sym, tf, since = job
            try:
                return job, self.ensure(sym, tf, since)
            except Exception as e:
                return job, e

        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            futures = [pool.submit(run, j) for j in jobs]
            for fut in as_completed(futures):
                job, res = fut.result()
                results[(job[0], job[1])] = res
                with lock:
                    done[0] += 1
                    if progress_cb:
                        progress_cb(done[0], total, job[0], job[1], res)
        return results

    def disk_usage_mb(self):
        total = 0
        for root, _, files in os.walk(self.cache_dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return round(total / 1024 / 1024, 1)
