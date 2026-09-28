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

    # ---------- خلاصه، خروجی گرفتن و بازگردانی ----------
    def summary(self):
        """خلاصه‌ی دیتای موجود: برای هر تایم‌فریم، تعداد نماد و بازه‌ی زمانی."""
        with self._index_lock:
            self.index = self._load_index()
        by_tf = {}
        for key, meta in list(self.index.items()):
            sym, tf = key.split("|", 1)
            path = self._path(meta["exchange"], sym, tf)
            if not os.path.exists(path):
                continue
            try:
                arr = np.load(path, mmap_mode="r")
                if len(arr) == 0:
                    continue
                first, last, n = int(arr[0, 0]), int(arr[-1, 0]), int(len(arr))
            except Exception:
                continue
            d = by_tf.setdefault(tf, {"symbols": 0, "candles": 0, "first": first, "last": last})
            d["symbols"] += 1
            d["candles"] += n
            d["first"] = min(d["first"], first)
            d["last"] = max(d["last"], last)
        return by_tf

    def export_archive(self, out_path, reports_dir=None):
        """کل دیتای تاریخی (و در صورت درخواست، گزارش‌ها) در یک فایل فشرده، مستقیم روی دیسک
        ساخته می‌شه (نه توی رم) تا روی سرور ۱ گیگی هم مشکلی پیش نیاد."""
        import tarfile
        tmp = out_path + ".tmp"
        with tarfile.open(tmp, "w:gz", compresslevel=3) as tar:
            tar.add(self.cache_dir, arcname="data_cache")
            if reports_dir and os.path.isdir(reports_dir):
                for f in os.listdir(reports_dir):
                    if f.startswith(("compare_", "results_", "summary_", "data_")) and \
                            f.endswith((".json", ".csv", ".txt")):
                        tar.add(os.path.join(reports_dir, f), arcname=f"reports/{f}")
        os.replace(tmp, out_path)
        return out_path

    def import_archive(self, archive_path, reports_dir=None):
        """
        بازگردانی از فایل خروجی. دیتای فایل با دیتای موجود «ادغام» می‌شه (نه جایگزین):
        کندل‌های تکراری حذف و کندل‌های جدیدتر/قدیمی‌تر هر دو نگه داشته می‌شن. فقط فایل‌های
        مجاز (npy / json / csv / txt) و فقط داخل پوشه‌های data_cache و reports قبول می‌شن.
        """
        import shutil
        import tarfile
        import tempfile
        allowed_ext = (".npy", ".json", ".csv", ".txt")
        stats = {"files": 0, "merged": 0, "new": 0, "reports": 0, "skipped": 0}
        with self._index_lock:
            self.index = self._load_index()
        tmpdir = tempfile.mkdtemp(prefix="tb_import_", dir=os.path.dirname(os.path.abspath(self.cache_dir)))
        try:
            with tarfile.open(archive_path, "r:*") as tar:
                for m in tar.getmembers():
                    name = m.name.replace("\\", "/").lstrip("./")
                    if m.isdir():
                        continue
                    if not m.isfile() or ".." in name.split("/") or name.startswith("/") \
                            or not name.endswith(allowed_ext) or not name.startswith(("data_cache/", "reports/")):
                        stats["skipped"] += 1
                        continue
                    dest = os.path.join(tmpdir, *name.split("/"))
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    src = tar.extractfile(m)
                    with open(dest, "wb") as f:
                        shutil.copyfileobj(src, f, 1024 * 1024)
                    stats["files"] += 1

            # گزارش‌ها
            rep_src = os.path.join(tmpdir, "reports")
            if reports_dir and os.path.isdir(rep_src):
                os.makedirs(reports_dir, exist_ok=True)
                for f in os.listdir(rep_src):
                    shutil.copy2(os.path.join(rep_src, f), os.path.join(reports_dir, f))
                    stats["reports"] += 1

            # دیتا: ادغام بر اساس index فایل واردشده
            src_root = os.path.join(tmpdir, "data_cache")
            try:
                with open(os.path.join(src_root, "index.json"), "r", encoding="utf-8") as f:
                    src_index = json.load(f)
            except Exception:
                src_index = {}
            for key, meta in src_index.items():
                sym, tf = key.split("|", 1)
                safe = sym.replace("/", "_").replace(":", "_")
                src_path = os.path.join(src_root, meta["exchange"], f"{safe}__{tf}.npy")
                if not os.path.exists(src_path):
                    continue
                try:
                    incoming = np.load(src_path)
                    if incoming.ndim != 2 or incoming.shape[1] != 6:
                        stats["skipped"] += 1
                        continue
                except Exception:
                    stats["skipped"] += 1
                    continue
                with self._index_lock:
                    local = self.index.get(key)
                if local and local.get("exchange") != meta["exchange"]:
                    stats["skipped"] += 1   # دیتای یک صرافی دیگه؛ قاطی نمی‌کنیم
                    continue
                dst = self._path(meta["exchange"], sym, tf)
                if local and os.path.exists(dst):
                    arr = np.vstack([np.load(dst), incoming])
                    _, uniq = np.unique(arr[:, 0], return_index=True)
                    arr = arr[np.sort(uniq)]
                    arr = arr[np.argsort(arr[:, 0], kind="stable")]
                    stats["merged"] += 1
                else:
                    arr = incoming
                    stats["new"] += 1
                tmp = dst + ".tmp.npy"
                np.save(tmp, arr)
                os.replace(tmp, dst)
                with self._index_lock:
                    prev = self.index.get(key) or {}
                    er = [x for x in (prev.get("earliest_requested"), meta.get("earliest_requested")) if x is not None]
                    self.index[key] = {"exchange": meta["exchange"], "earliest_requested": min(er) if er else None,
                                       "updated": max(prev.get("updated", 0) or 0, meta.get("updated", 0) or 0)}
            with self._index_lock:
                self._save_index()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return stats


def data_quality(arr, timeframe, start_ms, end_ms):
    """
    کیفیت دیتای یک نماد در بازه‌ی خواسته‌شده:
      coverage_pct: چند درصد کندل‌های مورد انتظار واقعاً هست
      listed_after_start: اولین کندل بعد از شروع بازه‌ست (ارز جوان‌تر از بازه‌ی تست)
      gaps / max_gap_hours: وقفه‌های دیتا (مثلاً قطعی صرافی)
      spikes: کندل‌هایی با پرش بیش از ۵۰٪ (احتمال خطای دیتا یا تغییر نماد)
    """
    out = {"coverage_pct": 0.0, "candles": 0, "expected": 0, "gaps": 0, "max_gap_hours": 0.0,
           "spikes": 0, "zero_volume_pct": 0.0, "first": None, "last": None, "listed_after_start": False}
    if arr is None or len(arr) == 0 or timeframe not in TF_MS:
        return out
    tf_ms = TF_MS[timeframe]
    a = np.asarray(arr)
    ts = a[:, 0].astype(np.int64)
    start_al = -(-int(start_ms) // tf_ms) * tf_ms
    end_al = (int(end_ms) // tf_ms) * tf_ms
    sel = (ts >= start_al) & (ts < end_al)
    t = ts[sel]
    expected = max(1, (end_al - start_al) // tf_ms)
    out.update({"candles": int(len(t)), "expected": int(expected),
                "coverage_pct": round(min(100.0, len(t) / expected * 100), 2),
                "first": int(ts[0]), "last": int(ts[-1]),
                "listed_after_start": bool(ts[0] > start_al + tf_ms)})
    if len(t) > 1:
        d = np.diff(t)
        g = d[d > tf_ms]
        out["gaps"] = int(len(g))
        out["max_gap_hours"] = round(float(g.max()) / 3_600_000, 2) if len(g) else 0.0
        c = a[sel, 4]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.abs(np.log(c[1:] / c[:-1]))
        out["spikes"] = int(np.nansum(r > np.log(1.5)))
        out["zero_volume_pct"] = round(float((a[sel, 5] <= 0).mean() * 100), 2)
    return out
