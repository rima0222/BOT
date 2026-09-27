# -*- coding: utf-8 -*-
"""
اندیکاتورهای مشترک بین ربات زنده (روی پنجره‌ی CANDLE_LIMIT کندلی) و موتور بک‌تست
(روی کل سری). همه‌شون طوری تعریف شدن که مقدارشون در کندل t فقط به همون پنجره
وابسته باشه، تا بک‌تست دقیقاً همون چیزی رو ببینه که ربات زنده می‌دید:
  - میانگین/انحراف معیار/RSI: پنجره‌ی ثابت (rolling) — مستقل از شروع پنجره.
  - EMA: بازگشتیه و به نقطه‌ی شروع وابسته‌ست. زنده روی پنجره حساب می‌کنه؛ موتور بک‌تست
    با فرمول دقیق ema_window_at همون عدد رو برای هر کندل بازسازی می‌کنه.
"""
import numpy as np
import pandas as pd


def sma(x, n):
    return pd.Series(x).rolling(n).mean().values


def rolling_std(x, n):
    return pd.Series(x).rolling(n).std(ddof=0).values


def rsi_sma(close, n):
    """RSI با میانگین ساده (نسخه‌ی Cutler) — به نقطه‌ی شروع دیتا وابسته نیست."""
    d = pd.Series(close).diff()
    gain = d.clip(lower=0).rolling(n).mean().values
    loss = (-d.clip(upper=0)).rolling(n).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - 100.0 / (1.0 + gain / loss)
    rsi = np.where(loss == 0, np.where(gain == 0, 50.0, 100.0), rsi)
    return np.where(np.isnan(gain) | np.isnan(loss), np.nan, rsi)


def ema(x, span):
    """EMA معمولی (adjust=False) از ابتدای آرایه‌ی داده‌شده."""
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


def ema_window_at(x, ema_full, span, u, s):
    """
    مقدار EMAای که اگه از کندل s شروع می‌شد، در کندل u داشت (u, s آرایه).
    y_u(از s) = EMA_full(u) - β^(u-s) · (EMA_full(s) - x(s))   که β = 1 - 2/(span+1)
    """
    beta = 1.0 - 2.0 / (span + 1.0)
    return ema_full[u] - np.power(beta, (u - s).astype(np.float64)) * (ema_full[s] - x[s])


def bollinger(close, n, k):
    mid = sma(close, n)
    sd = rolling_std(close, n)
    return mid, mid + k * sd, mid - k * sd


def prev_max(x, n):
    """بیشینه‌ی n کندل قبلی (بدون کندل فعلی)."""
    return pd.Series(x).rolling(n).max().shift(1).values


def prev_min(x, n):
    return pd.Series(x).rolling(n).min().shift(1).values


def prev_mean(x, n):
    return pd.Series(x).rolling(n).mean().shift(1).values


def last_min(x, n):
    """کمینه‌ی n کندل اخیر شامل کندل فعلی."""
    return pd.Series(x).rolling(n, min_periods=1).min().values


def last_max(x, n):
    return pd.Series(x).rolling(n, min_periods=1).max().values
