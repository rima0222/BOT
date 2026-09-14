# -*- coding: utf-8 -*-
"""
موتور تحلیل:
  1) تشخیص روند بر اساس تئوری داو (سقف/کف‌های بالاتر یا پایین‌تر) + تایید حجم
  2) شناسایی سطوح حمایت/مقاومت از نقاط سوینگ
  3) تولید سیگنال ورود فقط وقتی روند + نزدیکی به سطح + حداقل ریسک‌به‌ریوارد هم‌زمان جور باشن
"""
import pandas as pd


def find_swings(df, order=3):
    """نقاط سوینگ های/لو (فراکتال ساده)."""
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swing_highs, swing_lows = [], []
    for i in range(order, n - order):
        window_h = highs[i - order: i + order + 1]
        if highs[i] == window_h.max():
            swing_highs.append((i, highs[i]))
        window_l = lows[i - order: i + order + 1]
        if lows[i] == window_l.min():
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def determine_trend(swing_highs, swing_lows):
    """تئوری داو: روند صعودی = سقف و کف بالاتر؛ نزولی = سقف و کف پایین‌تر."""
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "sideways"
    h1, h2 = swing_highs[-2][1], swing_highs[-1][1]
    l1, l2 = swing_lows[-2][1], swing_lows[-1][1]
    if h2 > h1 and l2 > l1:
        return "uptrend"
    if h2 < h1 and l2 < l1:
        return "downtrend"
    return "sideways"


def volume_confirms(df, trend, lookback=20):
    """تایید حجمی طبق تئوری داو: حجم باید هم‌جهت با روند غالب باشه."""
    recent = df.tail(lookback).copy()
    recent["change"] = recent["close"] - recent["open"]
    up_vol = recent.loc[recent["change"] > 0, "volume"].mean()
    down_vol = recent.loc[recent["change"] < 0, "volume"].mean()
    if pd.isna(up_vol) or pd.isna(down_vol):
        return False
    if trend == "uptrend":
        return up_vol > down_vol
    if trend == "downtrend":
        return down_vol > up_vol
    return False


def compute_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _cluster_levels(levels, cluster_pct):
    if not levels:
        return []
    levels = sorted(levels)
    clusters, current = [], [levels[0]]
    for lv in levels[1:]:
        if abs(lv - current[-1]) / current[-1] * 100 <= cluster_pct:
            current.append(lv)
        else:
            clusters.append(sum(current) / len(current))
            current = [lv]
    clusters.append(sum(current) / len(current))
    return clusters


def get_support_resistance(swing_highs, swing_lows, current_price, cluster_pct):
    resistance_levels = _cluster_levels([p for _, p in swing_highs], cluster_pct)
    support_levels = _cluster_levels([p for _, p in swing_lows], cluster_pct)
    supports_below = sorted([s for s in support_levels if s < current_price], reverse=True)
    resistances_above = sorted([r for r in resistance_levels if r > current_price])
    nearest_support = supports_below[0] if supports_below else None
    nearest_resistance = resistances_above[0] if resistances_above else None
    return nearest_support, nearest_resistance


def generate_signal(df, cfg):
    """
    خروجی: دیکشنری شامل روند فعلی، قیمت، نزدیک‌ترین حمایت/مقاومت،
    و در صورت وجود شرایط کامل، یک سیگنال ورود با SL/TP و نسبت R:R.
    """
    swing_highs, swing_lows = find_swings(df, order=cfg.SWING_ORDER)
    trend = determine_trend(swing_highs, swing_lows)
    current_price = float(df["close"].iloc[-1])
    atr_series = compute_atr(df, cfg.ATR_PERIOD)
    atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else None

    nearest_support, nearest_resistance = get_support_resistance(
        swing_highs, swing_lows, current_price, cfg.SR_CLUSTER_PCT
    )

    result = {
        "trend": trend,
        "price": current_price,
        "support": nearest_support,
        "resistance": nearest_resistance,
        "atr": atr,
        "signal": None,
    }

    if not atr or atr <= 0:
        return result

    vol_ok = volume_confirms(df, trend)

    # --- سناریوی خرید: روند صعودی + قیمت نزدیک حمایت ---
    if trend == "uptrend" and nearest_support and vol_ok:
        dist_pct = (current_price - nearest_support) / nearest_support * 100
        if 0 <= dist_pct <= cfg.PROXIMITY_PCT:
            entry = current_price
            sl = nearest_support - atr * cfg.ATR_SL_BUFFER
            risk = entry - sl
            if risk > 0:
                tp_min = entry + risk * cfg.MIN_RISK_REWARD
                tp = max(tp_min, nearest_resistance) if nearest_resistance else tp_min
                rr = (tp - entry) / risk
                if rr >= cfg.MIN_RISK_REWARD:
                    result["signal"] = {
                        "side": "LONG", "entry": entry, "sl": sl, "tp": tp, "rr": rr,
                    }

    # --- سناریوی فروش: روند نزولی + قیمت نزدیک مقاومت ---
    if trend == "downtrend" and nearest_resistance and vol_ok and result["signal"] is None:
        dist_pct = (nearest_resistance - current_price) / nearest_resistance * 100
        if 0 <= dist_pct <= cfg.PROXIMITY_PCT:
            entry = current_price
            sl = nearest_resistance + atr * cfg.ATR_SL_BUFFER
            risk = sl - entry
            if risk > 0:
                tp_min = entry - risk * cfg.MIN_RISK_REWARD
                tp = min(tp_min, nearest_support) if nearest_support else tp_min
                rr = (entry - tp) / risk
                if rr >= cfg.MIN_RISK_REWARD:
                    result["signal"] = {
                        "side": "SHORT", "entry": entry, "sl": sl, "tp": tp, "rr": rr,
                    }

    return result
