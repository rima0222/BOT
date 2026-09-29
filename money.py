# -*- coding: utf-8 -*-
"""
حساب‌وکتاب پول، مشترک بین ربات زنده و بک‌تست (همه‌ی توابع هم با عدد و هم با آرایه‌ی numpy
کار می‌کنن و ترتیب عملیاتشون یکیه، پس نتیجه‌ی هر دو دقیقاً برابره).

مدل کارمزد (همون paper_trader._fee_for_exit):
  ورود: لیمیت = میکر، بازار = تیکر (اسلیپیج ورود بازار از قبل توی قیمت ورود هست)
  خروج با حد سود: میکر
  خروج با حد ضرر/تریلینگ/حد زمانی: تیکر + اسلیپیج

  fi = کسر کارمزد ورود، fs = کسر کارمزد + اسلیپیج خروج با استاپ، fm = کسر کارمزد میکر

ضرر خالص هر واحد اگه حد ضرر بخوره:   (فاصله‌ی ورود تا SL) + ورود×fi + SL×fs
حد سود خالص: قیمتی که سود خالص (بعد از کارمزد ورود و خروج) = rr × ضرر خالص
سربه‌سر: قیمت استاپی که خروج باهاش دقیقاً صفر (بعد از کارمزد) می‌ده.
"""


def fee_fracs(maker_pct, taker_pct, slip_pct, entry_taker=False):
    fi = (taker_pct if entry_taker else maker_pct) / 100.0
    fs = taker_pct / 100.0 + slip_pct / 100.0
    fm = maker_pct / 100.0
    return fi, fs, fm


def cfg_fee_fracs(cfg):
    return fee_fracs(float(getattr(cfg, "MAKER_FEE_PCT", 0.0)), float(getattr(cfg, "TAKER_FEE_PCT", 0.0)),
                     float(getattr(cfg, "TAKER_SLIPPAGE_PCT", 0.0)), getattr(cfg, "ENTRY_MODE", "limit") == "market")


def loss_per_unit(is_long, entry, sl, fi, fs):
    risk = (entry - sl) if is_long else (sl - entry)
    return risk + entry * fi + sl * fs


def net_tp(is_long, entry, sl, rr, fi, fs, fm):
    loss = loss_per_unit(is_long, entry, sl, fi, fs)
    if is_long:
        return (entry * (1.0 + fi) + rr * loss) / (1.0 - fm)
    return (entry * (1.0 - fi) - rr * loss) / (1.0 + fm)


def breakeven_stop(is_long, entry, fi, fs):
    if is_long:
        return entry * (1.0 + fi) / (1.0 - fs)
    return entry * (1.0 - fi) / (1.0 + fs)
