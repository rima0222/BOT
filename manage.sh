#!/usr/bin/env bash
# استفاده:
#   bash manage.sh start     -> روشن کردن ربات
#   bash manage.sh stop      -> متوقف کردن ربات (بدون حذف چیزی)
#   bash manage.sh restart   -> ری‌استارت
#   bash manage.sh status    -> وضعیت فعلی
#   bash manage.sh logs      -> دیدن لاگ زنده
#   bash manage.sh compare [روز] [تعداد نماد] [quick|full]
#                            -> مقایسه‌ی خودکار استراتژی‌ها از ترمینال (مثلاً: bash manage.sh compare 730 20 quick)
#   bash manage.sh test      -> تست دقت موتور بک‌تست (بدون اینترنت)

case "$1" in
  start)
    sudo systemctl start tradingbot
    echo "✅ ربات استارت شد."
    ;;
  stop)
    sudo systemctl stop tradingbot
    echo "⏹️  ربات متوقف شد (فایل‌ها و دیتابیس دست‌نخورده باقی می‌مونن)."
    ;;
  restart)
    sudo systemctl restart tradingbot
    echo "🔄 ربات ری‌استارت شد."
    ;;
  status)
    sudo systemctl status tradingbot --no-pager
    ;;
  logs)
    sudo journalctl -u tradingbot -f
    ;;
  compare)
    cd "$(dirname "$0")"
    DAYS=${2:-730}; TOP=${3:-20}; GRID=${4:-quick}
    echo "🏆 مقایسه‌ی استراتژی‌ها: ${DAYS} روز، ${TOP} نماد، دامنه‌ی ${GRID} (نتیجه توی پنل هم نمایش داده می‌شه)"
    nice -n 10 ./venv/bin/python3 compare.py --days "$DAYS" --top "$TOP" --grid "$GRID"
    ;;
  test)
    cd "$(dirname "$0")"
    ./venv/bin/python3 tests/test_engine.py
    ;;
  *)
    echo "استفاده: bash manage.sh {start|stop|restart|status|logs|compare|test}"
    exit 1
    ;;
esac
