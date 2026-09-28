#!/usr/bin/env bash
# استفاده:
#   bash manage.sh start     -> روشن کردن ربات
#   bash manage.sh stop      -> متوقف کردن ربات (بدون حذف چیزی)
#   bash manage.sh restart   -> ری‌استارت
#   bash manage.sh status    -> وضعیت فعلی
#   bash manage.sh logs      -> دیدن لاگ زنده
#   bash manage.sh compare [روز] [تعداد نماد] [quick|full] [تایم‌فریم‌ها مثلاً 1m,5m یا 15m,1h,4h]
#                            -> مقایسه‌ی خودکار استراتژی‌ها از ترمینال (مثلاً: bash manage.sh compare 730 20 quick)
#   bash manage.sh test      -> تست دقت موتور بک‌تست (بدون اینترنت)
#   bash manage.sh fetch-data [روز] [تعداد نماد] [تایم‌فریم‌ها]  -> فقط دانلود/به‌روزرسانی دیتای تاریخی + بررسی اعتبار
#   bash manage.sh export-data         -> ساخت فایل بکاپ از کل دیتای تاریخی و گزارش‌ها (پوشه‌ی exports)
#   bash manage.sh import-data FILE    -> بازگردانی/ادغام دیتا از فایل بکاپ

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
    DAYS=${2:-730}; TOP=${3:-20}; GRID=${4:-quick}; TFS=${5:-15m,1h,4h}
    echo "🏆 مقایسه‌ی استراتژی‌ها: ${DAYS} روز، ${TOP} نماد، دامنه‌ی ${GRID}، تایم‌فریم‌ها ${TFS} (نتیجه توی پنل هم نمایش داده می‌شه)"
    nice -n 10 ./venv/bin/python3 compare.py --days "$DAYS" --top "$TOP" --grid "$GRID" --timeframes "$TFS"
    ;;
  test)
    cd "$(dirname "$0")"
    ./venv/bin/python3 tests/test_engine.py
    ;;
  fetch-data)
    cd "$(dirname "$0")"
    DAYS=${2:-730}; TOP=${3:-20}; TFS=${4:-15m,1h,4h}
    nice -n 10 ./venv/bin/python3 compare.py --data-only --days "$DAYS" --top "$TOP" --timeframes "$TFS"
    ;;
  export-data)
    cd "$(dirname "$0")"
    mkdir -p exports
    OUT="exports/tradingbot_data_$(date +%Y%m%d_%H%M).tar.gz"
    ./venv/bin/python3 -c "import config, market_data; c=market_data.MarketDataCache(config.DATA_CACHE_DIR, config.EXCHANGE_TRY_ORDER); c.export_archive('$OUT', config.REPORTS_DIR)"
    echo "✅ فایل دیتا ساخته شد: $(pwd)/$OUT ($(du -h "$OUT" | cut -f1))"
    ;;
  import-data)
    cd "$(dirname "$0")"
    if [ -z "$2" ] || [ ! -f "$2" ]; then echo "استفاده: bash manage.sh import-data مسیر_فایل"; exit 1; fi
    ./venv/bin/python3 -c "import sys, config, market_data; c=market_data.MarketDataCache(config.DATA_CACHE_DIR, config.EXCHANGE_TRY_ORDER); print('✅', c.import_archive(sys.argv[1], config.REPORTS_DIR))" "$2"
    ;;
  *)
    echo "استفاده: bash manage.sh {start|stop|restart|status|logs|compare|test|fetch-data|export-data|import-data}"
    exit 1
    ;;
esac
