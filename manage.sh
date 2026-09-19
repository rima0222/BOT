#!/usr/bin/env bash
# استفاده:
#   bash manage.sh start     -> روشن کردن ربات
#   bash manage.sh stop      -> متوقف کردن ربات (بدون حذف چیزی)
#   bash manage.sh restart   -> ری‌استارت
#   bash manage.sh status    -> وضعیت فعلی
#   bash manage.sh logs      -> دیدن لاگ زنده

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
  *)
    echo "استفاده: bash manage.sh {start|stop|restart|status|logs}"
    exit 1
    ;;
esac
