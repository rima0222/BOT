#!/usr/bin/env bash
# می‌گیره: دیتابیس معاملات (تاریخچه، آمار، موجودی مجازی) + فایل تنظیمات
# استفاده: bash backup.sh
set -e

mkdir -p backups
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILES_TO_BACKUP="config.py"

if [ -f tradingbot.db ]; then
  FILES_TO_BACKUP="$FILES_TO_BACKUP tradingbot.db"
else
  echo "⚠️  هنوز دیتابیسی ساخته نشده (ربات هنوز هیچ معامله‌ای ثبت نکرده)."
fi

tar -czf "backups/backup_${TIMESTAMP}.tar.gz" $FILES_TO_BACKUP
echo "✅ بکاپ ساخته شد: backups/backup_${TIMESTAMP}.tar.gz"
echo ""
echo "بکاپ‌های موجود:"
ls -la backups/

echo ""
echo "برای بازگردانی یک بکاپ:"
echo "  bash manage.sh stop"
echo "  tar -xzf backups/NAME.tar.gz"
echo "  bash manage.sh start"
