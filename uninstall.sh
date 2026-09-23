#!/usr/bin/env bash
# حذف کامل سرویس ربات از سرور.
# استفاده: bash uninstall.sh
set -e

echo "در حال متوقف و حذف سرویس..."
sudo systemctl stop tradingbot 2>/dev/null || true
sudo systemctl disable tradingbot 2>/dev/null || true
sudo rm -f /etc/systemd/system/tradingbot.service
sudo systemctl daemon-reload

echo "✅ سرویس ربات کاملاً از سیستم‌د حذف شد و دیگه اجرا نمی‌شه."
echo ""
echo "برای حذف کامل فایل‌های پروژه (کد + venv + دیتابیس)، دستور زیر رو بزن:"
echo ""
echo "    cd .. && rm -rf $(basename "$(pwd)")"
echo ""
echo "توجه: قبل از حذف کامل، اگه می‌خوای از تاریخچه معاملات و تنظیماتت بکاپ داشته باشی،"
echo "اول این رو بزن:   bash backup.sh"
