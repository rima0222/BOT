#!/usr/bin/env bash
# اسکریپت نصب و راه‌اندازی خودکار روی سرور اوبونتو
# استفاده: bash setup.sh
set -e

echo "=== نصب پیش‌نیازها ==="
sudo apt update -y
sudo apt install -y python3-venv python3-pip

echo "=== ساخت محیط مجازی پایتون ==="
python3 -m venv venv
source venv/bin/activate

echo "=== نصب وابستگی‌ها ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== تنظیم سرویس systemd ==="
CURRENT_USER=$(whoami)
CURRENT_DIR=$(pwd)

sudo tee /etc/systemd/system/tradingbot.service > /dev/null <<EOF
[Unit]
Description=Virtual Trading Bot (Paper Trading)
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${CURRENT_DIR}
ExecStart=${CURRENT_DIR}/venv/bin/python3 ${CURRENT_DIR}/bot.py
Restart=always
RestartSec=10
MemoryMax=300M
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable tradingbot
sudo systemctl restart tradingbot

chmod +x manage.sh uninstall.sh backup.sh 2>/dev/null || true

PANEL_PORT=$(grep -oP 'PORT\s*=\s*\K[0-9]+' config.py | head -1)
PANEL_PORT=${PANEL_PORT:-8899}

echo ""
echo "=== باز کردن پورت پنل در فایروال (اگه فعال باشه) ==="
sudo ufw allow ${PANEL_PORT}/tcp 2>/dev/null || true

echo ""
echo "=== نصب کامل شد ✅ ==="
echo "وضعیت سرویس:"
sudo systemctl status tradingbot --no-pager
echo ""
echo "📊 داشبورد در دسترسه روی: http://$(curl -s ifconfig.me 2>/dev/null || echo YOUR_SERVER_IP):${PANEL_PORT}"
echo ""
echo "دستورهای مدیریتی:"
echo "  bash manage.sh status    -> وضعیت"
echo "  bash manage.sh logs      -> لاگ زنده"
echo "  bash manage.sh stop      -> متوقف کردن (بدون حذف چیزی)"
echo "  bash manage.sh start     -> روشن کردن دوباره"
echo "  bash backup.sh           -> گرفتن بکاپ از دیتابیس و تنظیمات"
echo "  bash uninstall.sh        -> حذف کامل سرویس از سرور"
