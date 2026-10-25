#!/bin/bash
# One-paste installer for Google Cloud Shell:
#   bash <(curl -fsSL https://raw.githubusercontent.com/<owner>/ai-trading-desk/main/install.sh) <KEY>
# Creates a NEW, separate project "ai-trading-desk", a small VM in Taiwan (asia-east1),
# and starts the shadow desk. Touches no existing project.
set -euo pipefail
BUNDLE_URL="https://raw.githubusercontent.com/xchixmeng-svg/ai-trading-desk/main/bundle.enc"
BOOT_URL="https://raw.githubusercontent.com/xchixmeng-svg/ai-trading-desk/main/boot.sh"
KEY="${1:-}"
if [ -z "$KEY" ]; then echo "缺少金鑰：請完整貼上 Claude 給你的那一行指令（最後面有一串英數字）"; exit 1; fi
REGION=asia-east1
ZONE=asia-east1-b
VM=atd-worker

say() { printf '\n\033[1;33m%s\033[0m\n' "$*"; }

say "1/6 檢查帳單帳戶"
BILLING=$(gcloud billing accounts list --filter=open=true --format='value(name)' 2>/dev/null | head -1 || true)
if [ -z "$BILLING" ]; then
  say "找不到可用的帳單帳戶。請先到 console.cloud.google.com/billing 開啟免費試用或綁卡，再重新貼一次指令。"
  exit 1
fi

say "2/6 建立獨立專案（不會動到你原本的專案）"
PROJECT=$(gcloud projects list --filter='name=ai-trading-desk' --format='value(projectId)' | head -1 || true)
if [ -z "$PROJECT" ]; then
  PROJECT="atd-$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  gcloud projects create "$PROJECT" --name="ai-trading-desk" --quiet
fi
gcloud config set project "$PROJECT" --quiet >/dev/null
gcloud billing projects link "$PROJECT" --billing-account="$BILLING" --quiet >/dev/null

say "3/6 啟用 Compute Engine（約 1 分鐘）"
gcloud services enable compute.googleapis.com --quiet

say "5/6 建立主機（台灣 asia-east1，e2-small）"
gcloud compute addresses describe "$VM-ip" --region=$REGION >/dev/null 2>&1 || \
  gcloud compute addresses create "$VM-ip" --region=$REGION --quiet
IP=$(gcloud compute addresses describe "$VM-ip" --region=$REGION --format='value(address)')
gcloud compute firewall-rules describe atd-allow-web >/dev/null 2>&1 || \
  gcloud compute firewall-rules create atd-allow-web --allow=tcp:80,tcp:443 --target-tags=atd-web --quiet
curl -fsSL "$BOOT_URL?t=$(date +%s)" -o /tmp/atd_boot.sh
META="atd-bundle-url=$BUNDLE_URL,atd-bundle-key=$KEY"
if gcloud compute instances describe $VM --zone=$ZONE >/dev/null 2>&1; then
  gcloud compute instances add-metadata $VM --zone=$ZONE --metadata="$META" \
    --metadata-from-file=startup-script=/tmp/atd_boot.sh --quiet
  gcloud compute instances reset $VM --zone=$ZONE --quiet
else
  gcloud compute instances create $VM --zone=$ZONE --machine-type=e2-small \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=20GB --boot-disk-type=pd-balanced \
    --address="$IP" --tags=atd-web \
    --metadata="$META" --metadata-from-file=startup-script=/tmp/atd_boot.sh --quiet
fi

URL="https://$(echo "$IP" | tr . -).sslip.io"
say "6/6 完成。主機正在安裝，約 5～10 分鐘後打開："
echo "$URL"
echo
echo "（專案 ID：$PROJECT，主機：$VM，IP：$IP）"
