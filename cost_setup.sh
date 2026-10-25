#!/bin/bash
# One-paste (Cloud Shell): show this project's real Google Cloud spend on the dashboard.
#   bash <(curl -fsSL https://raw.githubusercontent.com/<owner>/ai-trading-desk/main/cost_setup.sh)
# Creates, only inside project atd-443c547a: a monthly budget (alerts only, it never stops anything),
# a Pub/Sub topic and a push subscription to the desk. Budgets and this Pub/Sub volume are free.
set -euo pipefail
PROJECT=atd-443c547a
ZONE=asia-east1-b
VM=atd-worker
BOOT_URL="https://raw.githubusercontent.com/xchixmeng-svg/ai-trading-desk/main/boot.sh"
say() { printf '\n\033[1;33m%s\033[0m\n' "$*"; }

gcloud config set project $PROJECT >/dev/null
BA=$(gcloud billing projects describe $PROJECT --format='value(billingAccountName)' | sed 's#billingAccounts/##')
[ -n "$BA" ] || { echo "這個專案沒有連結帳單帳戶"; exit 1; }
IP=$(gcloud compute instances describe $VM --zone=$ZONE --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
DOMAIN="$(echo "$IP" | tr . -).sslip.io"

say "1/4 開啟需要的服務"
gcloud services enable billingbudgets.googleapis.com pubsub.googleapis.com --quiet

say "2/4 建立推送通道"
gcloud pubsub topics describe atd-billing >/dev/null 2>&1 || gcloud pubsub topics create atd-billing --quiet
gcloud pubsub topics add-iam-policy-binding atd-billing \
  --member=serviceAccount:billing-budget-alert@system.gserviceaccount.com --role=roles/pubsub.publisher --quiet >/dev/null
TOKEN=$(gcloud compute instances describe $VM --zone=$ZONE \
  --format='value(metadata.items.filter(key:atd-billing-token).extract(value).flatten())' 2>/dev/null || true)
[ -n "$TOKEN" ] || TOKEN=$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32)
gcloud compute instances add-metadata $VM --zone=$ZONE --metadata=atd-billing-token="$TOKEN" --quiet
ENDPOINT="https://$DOMAIN/api/billing-push?token=$TOKEN"
if gcloud pubsub subscriptions describe atd-billing-push >/dev/null 2>&1; then
  gcloud pubsub subscriptions modify-push-config atd-billing-push --push-endpoint="$ENDPOINT" --quiet
else
  gcloud pubsub subscriptions create atd-billing-push --topic=atd-billing --push-endpoint="$ENDPOINT" \
    --ack-deadline=30 --quiet
fi

say "3/4 建立每月預算（只通知，不會停機）"
if gcloud billing budgets list --billing-account="$BA" --format='value(displayName)' | grep -qx atd-monthly; then
  echo "預算已存在"
else
  gcloud billing budgets create --billing-account="$BA" --display-name=atd-monthly \
    --budget-amount=30USD --filter-projects="projects/$PROJECT" --credit-types-treatment=exclude-all-credits \
    --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0 \
    --notifications-rule-pubsub-topic="projects/$PROJECT/topics/atd-billing" --quiet \
  || gcloud billing budgets create --billing-account="$BA" --display-name=atd-monthly \
    --budget-amount=1000 --filter-projects="projects/$PROJECT" --credit-types-treatment=exclude-all-credits \
    --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0 \
    --notifications-rule-pubsub-topic="projects/$PROJECT/topics/atd-billing" --quiet
fi

say "4/4 更新主機設定"
curl -fsSL "$BOOT_URL?t=$(date +%s)" -o /tmp/atd_boot.sh
gcloud compute instances add-metadata $VM --zone=$ZONE --metadata-from-file=startup-script=/tmp/atd_boot.sh --quiet
gcloud compute ssh $VM --zone=$ZONE --quiet \
  --command="sudo google_metadata_script_runner startup >/dev/null 2>&1; sudo /usr/local/bin/atd-update --force"

say "完成。Google 每天會回報幾次花費，第一次可能要等幾個小時；之後網站「資料來源」會顯示本月花費。"
