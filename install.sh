#!/bin/bash
# One paste in Google Cloud Shell (Chi, once):
#   bash <(curl -fsSL https://raw.githubusercontent.com/xchixmeng-svg/ai-trading-desk/main/setup.sh)
# Lets GitHub Actions of the private source repo (and only that repo) deploy to the VM, keyless
# (Workload Identity Federation - no key file exists anywhere), then starts the VM.
# Free: service accounts, identity pools and firewall rules cost nothing. Touches only project atd-443c547a.
set -euo pipefail
PROJECT=atd-443c547a
REGION=asia-east1
ZONE=asia-east1-b
VM=atd-worker
REPO=xchixmeng-svg/-ai-trading-desk-src
SA=atd-deployer
SA_EMAIL=$SA@$PROJECT.iam.gserviceaccount.com
BOOT_URL="https://raw.githubusercontent.com/xchixmeng-svg/ai-trading-desk/main/boot.sh"
say() { printf '\n\033[1;33m%s\033[0m\n' "$*"; }

gcloud config set project $PROJECT --quiet >/dev/null
NUM=$(gcloud projects describe $PROJECT --format='value(projectNumber)')

say "1/5 開啟需要的服務（約 1 分鐘）"
gcloud services enable compute.googleapis.com iam.googleapis.com iamcredentials.googleapis.com \
  sts.googleapis.com oslogin.googleapis.com --quiet

say "2/5 建立部署帳號（只有 GitHub 上這個 repo 能用，沒有金鑰檔）"
gcloud iam service-accounts describe $SA_EMAIL >/dev/null 2>&1 || \
  gcloud iam service-accounts create $SA --display-name="GitHub deploy" --quiet
for ROLE in roles/compute.instanceAdmin.v1 roles/compute.osAdminLogin roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:$SA_EMAIL --role=$ROLE \
    --condition=None --quiet >/dev/null
done
gcloud iam workload-identity-pools describe github --location=global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools create github --location=global --display-name=GitHub --quiet
gcloud iam workload-identity-pools providers describe github --workload-identity-pool=github --location=global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools providers create-oidc github --workload-identity-pool=github --location=global \
    --issuer-uri=https://token.actions.githubusercontent.com \
    --attribute-mapping=google.subject=assertion.sub,attribute.repository=assertion.repository \
    --attribute-condition="assertion.repository=='$REPO'" --quiet
gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$NUM/locations/global/workloadIdentityPools/github/attribute.repository/$REPO" \
  --quiet >/dev/null

say "3/5 開放 SSH（只接受每次部署臨時產生的金鑰）"
gcloud compute firewall-rules describe atd-allow-ssh >/dev/null 2>&1 || \
  gcloud compute firewall-rules create atd-allow-ssh --allow=tcp:22 --target-tags=atd-web --quiet
gcloud compute firewall-rules describe atd-allow-web >/dev/null 2>&1 || \
  gcloud compute firewall-rules create atd-allow-web --allow=tcp:80,tcp:443 --target-tags=atd-web --quiet

say "4/5 更新主機開機程式並啟動"
curl -fsSL "$BOOT_URL?t=$(date +%s)" -o /tmp/atd_boot.sh
if gcloud compute instances describe $VM --zone=$ZONE >/dev/null 2>&1; then
  gcloud compute instances add-metadata $VM --zone=$ZONE --metadata-from-file=startup-script=/tmp/atd_boot.sh --quiet
  if [ "$(gcloud compute instances describe $VM --zone=$ZONE --format='value(status)')" = RUNNING ]; then
    gcloud compute instances reset $VM --zone=$ZONE --quiet
  else
    gcloud compute instances start $VM --zone=$ZONE --quiet
  fi
else
  IP=$(gcloud compute addresses describe "$VM-ip" --region=$REGION --format='value(address)')
  gcloud compute instances create $VM --zone=$ZONE --machine-type=e2-small \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=20GB --boot-disk-type=pd-balanced --address="$IP" --tags=atd-web \
    --metadata-from-file=startup-script=/tmp/atd_boot.sh --quiet
fi

say "5/5 完成。把下面這串數字貼給 Claude："
echo "$NUM"
