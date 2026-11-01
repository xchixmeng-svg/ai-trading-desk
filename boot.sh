#!/bin/bash
# GCE startup script (runs on every boot, as root). Since 2026-11-01 the VM no longer pulls or builds
# releases itself: GitHub Actions builds the image and pushes it over SSH (scripts/remote_deploy.sh),
# so every deploy and its errors are visible in the Actions log.
# This script only: installs Docker once, adds swap, writes .env from instance metadata,
# installs the nightly backup cron, and (on a full boot) starts whatever release is installed.
set -u
MD=http://metadata.google.internal/computeMetadata/v1
md() { curl -fsS -H "Metadata-Flavor: Google" "$MD/instance/attributes/$1" 2>/dev/null || true; }
APP=/opt/ai-trading-desk
log() { echo "[atd-boot] $*"; logger -t atd-boot "$*" 2>/dev/null || true; }
TOOLS_ONLY=0; [ "${1:-}" = "--tools-only" ] && TOOLS_ONLY=1

if ! command -v docker >/dev/null 2>&1; then
  log "installing docker"
  apt-get update -y && apt-get install -y ca-certificates curl sqlite3
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
fi
command -v sqlite3 >/dev/null 2>&1 || apt-get install -y sqlite3 >/dev/null 2>&1 || true
# 1 GB swap so a small machine survives memory spikes
if [ ! -f /swapfile ]; then
  fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
swapon -a 2>/dev/null || true
mkdir -p /var/lib/ai-trading-desk /var/lib/ai-trading-desk-caddy /var/backups/ai-trading-desk
chown 10001 /var/lib/ai-trading-desk

# ---- .env from metadata (rewritten every run so key changes apply) -----------------------
IP=$(curl -fsS -H "Metadata-Flavor: Google" "$MD/instance/network-interfaces/0/access-configs/0/external-ip")
DOMAIN="$(echo "$IP" | tr . -).sslip.io"
umask 077
mkdir -p /etc/ai-trading-desk
cat > /etc/ai-trading-desk/atd.env <<ENV
LIVE_TRADING_ENABLED=false
ATD_DOMAIN=$DOMAIN
ATD_CORS_ORIGINS=https://$DOMAIN
ATD_BILLING_TOKEN=$(md atd-billing-token)
ENV
echo "ATD_DOMAIN=$DOMAIN" > /etc/ai-trading-desk/caddy.env
chmod 600 /etc/ai-trading-desk/atd.env
umask 022

# ---- the old pull-updater is retired: remove it ------------------------------------------
rm -f /usr/local/bin/atd-update

# ---- cron: nightly backup, keep the 2 newest (Chi: the VM keeps at most 2 days) ----------
cat > /etc/cron.d/ai-trading-desk <<'CRON'
15 3 * * * root sqlite3 /var/lib/ai-trading-desk/ai_trading_desk.sqlite3 ".backup '/var/backups/ai-trading-desk/atd-$(date -u +\%Y\%m\%d).sqlite3'"; ls -1t /var/backups/ai-trading-desk/atd-*.sqlite3 | tail -n +3 | xargs -r rm -f
CRON
chmod 644 /etc/cron.d/ai-trading-desk
ls -1t /var/backups/ai-trading-desk/atd-*.sqlite3 2>/dev/null | tail -n +3 | xargs -r rm -f

if [ "$TOOLS_ONLY" = 1 ]; then
  log "tools refreshed"
  exit 0
fi

if [ -f "$APP/infra/docker-compose.yml" ]; then
  (cd "$APP/infra" && docker compose up -d --remove-orphans) || log "compose up failed (next deploy will report why)"
fi
log "boot done, https://$DOMAIN"
