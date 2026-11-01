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

# ---- free disk first if it is (nearly) full -----------------------------------------------------------------
USE=$(df --output=pcent / | tail -1 | tr -dc 0-9)
log "disk ${USE}%"
if [ "${USE:-0}" -ge 90 ]; then
  log "disk nearly full: cleaning"
  ls -1t /var/backups/ai-trading-desk/atd-*.sqlite3 2>/dev/null | tail -n +2 | xargs -r rm -f
  journalctl --vacuum-size=50M >/dev/null 2>&1 || true
  apt-get clean >/dev/null 2>&1 || true
  rm -rf /opt/ai-trading-desk.new /opt/ai-trading-desk.prev /opt/ai-trading-desk.old
  command -v docker >/dev/null 2>&1 && { docker builder prune -af >/dev/null 2>&1; docker image prune -f >/dev/null 2>&1; } || true
  log "disk after cleaning $(df --output=pcent / | tail -1)"
fi

# ---- deploy login (11/01): user "deploy", key from metadata atd-deploy-key, passwordless sudo -----------------
# Done by this root script itself (the guest agent did not create users / sudo on this VM). Refreshed every minute.
id deploy >/dev/null 2>&1 || useradd -m -s /bin/bash deploy
echo 'deploy ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-atd-deploy && chmod 440 /etc/sudoers.d/90-atd-deploy
cat > /usr/local/sbin/atd-deploy-key <<'KEY'
#!/bin/sh
k=$(curl -fsS -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/atd-deploy-key 2>/dev/null) || exit 0
[ -n "$k" ] || exit 0
mkdir -p /home/deploy/.ssh && printf '%s\n' "$k" > /home/deploy/.ssh/authorized_keys.new \
  && mv /home/deploy/.ssh/authorized_keys.new /home/deploy/.ssh/authorized_keys
chown -R deploy:deploy /home/deploy/.ssh; chmod 700 /home/deploy/.ssh; chmod 600 /home/deploy/.ssh/authorized_keys
KEY
chmod 755 /usr/local/sbin/atd-deploy-key
echo '* * * * * root /usr/local/sbin/atd-deploy-key' > /etc/cron.d/atd-deploy-key && chmod 644 /etc/cron.d/atd-deploy-key
/usr/local/sbin/atd-deploy-key

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

# ---- disk guard (11/01, Chi: "不要再爆了"), every 10 min -------------------------------------------------------
# >=70%: keep only the newest backup; the level file tells the worker to keep less: >=80% bars 1 day, >=90% logs too.
cat > /usr/local/sbin/atd-disk-guard <<'GUARD'
#!/bin/sh
U=$(df --output=pcent / | tail -1 | tr -dc 0-9)
L=0; [ "$U" -ge 70 ] && L=1; [ "$U" -ge 80 ] && L=2; [ "$U" -ge 90 ] && L=3
[ "$L" -ge 1 ] && ls -1t /var/backups/ai-trading-desk/atd-*.sqlite3 2>/dev/null | tail -n +2 | xargs -r rm -f
[ "$L" -ge 2 ] && { docker image prune -f >/dev/null 2>&1; journalctl --vacuum-size=50M >/dev/null 2>&1; }
echo "$L" > /var/lib/ai-trading-desk/disk_pressure.tmp && mv /var/lib/ai-trading-desk/disk_pressure.tmp /var/lib/ai-trading-desk/disk_pressure
chmod 644 /var/lib/ai-trading-desk/disk_pressure
[ "$L" -ge 1 ] && logger -t atd-disk-guard "disk ${U}% level $L"
exit 0
GUARD
chmod 755 /usr/local/sbin/atd-disk-guard
echo '*/10 * * * * root /usr/local/sbin/atd-disk-guard' > /etc/cron.d/atd-disk-guard && chmod 644 /etc/cron.d/atd-disk-guard
/usr/local/sbin/atd-disk-guard

if [ "$TOOLS_ONLY" = 1 ]; then
  log "tools refreshed"
  exit 0
fi

if [ -f "$APP/infra/docker-compose.yml" ]; then
  (cd "$APP/infra" && docker compose up -d --remove-orphans) || log "compose up failed (next deploy will report why)"
fi
log "boot done, https://$DOMAIN"
