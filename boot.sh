#!/bin/bash
# GCE startup script (runs on every boot, as root).
# Installs Docker once, fetches the code bundle, writes .env from instance metadata,
# starts the stack, and installs an auto-updater + nightly DB backup.
set -u
MD=http://metadata.google.internal/computeMetadata/v1
md() { curl -fsS -H "Metadata-Flavor: Google" "$MD/instance/attributes/$1" 2>/dev/null || true; }
APP=/opt/ai-trading-desk
log() { echo "[atd-boot] $*"; logger -t atd-boot "$*"; }
TOOLS_ONLY=0; [ "${1:-}" = "--tools-only" ] && TOOLS_ONLY=1

if ! command -v docker >/dev/null 2>&1; then
  log "installing docker"
  apt-get update -y && apt-get install -y ca-certificates curl sqlite3 openssl
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
fi
mkdir -p /var/lib/ai-trading-desk /var/lib/ai-trading-desk-caddy /var/backups/ai-trading-desk
chown 10001 /var/lib/ai-trading-desk

# ---- .env from metadata (rewritten every boot so key changes apply) --------------------
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

# ---- updater: download bundle, rebuild only if it changed ----------------------------
cat > /usr/local/bin/atd-update <<'UPD'
#!/bin/bash
set -u
MD=http://metadata.google.internal/computeMetadata/v1
md() { curl -fsS -H "Metadata-Flavor: Google" "$MD/instance/attributes/$1" 2>/dev/null || true; }
APP=/opt/ai-trading-desk
URL=$(md atd-bundle-url)
KEY=$(md atd-bundle-key)
[ -z "$URL" ] && exit 0
TMP=$(mktemp -d)
curl -fsSL "$URL?t=$(date +%s)" -o "$TMP/b.enc" || { rm -rf "$TMP"; exit 0; }
# bundle is AES-256 encrypted in the public dist repo; key lives only in instance metadata
openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass "pass:$KEY" -in "$TMP/b.enc" -out "$TMP/b.tgz" \
  || { logger -t atd-update "decrypt failed"; rm -rf "$TMP"; exit 0; }
NEW=$(sha256sum "$TMP/b.tgz" | cut -d' ' -f1)
OLD=$(cat "$APP/.bundle_sha" 2>/dev/null || true)
if [ "$NEW" != "$OLD" ] || [ "${1:-}" = "--force" ]; then
  mkdir -p "$TMP/x" && tar -xzf "$TMP/b.tgz" -C "$TMP/x"
  rm -rf "$APP.new" && mv "$TMP/x" "$APP.new"
  # build first; tests run inside the build, so a broken bundle never replaces a running one
  if (cd "$APP.new/infra" && docker compose build worker); then
    rm -rf "$APP.old"; [ -d "$APP" ] && mv "$APP" "$APP.old"; mv "$APP.new" "$APP"
    echo "$NEW" > "$APP/.bundle_sha"
    # force-recreate: the old directory was moved away, containers must re-mount the new one
    (cd "$APP/infra" && docker compose up -d --force-recreate --remove-orphans)
    # refresh this updater + cron from the (encrypted, trusted) bundle
    bash "$APP/infra/gce/vm_boot.sh" --tools-only >/dev/null 2>&1 || true
    logger -t atd-update "deployed bundle $NEW"
  else
    logger -t atd-update "build failed for $NEW, keeping current version"
  fi
fi
rm -rf "$TMP"
UPD
chmod +x /usr/local/bin/atd-update

# ---- cron: auto-update every 15 min, backup nightly -----------------------------------
cat > /etc/cron.d/ai-trading-desk <<CRON
*/15 * * * * root /usr/local/bin/atd-update >/dev/null 2>&1
15 3 * * * root sqlite3 /var/lib/ai-trading-desk/ai_trading_desk.sqlite3 ".backup '/var/backups/ai-trading-desk/atd-\$(date -u +\%Y\%m\%d).sqlite3'" && find /var/backups/ai-trading-desk -name 'atd-*.sqlite3' -mtime +14 -delete
CRON
if [ "$TOOLS_ONLY" = 1 ]; then
  log "tools refreshed"
  exit 0
fi

# ---- first install -------------------------------------------------------------------
if [ ! -d "$APP" ]; then
  log "first install"
  /usr/local/bin/atd-update --force || true
fi

(cd "$APP/infra" && docker compose up -d --force-recreate --remove-orphans)

log "up at https://$DOMAIN"
