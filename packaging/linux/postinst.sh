#!/bin/sh
# strix-hip post-install hook (DEB postinst / RPM %post). Creates the service account and reloads systemd.
# It never enables or starts strix.service and never downloads anything.
set -e
case "$1" in
    configure|[0-9]*) ;;   # DEB: configure; RPM: 1 = install, 2 = upgrade
    *) exit 0 ;;
esac
if ! getent passwd strix >/dev/null 2>&1; then
    if command -v systemd-sysusers >/dev/null 2>&1 && [ -f /usr/lib/sysusers.d/strix.conf ]; then
        systemd-sysusers /usr/lib/sysusers.d/strix.conf || true
    fi
fi
if ! getent passwd strix >/dev/null 2>&1; then
    useradd --system --user-group --home-dir /var/lib/strix --no-create-home --shell /usr/sbin/nologin strix || true
fi
if [ -d /run/systemd/system ]; then
    systemctl daemon-reload >/dev/null 2>&1 || true
fi
exit 0
