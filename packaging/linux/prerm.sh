#!/bin/sh
# strix-hip pre-removal hook (DEB prerm / RPM %preun): stop the service on removal, not on upgrade.
# Configuration in /etc/strix and state in /var/lib/strix are kept.
set -e
case "$1" in
    remove|deconfigure|0)
        if [ -d /run/systemd/system ]; then
            systemctl stop strix.service >/dev/null 2>&1 || true
        fi
        ;;
esac
exit 0
