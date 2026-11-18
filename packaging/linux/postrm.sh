#!/bin/sh
# strix-hip post-removal hook (DEB postrm / RPM %postun): reload systemd. The strix account, /etc/strix and
# /var/lib/strix are left in place (remove them by hand after a purge if wanted).
set -e
if [ -d /run/systemd/system ]; then
    systemctl daemon-reload >/dev/null 2>&1 || true
fi
exit 0
