#!/bin/bash
# Hardware watchdog: auto-reboot the machine if the kernel hard-freezes (the GSP-style
# silent wedge). Intel i7-11700K -> iTCO_wdt. Hardware-level: needs no software running
# during the freeze. After a freeze, the board resets in ~60s; then SSH back in and
# restart the session (claude --resume <id> --channels ...).
#
# RUN WITH SUDO, then reboot once to fully activate:
#   sudo bash scripts/setup_watchdog.sh && sudo reboot
# Verify after reboot:
#   wdctl ; systemctl show -p RuntimeWatchdogUSec    (should be 60s)
set -e

# 1) load the Intel TCO watchdog now and on every boot
modprobe iTCO_wdt || { echo "modprobe iTCO_wdt failed (BIOS may disable TCO; check BIOS watchdog/WDT)"; exit 1; }
echo iTCO_wdt > /etc/modules-load.d/watchdog.conf

# 2) let systemd pet /dev/watchdog; hardware resets if systemd stops responding
mkdir -p /etc/systemd/system.conf.d
cat > /etc/systemd/system.conf.d/watchdog.conf <<'EOF'
[Manager]
RuntimeWatchdogSec=60
RebootWatchdogSec=10min
EOF
systemctl daemon-reexec

echo "OK. /dev/watchdog: $(ls /dev/watchdog 2>/dev/null || echo 'MISSING — check BIOS')"
wdctl 2>/dev/null | grep -iE 'Identity|Timeout' || true
echo "Now: sudo reboot, then verify with 'systemctl show -p RuntimeWatchdogUSec' (=60000000)."
