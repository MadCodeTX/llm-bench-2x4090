#!/bin/bash
# Queue lifecycle via systemd (survives ssh drops; auto-restarts on failure and
# resumes from models.json done-flags).
case "${1:-}" in
  start)
    systemctl --user stop llmbench 2>/dev/null
    systemctl --user reset-failed llmbench 2>/dev/null
    systemd-run --user --unit=llmbench --property=Restart=on-failure \
      --property=RestartSec=30 --working-directory="$(cd "$(dirname "$0")" && pwd)" \
      python3 bench.py queue
    ;;
  stop)   systemctl --user stop llmbench ;;
  status) systemctl --user status llmbench --no-pager -l | head -15
          journalctl --user -u llmbench -n 5 --no-pager ;;
  *) echo "usage: bench.sh {start|stop|status}"; exit 1 ;;
esac
