#!/usr/bin/env bash
# Removes the launcher and service. Keeps your documents, logs and ~/.config keys.
NAME=agentic-researcher
systemctl --user stop $NAME 2>/dev/null || true
rm -f ~/.config/systemd/user/$NAME.service ~/.local/share/applications/$NAME.desktop
systemctl --user daemon-reload
echo "Removed. Keys kept in ~/.config/$NAME/env"
