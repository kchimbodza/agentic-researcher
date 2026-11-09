#!/usr/bin/env bash
# Installs Agentic Researcher as a desktop app: a user service + an app launcher.
# The service only runs while you use it (the launcher starts it; it is not started at boot).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
UV="$(command -v uv)"
PORT=8770
NAME=agentic-researcher

mkdir -p ~/.config/$NAME ~/.config/systemd/user ~/.local/share/applications
# API keys live here (never in the repo). Phase 2+ reads ANTHROPIC_API_KEY.
[ -f ~/.config/$NAME/env ] || { printf 'ANTHROPIC_API_KEY=\n' > ~/.config/$NAME/env; chmod 600 ~/.config/$NAME/env; }
# Name shown on "Reviewed by …" and in the audit log.
if ! grep -q '^REVIEWER_NAME=' ~/.config/$NAME/env; then
  read -rp "Your name for review sign-off [$(id -un)]: " who || true
  printf 'REVIEWER_NAME=%s\n' "${who:-$(id -un)}" >> ~/.config/$NAME/env
fi

"$UV" sync --project "$DIR" -q

cat > ~/.config/systemd/user/$NAME.service <<UNIT
[Unit]
Description=Agentic Researcher (local web app)

[Service]
WorkingDirectory=$DIR
EnvironmentFile=%h/.config/$NAME/env
ExecStart=$UV run --project $DIR uvicorn app.server:app --host 127.0.0.1 --port $PORT
Restart=on-failure
RestartSec=5
UNIT

cat > ~/.local/share/applications/$NAME.desktop <<DESK
[Desktop Entry]
Type=Application
Name=Agentic Researcher
Comment=Answers you can verify: AI agents research your documents and check every quote
Exec=sh -c 'systemctl --user start $NAME; for i in \$(seq 1 40); do curl -s -o /dev/null http://127.0.0.1:$PORT/api/status && break; sleep 0.25; done; google-chrome --start-maximized --app=http://127.0.0.1:$PORT --class=AgenticResearcher'
Icon=$DIR/app/static/icon.svg
Categories=Science;Education;Development;
Keywords=AI;agent;research;papers;datasheets;
StartupWMClass=AgenticResearcher
DESK

systemctl --user daemon-reload
update-desktop-database ~/.local/share/applications 2>/dev/null || true
echo "Installed. Open 'Agentic Researcher' from your app menu."
echo "Stop it:  systemctl --user stop $NAME    Logs: journalctl --user -u $NAME -f"
