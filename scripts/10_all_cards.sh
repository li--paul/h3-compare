#!/usr/bin/env bash
# Render results/all-cards.md / .csv (Intel + AMD + NVIDIA).
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
start_log "10_all_cards"
"$VENV_PY" "$ROOT/scripts/10_all_cards.py"
echo "all-cards written: $ROOT/results/all-cards.md"
