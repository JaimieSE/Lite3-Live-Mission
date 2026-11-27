#!/usr/bin/env bash
set -e
integration_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mission_dir="$(dirname -- "$integration_dir")/mission_rgb_01a1edca/src/rescue_mission"
export PYTHONPATH="$mission_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$integration_dir/competition.py" "$@"
