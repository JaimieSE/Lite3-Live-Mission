#!/usr/bin/env bash
set -e -o pipefail
integration_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mission_dir="$(dirname -- "$integration_dir")/mission_rgb_01a1edca/src/rescue_mission"
export PYTHONPATH="$mission_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
view=summary
case "${1:-}" in
    --all) view=all; shift ;;
    --confirmed) view=confirmed; shift ;;
    --summary) view=summary; shift ;;
esac
if (( $# > 1 )) || [[ "${1:-}" == -* ]]; then
    printf 'Usage: %s [--summary|--all|--confirmed] [database]\n' "$0" >&2
    exit 2
fi
database="${1:-}"
if [[ -z "$database" ]]; then
    latest=""
    for pointer in "$integration_dir"/data/runs/*/database.path; do
        if [[ -f "$pointer" ]]; then
            latest="$pointer"
        fi
    done
    if [[ -z "$latest" ]]; then
        printf 'No run libraries found. Start ./run_live.sh or pass an existing database path.\n' >&2
        exit 1
    fi
    database="$(<"$latest")"
fi
if [[ ! -f "$database" ]]; then
    printf 'Database does not exist: %s\n' "$database" >&2
    exit 1
fi
if [[ "$view" == all ]]; then
    exec python3 -m rescue_mission.cli --db "$database" status
fi
if [[ "$view" == summary ]]; then
    printf 'Database %s\n\n' "$database"
    python3 -m rescue_mission.cli --db "$database" status | python3 "$integration_dir/status_summary.py"
    exit
fi

python3 -m rescue_mission.cli --db "$database" status | python3 -c '
import json
import sys

status = json.load(sys.stdin)
status["observation"] = [
    observation for observation in status["observation"]
    if observation.get("confirmed") is True
]
print(json.dumps(status, indent=2))
'
