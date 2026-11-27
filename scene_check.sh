#!/usr/bin/env bash
# Manual scene assessment / manual scene fallback on ROS domain 8 (see scene_check.py --help).
set -e
dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/lite3_ws/operator/env.sh"        # ROS, both workspaces, domain 8, Fast DDS profile
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$dir/scene_check.py" "$@"
