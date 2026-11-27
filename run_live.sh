#!/usr/bin/env bash
set -e
integration_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(dirname -- "$integration_dir")"
scene_dir="$source_dir/scene_understanding"
mission_dir="$source_dir/mission_rgb_01a1edca/src/rescue_mission"
export PYTHONPATH="$scene_dir:$mission_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
perception_rgbd=true
perception_vlm=false
perception_dry=false
for perception_arg in "$@"; do
 [[ "$perception_arg" != '--wide' ]] || perception_rgbd=false
 if [[ "$perception_arg" == --reid ]]; then
  export PYTHONPATH="$integration_dir/.reid-deps:$PYTHONPATH"
 fi
 case "$perception_arg" in --vlm|--scene-understanding) perception_vlm=true ;; esac
 [[ "$perception_arg" != '--dry-run' ]] || perception_dry=true
done
if "$perception_rgbd"; then
 source /opt/ros/humble/setup.bash
 export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-8}"
fi
if [[ "$perception_vlm" == true && "$perception_dry" == false && -z "${ANTHROPIC_API_KEY:-}" ]]; then
 read -r -s -p 'Anthropic API key (hidden, not saved): ' ANTHROPIC_API_KEY
 printf '\n'
 export ANTHROPIC_API_KEY
fi
exec "$scene_dir/.venv/bin/python" "$integration_dir/live_perception.py" "$@"
