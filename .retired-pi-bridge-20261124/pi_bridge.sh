#!/usr/bin/env bash
# Run pi_bridge.py on ROS domain 8 with the operator Fast DDS profile (same as the navigation stack).
set -e
integration_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=8
export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/lite3_ws/operator/fastdds_lite3.xml"
unset ROS_LOCALHOST_ONLY
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$integration_dir/pi_bridge.py" "$@"
