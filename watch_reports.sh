#!/usr/bin/env bash
# Watch the mission reports live: each room scan, Claude's answer, and every report sent with the server's reply.
#   ~/live_mission/watch_reports.sh            the current run (manual fallback's run if it is on, else the newest)
#   ~/live_mission/watch_reports.sh <folder>   a given competition-* run folder
# Ctrl+C to stop watching (it changes nothing).
LOG_ROOT="$HOME/lite3_ws/log/operator"
run="$1"
if [ -z "$run" ] && [ -f "$LOG_ROOT/scene_fallback.json" ]; then
  run=$(python3 -c "import json;print(json.load(open('$LOG_ROOT/scene_fallback.json'))['run'])" 2>/dev/null)
fi
[ -z "$run" ] && run=$(ls -d "$LOG_ROOT"/competition-* 2>/dev/null | grep -v scene-check | tail -1)
[ -d "$run" ] || { echo "No run folder found under $LOG_ROOT"; exit 1; }
mode=$(grep -m1 -ohE "POST http://[^ ]+|DRY RUN" "$run"/reporter*.log 2>/dev/null | tail -1)
echo "Run: $(basename "$run")   reporter: ${mode:-not started yet}"
echo "SENT = posted to the scorer; ACCEPTED/REJECTED = its reply; HELD = waiting for confirmed evidence"
echo "------------------------------------------------------------------------------------------------"
cd "$run" && touch scene_fallback.log reporter.log 2>/dev/null
tail -n 20 -F reporter*.log scene_fallback.log 2>/dev/null | grep --line-buffered -E \
  "scan fallback-|result fallback-|no result for|room scan |PAYLOAD|HTTP [0-9]|accepted|rejected|held|linked|pending:|rooms \(" \
  | sed -u -e 's/^PAYLOAD /SENT     /' -e 's/ HTTP 200: {"status":"accepted"/ ACCEPTED {"status":"accepted"/'
