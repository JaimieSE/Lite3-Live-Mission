"""Format a mission snapshot for an operator terminal."""

from collections import Counter
from datetime import datetime
import json
import sys


def recent(rows, limit=8):
    return sorted(rows, key=lambda row: row.get("observed_at", ""), reverse=True)[:limit]


def clock(row):
    stamp = row.get("observed_at")
    if not stamp:
        return "—"
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone().strftime("%H:%M:%S %Z").strip()


def map_position(row):
    location = row.get("location")
    if not isinstance(location, dict) or location.get("frame_id") != "map" or not all(
        isinstance(location.get(axis), (int, float)) for axis in ("x", "y", "z")
    ):
        return ""
    coordinates = ", ".join(f"{location[axis]:.2f}" for axis in ("x", "y", "z"))
    return f" · map ({coordinates}) m"


def location_summary(row):
    if row.get("location_issue"):
        return f" · no position ({row['location_issue']})"
    return map_position(row)


def main():
    snapshot = json.load(sys.stdin)
    objectives = snapshot["objective"]
    observations = snapshot["observation"]
    clues = snapshot["qr"]
    requests = snapshot["request"]

    complete = sum(row.get("status") == "complete" for row in objectives)
    confirmed = sum(row.get("confirmed") is True for row in observations)
    mapped = sum((row.get("location") or {}).get("frame_id") == "map" for row in observations)
    print(f"Objectives {complete}/{len(objectives)} complete | "
          f"Observations {len(observations)} ({confirmed} confirmed, {mapped} mapped) | "
          f"QR clues {len(clues)}")

    print("\nObjectives")
    if not objectives:
        print("  No objectives loaded.")
    for row in sorted(objectives, key=lambda item: (-item.get("priority", 0), item["id"])):
        description = row.get("description") or row.get("target", row["id"])
        print(f"  {row.get('status', 'unknown').upper():<9} {description}")
        print(f"            {row['id']} · {len(row.get('findings', []))} confirmed findings")
        if row.get("reason"):
            print(f"            {row['reason']}")

    print("\nLatest observations")
    if observations:
        labels = Counter(row["label"] for row in observations)
        print("  Counts by label: " + ", ".join(f"{name} {count}" for name, count in labels.most_common()))
    else:
        print("  No observations recorded.")
    for row in recent(observations):
        state = "confirmed" if row.get("confirmed") is True else "candidate"
        confidence = row.get("confidence")
        score = f" · {confidence:.0%}" if confidence is not None else ""
        print(f"  {clock(row)}  {row['label']} · {state}{score}{location_summary(row)}")
    if len(observations) > 8:
        print("  Use --all for the full observation records.")

    print("\nLatest QR clues")
    if not clues:
        print("  No decoded QR clues.")
    for row in recent(clues):
        text = row["text"]
        excerpt = text if len(text) <= 120 else text[:117] + "..."
        print(f"  {clock(row)}  {json.dumps(excerpt)}{location_summary(row) or ' · position unavailable'}")
    if len(clues) > 8 or any(len(row["text"]) > 120 for row in clues):
        print("  Use --all for the full clue text and evidence references.")

    pending = [row for row in requests if row.get("status") == "pending"]
    print(f"\nPending requests  {len(pending)}")
    for row in pending:
        print(f"  {row.get('action', 'unknown')} · {row.get('objective_id', row['id'])}")


if __name__ == "__main__":
    main()
