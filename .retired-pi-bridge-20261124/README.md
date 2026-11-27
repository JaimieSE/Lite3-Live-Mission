Retired 2026-11-24 at the operator's request. `pi_mission_reporter.py` (mission reports) and `nav_hold.py`
(scene holds) are now the only NUC consumers of the Pi topics. This was a dry-run-only consumer that never
moved the robot or submitted anything. Recordings stay in `../data/pi/`. To restore, move the files back to `~/live_mission`.
