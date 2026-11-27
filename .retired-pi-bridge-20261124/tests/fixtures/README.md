`pi_real_frames.jsonl`: real detector output from the Pi (`~/usb-camera-perception/stairs-*.log`,
2026-11-24). Those runs predate the `capture_stamp_ns`/`frame_id` fields, so both were added from each frame's own
camera stamp (`stair_curtain.stamp_ns`) and the Orbbec frame name. Everything else is as the Pi printed it.
