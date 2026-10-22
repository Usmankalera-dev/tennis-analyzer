#!/usr/bin/env python3
"""Tennis match analyzer - command line entry point.

Usage:
    python analyze.py input.mp4 [--trail neon|fire|ice]
                                [--out output.mp4] [--stats stats.json]

Pipeline: court keypoint calibration -> TrackNet ball tracking ->
YOLO player tracking -> hit/bounce event detection -> annotated
video + JSON shot-by-shot breakdown.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tennis_analyzer.court import CourtCalibrator
from tennis_analyzer.ball import BallTracker
from tennis_analyzer.players import PlayerTracker
from tennis_analyzer.events import detect_events
from tennis_analyzer.render import Renderer


def read_video(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, float(fps)


def build_render_state(events, ball_track, ball_court, fps, n):
    """Precompute per-frame info the renderer needs."""
    shots = events["shots"]
    bounces_so_far = []
    acc = []
    bi = 0
    blist = sorted(events["bounces"], key=lambda b: b["frame"])
    for i in range(n):
        while bi < len(blist) and blist[bi]["frame"] <= i:
            acc.append(blist[bi])
            bi += 1
        bounces_so_far.append(list(acc))

    current_shot = [None] * n
    last_call = [None] * n
    cur_shot, cur_call = None, None
    shot_by_hit = {sh["hit_frame"]: sh for sh in shots}
    bounce_by_frame = {}
    for sh in shots:
        if sh["bounce_frame"] is not None:
            bounce_by_frame[sh["bounce_frame"]] = sh
    for i in range(n):
        if i in shot_by_hit:
            cur_shot = shot_by_hit[i]
        if i in bounce_by_frame:
            b = bounce_by_frame[i]
            cur_call = {"call": b["call"], "margin_cm": b["margin_cm"], "line": b["line"]}
        current_shot[i] = cur_shot
        last_call[i] = cur_call

    flash = {}
    for sh in shots:
        if sh["bounce_frame"] is not None:
            bf = sh["bounce_frame"]
            pt = ball_track[bf]
            if pt is not None:
                for k in range(bf, min(bf + int(fps * 1.2), n)):
                    flash[k] = (pt[0], pt[1], sh["call"])

    trail_hist = []
    for i in range(n):
        hist = []
        j = i
        gap = 0
        while j >= 0 and len(hist) < 14:
            if ball_track[j] is not None:
                hist.append(ball_track[j])
                gap = 0
            else:
                gap += 1
                if gap > 2:
                    break
            j -= 1
        trail_hist.append(hist)

    return {
        "first_hit_frame": events["first_hit_frame"],
        "bounces_so_far": bounces_so_far,
        "current_shot": current_shot,
        "last_call": last_call,
        "bounce_flash": flash,
    }, trail_hist


def main():
    ap = argparse.ArgumentParser(description="Tennis match analyzer")
    ap.add_argument("input", help="input MP4 clip (broadcast view)")
    ap.add_argument("--trail", default="neon", choices=["neon", "fire", "ice"],
                    help="ball trail style")
    ap.add_argument("--out", default=None, help="output video path")
    ap.add_argument("--stats", default=None, help="output stats JSON path")
    ap.add_argument("--weights-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "weights"))
    ap.add_argument("--yolo", default="yolo11n.pt", help="YOLO weights")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.out or f"{base}_analyzed.mp4"
    stats_path = args.stats or f"{base}_stats.json"

    t0 = time.time()
    frames, fps = read_video(args.input, args.max_frames)
    if not frames:
        print("ERROR: could not read any frames from", args.input)
        return 1
    H0, W0 = frames[0].shape[:2]
    n = len(frames)
    print(f"[1/6] read {n} frames, {W0}x{H0} @ {fps:.1f} fps")

    print("[2/6] calibrating court ...")
    calib = CourtCalibrator(os.path.join(args.weights_dir, "court_keypoints.pt"))
    if not calib.calibrate(frames, sample_every=15):
        print("ERROR: court calibration failed (not enough keypoints).")
        return 1
    print(f"      homography ok, median keypoint residual "
          f"{calib.calib_error_cm:.1f} cm ({calib.calib_inliers} inliers)")

    print("[3/6] tracking ball (TrackNet) ...")
    bt = BallTracker(os.path.join(args.weights_dir, "tracknet_ball.pt"), batch=4)
    ball_track_raw, ball_detected, _ = bt.track(frames)
    det_pct = 100.0 * sum(ball_detected) / n
    print(f"      ball detected in {det_pct:.1f}% of frames")

    print("[4/6] tracking players (YOLO) ...")
    pt = PlayerTracker(args.yolo)
    players = pt.track(frames, calib, fps)
    for p in players:
        both = sum(b is not None for b in p["box"])
        print(f"      {p['name']} ({p['half']}): visible {100.0 * both / n:.0f}% of frames, "
              f"distance {p['distance_m']} m, top speed {p['top_speed_kmh']} km/h")

    print("[5/6] detecting events ...")
    events = detect_events(ball_track_raw, calib, players, fps)
    ball_track = events["ball_track"]
    ball_court = events["ball_court"]
    trk_pct = 100.0 * sum(t is not None for t in ball_track) / n
    print(f"      {len(events['shots'])} shots, "
          f"{len(events['bounces'])} bounces with calls; "
          f"ball tracked (cleaned) {trk_pct:.1f}% of frames")

    print("[6/6] rendering annotated video ...")
    state, trail_hist = build_render_state(events, ball_track, ball_court, fps, n)
    renderer = Renderer(trail=args.trail, scale=2)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W0 * 2, H0 * 2))
    for i in range(n):
        img = renderer.render(frames[i], i, fps, ball_track, ball_court,
                              players, state, trail_hist)
        writer.write(img)
    writer.release()

    stats = {
        "video": {"file": os.path.basename(args.input), "frames": n,
                  "fps": round(fps, 2), "width": W0, "height": H0,
                  "duration_s": round(n / fps, 1)},
        "court": {"calibrated": True,
                  "keypoint_residual_cm": round(calib.calib_error_cm, 1),
                  "ransac_inliers": calib.calib_inliers},
        "ball": {"model": "TrackNet (pretrained weights, yastrebksv/TrackNet)",
                 "detected_pct": round(det_pct, 1),
                 "tracked_pct_after_interpolation": round(trk_pct, 1)},
        "players": [{"name": p["name"], "half": p["half"],
                     "distance_m": p["distance_m"],
                     "top_speed_kmh": p["top_speed_kmh"]} for p in players],
        "shots": events["shots"],
        "notes": [
            "All speeds are estimates from a single broadcast camera.",
            "Shot speed = court-plane ball displacement in the first frames after the hit.",
            "In/out calls use singles lines and the calibrated homography.",
        ],
        "processing_time_s": round(time.time() - t0, 1),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"done in {time.time() - t0:.0f}s -> {out_path}, {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
