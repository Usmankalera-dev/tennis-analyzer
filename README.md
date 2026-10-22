# Tennis Analyzer

Turn a tennis broadcast clip into a "live breakdown": ball tracking
with a styled trail, bounce **IN/OUT calls with cm margins**, shot
speed estimates, live player speeds, and a 2D mini-court view — plus a
shot-by-shot JSON breakdown.

![stack](https://img.shields.io/badge/python-3.10%2B-blue) ![pytorch](https://img.shields.io/badge/PyTorch-CPU%2FGPU-orange) ![opencv](https://img.shields.io/badge/OpenCV-4.x-green)

## What it does

1. **Court calibration (automatic)** — a pretrained court-keypoint
   network finds the 14 court keypoints; a RANSAC homography maps
   image pixels to real court meters (23.77 m × 10.97 m doubles,
   8.23 m singles width).
2. **Ball tracking** — TrackNet (pretrained) reads 3 consecutive
   frames and outputs a heatmap of the tiny, fast ball. Spurious
   lock-ons (shoes, shirts) are removed with a median despike filter
   and short gaps (≤ 12 frames) are interpolated.
3. **Players** — YOLO (yolo11n, person class) detects people on
   keyframes; detections are linked into tracklets and the two
   on-court players are picked by court-half coverage and movement
   (line judges and ball kids are rejected). Feet are projected onto
   the court plane, median-filtered and Kalman-smoothed.
4. **Events** — bounce-anchored segmentation:
   - *Bounce*: lowest point of the ball in the image (image-y peak);
     long flights are re-scanned at a lower threshold because
     far-side bounces can be only ~2 px tall.
   - *Hit*: one per inter-bounce interval, scored from court-plane
     and image-plane direction reversal, closest approach to the
     player whose turn it is (hits alternate), and a speed-burst
     rule for the serve. A hitter must be on the opposite side of
     the net from where his shot bounces — this removes phantom
     "hits" the ground projection creates at bounces.
   - *Call*: the bounce is projected through the homography and
     measured against the singles lines → `IN +24 cm`, `OUT -9 cm`, …
   - *Shot speed*: court-plane ball displacement in the first frames
     after the hit (**estimate**, see limitations).
5. **Outputs** — annotated MP4 (trail, player boxes + live speeds,
   last-call panel, rally timer, mini-court inset with player dots and
   bounce markers) and a JSON file with every shot and player totals.

## Quick start

```bash
pip install -r requirements.txt
# weights (see "Model weights" below) must be in ./weights/
python analyze.py inputs/wimbledon_rally.mp4 --trail neon
# -> wimbledon_rally_analyzed.mp4 + wimbledon_rally_stats.json
```

### CLI options

| Option | Default | Meaning |
|---|---|---|
| `--trail` | `neon` | Ball trail style: `neon`, `fire`, or `ice` |
| `--out` | `<name>_analyzed.mp4` | Output video path |
| `--stats` | `<name>_stats.json` | Output stats path |
| `--weights-dir` | `./weights` | Folder with the two `.pt` files |
| `--yolo` | `yolo11n.pt` | YOLO weights (auto-downloaded by ultralytics) |
| `--max-frames` | all | Process only the first N frames |

## Model weights

Both networks are public research models by **yastrebksv**
(not official implementations):

| File | Model | Source |
|---|---|---|
| `weights/tracknet_ball.pt` | TrackNet ball tracker | https://github.com/yastrebksv/TrackNet |
| `weights/court_keypoints.pt` | Court keypoint detector (14 points) | https://github.com/yastrebksv/TennisCourtDetector |

TrackNet paper: *TrackNet: A Deep Learning Network for Tracking
High-speed and Tiny Objects in Sports Applications* (Huang et al.,
2019). The weights are also linked from those repos' READMEs
(Google Drive).

## Project layout

```
analyze.py                  CLI entry point
tennis_analyzer/
    nets.py                 TrackNet backbone (ball + court variants)
    court.py                Keypoint decoding, homography, in/out rules
    ball.py                 TrackNet inference, cleanup, interpolation
    players.py              YOLO detection, slot tracking, Kalman filter
    events.py               Hits, bounces, shots, calls, speed estimates
    render.py               HUD, trail styles, mini-court, video frames
inputs/                     Sample clips
weights/                    Pretrained .pt files (not in git)
```

## Demo results (test clip)

17 s rally, Djokovic v Federer, Wimbledon 2014 (632×360 @ 25 fps,
425 frames, single static broadcast camera):

- Court calibration: median keypoint residual **8.8 cm**.
- Ball detected raw in 342/425 frames (**80.5%**); most of the rest
  recovered by despiking + interpolation.
- **13 shots** segmented (1 serve + 12 rally shots), **12 bounces,
  all called IN** with margins — consistent with a rally that kept
  going. Hitter alternation and bounce positions match the footage
  on visual inspection, including Djokovic's on-the-rise shot taken
  ~2 frames after the bounce at frame ~87.
- Players: P1 (near) 26.0 m covered, top 17.2 km/h; P2 (far) 36.1 m,
  top 36.1 km/h (overestimate — see limitations).

## Honest limitations

- **Speeds are estimates.** One camera + ground-plane projection
  cannot measure true ball or player speed: the ball's height is
  unknown, and the projection biases results — on the test clip,
  shots hit *away* from the camera read high, and the far player's
  top speed (36.1 km/h) is likely an overestimate. Treat every
  number as ±20–30%; they are labeled "est." everywhere.
- **Calls are estimates too.** A bounce call is only as good as the
  homography (8.8 cm median keypoint residual on the test clip).
  Margins under ~10 cm should be read as "too close to call". This
  is a training/analysis aid, not an officiating system — no
  Hawk-Eye claims.
- Fast net exchanges are the hardest segment: in the test rally's
  half-volley scramble, one hitter is misattributed and three shots
  have no speed (track gaps at the contact). The serve contact is
  detected a few frames early (the toss fools the burst detector),
  so the serve reads 139 km/h est. — like a second serve.
- The homography assumes one **static broadcast camera** per clip.
  Cuts, zooms and replays inside a clip break calibration and ball
  continuity; trim your clip to one continuous rally from the main
  camera.
- Player identity ("PLAYER 1 / PLAYER 2") means near-court and
  far-court slots, not recognized athletes.
- Serve rules (foot faults, service-box-only calls, lets) are not
  implemented; calls use rally (singles) boundaries.
- CPU inference is slow: TrackNet takes ~4.3 s/frame on the
  development machine (8 GB RAM, 2 cores, no GPU), so the 17 s demo
  clip needed ~30 min. On a GPU it is near real-time. Short clips
  (15–30 s) are recommended.

## Sample

`inputs/wimbledon_rally.mp4` — a 17 s clip cut from a Wimbledon
rally video (Djokovic v Federer, 2014, via YouTube): serve + full
rally from the main broadcast camera, used for the demo output.
All rights to the footage belong to its owners; it is included only
as a test input.

## License

Code: MIT. Model weights: see the source repositories above.
