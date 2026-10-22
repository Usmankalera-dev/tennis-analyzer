"""Player detection, tracking and movement stats.

YOLO (yolo11n, person class) detects people in every frame. Detections
are linked into tracklets by greedy nearest-foot association. The two
match players are then chosen as the long tracklets with the largest
*smoothed* court-plane movement - line judges, ball kids and the
umpire stand still, the players do not. One player is taken from each
court half (near / far).

Foot positions are projected onto the court plane and smoothed with a
constant-velocity Kalman filter, which also bridges short detection
gaps. Speeds come from central differences of the filtered positions.

IMPORTANT: speeds are estimates from a single broadcast camera.
"""

import numpy as np
import cv2
from ultralytics import YOLO

from .court import NET_Y


class _Kalman2D:
    """Constant-velocity Kalman filter for one court-plane point."""

    def __init__(self, fps):
        dt = 1.0 / fps
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.08
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.04
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.started = False

    def step(self, meas):
        if meas is not None and not self.started:
            self.kf.statePost = np.array(
                [[meas[0]], [meas[1]], [0], [0]], np.float32)
            self.started = True
        pred = self.kf.predict()
        if meas is not None:
            post = self.kf.correct(np.array([[meas[0]], [meas[1]]], np.float32))
            x, y = float(post[0]), float(post[1])
        else:
            x, y = float(pred[0]), float(pred[1])
        return (x, y)


def _moving_average(seq, win):
    if len(seq) < 3:
        return seq
    k = np.ones(win) / win
    x = np.convolve(np.pad(seq, (win // 2, win // 2), mode="edge"), k, mode="valid")
    return x[:len(seq)]


class PlayerTracker:
    def __init__(self, yolo_weights="yolo11n.pt", conf=0.35):
        self.model = YOLO(yolo_weights)
        self.conf = conf

    def _detect(self, frame):
        res = self.model.predict(frame, classes=[0], conf=self.conf,
                                 verbose=False)[0]
        people = []
        if res.boxes is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                people.append({
                    "box": (x1, y1, x2, y2),
                    "foot": ((x1 + x2) / 2.0, y2),
                    "conf": float(b.conf[0]),
                })
        return people

    @staticmethod
    def _build_tracklets(detections, gate=70.0, max_gap=8):
        """Greedy frame-to-frame linking of detections into tracklets.
        Returns list of dicts: {frame_index: person}."""
        tracklets = []
        for i, people in enumerate(detections):
            used = set()
            for tr in tracklets:
                if i - tr["_last_i"] > max_gap:
                    continue
                best, bd = None, gate
                for j, p in enumerate(people):
                    if j in used:
                        continue
                    d = np.hypot(p["foot"][0] - tr["_last_foot"][0],
                                 p["foot"][1] - tr["_last_foot"][1])
                    if d < bd:
                        bd, best = d, j
                if best is not None:
                    tr["frames"][i] = people[best]
                    tr["_last_foot"] = people[best]["foot"]
                    tr["_last_i"] = i
                    used.add(best)
            for j, p in enumerate(people):
                if j not in used:
                    tracklets.append({"frames": {i: p}, "_last_foot": p["foot"],
                                      "_last_i": i})
        return tracklets

    @staticmethod
    def _smoothed_path(court_series):
        """court_series: list of (x, y)/None for all frames.
        Fill gaps, smooth, return path length in meters."""
        idx = [i for i, c in enumerate(court_series) if c is not None]
        if len(idx) < 5:
            return 0.0
        xs = np.array([court_series[i][0] for i in idx])
        ys = np.array([court_series[i][1] for i in idx])
        full_x = np.interp(range(len(court_series)), idx, xs)
        full_y = np.interp(range(len(court_series)), idx, ys)
        sx = _moving_average(full_x, 15)
        sy = _moving_average(full_y, 15)
        return float(np.sum(np.hypot(np.diff(sx), np.diff(sy))))

    def track(self, frames, calibrator, fps):
        """Returns a list of 2 player dicts (index 0 = near half):
        per-frame 'box', 'foot_img', 'court' (smoothed), 'speed_kmh',
        plus totals 'distance_m' and 'top_speed_kmh'."""
        n = len(frames)
        detections = []
        for f in frames:
            people = self._detect(f)
            for p in people:
                p["court"] = calibrator.to_court([p["foot"]])[0]
            detections.append(people)

        tracklets = self._build_tracklets(detections)

        # score tracklets and pick the best mover in each half
        cand = {0: [], 1: []}  # 0 = far half, 1 = near half
        for tr in tracklets:
            present = len(tr["frames"])
            if present < 0.5 * n:
                continue
            series = [None] * n
            for i, p in tr["frames"].items():
                series[i] = p["court"]
            courts = [c for c in series if c is not None]
            if len(courts) < 0.5 * n:
                continue
            med_y = float(np.median([c[1] for c in courts]))
            path = self._smoothed_path(series)
            half = 1 if med_y > NET_Y else 0
            cand[half].append((path, present, tr))
        chosen = []
        for half in (1, 0):  # near first -> player index 0
            if not cand[half]:
                raise RuntimeError(f"no player tracklet found in half {half}")
            cand[half].sort(key=lambda t: (t[0], t[1]), reverse=True)
            chosen.append(cand[half][0][2])

        players = []
        for new_idx, tr in enumerate(chosen):
            box = [None] * n
            foot = [None] * n
            raw = [None] * n
            for i, p in tr["frames"].items():
                box[i] = p["box"]
                foot[i] = p["foot"]
                raw[i] = p["court"]
            kal = _Kalman2D(fps)
            court_sm, missing = [None] * n, 0
            # median-filter the measured series first: single-frame
            # box glitches (common at the far side) otherwise leak
            # into the Kalman filter as impossible jumps
            present_idx = [i for i in range(n) if raw[i] is not None]
            meas_series = list(raw)
            if len(present_idx) >= 5:
                from scipy.signal import medfilt
                xs = np.interp(range(n), present_idx,
                               [raw[i][0] for i in present_idx])
                ys = np.interp(range(n), present_idx,
                               [raw[i][1] for i in present_idx])
                xs = medfilt(xs, 5)
                ys = medfilt(ys, 5)
                meas_series = [(float(xs[i]), float(ys[i])) if raw[i] is not None
                               else None for i in range(n)]
            for i in range(n):
                meas = meas_series[i]
                missing = missing + 1 if meas is None else 0
                pos = kal.step(meas if missing <= int(fps) else None)
                if kal.started and missing <= int(fps):
                    court_sm[i] = pos
            # speeds: central difference of filtered positions + EMA
            speeds = [0.0] * n
            ema = 0.0
            dist, top = 0.0, 0.0
            for i in range(n):
                a = court_sm[i - 2] if i >= 2 else None
                b = court_sm[i + 2] if i + 2 < n else None
                if a is not None and b is not None:
                    v = np.hypot(b[0] - a[0], b[1] - a[1]) * fps / 4.0 * 3.6
                    ema = 0.65 * ema + 0.35 * min(v, 38.0)
                speeds[i] = ema
                top = max(top, ema)
                if i > 0 and court_sm[i] is not None and court_sm[i - 1] is not None:
                    step = np.hypot(court_sm[i][0] - court_sm[i - 1][0],
                                    court_sm[i][1] - court_sm[i - 1][1])
                    if step < 1.2:
                        dist += step
            players.append({
                "name": f"PLAYER {new_idx + 1}",
                "half": "near" if new_idx == 0 else "far",
                "box": box,
                "foot_img": foot,
                "court": court_sm,
                "speed_kmh": speeds,
                "distance_m": round(dist, 1),
                "top_speed_kmh": round(top, 1),
            })
        return players
