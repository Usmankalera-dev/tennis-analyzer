"""Ball tracking with TrackNet.

The network sees 3 consecutive frames and outputs a quantized heatmap
(256 value classes per pixel). We decode it as an expected-value
heatmap: E = sum_v v * P(v). The peak of E is the ball position.
Detections then go through outlier removal and short-gap interpolation.
"""

import numpy as np
import cv2
import torch

from .nets import load_ball_net

MODEL_W, MODEL_H = 640, 360


class BallTracker:
    def __init__(self, weights_path, device="cpu", batch=4):
        self.model = load_ball_net(weights_path, device)
        self.device = device
        self.batch = batch

    def track(self, frames):
        """frames: list of BGR frames (any size).
        Returns (track, detected):
          track[i]    = (x, y) ball position in original pixels or None
          detected[i] = True only for real model detections (not filled)"""
        n = len(frames)
        H0, W0 = frames[0].shape[:2]

        def small(i):
            return cv2.resize(frames[i], (MODEL_W, MODEL_H)).astype(np.float32) / 255.0

        raw = [None] * n
        conf = [0.0] * n
        values = torch.arange(256, dtype=torch.float32, device=self.device)
        jobs = list(range(2, n))
        with torch.inference_mode():
            for s in range(0, len(jobs), self.batch):
                chunk = jobs[s:s + self.batch]
                inp = []
                for i in chunk:
                    stack = np.concatenate([small(i), small(i - 1), small(i - 2)], axis=2)
                    inp.append(np.rollaxis(stack, 2, 0))
                x = torch.from_numpy(np.stack(inp)).to(self.device)
                probs = self.model(x, testing=True)  # (B, 256, H*W)
                # expected-value heatmap, computed in channel chunks to
                # keep memory bounded (full tensor is ~240 MB per frame)
                exp = torch.zeros((len(chunk), MODEL_H * MODEL_W),
                                  dtype=torch.float32, device=self.device)
                for c0 in range(0, 256, 32):
                    exp += (probs[:, c0:c0 + 32] *
                            values[c0:c0 + 32].view(1, -1, 1)).sum(dim=1)
                exp = exp.reshape(-1, MODEL_H, MODEL_W).cpu().numpy()
                del probs, x
                for j, i in enumerate(chunk):
                    E = exp[j]
                    peak = float(E.max())
                    conf[i] = peak / 255.0
                    if peak < 45:  # too weak -> treat as no ball
                        continue
                    py, px = np.unravel_index(int(E.argmax()), E.shape)
                    r = 5
                    x0, x1 = max(px - r, 0), min(px + r + 1, MODEL_W)
                    y0, y1 = max(py - r, 0), min(py + r + 1, MODEL_H)
                    win = E[y0:y1, x0:x1]
                    w = np.clip(win - 0.45 * peak, 0, None)
                    if w.sum() > 0:
                        ys, xs = np.mgrid[y0:y1, x0:x1]
                        cx = float((xs * w).sum() / w.sum())
                        cy = float((ys * w).sum() / w.sum())
                    else:
                        cx, cy = float(px), float(py)
                    raw[i] = (cx * W0 / MODEL_W, cy * H0 / MODEL_H)
        track = _clean_track(raw, max_step=0.22 * W0)
        detected = [t is not None for t in track]
        track = _interpolate(track, max_gap=10)
        return track, detected, conf


def despike(track, thresh_px=55.0):
    """Remove points that disagree with the median of their
    neighbours (short TrackNet lock-on errors: shoes, shirts)."""
    t = list(track)
    n = len(t)
    for _ in range(2):
        for i in range(n):
            if t[i] is None:
                continue
            neigh = [t[j] for j in (i - 2, i - 1, i + 1, i + 2)
                     if 0 <= j < n and t[j] is not None]
            if len(neigh) < 3:
                continue
            med = (float(np.median([p[0] for p in neigh])),
                   float(np.median([p[1] for p in neigh])))
            if np.hypot(t[i][0] - med[0], t[i][1] - med[1]) > thresh_px:
                t[i] = None
    return t


def _clean_track(track, max_step):
    """Despike, then remove single-frame spikes: a point far from
    both neighbours."""
    t = despike(track)
    n = len(t)
    for _ in range(2):
        for i in range(n):
            if t[i] is None:
                continue
            prev = t[i - 1] if i > 0 else None
            nxt = t[i + 1] if i < n - 1 else None
            if prev is not None and nxt is not None:
                d1 = np.hypot(t[i][0] - prev[0], t[i][1] - prev[1])
                d2 = np.hypot(t[i][0] - nxt[0], t[i][1] - nxt[1])
                dnn = np.hypot(prev[0] - nxt[0], prev[1] - nxt[1])
                if d1 > max_step and d2 > max_step and dnn < max_step:
                    t[i] = None
            elif prev is not None or nxt is not None:
                ref = prev if prev is not None else nxt
                if np.hypot(t[i][0] - ref[0], t[i][1] - ref[1]) > 2.2 * max_step:
                    t[i] = None
    return t


def _interpolate(track, max_gap=10):
    """Linearly fill gaps of at most max_gap frames inside a track."""
    t = list(track)
    n = len(t)
    i = 0
    while i < n:
        if t[i] is not None:
            i += 1
            continue
        j = i
        while j < n and t[j] is None:
            j += 1
        gap = j - i
        if i > 0 and j < n and gap <= max_gap:
            p0, p1 = t[i - 1], t[j]
            for k in range(i, j):
                a = (k - i + 1) / (gap + 1)
                t[k] = (p0[0] + (p1[0] - p0[0]) * a,
                        p0[1] + (p1[1] - p0[1]) * a)
        i = j
    return t
