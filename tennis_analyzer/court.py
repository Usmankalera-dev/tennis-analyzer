"""Court calibration and geometry.

The court keypoint network predicts 14 keypoints. Their positions on a
real court are known exactly (ITF dimensions), so a homography between
image pixels and court meters is computed with RANSAC. After that, any
image point (player feet, ball bounce) can be expressed in meters.

Court coordinate system (meters):
  x: 0 .. 10.97 across the court (doubles width)
  y: 0 .. 23.77 along the court (baseline to baseline)
  y = 0 is the far baseline (top of the broadcast image), y = 23.77 is
  the near baseline. The net is at y = 11.885.
"""

import cv2
import numpy as np
import torch

from .nets import load_court_net

# ITF court dimensions (meters)
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
COURT_LENGTH = 23.77
SINGLES_INSET = (DOUBLES_WIDTH - SINGLES_WIDTH) / 2.0  # 1.37 m
NET_Y = COURT_LENGTH / 2.0                              # 11.885 m
SERVICE_FROM_BASELINE = NET_Y - 6.40                    # 5.485 m

# The 14 model keypoints, in the exact order the network outputs them,
# expressed in court meters.
KEYPOINTS_M = np.array([
    (0.0, 0.0),                                  # 0 far-left doubles corner
    (DOUBLES_WIDTH, 0.0),                        # 1 far-right doubles corner
    (0.0, COURT_LENGTH),                         # 2 near-left doubles corner
    (DOUBLES_WIDTH, COURT_LENGTH),               # 3 near-right doubles corner
    (SINGLES_INSET, 0.0),                        # 4 far singles, left
    (SINGLES_INSET, COURT_LENGTH),               # 5 near singles, left
    (DOUBLES_WIDTH - SINGLES_INSET, 0.0),        # 6 far singles, right
    (DOUBLES_WIDTH - SINGLES_INSET, COURT_LENGTH),  # 7 near singles, right
    (SINGLES_INSET, SERVICE_FROM_BASELINE),               # 8 far service line, left
    (DOUBLES_WIDTH - SINGLES_INSET, SERVICE_FROM_BASELINE),  # 9 far service line, right
    (SINGLES_INSET, COURT_LENGTH - SERVICE_FROM_BASELINE),   # 10 near service line, left
    (DOUBLES_WIDTH - SINGLES_INSET, COURT_LENGTH - SERVICE_FROM_BASELINE),  # 11 near service, right
    (DOUBLES_WIDTH / 2.0, SERVICE_FROM_BASELINE),            # 12 far center service
    (DOUBLES_WIDTH / 2.0, COURT_LENGTH - SERVICE_FROM_BASELINE),  # 13 near center service
], dtype=np.float64)

MODEL_W, MODEL_H = 640, 360


def _decode_keypoints(heatmaps):
    """heatmaps: (15, 360, 640) sigmoid probabilities.
    Returns list of (x, y, conf) in model pixels, or None per keypoint."""
    points = []
    for k in range(14):
        hm = heatmaps[k]
        peak = float(hm.max())
        if peak < 0.35:
            points.append(None)
            continue
        py, px = np.unravel_index(int(hm.argmax()), hm.shape)
        # sub-pixel centroid in a small window around the peak
        r = 6
        x0, x1 = max(px - r, 0), min(px + r + 1, hm.shape[1])
        y0, y1 = max(py - r, 0), min(py + r + 1, hm.shape[0])
        win = hm[y0:y1, x0:x1]
        w = np.clip(win - 0.5 * peak, 0, None)
        if w.sum() <= 0:
            cx, cy = float(px), float(py)
        else:
            ys, xs = np.mgrid[y0:y1, x0:x1]
            cx = float((xs * w).sum() / w.sum())
            cy = float((ys * w).sum() / w.sum())
        points.append((cx, cy, peak))
    return points


class CourtCalibrator:
    """Detects court keypoints on sampled frames and builds the
    image->meters homography (median of per-frame detections)."""

    def __init__(self, weights_path, device="cpu"):
        self.model = load_court_net(weights_path, device)
        self.device = device
        self.H = None              # 3x3 homography, image px -> court meters
        self.keypoints_px = None   # median keypoints in original image pixels
        self.calib_error_cm = None  # median keypoint residual after fit, cm
        self.calib_inliers = 0

    def _predict_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, (MODEL_W, MODEL_H)).astype(np.float32) / 255.0
        inp = torch.from_numpy(np.rollaxis(img, 2, 0)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(inp)[0]
        heat = torch.sigmoid(out).cpu().numpy()
        pts = _decode_keypoints(heat)
        scaled = []
        for p in pts:
            if p is None:
                scaled.append(None)
            else:
                scaled.append((p[0] * w / MODEL_W, p[1] * h / MODEL_H, p[2]))
        return scaled

    def calibrate(self, frames, sample_every=25):
        """frames: list of BGR frames. Returns True if calibration worked."""
        collected = [[] for _ in range(14)]
        idxs = list(range(0, len(frames), sample_every))
        for i in idxs:
            pts = self._predict_frame(frames[i])
            for k, p in enumerate(pts):
                if p is not None:
                    collected[k].append(p)
        median_pts = []
        for k in range(14):
            if len(collected[k]) >= max(2, len(idxs) // 3):
                arr = np.array(collected[k])
                median_pts.append((float(np.median(arr[:, 0])),
                                   float(np.median(arr[:, 1])),
                                   float(np.median(arr[:, 2]))))
            else:
                median_pts.append(None)
        self.keypoints_px = median_pts

        img_pts, m_pts = [], []
        for k, p in enumerate(median_pts):
            if p is not None:
                img_pts.append([p[0], p[1]])
                m_pts.append(KEYPOINTS_M[k])
        if len(img_pts) < 6:
            return False
        img_pts = np.array(img_pts, dtype=np.float64)
        m_pts = np.array(m_pts, dtype=np.float64)
        # NOTE: destination units are meters, so the RANSAC threshold
        # is in meters too (0.25 m), not pixels.
        H, inliers = cv2.findHomography(img_pts, m_pts, cv2.RANSAC, 0.25)
        if H is None:
            return False
        self.H = H
        # honest calibration quality: median distance (in cm) between
        # the detected keypoints projected to meters and their true
        # court positions
        proj_m = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        err_m = np.linalg.norm(proj_m - m_pts, axis=1)
        self.reproj_error_px = float(np.median(err_m) * 100.0)  # cm, see below
        self.calib_error_cm = float(np.median(err_m) * 100.0)
        self.calib_inliers = int(inliers.sum()) if inliers is not None else len(img_pts)
        return True

    def to_court(self, points_xy):
        """points_xy: (N,2) image pixels -> (N,2) court meters.
        Entries that are None stay None."""
        out = []
        arr = np.array([[p[0], p[1]] if p is not None else [np.nan, np.nan]
                        for p in points_xy], dtype=np.float64)
        valid = ~np.isnan(arr[:, 0])
        res = np.full_like(arr, np.nan)
        if valid.any():
            res[valid] = cv2.perspectiveTransform(
                arr[valid].reshape(-1, 1, 2), self.H).reshape(-1, 2)
        for i, p in enumerate(points_xy):
            out.append(None if p is None else (float(res[i, 0]), float(res[i, 1])))
        return out

    def to_image(self, points_m):
        Hinv = np.linalg.inv(self.H)
        arr = np.array(points_m, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(arr, Hinv).reshape(-1, 2)


def call_for_bounce(x_m, y_m):
    """In/out decision for a bounce at court coords (x_m, y_m), singles.
    Returns (call, margin_cm, line_name). Margin is positive when the
    ball is in (distance to the nearest line) and negative when out
    (distance past the nearest line)."""
    left = SINGLES_INSET
    right = DOUBLES_WIDTH - SINGLES_INSET
    dx_out = max(left - x_m, x_m - right, 0.0)     # outside sideways
    dy_out = max(0.0 - y_m, y_m - COURT_LENGTH, 0.0)  # outside long
    if dx_out == 0.0 and dy_out == 0.0:
        margins = {
            "singles sideline": min(x_m - left, right - x_m),
            "baseline": min(y_m - 0.0, COURT_LENGTH - y_m),
        }
        line = min(margins, key=margins.get)
        return "IN", int(round(margins[line] * 100)), line
    if dx_out > 0 and dy_out > 0:
        dist = float(np.hypot(dx_out, dy_out))
        line = "corner"
    elif dx_out >= dy_out:
        dist = dx_out
        line = "singles sideline"
    else:
        dist = dy_out
        line = "baseline"
    return "OUT", -int(round(dist * 100)), line
