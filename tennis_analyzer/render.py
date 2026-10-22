"""Annotated video rendering: ball trail, player boxes, HUD panels,
last-call display and the 2D mini-court inset."""

import cv2
import numpy as np

from .court import (DOUBLES_WIDTH, COURT_LENGTH, SINGLES_INSET, NET_Y,
                    SERVICE_FROM_BASELINE)

# trail styles: list of BGR colors from head (current) to tail
TRAIL_STYLES = {
    "neon": [(20, 255, 57)] * 14,
    "fire": [(0, 165, 255), (0, 120, 255), (0, 60, 255), (0, 0, 255)] * 4,
    "ice": [(255, 255, 255), (255, 240, 180), (255, 210, 120), (255, 180, 60)] * 4,
}

PLAYER_COLORS = [(255, 220, 0), (255, 0, 200)]  # BGR: cyan-ish, magenta


def _panel(img, x, y, w, h, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _text(img, txt, x, y, scale=0.62, color=(255, 255, 255), thick=2):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


class MiniCourt:
    """Top-view 2D court. Court meters -> mini pixels with margins."""

    def __init__(self, px_per_m=11):
        self.ppm = px_per_m
        self.mx, self.my = 1.9, 1.9  # meter margins
        self.W = int((DOUBLES_WIDTH + 2 * self.mx) * self.ppm)
        self.H = int((COURT_LENGTH + 2 * self.my) * self.ppm)
        self.base = self._draw_base()

    def mp(self, x_m, y_m):
        return (int((x_m + self.mx) * self.ppm), int((y_m + self.my) * self.ppm))

    def _draw_base(self):
        img = np.zeros((self.H, self.W, 3), np.uint8)
        img[:] = (25, 60, 25)  # dark grass green
        inner = np.zeros_like(img)
        inner[:] = (40, 110, 45)
        p0 = self.mp(-1.2, -1.2)
        p1 = self.mp(DOUBLES_WIDTH + 1.2, COURT_LENGTH + 1.2)
        cv2.rectangle(inner, p0, p1, (40, 110, 45), -1)
        img = inner
        w = (240, 240, 240)
        # doubles box
        cv2.rectangle(img, self.mp(0, 0), self.mp(DOUBLES_WIDTH, COURT_LENGTH), w, 2)
        # singles lines
        cv2.line(img, self.mp(SINGLES_INSET, 0), self.mp(SINGLES_INSET, COURT_LENGTH), w, 1)
        cv2.line(img, self.mp(DOUBLES_WIDTH - SINGLES_INSET, 0),
                 self.mp(DOUBLES_WIDTH - SINGLES_INSET, COURT_LENGTH), w, 1)
        # service lines + center service line
        for yy in (SERVICE_FROM_BASELINE, COURT_LENGTH - SERVICE_FROM_BASELINE):
            cv2.line(img, self.mp(SINGLES_INSET, yy),
                     self.mp(DOUBLES_WIDTH - SINGLES_INSET, yy), w, 1)
        cv2.line(img, self.mp(DOUBLES_WIDTH / 2, SERVICE_FROM_BASELINE),
                 self.mp(DOUBLES_WIDTH / 2, COURT_LENGTH - SERVICE_FROM_BASELINE), w, 1)
        # net
        cv2.line(img, self.mp(-0.6, NET_Y), self.mp(DOUBLES_WIDTH + 0.6, NET_Y),
                 (255, 255, 255), 3)
        return img

    def draw(self, players_court, bounces, ball_court):
        img = self.base.copy()
        for b in bounces:
            x, y = b["court"]
            c = (0, 220, 0) if b["call"] == "IN" else (0, 0, 255)
            cv2.circle(img, self.mp(x, y), 5, c, -1)
            cv2.circle(img, self.mp(x, y), 5, (255, 255, 255), 1)
        for i, pc in enumerate(players_court):
            if pc is not None:
                cv2.circle(img, self.mp(pc[0], pc[1]), 6, PLAYER_COLORS[i], -1)
                cv2.circle(img, self.mp(pc[0], pc[1]), 6, (255, 255, 255), 1)
        if ball_court is not None:
            cv2.circle(img, self.mp(ball_court[0], ball_court[1]), 3, (0, 255, 255), -1)
        return img


class Renderer:
    def __init__(self, trail="neon", scale=2):
        self.trail_colors = TRAIL_STYLES.get(trail, TRAIL_STYLES["neon"])
        self.trail_name = trail
        self.s = scale
        self.mini = MiniCourt()

    def render(self, frame, idx, fps, ball_track, ball_court_track,
               players, events, trail_hist):
        s = self.s
        img = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        H, W = img.shape[:2]

        # ---- mini court, top-right ----
        mini = self.mini.draw([p["court"][idx] for p in players],
                              events["bounces_so_far"][idx],
                              ball_court_track[idx])
        mh, mw = mini.shape[:2]
        mx0, my0 = W - mw - 12, 10
        img[my0:my0 + mh, mx0:mx0 + mw] = mini
        cv2.rectangle(img, (mx0, my0), (mx0 + mw, my0 + mh), (255, 255, 255), 1)
        _text(img, "COURT VIEW", mx0 + 6, my0 + 18, 0.5)

        # ---- HUD top-left ----
        fh = events["first_hit_frame"]
        rally_t = max(0.0, (idx - fh) / fps) if fh is not None and idx >= fh else 0.0
        cur = events["current_shot"][idx]
        _panel(img, 10, 10, 330, 108)
        _text(img, "TENNIS ANALYZER", 22, 38, 0.66, (0, 255, 200))
        _text(img, f"RALLY  {rally_t:05.1f}s", 22, 66, 0.62)
        shot_txt = f"SHOT {cur['shot_no']}" if cur else "SHOT -"
        spd = cur.get("speed_kmh_est") if cur else None
        _text(img, f"{shot_txt}   {spd:.0f} km/h est." if spd else shot_txt, 22, 94, 0.62,
              (0, 255, 255) if spd else (255, 255, 255))

        # ---- last call panel, bottom-left ----
        last = events["last_call"][idx]
        if last is not None:
            col = (0, 200, 0) if last["call"] == "IN" else (0, 0, 220)
            _panel(img, 10, H - 118, 430, 66)
            cv2.rectangle(img, (10, H - 118), (440, H - 52), col, 3)
            sign = "+" if last["margin_cm"] >= 0 else ""
            _text(img, f"LAST CALL: {last['call']} {sign}{last['margin_cm']} cm",
                  24, H - 88, 0.72, col if last["call"] == "OUT" else (80, 255, 80))
            _text(img, f"({last['line']})", 24, H - 62, 0.5, (200, 200, 200))

        # ---- bounce flash marker on main view ----
        bf = events["bounce_flash"].get(idx)
        if bf is not None:
            bx, by = int(bf[0] * s), int(bf[1] * s)
            col = (0, 255, 0) if bf[2] == "IN" else (0, 0, 255)
            cv2.circle(img, (bx, by), 14, col, 3)
            cv2.circle(img, (bx, by), 4, col, -1)

        # ---- players ----
        for pi, p in enumerate(players):
            box = p["box"][idx]
            if box is None:
                continue
            x1, y1, x2, y2 = [int(v * s) for v in box]
            col = PLAYER_COLORS[pi]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            label = f"P{pi + 1}  {p['speed_kmh'][idx]:.0f} km/h"
            _panel(img, x1, max(y1 - 26, 0), 150, 24, 0.45)
            _text(img, label, x1 + 5, max(y1 - 8, 16), 0.55, col)

        # ---- player totals strip, bottom-right ----
        totals = "   |   ".join(
            f"P{i + 1}: {p['distance_m']:.0f} m, top {p['top_speed_kmh']:.0f} km/h"
            for i, p in enumerate(players))
        _panel(img, W - 620, H - 44, 608, 34)
        _text(img, totals, W - 608, H - 20, 0.52)

        # ---- ball trail ----
        hist = trail_hist[idx]
        pts = [(int(x * s), int(y * s)) for (x, y) in hist]
        for j in range(len(pts) - 1):
            c = self.trail_colors[min(j, len(self.trail_colors) - 1)]
            thick = max(2, 7 - j // 2)
            cv2.line(img, pts[j], pts[j + 1], c, thick, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[0], 6, (255, 255, 255), -1)
            cv2.circle(img, pts[0], 6, self.trail_colors[0], 2)
        return img
