"""Event detection: bounces, hits, shots, in/out calls.

Method (bounce-anchored, alternation-aware):
1. The ball track is despiked (median test) - TrackNet sometimes
   locks onto shoes or shirts for a few frames, creating impossible
   trajectory spikes.
2. BOUNCES are local lowest points of the ball in the image
   (image-y maxima). They are the most reliable events in the track.
3. Between two bounces there is one HIT (volleys aside). It is found
   by combining: court-plane direction reversal, image-plane
   direction reversal, and closest approach to the player who is due
   to hit (rally hits alternate between the two players). A serve is
   found by its speed burst near the server.
4. A second pass rejects "approach dips": y-maxima that are really
   the ball dipping to the receiver's racket (they project out of
   the court, sit next to the defender, and are followed at once by
   that defender's hit).
5. Shot speed comes from the smoothed court-plane trajectory right
   after the hit. It is an ESTIMATE: one camera cannot measure true
   ball speed.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .court import call_for_bounce, DOUBLES_WIDTH, COURT_LENGTH, NET_Y
from .ball import despike


def _interpolate(track, max_gap=12):
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
        if i > 0 and j < n and (j - i) <= max_gap:
            p0, p1 = t[i - 1], t[j]
            for k in range(i, j):
                a = (k - i + 1) / (j - i + 1)
                t[k] = (p0[0] + (p1[0] - p0[0]) * a, p0[1] + (p1[1] - p0[1]) * a)
        i = j
    return t


def _runs(track):
    n = len(track)
    i = 0
    while i < n:
        if track[i] is None:
            i += 1
            continue
        j = i
        while j < n and track[j] is not None:
            j += 1
        yield i, j
        i = j


def _smooth_series(ball_court, n):
    idx = [i for i, c in enumerate(ball_court) if c is not None]
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    if len(idx) < 12:
        return xs, ys
    xs[idx] = [ball_court[i][0] for i in idx]
    ys[idx] = [ball_court[i][1] for i in idx]
    full = np.arange(n)
    xs = np.interp(full, idx, xs[idx])
    ys = np.interp(full, idx, ys[idx])
    xs = savgol_filter(xs, 9, 3)
    ys = savgol_filter(ys, 9, 3)
    xs[:idx[0]] = np.nan
    xs[idx[-1] + 1:] = np.nan
    ys[:idx[0]] = np.nan
    ys[idx[-1] + 1:] = np.nan
    return xs, ys


def _angles(vx, vy, half=4):
    """Direction-change angle between velocity at i-half and i+half."""
    n = len(vx)
    ang = np.zeros(n)
    spd = np.hypot(vx, vy)
    for i in range(n):
        a, b = max(i - half, 0), min(i + half, n - 1)
        s1, s2 = spd[a], spd[b]
        if s1 > 0.5 and s2 > 0.5:
            cosang = (vx[a] * vx[b] + vy[a] * vy[b]) / (s1 * s2)
            ang[i] = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    return ang, spd


class _EventFinder:
    def __init__(self, ball_img_raw, calibrator, players, fps):
        self.fps = fps
        self.players = players
        self.n = len(ball_img_raw)
        self.ball_img = _interpolate(despike(ball_img_raw), max_gap=12)
        self.ball_court = calibrator.to_court(self.ball_img)
        n = self.n
        self.xs, self.ys = _smooth_series(self.ball_court, n)
        # court-plane velocity + reversal angles
        vx = np.zeros(n)
        vy = np.zeros(n)
        for i in range(n):
            a, b = max(i - 2, 0), min(i + 2, n - 1)
            vx[i] = (self.xs[b] - self.xs[a]) * fps / max(b - a, 1)
            vy[i] = (self.ys[b] - self.ys[a]) * fps / max(b - a, 1)
        self.ang_c, self.spd_c = _angles(vx, vy)
        # image-plane velocity + reversal angles
        ix = np.array([p[0] if p else np.nan for p in self.ball_img])
        iy = np.array([p[1] if p else np.nan for p in self.ball_img])
        ok = ~np.isnan(ix)
        if ok.sum() > 12:
            ix[ok] = savgol_filter(ix[ok], 7, 3)
            iy[ok] = savgol_filter(iy[ok], 7, 3)
        ivx = np.zeros(n)
        ivy = np.zeros(n)
        for i in range(n):
            a, b = max(i - 2, 0), min(i + 2, n - 1)
            ivx[i] = (ix[b] - ix[a]) * fps / max(b - a, 1)
            ivy[i] = (iy[b] - iy[a]) * fps / max(b - a, 1)
        self.ang_i, _ = _angles(ivx, ivy, half=3)
        self.ball_ix, self.ball_iy = ix, iy

    # ----- helpers -----
    def court_dist(self, i, p):
        pc = p["court"][i]
        if pc is None or np.isnan(self.xs[i]):
            return 99.0
        return float(np.hypot(pc[0] - self.xs[i], pc[1] - self.ys[i]))

    def img_dist(self, i, p):
        foot = p["foot_img"][i]
        if foot is None or np.isnan(self.ball_ix[i]):
            return 999.0
        return float(np.hypot(foot[0] - self.ball_ix[i], foot[1] - self.ball_iy[i]))

    def find_hit(self, lo, hi, expected=None, serve=False, bounce_y=None):
        """Best hit frame in [lo, hi]. expected = player index due to
        hit (alternation). bounce_y = court y of the bounce this hit
        must produce: a real hitter stands on the OPPOSITE side of
        the net from where his shot lands (this kills the phantom
        'hits' the projection creates right at a bounce).
        Returns (frame, score, hitter_index).
        In serve mode only a true serve contact counts: the first
        frame where the ball bursts to full speed next to the server
        (pre-serve hand bounces also reverse the ball, but slowly)."""
        best = (None, -1.0, None)
        for i in range(max(lo, 2), min(hi, self.n - 2)):
            if np.isnan(self.ys[i]):
                continue
            if bounce_y is not None and abs(self.ys[i] - NET_Y) > 0.6 and \
                    (self.ys[i] - NET_Y) * (bounce_y - NET_Y) > 0:
                continue  # same side as its bounce: not a real hit
            dc = [self.court_dist(i, p) for p in self.players]
            di = [self.img_dist(i, p) for p in self.players]
            near = int(np.argmin(dc))
            if dc[near] > 6.5:
                continue
            if serve:
                # serve contact = strongest speed burst next to the
                # server inside the baseline zone (the toss can fake
                # a weak burst; the real contact dwarfs it)
                burst = float(self.spd_c[min(i + 2, self.n - 1)])
                if burst > 12.0 and \
                        (self.ys[i] < 3.5 or self.ys[i] > COURT_LENGTH - 3.5) and \
                        burst > best[1]:
                    best = (i, burst, near)
                continue
            score = 0.0
            if self.spd_c[max(i - 4, 0)] > 3.0 and self.spd_c[min(i + 4, self.n - 1)] > 3.0:
                score += self.ang_c[i]
            score += 0.7 * self.ang_i[i]
            if di[near] < 70:
                score += 55
            elif di[near] < 120:
                score += 28
            if expected is not None:
                if near == expected:
                    score *= 1.35
                else:
                    score *= 0.75
            if score > best[1]:
                best = (i, score, near)
        if serve:
            # serve mode uses the burst scan only
            if best[0] is not None:
                return (best[0], 120.0, best[2])
            return best
        # closest-approach candidate: at contact the ball passes
        # within about a racket length of the hitter's feet (image)
        cands = [expected] if expected is not None else [0, 1]
        for pi in cands:
            seg = range(max(lo, 2), min(hi, self.n - 2))
            dmin, imin = 999.0, None
            for i in seg:
                d = self.img_dist(i, self.players[pi])
                if d < dmin:
                    dmin, imin = d, i
            if imin is not None and dmin < 65 and not np.isnan(self.ys[imin]):
                if bounce_y is None or abs(self.ys[imin] - NET_Y) <= 0.6 or \
                        (self.ys[imin] - NET_Y) * (bounce_y - NET_Y) < 0:
                    sc = 62.0 + 0.3 * self.ang_i[imin] + (65.0 - dmin) / 3.0
                    if sc > best[1]:
                        best = (imin, sc, pi)
        return best

    def find_bounces(self):
        cand = []
        for s, e in _runs(self.ball_img):
            if e - s < 9:
                continue
            ys_img = np.array([self.ball_img[i][1] for i in range(s, e)], dtype=float)
            ys_s = np.convolve(ys_img, np.ones(3) / 3.0, mode="same")
            peaks, _ = find_peaks(ys_s, prominence=2.2, distance=9)
            for p in peaks:
                idx = s + int(p)
                c = self.ball_court[idx]
                if c is not None and -4.0 <= c[0] <= DOUBLES_WIDTH + 4.0 \
                        and -4.0 <= c[1] <= COURT_LENGTH + 4.0:
                    cand.append(idx)
        return sorted(cand)

    def second_chance_bounces(self, shots):
        """A long flight (hit -> next event > 30 frames) must contain
        a bounce. Far-side bounces can be only ~2px tall in the image
        and miss the main prominence threshold, so re-scan those
        flights with a low threshold, on the receiver's side only."""
        extra = []
        for k, sh in enumerate(shots):
            hf = sh["hit_frame"]
            end = sh["bounce_frame"]
            if end is None and k + 1 < len(shots):
                end = shots[k + 1]["hit_frame"]
            if end is None:
                end = min(hf + 45, self.n - 1)
            if end - hf <= 30:
                continue
            lo, hi = hf + 9, min(end - 3, hf + 45)
            if hi - lo < 6:
                continue
            hit_c = self.ball_court[hf]
            if hit_c is None:
                continue
            seg = [i for i in range(lo, hi)
                   if self.ball_img[i] is not None and self.ball_court[i] is not None
                   and (self.ball_court[i][1] - NET_Y) * (hit_c[1] - NET_Y) < 0]
            if len(seg) < 6:
                continue
            ys_img = np.array([self.ball_img[i][1] for i in seg], dtype=float)
            peaks, props = find_peaks(ys_img, prominence=0.8, distance=8)
            if len(peaks):
                j = int(peaks[int(np.argmax(props["prominences"]))])
                extra.append(seg[j])
        return extra

    def assemble(self, bounce_cand):
        """Pair hits with bounces. Returns list of shot dicts with
        hit_frame / bounce_frame / hitter_index."""
        n = self.n
        shots = []
        pending = None  # (hit_frame, hitter_idx)
        last_b = 0
        expected = None
        bi = 0
        while bi < len(bounce_cand):
            b = bounce_cand[bi]
            if pending is None:
                h, sc, who = self.find_hit(last_b + 2 if last_b else 2, b - 3,
                                           expected=expected, serve=(last_b == 0),
                                           bounce_y=self.ball_court[b][1])
                if h is not None and sc > 60:
                    pending = (h, who)
            if pending is None:
                bi += 1
                continue
            if b - pending[0] < 9:
                # too soon after the hit to be its bounce: this is
                # the ball dipping just after contact - skip it
                bi += 1
                continue
            shots.append({"hit_frame": pending[0], "bounce_frame": b,
                          "hitter_index": pending[1]})
            expected = 1 - pending[1] if pending[1] is not None else None
            last_b = b
            pending = None
            # look for the next hit before some later bounce, skipping
            # spurious bounce candidates that have no hit before them
            j = bi + 1
            while j < len(bounce_cand):
                nxt = bounce_cand[j]
                h, sc, who = self.find_hit(b + 2, nxt - 3, expected=expected,
                                           bounce_y=self.ball_court[nxt][1])
                if h is not None and sc > 60:
                    pending = (h, who)
                    bi = j - 1  # next iteration closes at bounce_cand[j]
                    break
                j += 1
            else:
                bi = len(bounce_cand)
            bi += 1
        if pending is not None:
            shots.append({"hit_frame": pending[0], "bounce_frame": None,
                          "hitter_index": pending[1]})
        if bounce_cand:
            h, sc, who = self.find_hit(bounce_cand[-1] + 2, n - 3, expected=expected)
            if h is not None and sc > 60 and (not shots or shots[-1]["hit_frame"] != h):
                shots.append({"hit_frame": h, "bounce_frame": None,
                              "hitter_index": who})
        return shots

    def reject_dips(self, shots):
        """Drop shots whose 'bounce' is an approach dip: the bounce
        projects OUT, sits next to the defender, and that defender
        hits the ball within 12 frames."""
        drop = set()
        for k, sh in enumerate(shots[:-1]):
            bf = sh["bounce_frame"]
            if bf is None:
                continue
            nxt = shots[k + 1]
            if nxt["hit_frame"] - bf > 12:
                continue
            c = self.ball_court[bf]
            call, _, _ = call_for_bounce(c[0], c[1])
            if call != "OUT":
                continue
            defender = 0 if c[1] > NET_Y else 1  # near player defends y>net
            if nxt["hitter_index"] == defender and \
                    self.court_dist(bf, self.players[defender]) < 3.2:
                drop.add(bf)
        return drop


def detect_events(ball_img_raw, calibrator, players, fps):
    """Returns dict: shots, bounces, first_hit_frame, ball_track,
    ball_court (cleaned tracks used for rendering)."""
    ef = _EventFinder(ball_img_raw, calibrator, players, fps)
    cand = ef.find_bounces()
    shots = ef.assemble(cand)
    dips = ef.reject_dips(shots)
    if dips:
        cand = [b for b in cand if b not in dips]
        shots = ef.assemble(cand)
    extra = ef.second_chance_bounces(shots)
    if extra:
        cand = sorted(set(cand) | set(extra))
        shots = ef.assemble(cand)
        dips = ef.reject_dips(shots)
        if dips:
            cand = [b for b in cand if b not in dips]
            shots = ef.assemble(cand)

    n = ef.n
    out_shots = []
    for k, sh in enumerate(shots):
        hf = sh["hit_frame"]
        pos = (float(ef.xs[hf]), float(ef.ys[hf])) if not np.isnan(ef.xs[hf]) else None
        speeds = []
        for i in range(hf + 2, min(hf + 12, n - 1)):
            if not (np.isnan(ef.xs[i - 1]) or np.isnan(ef.xs[i + 1])):
                speeds.append(np.hypot(ef.xs[i + 1] - ef.xs[i - 1],
                                       ef.ys[i + 1] - ef.ys[i - 1]) * fps / 2.0)
        # typical speed shortly after the hit: mean of the 3 fastest
        # instantaneous speeds (peak-like, but spike-resistant)
        speed = float(np.mean(sorted(speeds)[-3:])) * 3.6 if speeds else None
        if speed is not None and (speed > 235 or speed < 8):
            speed = None
        if pos and (pos[1] < -4.0 or pos[1] > COURT_LENGTH + 4.0
                    or pos[0] < -4.0 or pos[0] > DOUBLES_WIDTH + 4.0):
            speed = None  # hit localized off-court: track artifact
        hitter = None
        if sh["hitter_index"] is not None:
            hitter = players[sh["hitter_index"]]["name"]
        rec = {
            "shot_no": k + 1,
            "type": "shot",
            "hitter": hitter,
            "hit_frame": hf,
            "hit_time_s": round(hf / fps, 2),
            "hit_court_m": [round(pos[0], 2), round(pos[1], 2)] if pos else None,
            "speed_kmh_est": round(speed, 1) if speed else None,
            "bounce_frame": sh["bounce_frame"],
            "bounce_court_m": None,
            "call": None,
            "margin_cm": None,
            "line": None,
        }
        if sh["bounce_frame"] is not None:
            c = ef.ball_court[sh["bounce_frame"]]
            call, margin, line = call_for_bounce(c[0], c[1])
            rec.update({"bounce_court_m": [round(c[0], 2), round(c[1], 2)],
                        "call": call, "margin_cm": margin, "line": line})
        if k == 0 and pos and (pos[1] < 3.3 or pos[1] > COURT_LENGTH - 3.3):
            rec["type"] = "serve"
        out_shots.append(rec)

    bounce_recs = [{"frame": sh["bounce_frame"],
                    "court": [float(ef.ball_court[sh["bounce_frame"]][0]),
                              float(ef.ball_court[sh["bounce_frame"]][1])],
                    "call": sh["call"]}
                   for sh in out_shots if sh["bounce_frame"] is not None]
    return {
        "shots": out_shots,
        "bounces": bounce_recs,
        "first_hit_frame": out_shots[0]["hit_frame"] if out_shots else None,
        "ball_track": ef.ball_img,
        "ball_court": ef.ball_court,
    }
