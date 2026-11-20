"""Join smoothing for ComfyUI-VideoExtend: makes the place where a generated part meets the source less visible,
changing only the generated part (the source's frames are never touched).

At each join, on the generated side, starting next to the join:
  1. repeats: generated frames in the first four that (nearly) repeat the frame before them are dropped (a stutter);
  2. camera: the generated part's camera motion (scale + shift, fitted on the background) is ramped in from the
     source's over 1 s, including the step across the join, then the correction relaxes over 3 s; it only ever zooms
     in, so no empty borders appear;
  3. speed: when the generated motion is slower or faster than the source's, the frames next to the join are
     re-timed so the speed ramps from the source's to the part's own (optical-flow in-between frames);
  4. align: the first half second is pulled toward the source's edge frame by a fading part of the optical flow,
     only where the flow is consistent both ways;
  5. colour: the low-frequency colour difference at the join is added to the generated part: its overall part is
     kept for the whole part, the local remainder fades out over 1 s.
Steps 2-4 make new pixels from optical flow, so they switch themselves off when the motion is too fast for it
(tracking shots, fast action): average motion above 16 px/frame or the fastest 5% above 40 px/frame (1920 wide).
They need OpenCV; without it only steps 1 and 5 run.
"""
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # the build has no OpenCV: colour and repeats only
    cv2 = None

FADE_S = 1.0          # colour: local remainder fades over this
EDGE = 3              # frames averaged on each side for the colour offset
CLAMP = 16.0          # largest colour correction, levels
DUP = 0.25            # repeat: mean change below this share of the clip's usual frame change
LOOK = 4              # repeats are looked for in the first frames only
EASE_S = 0.75         # speed ramp length
EASE_MIN = 1.15       # only re-time when the speeds differ by more than this
RATIO_MAX = 1.5
ALIGN = 0.8           # share of the flow pulled at the join
ALIGN_FADE_S = 0.5
ALIGN_TOL = 1.5       # px of forward-backward flow mismatch tolerated
ALIGN_MIN_TRUST = 0.75
ALIGN_MAX_PX = 8.0
MAX_SPEED = 16.0      # px/frame at 1920 wide
MAX_PEAK = 40.0
CAM_S = 1.0
CAM_RELAX_S = 3.0
CAM_CAP_SCALE = 0.05
CAM_CAP_PX = 48.0
CAM_MIN_SCALE = 0.98   # a zoom-out correction (the part framed tighter than the source) may go down to this; the
                       # thin edge it opens is filled by mirroring and closes as the correction relaxes
CAM_BORDER = 0.012     # share of the frame that may open at an edge (mirrored) before a zoom-in margin is used
CAM_MARGIN_STEP = 0.001  # the anti-border zoom grows by at most 0.1% per frame (no zoom pop at the join)
SHARPEN_MIN = 1.08    # detail: sharpen the generated part when the source has this much more fine detail
SHARPEN_MAX = 0.5     # largest unsharp-mask amount
WINDOW_S = CAM_S + CAM_RELAX_S + 0.5   # generated frames held per join


def window_frames(fps):
    return int(round(WINDOW_S * float(fps))) + 8


def cos_w(d, n):
    return 0.0 if d >= n else 0.5 * (1 + np.cos(np.pi * d / n))


def small_gray(f, w=160):
    h = max(1, int(round(f.shape[0] * w / f.shape[1])))
    im = Image.fromarray(f).convert("L").resize((w, h), Image.BOX)
    return np.asarray(im, np.float32)


def _blur(a, sigma):
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    p = np.pad(a, r, mode="edge")
    p = np.apply_along_axis(lambda x: np.convolve(x, k, "valid"), 1, p)
    return np.apply_along_axis(lambda x: np.convolve(x, k, "valid"), 0, p)


def lowfreq(f, size=(48, 27), sigma=2.0):
    """Low-frequency colour field of a frame (float), at the frame's size."""
    H, W = f.shape[:2]
    if cv2 is not None:
        s = cv2.resize(f.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        s = cv2.GaussianBlur(s, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)
    ch = []
    for c in range(3):
        im = Image.fromarray(np.ascontiguousarray(f[..., c], dtype=np.float32), mode="F")
        s = _blur(np.asarray(im.resize(size, Image.BOX), np.float32), sigma)
        ch.append(np.asarray(Image.fromarray(np.ascontiguousarray(s, dtype=np.float32), mode="F")
                             .resize((W, H), Image.BILINEAR), np.float32))
    return np.stack(ch, 2)


class Flow:
    """DIS optical flow at up to 960 px wide."""
    def __init__(self, W, H, work=960):
        self.W, self.H = W, H
        self.w = min(work, W); self.h = max(8, int(round(H * self.w / W)))
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def _g(self, a):
        return cv2.resize(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (self.w, self.h), interpolation=cv2.INTER_AREA)

    def __call__(self, a, b):
        return self.dis.calc(self._g(a), self._g(b), None)

    def mag(self, a, b):
        """(mean, 95th percentile) motion, px/frame scaled to a 1920-wide frame."""
        m = np.linalg.norm(self(a, b), axis=2) * (1920.0 / self.w)
        return float(m.mean()), float(np.percentile(m, 95))

    def between(self, a, b, t):
        if t < 0.02:
            return a
        if t > 0.98:
            return b
        s = self.W / self.w
        fab = cv2.resize(self(a, b), (self.W, self.H)) * s
        fba = cv2.resize(self(b, a), (self.W, self.H)) * s
        t = np.float32(t)
        gx, gy = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32))
        wa = cv2.remap(a, gx - t * fab[..., 0], gy - t * fab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(b, gx - (1 - t) * fba[..., 0], gy - (1 - t) * fba[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        return np.clip((1 - t) * wa.astype(np.float32) + t * wb.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)


_ORB = None


def cam_step(a, b, min_pts=80):
    """Camera motion a -> b about the frame centre: (log scale, dx, dy) in full-size px, or None."""
    global _ORB
    if _ORB is None:
        _ORB = (cv2.ORB_create(3000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True))
    orb, bf = _ORB
    W = a.shape[1]; f = min(1.0, 960.0 / W)
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ka, da = orb.detectAndCompute(ga, None); kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None:
        return None
    m = bf.match(da, db)
    if len(m) < min_pts:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=4000)
    if M is None or inl is None or inl.sum() < min_pts:
        return None
    sc = float(np.hypot(M[0, 0], M[1, 0]))
    cx, cy = ga.shape[1] / 2.0, ga.shape[0] / 2.0
    dx = M[0, 2] + M[0, 0] * cx + M[0, 1] * cy - cx
    dy = M[1, 2] + M[1, 0] * cx + M[1, 1] * cy - cy
    return np.array([np.log(sc), dx / f, dy / f])


def _camera(fr, ref, fps, info):
    vs = [cam_step(ref[j + 1], ref[j]) for j in range(min(4, len(ref) - 1))]
    vs = [v for v in vs if v is not None]
    if len(vs) < 2:
        info["camera"] = "skipped (no reliable background)"
        return fr
    v_s = np.median(vs, 0)
    N = max(4, int(round(CAM_S * fps))); Rx = int(round(CAM_RELAX_S * fps))
    steps = [cam_step(ref[0], fr[0])] + [cam_step(fr[k], fr[k + 1]) for k in range(min(N, len(fr) - 1))]
    if sum(v is None for v in steps) > len(steps) // 3:
        info["camera"] = "skipped (no reliable background)"
        return fr
    steps = [v if v is not None else v_s for v in steps]
    C, Cs = np.zeros(3), []
    lim = np.array([CAM_CAP_SCALE, CAM_CAP_PX, CAM_CAP_PX])
    for k, v_g in enumerate(steps):
        C = np.clip(C + cos_w(k, N) * (v_s - v_g), -lim, lim)
        Cs.append(C.copy())
    H, W = fr[0].shape[:2]
    out, m = [], 1.0
    for j, x in enumerate(fr):
        c = Cs[j] if j < len(Cs) else Cs[-1] * max(0.0, 1.0 - (j - len(Cs) + 1) / float(max(1, Rx)))
        if np.abs(c).max() < 1e-4 and m <= 1.0 + 1e-6:
            out.append(x); continue
        dx, dy = float(c[1]), float(c[2])
        S = max(float(np.exp(c[0])), CAM_MIN_SCALE)
        # an edge up to CAM_BORDER of the frame may open (filled by mirroring); beyond that a zoom-in margin covers
        # it, and that margin may only grow or shrink slowly, so nothing pops at the join
        bx, by = CAM_BORDER * W, CAM_BORDER * H
        ux = (1 - S) * W / 2 + abs(dx) - bx
        uy = (1 - S) * H / 2 + abs(dy) - by
        need = max(1.0, 1 + 2 * max(0.0, ux) / W, 1 + 2 * max(0.0, uy) / H)
        m = min(need, m + CAM_MARGIN_STEP) if need > m else max(need, m - CAM_MARGIN_STEP)
        S = S * m
        lim_x = max(0.0, (S - 1) * W / 2 + bx); lim_y = max(0.0, (S - 1) * H / 2 + by)
        dx = float(np.clip(dx, -lim_x, lim_x)); dy = float(np.clip(dy, -lim_y, lim_y))
        M = np.float32([[S, 0, (1 - S) * W / 2 + dx], [0, S, (1 - S) * H / 2 + dy]])
        out.append(cv2.warpAffine(x, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT))
    Ca = np.array(Cs)
    info["camera"] = {"max_zoom": round(float(np.exp(np.abs(Ca[:, 0]).max())), 4),
                      "max_shift_px": round(float(np.abs(Ca[:, 1:]).max()), 1)}
    return out


def fix_window(fr, ref, fps, motion_median):
    """fr: generated frames, index 0 next to the join (at most the window); ref: the source's frames, index 0 next
    to the join (a few). Returns (new frames, info, overall colour offset for the rest of the part)."""
    info = {}
    # 1. repeats
    keep, prev, dropped = [], ref[0], 0
    for j, f in enumerate(fr):
        if j < LOOK and np.abs(small_gray(f) - small_gray(prev)).mean() < DUP * motion_median:
            dropped += 1; continue
        keep.append(f); prev = f
    fr = keep
    info["repeats_dropped"] = dropped
    if cv2 is not None and len(fr) > 8 and len(ref) > 2:
        flow = Flow(fr[0].shape[1], fr[0].shape[0])
        mr = [flow.mag(ref[j + 1], ref[j]) for j in range(min(4, len(ref) - 1))]
        mg = [flow.mag(fr[j], fr[j + 1]) for j in range(min(4, len(fr) - 1))]
        mean = max(np.median([m[0] for m in mr]), np.median([m[0] for m in mg]))
        peak = max(m[1] for m in mr[:3] + mg[:3])
        fast = mean > MAX_SPEED or peak > MAX_PEAK
        info["motion_px"] = [round(float(mean), 1), round(float(peak), 1)]
        info["flow_steps"] = "off (fast motion)" if fast else "on"
        # 2. camera: fitted on the background, so only a fast-moving whole frame turns it off
        if mean <= MAX_SPEED:
            fr = _camera(fr, ref, fps, info)
        else:
            info["camera"] = "off (fast motion)"
        if not fast:
            # 3. speed
            vr = np.median([m[0] for m in mr]); vg = np.median([m[0] for m in mg])
            R = float(np.clip(vr / max(vg, 1e-3), 1 / RATIO_MAX, RATIO_MAX)) if vg > 0.3 else 1.0
            if max(R, 1 / R) > EASE_MIN:
                E = max(4, int(round(EASE_S * fps)))
                taus, t, d = [], 0.0, 0
                while t <= len(fr) - 1 + 1e-6:
                    taus.append(t); t += 1 + (R - 1) * cos_w(d, E); d += 1
                new = []
                for t in taus:
                    i = int(np.floor(t))
                    new.append(fr[i] if i + 1 >= len(fr) else flow.between(fr[i], fr[i + 1], t - i))
                info["speed"] = {"ratio": round(R, 2), "frames_change": len(new) - len(fr)}
                fr = new
            # 4. align
            Fw = flow(fr[0], ref[0]); Bw = flow(ref[0], fr[0])
            hw, ww = Fw.shape[:2]
            gxw, gyw = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hw, dtype=np.float32))
            Bat = cv2.remap(Bw, gxw + Fw[..., 0], gyw + Fw[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            err = np.linalg.norm(Fw + Bat, axis=2)
            conf = np.exp(-(err / ALIGN_TOL) ** 2).astype(np.float32)
            conf = cv2.GaussianBlur(cv2.erode(conf, np.ones((9, 9), np.uint8)), (0, 0), 6)
            F = cv2.resize(Fw * conf[..., None], (flow.W, flow.H)) * (flow.W / flow.w)
            magp = float(np.mean(np.linalg.norm(F, axis=2))) * 1920.0 / flow.W
            if conf.mean() >= ALIGN_MIN_TRUST and magp <= ALIGN_MAX_PX:
                Na = max(1, int(round(ALIGN_FADE_S * fps)))
                gx, gy = np.meshgrid(np.arange(flow.W, dtype=np.float32), np.arange(flow.H, dtype=np.float32))
                for j in range(min(Na, len(fr))):
                    wgt = np.float32(ALIGN * cos_w(j, Na))
                    fr[j] = cv2.remap(fr[j], gx - wgt * F[..., 0], gy - wgt * F[..., 1], cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
                info["align_px"] = round(magp, 2)
            else:
                info["align"] = "skipped (flow not reliable)"
    elif cv2 is None:
        info["flow_steps"] = "off (no OpenCV)"
    # 5. colour
    e = min(EDGE, len(fr), len(ref))
    D = lowfreq(np.mean([x.astype(np.float32) for x in ref[:e]], 0)) - \
        lowfreq(np.mean([x.astype(np.float32) for x in fr[:e]], 0))
    D = np.clip(D, -CLAMP, CLAMP)
    Dg = D.mean((0, 1)).astype(np.float32)
    N = max(1, int(round(FADE_S * fps)))
    fr = [np.clip(x.astype(np.float32) + Dg + (D - Dg) * cos_w(j, N) + 0.5, 0, 255).astype(np.uint8)
          for j, x in enumerate(fr)]
    info["colour_offset"] = [round(float(v), 2) for v in Dg]
    # 6. detail: a generated part softer than the source gets a matched unsharp mask (kept for the whole part)
    amt = 0.0
    if cv2 is not None:
        hs = np.mean([detail(x) for x in ref[:e]]); hg = np.mean([detail(x) for x in fr[:e]])
        r = hs / max(hg, 1e-3)
        if r > SHARPEN_MIN:
            amt = float(min(SHARPEN_MAX, r - 1.0))
            fr = [sharpen(x, amt) for x in fr]
        info["detail_ratio"] = round(float(r), 2)
    info["sharpen"] = round(amt, 2)
    return fr, info, Dg, amt


def detail(f):
    g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())


def sharpen(f, amt, sigma=1.2):
    if amt <= 0:
        return f
    x = f.astype(np.float32)
    return np.clip(x + amt * (x - cv2.GaussianBlur(x, (0, 0), sigma)) + 0.5, 0, 255).astype(np.uint8)
