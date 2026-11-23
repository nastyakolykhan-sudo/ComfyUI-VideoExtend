"""How fast a clip moves at its two edges (its first and last second), measured with optical flow and put into words
a prompt writer can use: an LLM that watches a clip at about one frame per second cannot see speed, so prompts it
writes tend to let the motion wind down after a join. No ComfyUI imports; runs offline.

Speeds are in frame widths per second, so they mean the same at any size and frame rate:
  camera  = the frame's overall movement (median flow), and its zoom (scale change of a similarity fit)
  subject = how fast things move against that (95th percentile of the flow minus the camera's)
"""
import numpy as np

try:
    import cv2
except ImportError:  # the build has no OpenCV: no measurement, the prompt writer works without it
    cv2 = None

try:
    from . import extend_core as ec
except ImportError:  # run as a plain module (offline tests)
    import extend_core as ec

WORK_W = 320  # flow is measured at this width (speed is relative to the width, so the size hardly matters)
SUBJECT_WORDS = ((0.04, "still"), (0.15, "slow"), (0.40, "moderate"), (0.80, "fast"), (9e9, "very fast"))
CAMERA_WORDS = ((0.03, "still"), (0.06, "slow"), (0.20, "moderate"), (0.45, "fast"), (9e9, "very fast"))
ZOOM_STILL = 0.03  # scale change per second below this is no zoom


def _word(v, table):
    for limit, w in table:
        if v < limit:
            return w
    return table[-1][1]


def _gray(a, w):
    h = max(8, int(round(a.shape[0] * w / float(a.shape[1]))))
    return cv2.resize(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (w, h), interpolation=cv2.INTER_AREA)


def _steps(frames):
    """Per frame step: (dx, dy, zoom, subject), dx/dy/subject in frame widths, zoom as scale change (1.0 = none)."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = []
    for a, b in zip(frames, frames[1:]):
        f = dis.calc(a, b, None)
        h, w = a.shape
        ys, xs = np.mgrid[0:h:4, 0:w:4]
        u, v = f[ys, xs, 0].ravel(), f[ys, xs, 1].ravel()
        x, y = (xs.ravel() - w / 2.0), (ys.ravel() - h / 2.0)
        # similarity fit u = (s-1) x - r y + tx, v = r x + (s-1) y + ty  (least squares)
        A = np.zeros((2 * x.size, 4), np.float32)
        A[0::2] = np.stack([x, -y, np.ones_like(x), np.zeros_like(x)], 1)
        A[1::2] = np.stack([y, x, np.zeros_like(x), np.ones_like(x)], 1)
        rhs = np.empty(2 * x.size, np.float32); rhs[0::2], rhs[1::2] = u, v
        p = np.linalg.lstsq(A, rhs, rcond=None)[0]
        cam_u = p[0] * x - p[1] * y + p[2]
        cam_v = p[1] * x + p[0] * y + p[3]
        res = np.hypot(u - cam_u, v - cam_v)
        out.append((p[2] / w, p[3] / w, 1.0 + p[0], float(np.percentile(res, 95)) / w))
    return np.array(out, np.float64)


def _describe(st, fps):
    """A dict of measured speeds and a sentence for one edge."""
    dx, dy = np.median(st[:, 0]) * fps, np.median(st[:, 1]) * fps          # widths per second, content motion
    cam = float(np.hypot(dx, dy))
    zoom = float(np.prod(st[:, 2]) ** (fps / float(len(st))) - 1.0)       # scale change per second
    subj = float(np.median(st[:, 3]) * fps)
    cw, sw = _word(cam, CAMERA_WORDS), _word(subj, SUBJECT_WORDS)
    parts = []
    if cw == "still" and abs(zoom) < ZOOM_STILL:
        parts.append("the camera is still")
    else:
        if cw != "still":
            # content moving left means the camera turns or travels right
            horiz = "to the right" if dx < 0 else "to the left"
            vert = "downward" if dy < 0 else "upward"
            d = horiz if abs(dx) >= 2 * abs(dy) else vert if abs(dy) >= 2 * abs(dx) else f"{vert} and {horiz}"
            parts.append(f"the camera moves {d} at a {cw} pace ({cam:.2f} frame widths per second)")
        if abs(zoom) >= ZOOM_STILL:
            parts.append("pushing in" if zoom > 0 else "pulling back")
    parts.append(f"the motion in the frame is {sw} ({subj:.2f} frame widths per second)")
    return {"camera": round(cam, 3), "camera_word": cw, "zoom_per_s": round(zoom, 3), "subject": round(subj, 3),
            "subject_word": sw, "text": ", ".join(parts)}


def edge_motion(path, seconds=1.0):
    """Speeds over the clip's first and last `seconds`: {"start": {...}, "end": {...}, "text": "..."}; the text is
    empty when OpenCV is missing or the clip is too short to measure."""
    if cv2 is None:
        return {"start": None, "end": None, "text": ""}
    p = ec.probe(path)
    fps, n = float(p["fps"]), p["frames"]
    k = max(2, min(n, int(round(seconds * fps)) + 1))
    head, tail = [], []
    for i, a in enumerate(ec.iter_frames(path)):
        g = _gray(a, WORK_W)
        if i < k:
            head.append(g)
        if i >= n - k:
            tail.append(g)
    tail = tail[-k:]
    if len(head) < 2 or len(tail) < 2:
        return {"start": None, "end": None, "text": ""}
    s, e = _describe(_steps(head), fps), _describe(_steps(tail), fps)
    text = ("Measured motion (the source's own speed at each edge): in its first %.1f s %s; in its last %.1f s %s."
            % (seconds, s["text"], seconds, e["text"]))
    return {"start": s, "end": e, "text": text}


STILL_CAM, STILL_SUBJ = 0.03, 0.04   # frame widths per second below which a value counts as still


def window_speed(path, i0, i1, cover=None):
    """Camera speed and in-frame motion (frame widths per second) over raw frames i0 .. i1-1 of a file, or None.
    cover=(W, H): measure the frames as they will be shown, scaled to cover W x H and centre-cropped (a part made
    at another shape or padded with bars)."""
    if cv2 is None:
        return None
    p = ec.probe(path)
    fr = []
    for i, a in enumerate(ec.iter_frames(path)):
        if i >= i1:
            break
        if i >= i0:
            if cover is not None:
                W, H = cover
                sc = max(W / float(a.shape[1]), H / float(a.shape[0]))
                cw, ch = int(round(W / sc)), int(round(H / sc))
                x, y = (a.shape[1] - cw) // 2, (a.shape[0] - ch) // 2
                a = a[y:y + ch, x:x + cw]
            fr.append(_gray(a, WORK_W))
    if len(fr) < 4:
        return None
    st, fps = _steps(fr), float(p["fps"])
    return {"camera": float(np.hypot(*np.median(st[:, :2], axis=0)) * fps), "subject": float(np.median(st[:, 3]) * fps)}


def speed_ratio(src, gen):
    """How much faster the source moves than the generated footage (geometric mean over camera and in-frame motion,
    each only when the generated footage has that motion: a still camera can't be sped up into a moving one, that is
    a content difference, not a speed one); 1.0 when nothing can be compared."""
    if not src or not gen:
        return 1.0
    rs = []
    for k, still in (("camera", STILL_CAM), ("subject", STILL_SUBJ)):
        if gen[k] >= still:
            rs.append(max(src[k], still / 2) / gen[k])
    return float(np.exp(np.mean(np.log(rs)))) if rs else 1.0
