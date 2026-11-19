"""Core of ComfyUI-VideoExtend: pure functions over video files (PyAV + numpy + Pillow), no ComfyUI imports, so they
run and test offline. Frames are streamed; only a reversed part is held in memory (at the generated size, uint8).

Terms: the *source* is the clip being extended; *parts* are generated clips that go before it (start) or after it
(end). The join keeps the source's own frames, size, frame rate and sound, and fits every part to them.
"""
import json, math, os
from fractions import Fraction

import av
import numpy as np
from PIL import Image

AUDIO_RATE = 48000


# ------------------------------------------------------------------------------------------------------- reading

def probe(path):
    """Size, frame rate, frame count, duration and whether there is audio, without decoding frames."""
    with av.open(path) as c:
        if not c.streams.video:
            raise ValueError(f"No video stream in {os.path.basename(str(path))}")
        vs = c.streams.video[0]
        fps = Fraction(vs.average_rate or vs.guessed_rate or 24).limit_denominator(100_000)
        n = 0
        for p in c.demux(vs):
            if p.pts is not None and p.size and p.pts >= 0:
                n += 1
        dur = n / float(fps)
        return {"width": vs.codec_context.width, "height": vs.codec_context.height, "fps": fps, "frames": n,
                "duration": dur, "audio": bool(c.streams.audio)}


try:
    from av.video.reformatter import Colorspace as _CS, ColorRange as _CR
except ImportError:  # older PyAV
    _CS = _CR = None


def _to_rgb(f):
    """Decode to RGB with the frame's own colour tags; untagged HD is read as BT.709 and untagged range as limited
    (what players do), so a clip written by this pack, or by most tools, comes back with the same levels."""
    if _CS is None:
        return f.to_ndarray(format="rgb24")
    cs = int(getattr(f, "colorspace", 2) or 2)
    rng = int(getattr(f, "color_range", 0) or 0)
    src_cs = cs if cs not in (2, 3) else (_CS.ITU709 if f.height >= 720 else _CS.ITU601)
    src_rng = rng if rng in (1, 2) else _CR.MPEG
    try:
        return f.reformat(format="rgb24", src_colorspace=src_cs, src_color_range=src_rng,
                          dst_color_range=_CR.JPEG).to_ndarray()
    except (TypeError, ValueError):
        return f.to_ndarray(format="rgb24")


def iter_frames(path, fmt="rgb24"):
    with av.open(path) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        for f in c.decode(vs):
            yield _to_rgb(f) if fmt == "rgb24" else f.to_ndarray(format=fmt)


def edge_frames(path):
    """(first, last) frames as uint8 HxWx3 arrays, decoding the clip once without keeping it."""
    first = last = None
    for a in iter_frames(path):
        if first is None:
            first = a
        last = a
    if first is None:
        raise ValueError(f"No frames in {os.path.basename(str(path))}")
    return first, last


def read_audio(path, start=0.0, duration=None):
    """Audio as float32 (2, samples) at 48 kHz, from `start` seconds for `duration` seconds; None when silent."""
    with av.open(path) as c:
        if not c.streams.audio:
            return None
        a = c.streams.audio[0]
        res = av.AudioResampler(format="fltp", layout="stereo", rate=AUDIO_RATE)
        chunks = []
        for fr in c.decode(a):
            for r in res.resample(fr):
                chunks.append(r.to_ndarray())
        for r in res.resample(None):
            chunks.append(r.to_ndarray())
    if not chunks:
        return None
    wave = np.concatenate(chunks, axis=1).astype(np.float32)
    s = int(round(start * AUDIO_RATE))
    wave = wave[:, s:]
    if duration is not None:
        wave = _fit_len(wave, int(round(duration * AUDIO_RATE)))
    return wave


def _fit_len(wave, n):
    if wave.shape[1] >= n:
        return wave[:, :n]
    return np.pad(wave, ((0, 0), (0, n - wave.shape[1])))


# ------------------------------------------------------------------------------------------------------- writing

def _open_writer(path, fps, width, height, crf=12, preset="fast", audio=False):
    out = av.open(path, mode="w")
    vs = out.add_stream("libx264", rate=fps, options={"crf": str(crf), "preset": preset})
    vs.width, vs.height, vs.pix_fmt = width, height, "yuv420p"
    vs.codec_context.time_base = Fraction(1) / fps
    for k, v in (("colorspace", 1), ("color_primaries", 1), ("color_trc", 1), ("color_range", 1)):  # BT.709, limited
        try:
            setattr(vs.codec_context, k, v)
        except (AttributeError, TypeError, ValueError):
            pass
    oa = out.add_stream("aac", rate=AUDIO_RATE, layout="stereo") if audio else None
    return out, vs, oa


def _put(out, vs, arr, pts):
    f = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
    if _CS is not None:
        try:
            f = f.reformat(format="yuv420p", src_colorspace=_CS.ITU709, dst_colorspace=_CS.ITU709,
                           src_color_range=_CR.JPEG, dst_color_range=_CR.MPEG, dst_color_trc=1, dst_color_primaries=1)
        except (TypeError, ValueError):
            pass
    f.pts = pts
    f.time_base = vs.codec_context.time_base
    for p in vs.encode(f):
        out.mux(p)


def _write_audio(out, oa, wave):
    step = 1024
    for i in range(0, wave.shape[1], step):
        fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(wave[:, i:i + step]), format="fltp", layout="stereo")
        fr.sample_rate = AUDIO_RATE
        fr.pts = i
        for p in oa.encode(fr):
            out.mux(p)
    for p in oa.encode(None):
        out.mux(p)


def segment(src, out_path, part="tail", seconds=5.0, reverse=False, keep_audio=True, crf=12, preset="fast"):
    """The first (`head`) or last (`tail`) `seconds` of a clip (the whole clip when shorter), optionally played
    backwards (audio is then dropped). Returns its real length in seconds."""
    sp = probe(src)
    fps, n = sp["fps"], sp["frames"]
    k = max(1, min(n, int(round(seconds * float(fps))))) if seconds > 0 else n
    lo = 0 if part == "head" else n - k
    W, H = sp["width"] - sp["width"] % 2, sp["height"] - sp["height"] % 2
    audio = keep_audio and not reverse and sp["audio"]
    wave = read_audio(src, lo / float(fps), k / float(fps)) if audio else None
    out, vs, oa = _open_writer(out_path, fps, W, H, crf, preset, audio=wave is not None)
    try:
        if reverse:
            frames = [a[:H, :W] for i, a in enumerate(iter_frames(src)) if lo <= i < lo + k]
            for j, a in enumerate(reversed(frames)):
                _put(out, vs, a, j)
        else:
            j = 0
            for i, a in enumerate(iter_frames(src)):
                if i >= lo + k:
                    break
                if i >= lo:
                    _put(out, vs, a[:H, :W], j)
                    j += 1
        for p in vs.encode():
            out.mux(p)
        if wave is not None:
            _write_audio(out, oa, wave)
    finally:
        out.close()
    return k / float(fps)


# -------------------------------------------------------------------------------------------------------- joining

def _cover(a, W, H):
    """Scale to cover W x H (Lanczos), centre-crop the excess: no stretching, no borders."""
    h, w = a.shape[:2]
    if (w, h) == (W, H):
        return a
    s = max(W / w, H / h)
    nw, nh = max(W, int(math.ceil(w * s))), max(H, int(math.ceil(h * s)))
    im = Image.fromarray(a).resize((nw, nh), Image.LANCZOS)
    x, y = (nw - W) // 2, (nh - H) // 2
    return np.asarray(im)[y:y + H, x:x + W]


def _part_plan(path, fps_out, drop_first, drop_last, max_seconds, keep="start"):
    """Which frames of a part are kept, and which of them land on each output frame (matched by time).
    When max_seconds cuts the part, keep="start" keeps its beginning, keep="end" its end (a part before the source)."""
    p = probe(path)
    n = p["frames"]
    lo, hi = min(drop_first, n - 1), max(min(drop_first, n - 1) + 1, n - drop_last)
    if max_seconds and keep == "end":
        lo = max(lo, hi - max(1, int(round(max_seconds * float(p["fps"])))))
    kept = hi - lo
    dur = kept / float(p["fps"])
    if max_seconds and dur > max_seconds:
        dur = max_seconds
    m = max(1, int(round(dur * float(fps_out))))
    ratio = float(p["fps"]) / float(fps_out)
    picks = [min(lo + kept - 1, lo + int(math.floor(j * ratio + 1e-6))) for j in range(m)]
    return {"path": path, "probe": p, "lo": lo, "kept": kept, "picks": picks, "seconds": m / float(fps_out)}


def _fade(wave, n_in, n_out):
    w = wave.copy()
    if n_in and w.shape[1]:
        k = min(n_in, w.shape[1]); w[:, :k] *= np.linspace(0, 1, k, dtype=np.float32)
    if n_out and w.shape[1]:
        k = min(n_out, w.shape[1]); w[:, -k:] *= np.linspace(1, 0, k, dtype=np.float32)
    return w


def join(source, out_path, before=(), after=(), before_reversed=False, drop_before_last=0, drop_after_first=0,
         max_seconds=0.0, fade=0.08, crf=12, preset="fast"):
    """before parts + source + after parts as one clip at the source's size and frame rate.

    before / after: generated clips in time order (for before_reversed: in the order they were generated on the
    reversed source, so they are joined, then played backwards, and their audio is dropped).
    drop_before_last / drop_after_first: frames to drop where a part meets the source (a part that starts or ends on
    the source's own frame). max_seconds: cap on each side's total length (0 = off). Parts are matched to the
    source's frame rate by time and scaled to cover its size (Lanczos, centre crop). The source's frames and sound
    stay as they are; each part keeps its own sound with short fades at the joins.
    """
    sp = probe(source)
    fps, W, H = sp["fps"], sp["width"] - sp["width"] % 2, sp["height"] - sp["height"] % 2
    before, after = [p for p in before if p], [p for p in after if p]

    def side_plans(paths, which):
        # walk outwards from the source: after parts in order, before parts from the last (or, reversed, the first)
        backwards = which == "before" and not before_reversed
        order = list(reversed(paths)) if backwards else list(paths)
        plans, left = [], max_seconds or 0.0
        for i, path in enumerate(order):
            touching = i == 0
            if which == "after":
                df, dl = (drop_after_first if touching else 0), 0
            elif before_reversed:
                df, dl = (drop_before_last if touching else 0), 0  # its first frames meet the source
            else:
                df, dl = 0, (drop_before_last if touching else 0)
            plan = _part_plan(path, fps, df, dl, left if max_seconds else 0.0, keep="end" if backwards else "start")
            plans.append(plan)
            if max_seconds:
                left -= plan["seconds"]
                if left <= 1.0 / float(fps):
                    break
        return list(reversed(plans)) if backwards else plans

    bplans, aplans = side_plans(before, "before"), side_plans(after, "after")

    # audio: before parts + source + after parts, each its own length in output frames
    has_audio = sp["audio"] or any(p["probe"]["audio"] for p in aplans) or \
        (not before_reversed and any(p["probe"]["audio"] for p in bplans))
    waves = []
    nf = int(round(fade * AUDIO_RATE))

    def part_audio(plan, silent=False):
        n = int(round(plan["seconds"] * AUDIO_RATE))
        if silent or not plan["probe"]["audio"]:
            return np.zeros((2, n), np.float32)
        w = read_audio(plan["path"], plan["lo"] / float(plan["probe"]["fps"]), plan["seconds"])
        return _fade(w, nf, nf) if w is not None else np.zeros((2, n), np.float32)

    if has_audio:
        if before_reversed:
            total = sum(int(round(p["seconds"] * AUDIO_RATE)) for p in bplans)
            waves.append(np.zeros((2, total), np.float32))
        else:
            waves += [part_audio(p) for p in bplans]
        sn = int(round(sp["frames"] / float(fps) * AUDIO_RATE))
        sw = read_audio(source, 0.0, sp["frames"] / float(fps)) if sp["audio"] else None
        waves.append(sw if sw is not None else np.zeros((2, sn), np.float32))
        waves += [part_audio(p) for p in aplans]

    out, vs, oa = _open_writer(out_path, fps, W, H, crf, preset, audio=has_audio)
    pts = 0
    report = {"fps": str(fps), "size": [W, H], "before": [], "after": [], "source_frames": sp["frames"]}

    def write_part(plan, reverse_frames=None):
        nonlocal pts
        if reverse_frames is not None:
            for a in reverse_frames:
                _put(out, vs, a, pts); pts += 1
            return
        want = plan["picks"]
        need, cache, it_i = set(want), {}, 0
        frames = iter_frames(plan["path"])
        for i, a in enumerate(frames):
            if i in need:
                cache[i] = _cover(a, W, H)
            while it_i < len(want) and want[it_i] in cache:
                _put(out, vs, cache[want[it_i]], pts); pts += 1
                if it_i + 1 >= len(want) or want[it_i + 1] != want[it_i]:
                    cache.pop(want[it_i], None)
                it_i += 1
            if it_i >= len(want):
                break
        while it_i < len(want):  # part ended early: hold its last frame
            a = cache.get(want[it_i - 1]) if it_i else None
            if a is None:
                break
            _put(out, vs, a, pts); pts += 1; it_i += 1

    try:
        if before_reversed and bplans:
            seq = []
            for plan in bplans:
                need = set(plan["picks"]); got = {}
                for i, a in enumerate(iter_frames(plan["path"])):
                    if i in need:
                        got[i] = _cover(a, W, H)
                    if i >= max(need):
                        break
                seq += [got[i] for i in plan["picks"] if i in got]
            write_part(None, reverse_frames=list(reversed(seq)))
        else:
            for plan in bplans:
                write_part(plan)
        report["before"] = [{"frames": len(p["picks"]), "seconds": round(p["seconds"], 3),
                             "from": [p["probe"]["width"], p["probe"]["height"], str(p["probe"]["fps"])]} for p in bplans]
        start = pts
        for a in iter_frames(source):
            _put(out, vs, a[:H, :W], pts); pts += 1
        report["source_at"] = start
        for plan in aplans:
            write_part(plan)
        report["after"] = [{"frames": len(p["picks"]), "seconds": round(p["seconds"], 3),
                            "from": [p["probe"]["width"], p["probe"]["height"], str(p["probe"]["fps"])]} for p in aplans]
        for p in vs.encode():
            out.mux(p)
        if has_audio:
            wave = np.concatenate(waves, axis=1)
            _write_audio(out, oa, _fit_len(wave, int(round(pts / float(fps) * AUDIO_RATE))))
    finally:
        out.close()
    report["frames"] = pts
    report["seconds"] = round(pts / float(fps), 3)
    # every place two pieces meet: between before parts, before part / source, source / after part, after parts
    seams, at = [], 0
    blens = [len(p["picks"]) for p in (reversed(bplans) if before_reversed else bplans)]
    for k in blens:
        at += k
        seams.append(at)
    at = report["source_at"] + sp["frames"]
    if aplans:
        seams.append(at)
        for p in aplans[:-1]:
            at += len(p["picks"])
            seams.append(at)
    report["seams"] = seams
    report["audio"] = has_audio
    return report


# ------------------------------------------------------------------------------------------------------ seam repair

def gen_size(W, H, pixels, multiple=32):
    """The clip scaled to about `pixels` area keeping its exact shape (sw x sh, never above the clip), and the canvas
    it sits on (gw x gh, padded right and bottom to multiples of `multiple`). Returns gw, gh, sw, sh."""
    s = min(1.0, math.sqrt(pixels / float(W * H)))
    sw, sh = max(2, int(round(W * s))), max(2, int(round(H * s)))
    gw, gh = -(-sw // multiple) * multiple, -(-sh // multiple) * multiple
    return gw, gh, sw, sh


def _to_canvas(a, gw, gh, sw, sh):
    """Scale to sw x sh (Lanczos, same shape as the clip) and pad to gw x gh by repeating the edge pixels."""
    im = np.asarray(Image.fromarray(a).resize((sw, sh), Image.LANCZOS))
    return np.pad(im, ((0, gh - sh), (0, gw - sw), (0, 0)), mode="edge")


def _from_canvas(g, W, H, sw, sh):
    """The exact inverse of _to_canvas: drop the padding, scale back to W x H."""
    g = np.asarray(g)[:sh, :sw]
    return np.asarray(Image.fromarray(g).resize((W, H), Image.LANCZOS))


def _small_gray(a, w=160):
    h = max(1, int(round(a.shape[0] * w / a.shape[1])))
    return np.asarray(Image.fromarray(a).convert("L").resize((w, h), Image.BILINEAR), np.float32)


def seam_window(path, seam, context=9, gap=23, pixels=921600):
    """Frames around a join, for a model that regenerates the `gap` frames centred on it from `context` frames kept
    on each side (window = 2 * context + gap frames, e.g. 9 + 23 + 9 = 41 = 8n + 1 for LTX).

    seam: index of the first frame after the join (-1 = no join). Returns frames (uint8, gen size), width, height,
    fps, start (first window frame, -1 when the window doesn't fit) and jump: the frame change across the join over
    the median change of the other frames in the window (about 1 = invisible, 2 = a visible jump)."""
    p = probe(path)
    n, fps = p["frames"], p["fps"]
    W, H, sw, sh = gen_size(p["width"], p["height"], pixels)
    total = 2 * context + gap
    start = seam - context - gap // 2
    if seam <= 0 or start < 0 or start + total > n:
        return np.zeros((1, H, W, 3), np.uint8), W, H, fps, -1, 0.0
    frames, smalls = [], []
    for i, a in enumerate(iter_frames(path)):
        if i >= start + total:
            break
        if i >= start:
            frames.append(_to_canvas(a, W, H, sw, sh))
            smalls.append(_small_gray(a))
    diffs = [float(np.mean(np.abs(smalls[i] - smalls[i - 1]))) for i in range(1, len(smalls))]
    k = seam - start - 1  # diffs[k] = change from frame seam-1 to frame seam
    others = [d for j, d in enumerate(diffs) if j != k]
    base = float(np.median(others)) if others else 0.0
    jump = diffs[k] / max(base, 0.25) if diffs else 0.0
    return np.stack(frames), W, H, fps, start, round(jump, 3)


def seam_splice(path, out_path, windows, context=9, pixels=921600, crf=12, preset="fast"):
    """Put regenerated windows back, all in one pass (one encode). Each window (frames, start) covers the video's
    frames start .. start + len - 1 and is mapped back exactly (padding dropped, scaled to the video's size). The
    regenerated gap replaces the original; over the guide frames on each side, where the model reproduces the
    original, it fades in and out (no double image in the gap). Every other frame and the sound stay as they are."""
    p = probe(path)
    fps, W, H = p["fps"], p["width"] - p["width"] % 2, p["height"] - p["height"] % 2
    _, _, sw, sh = gen_size(p["width"], p["height"], pixels)
    spans = [(start, start + len(fr), fr) for fr, start in windows if start >= 0]
    wave = read_audio(path, 0.0, p["frames"] / float(fps)) if p["audio"] else None
    out, vs, oa = _open_writer(out_path, fps, W, H, crf, preset, audio=wave is not None)
    try:
        for i, a in enumerate(iter_frames(path)):
            a = a[:H, :W]
            for lo, hi, fr in spans:
                if lo <= i < hi:
                    j, n = i - lo, hi - lo
                    t = 1.0
                    if j < context:
                        t = (j + 1) / float(context + 1)
                    elif j >= n - context:
                        t = (n - j) / float(context + 1)
                    g = _from_canvas(fr[j], W, H, sw, sh).astype(np.float32)
                    a = np.clip(g * t + a.astype(np.float32) * (1 - t), 0, 255).astype(np.uint8)
            _put(out, vs, a, i)
        for pk in vs.encode():
            out.mux(pk)
        if wave is not None:
            _write_audio(out, oa, wave)
    finally:
        out.close()
    return {"replaced": [[lo, hi] for lo, hi, _ in spans]}


if __name__ == "__main__":  # offline: python extend_core.py join SRC OUT --after A [--before B] ...
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["join", "segment", "edges", "probe"])
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--before", nargs="*", default=[])
    ap.add_argument("--after", nargs="*", default=[])
    ap.add_argument("--before-reversed", action="store_true")
    ap.add_argument("--drop-before-last", type=int, default=0)
    ap.add_argument("--drop-after-first", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--part", default="tail")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    if a.cmd == "probe":
        print({k: str(v) for k, v in probe(a.src).items()})
    elif a.cmd == "edges":
        f, l = edge_frames(a.src)
        Image.fromarray(f).save(a.out + "_first.png"); Image.fromarray(l).save(a.out + "_last.png")
    elif a.cmd == "segment":
        print(segment(a.src, a.out, a.part, a.seconds, a.reverse))
    else:
        print(json.dumps(join(a.src, a.out, a.before, a.after, a.before_reversed, a.drop_before_last,
                              a.drop_after_first, a.max_seconds), indent=1))
