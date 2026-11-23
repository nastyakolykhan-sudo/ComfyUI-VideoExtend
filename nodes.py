"""ComfyUI nodes for extending a video at its start and/or end with generated clips (classic node API). The work is in
extend_core; nothing here knows which model made the clips."""
import json, os, uuid

import numpy as np
import torch

import folder_paths

from . import extend_core as ec
from . import extend_motion as em

try:
    from comfy_api.latest import InputImpl
    VideoFromFile = InputImpl.VideoFromFile
except ImportError:  # older ComfyUI
    from comfy_api.input_impl import VideoFromFile

CATEGORY = "video/extend"
STEPS = 3  # generated parts per side the join accepts


def _temp(prefix, ext=".mp4"):
    d = folder_paths.get_temp_directory()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{prefix}_{uuid.uuid4().hex[:12]}{ext}")


def _as_file(video):
    """A plain file path for `video`: the file itself when it is an untrimmed, uncropped file on disk, else a temp
    copy written by the video's own save_to (trims, crops, in-memory files and tensor videos)."""
    if isinstance(video, VideoFromFile) and getattr(video, "_VideoFromFile__crop", None) is None:
        start, duration = video.get_active_trim_window()
        src = video.get_stream_source()
        if isinstance(src, str) and not start and not duration:
            return src
    path = _temp("extend_in")
    video.save_to(path)
    return path


def _image(a):
    return torch.from_numpy(np.asarray(a, np.float32) / 255.0)[None]


def _always(*args, **kwargs):
    """The node writes its result to ComfyUI's temp folder, which a server may clear between jobs: never reuse a
    cached result (its file can be gone), always run again (cheap)."""
    return float("nan")


class VideoExtendEdgeFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "FLOAT", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("first_frame", "last_frame", "width", "height", "fps", "frames", "seconds", "has_audio")
    OUTPUT_TOOLTIPS = ("The clip's first frame.", "The clip's last frame.", "Width.", "Height.", "Frame rate.",
                       "Frame count.", "Length in seconds (frames / fps).", "Whether the clip has sound.")
    FUNCTION = "edges"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The first and last frames of a video (e.g. to start or end a generated clip on them) and its "
                   "size, frame rate and length. Streams the clip once; no frames are held.")

    def edges(self, video):
        path = _as_file(video)
        p = ec.probe(path)
        first, last = ec.edge_frames(path)
        return (_image(first), _image(last), p["width"], p["height"], float(p["fps"]), p["frames"],
                float(p["duration"]), p["audio"])


class VideoExtendSegment:
    IS_CHANGED = classmethod(_always)
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "part": (["tail", "head"], {"default": "tail",
                         "tooltip": "tail: the last seconds of the clip. head: the first seconds."}),
                "seconds": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                            "tooltip": "How many seconds (the whole clip when it is shorter; 0 = the whole clip)."}),
                "reverse": ("BOOLEAN", {"default": False,
                            "tooltip": "Play the part backwards (its sound is dropped). With head: the clip's first "
                                       "frame becomes the part's last, so a model that only continues a clip forward "
                                       "can make what comes before it."}),
                "audio": (["keep", "drop"], {"default": "keep"}),
            },
            "optional": {
                "pad": ("BOOLEAN", {"default": False,
                        "tooltip": "Centre the part on a 16:9 (landscape) or 9:16 (portrait) canvas with black bars, for a "
                                   "model that only makes those shapes (Gemini Omni crops anything else). Extend Join "
                                   "removes the bars again when it fits the result to the clip."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "FLOAT")
    RETURN_NAMES = ("video", "seconds")
    OUTPUT_TOOLTIPS = ("The part, H.264 (crf 12).", "Its length in seconds.")
    FUNCTION = "segment"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The first or last seconds of a video, optionally played backwards: what a model that continues a "
                   "clip needs to see (most read only the last seconds).")

    def segment(self, video, part, seconds, reverse, audio, pad=False):
        out = _temp("extend_segment")
        secs = ec.segment(_as_file(video), out, part, float(seconds), bool(reverse), audio == "keep", pad=bool(pad))
        return (VideoFromFile(out), float(secs))


class VideoExtendJoin:
    IS_CHANGED = classmethod(_always)
    @classmethod
    def INPUT_TYPES(cls):
        lazy = {"lazy": True}
        return {
            "required": {
                "source": ("VIDEO", {"tooltip": "The clip being extended. Its frames, size, frame rate and sound stay "
                                                "as they are."}),
                "add_before": ("BOOLEAN", {"default": False, "tooltip": "Put the before parts in front of the source."}),
                "add_after": ("BOOLEAN", {"default": True, "tooltip": "Put the after parts behind the source."}),
                "before_parts": ("INT", {"default": 1, "min": 1, "max": STEPS,
                                 "tooltip": "How many before parts to use (before_1, before_2, ...)."}),
                "after_parts": ("INT", {"default": 1, "min": 1, "max": STEPS,
                                "tooltip": "How many after parts to use (after_1, after_2, ...)."}),
                "before_reversed": ("BOOLEAN", {"default": False,
                                    "tooltip": "The before parts were made by continuing the source played backwards "
                                               "(Extend Segment head + reverse): they are joined in order, played "
                                               "backwards and put in front, without their sound."}),
                "drop_before_last": ("INT", {"default": 0, "min": 0, "max": 1000,
                                     "tooltip": "Frames to drop where the before side meets the source, e.g. 1 when "
                                                "the part ends on the source's own first frame."}),
                "drop_after_first": ("INT", {"default": 0, "min": 0, "max": 1000,
                                     "tooltip": "Frames to drop where the after side meets the source, e.g. 1 when "
                                                "the part starts on the source's own last frame."}),
                "max_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                                "tooltip": "Longest each side may be (the far end is cut); 0 = keep everything."}),
            },
            "optional": {
                **{f"before_{i}": ("VIDEO", lazy) for i in range(1, STEPS + 1)},
                **{f"after_{i}": ("VIDEO", lazy) for i in range(1, STEPS + 1)},
                "overlap_before": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05,
                                   "tooltip": "Seconds each before part repeats of the piece next to it (a model that "
                                              "returns its input + the new footage, sliced that much early). The two "
                                              "copies are cross-dissolved instead of cut. 0 = cut."}),
                "overlap_after": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05,
                                  "tooltip": "The same for the after parts."}),
                "smooth_joins": ("BOOLEAN", {"default": False,
                                 "tooltip": "Smooth every join with the source that is cut (no overlap), on the "
                                            "generated side only: repeated frames, camera motion, speed, alignment "
                                            "and colour. The source's frames stay as they are."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "report", "seams")
    OUTPUT_TOOLTIPS = ("before + source + after at the source's size and frame rate, H.264 (crf 12), with sound.",
                       "JSON: frames and seconds per part, where the source starts, the output length.",
                       "Frame index of every join (first frame after it), comma-separated; for Extend Seam Windows.")
    FUNCTION = "join"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Join generated clips before and/or after a video. Each part is matched to the source's frame rate "
                   "by time and scaled to cover its size (Lanczos, centred), so a part generated at a lower resolution "
                   "comes back at the source's; the source's own frames and sound are kept. Only the parts in use are "
                   "run (lazy inputs).")

    def check_lazy_status(self, source, add_before, add_after, before_parts, after_parts, before_reversed=False,
                          drop_before_last=0, drop_after_first=0, max_seconds=0.0, overlap_before=0.0,
                          overlap_after=0.0, smooth_joins=False, **parts):
        need = []
        if add_before:
            need += [f"before_{i}" for i in range(1, before_parts + 1) if parts.get(f"before_{i}") is None]
        if add_after:
            need += [f"after_{i}" for i in range(1, after_parts + 1) if parts.get(f"after_{i}") is None]
        return need

    def join(self, source, add_before, add_after, before_parts, after_parts, before_reversed=False,
             drop_before_last=0, drop_after_first=0, max_seconds=0.0, overlap_before=0.0, overlap_after=0.0,
             smooth_joins=False, **parts):
        def side(name, on, n):
            if not on:
                return []
            vids = [parts.get(f"{name}_{i}") for i in range(1, n + 1)]
            missing = [f"{name}_{i}" for i, v in enumerate(vids, 1) if v is None]
            if missing:
                raise ValueError(f"Extend Join: {', '.join(missing)} not connected.")
            return [_as_file(v) for v in vids]

        before, after = side("before", add_before, before_parts), side("after", add_after, after_parts)
        if not before and not after:
            return (source, json.dumps({"note": "nothing to add"}), "")
        out = _temp("extend_join")
        if smooth_joins or overlap_before or overlap_after:
            report = ec.join_v2(_as_file(source), out, before, after, bool(before_reversed), int(drop_before_last),
                                int(drop_after_first), float(max_seconds), float(overlap_before or 0.0),
                                float(overlap_after or 0.0), bool(smooth_joins), tmp_path=_temp("extend_join_a"))
        else:
            report = ec.join(_as_file(source), out, before, after, bool(before_reversed), int(drop_before_last),
                             int(drop_after_first), float(max_seconds))
        return (VideoFromFile(out), json.dumps(report), ",".join(str(x) for x in report["seams"]))


class VideoExtendSeamWindows:
    IS_CHANGED = classmethod(_always)
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The joined video (Extend Join)."}),
                "seams": ("STRING", {"default": "", "forceInput": True,
                          "tooltip": "Frame indices of the joins, comma-separated (Extend Join's seams)."}),
                "mode": (["auto", "always", "off"], {"default": "auto",
                         "tooltip": "auto: only joins whose jump is above the threshold. always: every join. off: none."}),
                "threshold": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 10.0, "step": 0.05,
                              "tooltip": "Frame change across the join over the usual change around it; about 1 is "
                                         "invisible, 2 a clear jump."}),
                "context": ("INT", {"default": 9, "min": 1, "max": 49,
                            "tooltip": "Frames kept on each side as guides (8n + 1 for LTX: 9, 17, ...)."}),
                "gap": ("INT", {"default": 23, "min": 1, "max": 97,
                        "tooltip": "Frames regenerated across the join. 2 x context + gap must be 8n + 1 for LTX "
                                   "(9 + 23 + 9 = 41)."}),
                "pixels": ("INT", {"default": 921600, "min": 65536, "max": 8294400, "step": 1024,
                           "tooltip": "Area of the windows (921600 = 1280x720), the clip's shape, sides multiples of 32."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN", "INT", "INT", "INT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("windows", "starts", "any", "width", "height", "frames", "fps", "fps_int", "guide_at", "report")
    # windows keep the clip's exact shape: scaled, then padded right/bottom to multiples of 32 (Extend Seam Splice
    # undoes exactly that)
    OUTPUT_IS_LIST = (True, True, False, False, False, False, False, False, False, False)
    OUTPUT_TOOLTIPS = ("One frame batch per join to repair (a list: nodes after it run once per join).",
                       "Where each window starts in the video (a list, same order).",
                       "Whether any join needs repair.", "Window width.", "Window height.",
                       "Frames per window (2 x context + gap).", "Frame rate.", "Frame rate, rounded.",
                       "Index of the second guide (frames - context).", "JSON: every join, its jump and whether it is repaired.")
    FUNCTION = "windows"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Find the joins worth repairing and cut a window of frames around each: guides on both sides and the "
                   "frames in between for a model (e.g. LTX with two guides) to regenerate. Measures each join's jump.")

    def windows(self, video, seams, mode, threshold, context, gap, pixels):
        path = _as_file(video)
        idx = [int(x) for x in str(seams or "").replace(" ", "").split(",") if x.strip().lstrip("-").isdigit()]
        wins, starts, rows, W, H, fps = [], [], [], 64, 64, 24.0
        for s in idx:
            fr, W, H, f, start, jump = ec.seam_window(path, s, context, gap, pixels)
            fps = float(f)
            fix = start >= 0 and mode != "off" and (mode == "always" or jump > threshold)
            rows.append({"seam": s, "jump": jump, "repair": fix, "start": start})
            if fix:
                wins.append(torch.from_numpy(fr.astype(np.float32) / 255.0))
                starts.append(start)
        n = 2 * context + gap
        anyfix = bool(wins)
        if not anyfix:  # a placeholder so the list is never empty; nothing downstream runs (lazy splice)
            wins, starts = [torch.zeros((n, H, W, 3))], [-1]
        return (wins, starts, anyfix, W, H, n, fps, max(1, int(round(fps))), n - context, json.dumps(rows))


class VideoExtendSeamSplice:
    IS_CHANGED = classmethod(_always)
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "starts": ("INT", {"forceInput": True, "tooltip": "Extend Seam Windows' starts."}),
                "any": ("BOOLEAN", {"forceInput": True, "tooltip": "Extend Seam Windows' any: nothing runs when false."}),
                "context": ("INT", {"default": 9, "min": 1, "max": 49,
                            "tooltip": "The same context as the windows: the repaired window fades in and out over "
                                       "these guide frames, where it matches the original."}),
                "pixels": ("INT", {"default": 921600, "min": 65536, "max": 8294400, "step": 1024,
                           "tooltip": "The same pixels as the windows (to undo their scaling exactly)."}),
            },
            "optional": {
                "frames": ("IMAGE", {"lazy": True, "tooltip": "The repaired windows (same order as starts); only run "
                                                               "when a join needs repair."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "splice"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Put repaired windows back into the video at their places, blended into the original at the edges. "
                   "Every other frame and the sound stay as they are.")

    def check_lazy_status(self, video, starts, any, context, pixels, frames=None):
        if any and any[0] and (frames is None or all(f is None for f in frames)):
            return ["frames"]
        return []

    def splice(self, video, starts, any, context, pixels, frames=None):
        video = video[0]
        if not (any and any[0]):
            return (video,)
        context, pixels = int(context[0]), int(pixels[0])
        wins = [((f.clamp(0, 1) * 255).round().byte().cpu().numpy(), int(s)) for f, s in zip(frames, starts) if s >= 0]
        if not wins:
            return (video,)
        out = _temp("extend_seam")
        ec.seam_splice(_as_file(video), out, wins, context, pixels)
        return (VideoFromFile(out),)



class VideoExtendAutoResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)}}

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("resolution", "short_side", "aspect_16_9")
    OUTPUT_TOOLTIPS = ("480p, 720p or 1080p: the generation resolution nearest the clip (up to 600 px on the short "
                       "side -> 480p, up to 900 -> 720p, larger -> 1080p).", "The clip's short side in px.",
                       "16:9 for a landscape (or square) clip, 9:16 for a portrait one: the canvas Extend Segment's pad "
                       "uses, for models that only make those shapes.")
    FUNCTION = "pick"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Picks the generation resolution from the clip itself, so the new footage is made as close to the "
                   "clip's own sharpness as the models allow (1080p at most).")

    def pick(self, video):
        p = ec.probe(_as_file(video))
        short = min(p["width"], p["height"])
        res = "480p" if short <= 600 else "720p" if short <= 900 else "1080p"
        return (res, int(short), "16:9" if p["width"] >= p["height"] else "9:16")


class VideoExtendEdgeMotion:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)},
                "optional": {"seconds": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.1,
                                                   "tooltip": "How much of each edge is measured."})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("motion", "report")
    OUTPUT_TOOLTIPS = ("One sentence with the camera's and the subject's speed (and the camera's direction and zoom) "
                       "in the clip's first and last seconds, in words and frame widths per second; empty when it "
                       "cannot be measured.", "The measurements as JSON.")
    FUNCTION = "measure"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Measures how fast the clip moves at its start and its end (optical flow), for a prompt writer: "
                   "so that generated footage joined there keeps the same pace instead of winding down.")

    def measure(self, video, seconds=1.0):
        r = em.edge_motion(_as_file(video), float(seconds))
        return (r["text"], json.dumps(r))


NODE_CLASS_MAPPINGS = {
    "VideoExtendEdgeFrames": VideoExtendEdgeFrames,
    "VideoExtendSegment": VideoExtendSegment,
    "VideoExtendJoin": VideoExtendJoin,
    "VideoExtendSeamWindows": VideoExtendSeamWindows,
    "VideoExtendSeamSplice": VideoExtendSeamSplice,
    "VideoExtendAutoResolution": VideoExtendAutoResolution,
    "VideoExtendEdgeMotion": VideoExtendEdgeMotion,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoExtendEdgeFrames": "Video Edge Frames",
    "VideoExtendSegment": "Extend Segment (Head / Tail)",
    "VideoExtendJoin": "Extend Join",
    "VideoExtendSeamWindows": "Extend Seam Windows",
    "VideoExtendSeamSplice": "Extend Seam Splice",
    "VideoExtendAutoResolution": "Extend Auto Resolution",
    "VideoExtendEdgeMotion": "Extend Edge Motion",
}
