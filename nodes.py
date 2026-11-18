"""ComfyUI nodes for extending a video at its start and/or end with generated clips (classic node API). The work is in
extend_core; nothing here knows which model made the clips."""
import json, os, uuid

import numpy as np
import torch

import folder_paths

from . import extend_core as ec

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
        }

    RETURN_TYPES = ("VIDEO", "FLOAT")
    RETURN_NAMES = ("video", "seconds")
    OUTPUT_TOOLTIPS = ("The part, H.264 (crf 12).", "Its length in seconds.")
    FUNCTION = "segment"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The first or last seconds of a video, optionally played backwards: what a model that continues a "
                   "clip needs to see (most read only the last seconds).")

    def segment(self, video, part, seconds, reverse, audio):
        out = _temp("extend_segment")
        secs = ec.segment(_as_file(video), out, part, float(seconds), bool(reverse), audio == "keep")
        return (VideoFromFile(out), float(secs))


class VideoExtendJoin:
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
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    OUTPUT_TOOLTIPS = ("before + source + after at the source's size and frame rate, H.264 (crf 12), with sound.",
                       "JSON: frames and seconds per part, where the source starts, the output length.")
    FUNCTION = "join"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Join generated clips before and/or after a video. Each part is matched to the source's frame rate "
                   "by time and scaled to cover its size (Lanczos, centred), so a part generated at a lower resolution "
                   "comes back at the source's; the source's own frames and sound are kept. Only the parts in use are "
                   "run (lazy inputs).")

    def check_lazy_status(self, source, add_before, add_after, before_parts, after_parts, before_reversed=False,
                          drop_before_last=0, drop_after_first=0, max_seconds=0.0, **parts):
        need = []
        if add_before:
            need += [f"before_{i}" for i in range(1, before_parts + 1) if parts.get(f"before_{i}") is None]
        if add_after:
            need += [f"after_{i}" for i in range(1, after_parts + 1) if parts.get(f"after_{i}") is None]
        return need

    def join(self, source, add_before, add_after, before_parts, after_parts, before_reversed=False,
             drop_before_last=0, drop_after_first=0, max_seconds=0.0, **parts):
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
            return (source, json.dumps({"note": "nothing to add"}))
        out = _temp("extend_join")
        report = ec.join(_as_file(source), out, before, after, bool(before_reversed), int(drop_before_last),
                         int(drop_after_first), float(max_seconds))
        return (VideoFromFile(out), json.dumps(report))


NODE_CLASS_MAPPINGS = {
    "VideoExtendEdgeFrames": VideoExtendEdgeFrames,
    "VideoExtendSegment": VideoExtendSegment,
    "VideoExtendJoin": VideoExtendJoin,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoExtendEdgeFrames": "Video Edge Frames",
    "VideoExtendSegment": "Extend Segment (Head / Tail)",
    "VideoExtendJoin": "Extend Join",
}
