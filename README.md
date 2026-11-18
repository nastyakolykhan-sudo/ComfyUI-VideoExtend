# ComfyUI-VideoExtend

Nodes for extending a video at its start and/or end with clips from any video model, while keeping the original
clip's own frames, size, frame rate and sound. Built for the Silverside **Video Extend** workflow. ComfyUI 0.45+; no
extra dependencies (PyAV, numpy, Pillow and torch ship with ComfyUI). Model-agnostic: nothing here knows which model
made the clips.

| Node | What it does |
|---|---|
| **Video Edge Frames** (`VideoExtendEdgeFrames`) | The clip's first and last frames (IMAGE), plus width, height, fps, frames, seconds and whether it has sound. Streams the clip once; no frames are held. For image-to-video models that start or end on a given frame. |
| **Extend Segment (Head / Tail)** (`VideoExtendSegment`) | The first (`head`) or last (`tail`) N seconds of a clip, optionally played backwards (sound dropped). What a model that continues a clip needs to see; `head` + `reverse` lets a forward-only model make what comes before the clip. Outputs the part and its length. |
| **Extend Join** (`VideoExtendJoin`) | `before` parts + source + `after` parts as one clip (H.264 crf 12, yuv420p). Each part is matched to the source's frame rate by time and scaled to cover its size (Lanczos, centre crop: no stretching, no borders), so a part generated at 720p comes back at the source's 1080p. The source's frames and sound stay as they are; each part keeps its own sound with 80 ms fades at the joins (silence when it has none, or when the before side was made reversed). Up to 3 parts per side, lazy: only the parts in use run. `drop_before_last` / `drop_after_first` drop frames where a part repeats the source's own edge frame (image-to-video from the source's last frame); `max_seconds` caps each side (the far end is cut). Outputs the video and a JSON report. |

Offline use (no ComfyUI): `python extend_core.py probe SRC`, `edges SRC OUTPREFIX`,
`segment SRC OUT --part head --seconds 10 --reverse`,
`join SRC OUT --before B1 --after A1 A2 [--before-reversed] [--drop-after-first 1] [--max-seconds 5]`.
