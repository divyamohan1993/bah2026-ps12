"""All-intra video encoding for O(1) seek on the web (SERVE+VIZ, research/04 §3).

The dashboard's "play" path is a hardware-decoded video, but inter-frame (delta) frames
make seeking expensive: a seek must decode from the nearest preceding keyframe. Time-lapses
here are short (dozens-to-hundreds of frames), so we encode **all-intra** — *every frame is
a keyframe* (GOP size 1, no B-frames, no scene-cut) — which makes every frame an O(1) seek
target while keeping the file small enough for a short loop (research/04 §2-§3, §10).

Functions:
    * :func:`encode_video`        — encode a list of RGBA frames to MP4/WebM (all-intra).
    * :func:`encode_observed_video` / :func:`encode_interpolated_video` — convenience wrappers.
    * :func:`encode_side_by_side`  — a GT-vs-interpolated compare video (frames placed
      left/right) for the synced compare UX.

Frames may be ``(H, W, 4)`` RGBA or ``(H, W, 3)`` RGB ``uint8`` arrays (or 2D BT fields,
which are colorized on the fixed range). ``imageio``/``imageio-ffmpeg`` are imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .. import constants as C
from .render import render_rgba

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# Container suffix -> (imageio ffmpeg codec, pixel format). H.264 in MP4, VP9 in WebM.
_CODEC_BY_SUFFIX = {
    ".mp4": ("libx264", "yuv420p"),
    ".webm": ("libvpx-vp9", "yuv420p"),
    ".mkv": ("libx264", "yuv420p"),
}


def _to_rgb_uint8(frame: "np.ndarray", cmap: str) -> "np.ndarray":
    """Coerce a frame (BT 2D, RGB, or RGBA) to an ``(H, W, 3)`` ``uint8`` RGB array.

    RGBA is composited onto an opaque black background (matches the dark dashboard theme and
    keeps transparent off-disk pixels from turning an arbitrary colour).
    """
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim == 2:  # BT field -> colorize on the fixed range
        arr = render_rgba(arr, cmap=cmap)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        rgba = arr.astype(np.float32)
        alpha = rgba[..., 3:4] / 255.0
        rgb = (rgba[..., :3] * alpha).round().astype(np.uint8)  # over black
        return rgb
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr.astype(np.uint8)
    raise ValueError(f"unsupported frame shape {arr.shape}; expected 2D BT, RGB, or RGBA")


def _all_intra_output_params(codec: str) -> list[str]:
    """ffmpeg output params forcing an all-intra stream (every frame a keyframe).

    ``-g 1`` sets GOP size 1 (keyframe every frame). For H.264 we additionally disable
    B-frames and scene-cut insertion so the cadence is strictly intra; for VP9 we request
    error-resilient + keyframe-every-frame behaviour. This is what makes web seeking O(1).
    """
    params = ["-g", "1"]
    if codec == "libx264":
        # No B-frames, force IDR every frame, disable scene-cut GOP shortening.
        params += ["-bf", "0", "-x264-params", "keyint=1:min-keyint=1:scenecut=0", "-preset", "fast"]
    elif codec == "libvpx-vp9":
        params += ["-keyint_min", "1", "-error-resilient", "1", "-b:v", "0", "-crf", "30"]
    return params


def encode_video(
    frames_rgba: Sequence["np.ndarray"],
    out_path: str | Path,
    fps: int = 10,
    all_intra: bool = True,
    cmap: str = C.DEFAULT_COLORMAP,
) -> Path:
    """Encode a sequence of frames to a video file (all-intra by default).

    Args:
        frames_rgba: ordered frames; each ``(H, W, 4)`` RGBA, ``(H, W, 3)`` RGB ``uint8``,
            or a 2D BT field (colorized on the fixed range).
        out_path: output video path; the suffix selects the container/codec
            (``.mp4`` -> H.264, ``.webm`` -> VP9).
        fps: playback frames per second.
        all_intra: when True (default) every frame is a keyframe (O(1) seek; research/04).
        cmap: colormap used when frames are 2D BT fields.

    Returns:
        The :class:`pathlib.Path` to the written video.

    Raises:
        ValueError: if ``frames_rgba`` is empty.
    """
    import imageio.v2 as imageio
    import numpy as np

    frames = list(frames_rgba)
    if not frames:
        raise ValueError("encode_video requires at least one frame")

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    codec, pix_fmt = _CODEC_BY_SUFFIX.get(suffix, ("libx264", "yuv420p"))

    rgb_frames = [_to_rgb_uint8(f, cmap) for f in frames]
    # H.264/VP9 with yuv420p need even dimensions; pad odd height/width by one pixel.
    h, w = rgb_frames[0].shape[:2]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        rgb_frames = [
            np.pad(fr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge") for fr in rgb_frames
        ]

    output_params = _all_intra_output_params(codec) if all_intra else []

    writer = imageio.get_writer(
        str(p),
        fps=fps,
        codec=codec,
        format="FFMPEG",
        pixelformat=pix_fmt,
        macro_block_size=None,  # we already ensured even dims; don't let imageio rescale
        output_params=output_params,
    )
    try:
        for fr in rgb_frames:
            writer.append_data(fr)
    finally:
        writer.close()
    return p


def encode_observed_video(
    observed_frames: Sequence["np.ndarray"],
    out_path: str | Path,
    fps: int = 10,
    cmap: str = C.DEFAULT_COLORMAP,
) -> Path:
    """Encode the observed-only (ground-truth) time-lapse (all-intra)."""
    return encode_video(observed_frames, out_path, fps=fps, all_intra=True, cmap=cmap)


def encode_interpolated_video(
    all_frames: Sequence["np.ndarray"],
    out_path: str | Path,
    fps: int = 10,
    cmap: str = C.DEFAULT_COLORMAP,
) -> Path:
    """Encode the densified (observed + interpolated) time-lapse (all-intra).

    This is the headline "fill in the frames" loop; play it at a higher fps than the
    observed video to show the smoother motion the interpolation provides.
    """
    return encode_video(all_frames, out_path, fps=fps, all_intra=True, cmap=cmap)


def encode_side_by_side(
    left_frames: Sequence["np.ndarray"],
    right_frames: Sequence["np.ndarray"],
    out_path: str | Path,
    fps: int = 10,
    gap_px: int = 4,
    cmap: str = C.DEFAULT_COLORMAP,
) -> Path:
    """Encode a left/right side-by-side compare video (e.g. GT vs interpolated).

    Frames are paired by index (truncated to the shorter sequence), placed left and right
    with a thin separator gap, and encoded all-intra. Heights are matched to the smaller of
    the two; this drives the synced compare UX (research/04 §7).

    Args:
        left_frames: left-pane frames (BT 2D / RGB / RGBA).
        right_frames: right-pane frames (same accepted types).
        out_path: output video path (suffix selects codec).
        fps: playback frames per second.
        gap_px: width in pixels of the separator between the two panes.
        cmap: colormap for 2D BT inputs.

    Returns:
        The :class:`pathlib.Path` to the written video.

    Raises:
        ValueError: if either sequence is empty.
    """
    import numpy as np

    left = [_to_rgb_uint8(f, cmap) for f in left_frames]
    right = [_to_rgb_uint8(f, cmap) for f in right_frames]
    if not left or not right:
        raise ValueError("encode_side_by_side requires non-empty left and right sequences")

    n = min(len(left), len(right))
    target_h = min(left[0].shape[0], right[0].shape[0])

    def _fit_height(fr: "np.ndarray") -> "np.ndarray":
        if fr.shape[0] == target_h:
            return fr
        from PIL import Image

        scale = target_h / fr.shape[0]
        new_w = max(1, int(round(fr.shape[1] * scale)))
        img = Image.fromarray(fr, mode="RGB").resize(
            (new_w, target_h), getattr(Image, "Resampling", Image).LANCZOS
        )
        return np.asarray(img, dtype=np.uint8)

    sep = np.zeros((target_h, max(0, gap_px), 3), dtype=np.uint8)
    combined: list[np.ndarray] = []
    for i in range(n):
        l = _fit_height(left[i])
        r = _fit_height(right[i])
        combined.append(np.concatenate([l, sep, r], axis=1))

    return encode_video(combined, out_path, fps=fps, all_intra=True, cmap=cmap)


__all__ = [
    "encode_video",
    "encode_observed_video",
    "encode_interpolated_video",
    "encode_side_by_side",
]
