"""Core frame-interpolation routines (the inference scientific core).

Given a trained :class:`~frameflow.models.base.VFIModel` (any module exposing
``forward(I0, I1, t) -> It`` with ``I0,I1`` of shape ``(B,1,H,W)`` and ``t`` of shape
``(B,1)``), these helpers synthesize intermediate brightness-temperature frames:

    * :func:`interpolate_pair` — one frame at an arbitrary fraction ``t`` in (0,1).
      Normalizes the inputs (using ``frameflow.data.preprocess.normalize`` when available,
      else a fixed-range fallback / a caller-supplied normalizer / identity), builds the
      **batched** timestep ``t`` of shape ``(1,1)`` (P2-ONNX convention — never a scalar),
      runs the model, denormalizes back to Kelvin, and restores the NaN (off-disk) mask.
    * :func:`interpolate_recursive` — binary subdivision producing a 2x / 4x / 8x densified
      sequence (30 -> 15 -> 7.5 min cadence) with per-frame provenance (observed vs
      interpolated, and the fractional ``t`` of each inserted frame). Arbitrary input
      cadence is supported via the per-frame ``time`` metadata.

Design rules honoured here:
    * **P2-ONNX batched t.** Every model call passes ``t`` as a ``(B,1)`` tensor.
    * **NaN/off-disk preservation.** The union of input NaN masks is re-applied to the
      output so space pixels never become finite garbage.
    * **No dependency on the models package.** A model is duck-typed (anything with a
      ``forward``); the in-test dummy models and the real IFNet both satisfy it. We never
      ``import frameflow.models`` here (it may be mid-build).
    * **Kelvin in, Kelvin out.** Inputs and outputs are physical Kelvin float32 arrays.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import constants as C

if TYPE_CHECKING:  # typing only
    import numpy as np


__all__ = [
    "InterpolatedFrame",
    "interpolate_pair",
    "interpolate_recursive",
    "interpolate_pair_nc",
]

Normalizer = Callable[["np.ndarray"], "np.ndarray"]


# ---------------------------------------------------------------------------
# Normalization helpers (model INPUT scaling — distinct from the metric range, P1)
# ---------------------------------------------------------------------------
def _resolve_normalizers(
    dataset: str,
    normalizer: Normalizer | None,
    denormalizer: Normalizer | None,
) -> tuple[Normalizer, Normalizer]:
    """Resolve (normalize, denormalize) callables Kelvin<->[0,1] for model input.

    Preference order:
        1. Explicit ``normalizer`` / ``denormalizer`` passed by the caller.
        2. ``frameflow.data.preprocess.normalize`` / ``denormalize`` if that module is
           importable (the DATA team's real, dataset-aware implementation).
        3. A built-in fixed-range fallback using the shared NORM constants
           (:data:`BT_NORM_VMIN_K` / :data:`BT_NORM_VMAX_K`), NaN-safe.

    The fixed-range fallback guarantees inference works before the DATA package lands.
    """
    import numpy as np  # lazy

    if normalizer is not None and denormalizer is not None:
        return normalizer, denormalizer

    # Try the DATA team's implementation (optional; may not exist yet).
    data_norm: Normalizer | None = None
    data_denorm: Normalizer | None = None
    try:  # pragma: no cover - exercised only when the DATA module is present
        from ..data.preprocess import denormalize as _dn  # type: ignore
        from ..data.preprocess import normalize as _nm  # type: ignore

        data_norm = lambda a: np.asarray(_nm(a), dtype=np.float32)  # noqa: E731
        data_denorm = lambda a: np.asarray(_dn(a), dtype=np.float32)  # noqa: E731
    except Exception:
        data_norm = data_denorm = None

    vmin = float(C.BT_NORM_VMIN_K)
    vmax = float(C.BT_NORM_VMAX_K)
    span = max(vmax - vmin, 1e-6)

    def _fallback_norm(a: np.ndarray) -> np.ndarray:
        x = (np.asarray(a, dtype=np.float32) - vmin) / span
        return x

    def _fallback_denorm(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=np.float32) * span + vmin

    norm = normalizer or data_norm or _fallback_norm
    denorm = denormalizer or data_denorm or _fallback_denorm
    return norm, denorm


def _to_chw(arr: np.ndarray) -> np.ndarray:
    """Return a single-channel ``(1, H, W)`` float32 view of a 2D ``(H, W)`` or ``(1,H,W)`` array."""
    import numpy as np  # lazy

    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        return a[np.newaxis, :, :]
    if a.ndim == 3 and a.shape[0] == 1:
        return a
    raise ValueError(f"expected (H,W) or (1,H,W) frame, got shape {a.shape!r}")


def interpolate_pair(
    model: Any,
    I0: Any,
    I1: Any,
    t: float = 0.5,
    *,
    dataset: str = "goes",
    device: str = "cpu",
    normalizer: Normalizer | None = None,
    denormalizer: Normalizer | None = None,
    fill_nan_for_model: float = 0.0,
    **kw: Any,
) -> np.ndarray:
    """Synthesize one intermediate frame at fraction ``t`` from two bracketing frames.

    Pipeline: NaN-mask the inputs -> normalize Kelvin to the model's input convention ->
    build the **batched** timestep tensor ``t`` of shape ``(1, 1)`` (P2-ONNX; never a bare
    scalar) -> run ``model.forward(I0, I1, t)`` under ``torch.no_grad()`` -> denormalize to
    Kelvin -> restore the union NaN mask. Single-channel ``(1, H, W)`` tensors are used
    throughout (the VFI contract).

    This function is **polymorphic** for backward compatibility with the CLI / CONTRACTS
    §6.1 path form: if ``model`` is a string / :class:`os.PathLike` (an input ``.nc``), the
    call is dispatched to :func:`interpolate_pair_nc` with ``(in0_nc=model, in1_nc=I0, ...)``
    so ``interpolate_pair(in0_nc, in1_nc, t=..., ckpt=..., out_nc=...)`` keeps working.

    Args:
        model: a trained VFI model (duck-typed: anything with ``forward(I0, I1, t)``), or a
            path string (dispatches to the ``.nc`` form).
        I0: earlier bracketing frame, Kelvin ``(H, W)`` or ``(1, H, W)``.
        I1: later bracketing frame, same shape.
        t: interpolation fraction in (0, 1) (``0.5`` == midpoint). Arbitrary-``t`` supported.
        dataset: dataset key for normalization (advisory; used by the DATA normalizer).
        device: torch device for inference (``"cpu"`` here; ``"cuda"`` when available).
        normalizer / denormalizer: optional explicit Kelvin<->model-input callables; if
            omitted, the DATA team's ``preprocess.normalize`` is used when importable, else a
            fixed-range fallback.
        fill_nan_for_model: value substituted for NaN pixels before the model sees them (the
            mask is restored afterwards). 0.0 (after normalization) is a safe neutral value.
        **kw: ignored extras (keeps the call tolerant of caller-specific kwargs).

    Returns:
        The interpolated frame as a Kelvin float32 ``(H, W)`` ``np.ndarray`` with NaN at
        off-disk pixels.
    """
    # --- polymorphic dispatch: path form (CLI / CONTRACTS §6.1) ---------------------------
    if isinstance(model, (str, Path)):
        return interpolate_pair_nc(  # type: ignore[return-value]
            in0_nc=model, in1_nc=I0, t=(I1 if isinstance(I1, (int, float)) else t), **kw
        )

    import numpy as np  # lazy
    import torch  # lazy

    a0 = _to_chw(I0)
    a1 = _to_chw(I1)
    if a0.shape != a1.shape:
        raise ValueError(f"I0 {a0.shape} and I1 {a1.shape} must have the same shape")

    # Union of off-disk (NaN) masks: any pixel NaN in either input is NaN in the output.
    mask_nan = np.isnan(a0) | np.isnan(a1)

    norm, denorm = _resolve_normalizers(dataset, normalizer, denormalizer)

    # Normalize, replacing NaN with a neutral fill so the model never sees NaN.
    n0 = norm(np.where(np.isnan(a0), C.BT_NORM_VMIN_K, a0))
    n1 = norm(np.where(np.isnan(a1), C.BT_NORM_VMIN_K, a1))
    n0 = np.where(mask_nan, fill_nan_for_model, n0).astype(np.float32)
    n1 = np.where(mask_nan, fill_nan_for_model, n1).astype(np.float32)

    dev = torch.device(device)
    t0 = torch.from_numpy(n0[np.newaxis]).to(dev)  # (1, 1, H, W)
    t1 = torch.from_numpy(n1[np.newaxis]).to(dev)
    # P2-ONNX: t MUST be (B, 1) with a leading BATCH dim — NEVER a python/0-d scalar.
    t_tensor = torch.full((1, 1), float(t), dtype=torch.float32, device=dev)

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "to"):
        try:
            model.to(dev)
        except Exception:  # pragma: no cover - some dummies don't implement .to with device
            pass

    with torch.no_grad():
        out = model.forward(t0, t1, t_tensor)
    # Normalize the model output to a single prediction tensor. The MODELS registry
    # contract returns a dict with a "pred" key (IFNet: {"pred","flow","mask"});
    # forward_with_flow-style models return a (It, flow) tuple; some dummies return a
    # bare tensor. Accept all three.
    if isinstance(out, dict):
        out = out.get("pred", out.get("It"))
        if out is None:
            raise KeyError("model output dict has no 'pred'/'It' key")
    elif isinstance(out, (tuple, list)):  # (It, flow, ...) — take the predicted frame
        out = out[0]
    out_np = out.detach().to("cpu").numpy()

    if was_training and hasattr(model, "train"):
        model.train()

    # Collapse (B, C, H, W) -> (H, W) (B=C=1) and denormalize back to Kelvin.
    out_np = np.asarray(out_np, dtype=np.float32)
    while out_np.ndim > 2:
        out_np = out_np[0]
    out_k = denorm(out_np).astype(np.float32)

    out_k[mask_nan[0]] = np.nan
    return out_k


@dataclass
class InterpolatedFrame:
    """One frame in a densified sequence, with provenance.

    Attributes:
        bt: the brightness-temperature field, Kelvin float32 ``(H, W)`` (NaN off-disk).
        t_global: position along the whole sequence in [0, 1] (0 == first observed frame).
        kind: ``"observed"`` (a real input frame, passed through) or ``"interpolated"``.
        t_local: for interpolated frames, the fraction in (0, 1) **within the bracketing
            observed pair** the model was queried at (``0.5`` == midpoint). ``None`` for
            observed frames.
        bracket: ``[i, j]`` indices (into the ORIGINAL observed-frame list) of the two
            observed frames this frame sits between; ``None`` for observed frames.
        time: optional timestamp for this instant (``np.datetime64`` / ISO string), filled
            when the caller provides observed-frame times.
    """

    bt: np.ndarray
    t_global: float
    kind: str
    t_local: float | None = None
    bracket: list[int] | None = None
    time: Any | None = None


def _subdivision_levels(factor: int) -> int:
    """Return the number of binary-subdivision rounds for a power-of-two ``factor``.

    ``factor`` must be one of ``{2, 4, 8}`` (the PS cadence targets 30->15->7.5->3.75 min).
    2 -> 1 round (insert the midpoint), 4 -> 2 rounds, 8 -> 3 rounds.
    """
    if factor not in (2, 4, 8):
        raise ValueError(f"factor must be one of {{2, 4, 8}}, got {factor}")
    return {2: 1, 4: 2, 8: 3}[factor]


def _to_dt64ns(when: Any) -> np.datetime64:
    """Coerce a timestamp to ``datetime64[ns]``, accepting ints (nanoseconds since epoch).

    ``np.datetime64(int)`` raises ("requires a specified unit"), which bites when a caller
    passes ``times`` as a list produced by ``datetime64[ns].tolist()`` (that yields raw int64
    nanoseconds). We therefore treat a plain integer as ns-since-epoch and otherwise defer to
    numpy's normal parsing (ISO strings, ``datetime``, ``np.datetime64`` all work).
    """
    import numpy as np  # lazy

    if isinstance(when, (int, np.integer)):
        return np.datetime64(int(when), "ns")
    return np.datetime64(when)


def _interp_times(t_start: Any, t_end: Any, frac: float) -> Any:
    """Linearly interpolate a timestamp ``frac`` of the way from ``t_start`` to ``t_end``.

    Returns ``None`` if either endpoint is ``None``. Works with ``np.datetime64`` / ISO
    strings / ``datetime`` / integer-ns (coerced to ``datetime64[ns]``).
    """
    if t_start is None or t_end is None:
        return None
    import numpy as np  # lazy

    a = _to_dt64ns(t_start)
    b = _to_dt64ns(t_end)
    delta = (b - a) / np.timedelta64(1, "ns")
    return a + np.timedelta64(int(round(delta * frac)), "ns")


def interpolate_recursive(
    model: Any,
    frames: Sequence[Any],
    factor: int = 2,
    *,
    times: Sequence[Any] | None = None,
    dataset: str = "goes",
    device: str = "cpu",
    normalizer: Normalizer | None = None,
    denormalizer: Normalizer | None = None,
    **kw: Any,
) -> list[InterpolatedFrame]:
    """Densify a sequence of observed frames by ``factor`` via binary subdivision.

    For ``factor`` in ``{2, 4, 8}`` this inserts ``factor - 1`` frames between every pair of
    consecutive observed frames by **recursive halving**: round 1 inserts the midpoint of
    each observed pair (30 -> 15 min); round 2 inserts the midpoints of the resulting
    sub-intervals (15 -> 7.5 min); round 3 again (7.5 -> 3.75 min). Each inserted frame is
    synthesized with :func:`interpolate_pair` from the two **already-available** bracketing
    frames at that round (so later rounds interpolate between an observed and a previously
    interpolated frame — genuine recursion, which the validation suite measures for error
    accumulation, R5 M4).

    The result is the FULL densified sequence in temporal order, each element tagged
    observed-vs-interpolated with its global position, the local fraction it was queried at,
    its originating observed-pair ``bracket``, and (if ``times`` given) its timestamp.

    Args:
        model: a VFI model (duck-typed ``forward(I0, I1, t)``).
        frames: the observed frames, each Kelvin ``(H, W)`` or ``(1, H, W)``; ``>= 2``.
        factor: up-sampling factor, one of ``{2, 4, 8}``.
        times: optional observed-frame timestamps (len == ``len(frames)``); inserted frames
            get linearly interpolated timestamps.
        dataset / device / normalizer / denormalizer: forwarded to :func:`interpolate_pair`.
        **kw: forwarded to :func:`interpolate_pair`.

    Returns:
        ``list[InterpolatedFrame]`` of length ``(len(frames) - 1) * factor + 1``.

    Raises:
        ValueError: if fewer than two frames, or ``factor`` is not in ``{2, 4, 8}``.
    """
    import numpy as np  # lazy

    obs = [np.asarray(f, dtype=np.float32) for f in frames]
    if len(obs) < 2:
        raise ValueError("interpolate_recursive needs at least two observed frames")
    rounds = _subdivision_levels(factor)
    if times is not None and len(times) != len(obs):
        raise ValueError("times must have the same length as frames")

    n_obs = len(obs)

    def _do(I0: Any, I1: Any, frac: float) -> np.ndarray:
        return interpolate_pair(
            model, I0, I1, frac,
            dataset=dataset, device=device,
            normalizer=normalizer, denormalizer=denormalizer, **kw,
        )

    out: list[InterpolatedFrame] = []
    # Process each consecutive observed interval [k, k+1] independently, then concatenate
    # (sharing the boundary observed frame). Within an interval we recursively subdivide.
    for k in range(n_obs - 1):
        left_obs, right_obs = obs[k], obs[k + 1]
        t_left = times[k] if times is not None else None
        t_right = times[k + 1] if times is not None else None
        bracket = [k, k + 1]

        # nodes: list of (global_fraction_within_interval, frame_array). Start with the two
        # observed endpoints; recursively bisect every adjacent gap `rounds` times.
        nodes: list[tuple[float, np.ndarray]] = [(0.0, left_obs), (1.0, right_obs)]
        for _ in range(rounds):
            new_nodes: list[tuple[float, np.ndarray]] = [nodes[0]]
            for a, b in zip(nodes[:-1], nodes[1:], strict=False):
                fa, fA = a
                fb, fB = b
                mid_frac = 0.5 * (fa + fb)
                # local fraction within [fa, fb] is always 0.5 for a bisection
                mid_frame = _do(fA, fB, 0.5)
                new_nodes.append((mid_frac, mid_frame))
                new_nodes.append(b)
            nodes = new_nodes

        # Emit this interval's nodes. Skip the right endpoint except on the last interval
        # (it is the left endpoint of the next interval — avoid duplicating observed frames).
        last_interval = k == n_obs - 2
        for idx, (frac_in_interval, frame) in enumerate(nodes):
            is_right_endpoint = idx == len(nodes) - 1
            if is_right_endpoint and not last_interval:
                continue
            is_observed = frac_in_interval in (0.0, 1.0)
            # global position across the whole observed sequence in [0, 1]
            t_global = (k + frac_in_interval) / (n_obs - 1)
            if is_observed:
                obs_index = k if frac_in_interval == 0.0 else k + 1
                out.append(
                    InterpolatedFrame(
                        bt=obs[obs_index],
                        t_global=float(t_global),
                        kind="observed",
                        t_local=None,
                        bracket=None,
                        time=(times[obs_index] if times is not None else None),
                    )
                )
            else:
                out.append(
                    InterpolatedFrame(
                        bt=frame,
                        t_global=float(t_global),
                        kind="interpolated",
                        t_local=float(frac_in_interval),
                        bracket=list(bracket),
                        time=_interp_times(t_left, t_right, frac_in_interval),
                    )
                )
    return out


# ---------------------------------------------------------------------------
# Path / .nc form (CLI + CONTRACTS §6.1) — reads two .nc frames, writes one .nc
# ---------------------------------------------------------------------------
def interpolate_pair_nc(
    in0_nc: str | Path,
    in1_nc: str | Path,
    *,
    t: float = 0.5,
    ckpt: str = "runs/ckpt/best_ssim.ckpt",
    out_nc: str | Path = "out/interp_nc/mid.nc",
    model: Any | None = None,
    model_name: str = "RIFE",
    model_version: str = "v0.1.0",
    dataset: str = "goes",
    device: str = "cpu",
    **kw: Any,
) -> Path:
    """Read two ``.nc`` frames, synthesize the frame at ``t``, write a schema-correct ``.nc``.

    This is the CONTRACTS §6.1 / CLI entry point. It loads the bracketing frames with
    :func:`frameflow.infer.netcdf_io.read_netcdf`, builds (or accepts) a model, calls
    :func:`interpolate_pair`, and writes the result via
    :func:`frameflow.infer.netcdf_io.write_frame_nc` with attributes from
    :func:`frameflow.contracts.netcdf_attrs`.

    If no ``model`` is supplied, it attempts to load one from ``ckpt`` via
    ``frameflow.models.get_model`` / a checkpoint; should the models package be unavailable
    (e.g. still mid-build) a clear ``RuntimeError`` is raised telling the caller to pass an
    explicit ``model=``. This keeps the inference I/O usable in isolation while still wiring
    cleanly to the trained model in the full pipeline.

    Args:
        in0_nc / in1_nc: paths to the two bracketing ``.nc`` frames.
        t: interpolation fraction in (0, 1).
        ckpt: checkpoint path used to build a model when ``model`` is None.
        out_nc: destination ``.nc`` path.
        model: an explicit VFI model (skips checkpoint loading).
        model_name / model_version: recorded in the output ``.nc`` attributes.
        dataset / device: forwarded to :func:`interpolate_pair`.

    Returns:
        The :class:`pathlib.Path` of the written ``.nc``.
    """
    import numpy as np  # lazy

    from .netcdf_io import read_netcdf, write_netcdf

    ds0 = read_netcdf(in0_nc)
    ds1 = read_netcdf(in1_nc)
    var = "bt"
    a0 = np.asarray(ds0[var].values, dtype=np.float32)
    a1 = np.asarray(ds1[var].values, dtype=np.float32)
    # Drop a leading singleton time axis if present -> (H, W).
    if a0.ndim == 3:
        a0 = a0[0]
    if a1.ndim == 3:
        a1 = a1[0]

    if model is None:
        model = _load_model_from_ckpt(ckpt)

    out_k = interpolate_pair(model, a0, a1, t, dataset=dataset, device=device, **kw)

    # Build coords + time. Use input grid coords; the synthesized instant's time is the
    # linear blend of the two input times.
    lat = np.asarray(ds0["lat"].values, dtype=np.float64)
    lon = np.asarray(ds0["lon"].values, dtype=np.float64)
    t0 = np.datetime64(ds0["time"].values.reshape(-1)[0])
    t1 = np.datetime64(ds1["time"].values.reshape(-1)[0])
    delta = (t1 - t0) / np.timedelta64(1, "ns")
    when = t0 + np.timedelta64(int(round(delta * float(t))), "ns")

    from ..contracts import netcdf_attrs

    attrs = netcdf_attrs(
        source_frames=[str(in0_nc), str(in1_nc)],
        t=float(t),
        model=str(model_name),
        model_version=str(model_version),
        kind="interpolated",
        extra={"crs": str(ds0.attrs.get("crs", C.DEFAULT_GRID_CRS))},
    )
    return write_netcdf(
        bt=out_k,
        lat=lat,
        lon=lon,
        time=np.array([when], dtype="datetime64[ns]"),
        attrs=attrs,
        path=out_nc,
    )


def _load_model_from_ckpt(ckpt: str | Path) -> Any:
    """Best-effort load of a trained VFI model from a checkpoint.

    Tries the MODELS package factory; raises a clear, actionable error if it (or the
    checkpoint) is unavailable so callers know to pass an explicit ``model=``.
    """
    try:  # pragma: no cover - depends on the MODELS package + a real checkpoint
        import torch

        from ..models import get_model  # type: ignore

        obj = torch.load(str(ckpt), map_location="cpu")
        if hasattr(obj, "forward"):
            return obj
        cfg = obj.get("config") if isinstance(obj, dict) else None
        model = get_model(cfg) if cfg is not None else get_model("rife")  # type: ignore[arg-type]
        state = obj.get("model") if isinstance(obj, dict) else obj
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
        return model
    except Exception as exc:
        raise RuntimeError(
            f"interpolate_pair_nc could not load a model from {ckpt!r} "
            f"({type(exc).__name__}: {exc}). Pass an explicit model= to run without a "
            f"checkpoint (the models package may still be in development)."
        ) from exc
