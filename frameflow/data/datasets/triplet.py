"""``TripletDataset`` — a torch-compatible dataset yielding VFI ``Sample`` triplets.

Consumes the canonical Zarr cube (CubeSchema) and serves leave-the-middle-out triplets: for
an interior frame index ``i`` the bracketing frames are ``i-1`` (``I0``) and ``i+1`` (``I1``)
and the target is the true middle frame ``i`` (``It``) at fraction ``t`` (research/03 §5.4,
research/06 §6). Each item is a :class:`~frameflow.contracts.Sample` whose tensors are
single-channel ``(1, H, W)`` (channel-first), normalized to the model-input range via
:func:`frameflow.data.preprocess.normalize`, and NaN-safe (off-disk pixels are filled with a
neutral value and recorded in a returned validity mask).

Design rules honoured:

* **Split by time, never randomly** (avoids temporal leakage; research/06 §6). ``split`` in
  ``{"train", "val", "test", "all"}`` slices contiguous time ranges.
* **t handling:** ``t_mode="fixed"`` always returns ``t=0.5`` (the true geometric midpoint
  for an evenly-spaced triplet); ``t_mode="random"`` draws ``t`` in the open interval
  ``(0, 1)`` per item (still trained against the true middle frame — useful for arbitrary-t
  models that condition on ``t``). ``t`` is always strictly inside ``(0, 1)``.
* **torch is imported lazily** so ``import frameflow.data`` never pulls torch in.

The class subclasses ``torch.utils.data.Dataset`` when torch is present (so it composes with
``DataLoader``); it is fully usable without a DataLoader too.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ... import constants as C
from ...contracts import Sample

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import torch


__all__ = ["TripletDataset"]


def _dataset_base() -> type:
    """Return ``torch.utils.data.Dataset`` if torch is importable, else ``object``.

    Importing torch lazily here keeps ``import frameflow.data`` torch-free while still letting
    :class:`TripletDataset` be a proper ``Dataset`` subclass (so it works with ``DataLoader``)
    whenever torch is installed.
    """
    try:
        from torch.utils.data import Dataset  # lazy
    except Exception:  # pragma: no cover - torch is installed in this env
        return object
    return Dataset


class TripletDataset(_dataset_base()):  # type: ignore[misc]
    """A ``torch.utils.data.Dataset`` of ``(I0, It, I1, t)`` brightness-temperature triplets.

    Args:
        cube_path: path to a Zarr cube (:class:`~frameflow.contracts.CubeSchema`).
        split: ``"train"`` / ``"val"`` / ``"test"`` (contiguous time slices) or ``"all"``.
        patch_size: spatial crop size in pixels. ``None`` (or a size >= the grid) uses the
            full frame. Crops are spatially aligned across the three frames of a triplet.
        normalized: if True, map Kelvin -> model-input range via
            :func:`frameflow.data.preprocess.normalize` (mode ``norm_mode``); if False, return
            raw Kelvin.
        t_mode: ``"fixed"`` (always ``t=0.5``) or ``"random"`` (draw ``t`` in ``(0, 1)``).
        norm_mode: normalization mode passed through (``"fixed_range"`` or ``"zscore"``).
        dataset: dataset key for z-score stats (``norm_mode="zscore"`` only).
        nan_fill: value used to replace NaN (space) pixels AFTER normalization so the network
            never sees NaN; the boolean validity mask is returned in ``meta["mask"]``.
        train_frac / val_frac: fractions of the time axis assigned to train / val (the
            remainder is test). Only used to compute the contiguous split boundaries.
        seed: base RNG seed (random crop origin + random ``t`` are derived per index, so item
            access is deterministic and DataLoader-worker safe).
        var: cube data-variable name (defaults to ``"bt"``).
        **kw: extra keyword args are accepted and ignored (CONTRACTS.md §3.3 ``**kw``), so
            callers like the train ``DataModule`` can forward arbitrary ``dataset_kwargs``.

    The dataset is **lazy**: the cube is opened on first access (not at construction) and the
    ``bt`` array is read frame-slice by frame-slice, so very large cubes are never fully
    materialized.
    """

    def __init__(
        self,
        cube_path: str | Path,
        *,
        split: str = "train",
        patch_size: int | None = 256,
        normalized: bool = True,
        t_mode: str = "fixed",
        norm_mode: str = "fixed_range",
        dataset: str = "synthetic",
        nan_fill: float = 0.0,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 0,
        var: str = C.CUBE_DATA_VAR,
        **kw: Any,
    ) -> None:
        self.cube_path = str(cube_path)
        self.split = split
        self.patch_size = patch_size
        self.normalized = bool(normalized)
        self.t_mode = t_mode
        self.norm_mode = norm_mode
        self.dataset = dataset
        self.nan_fill = float(nan_fill)
        self.train_frac = float(train_frac)
        self.val_frac = float(val_frac)
        self.seed = int(seed)
        self.var = var
        # Forward-compat: accept (and ignore) any extra kwargs the contract's `**kw` allows
        # callers (e.g. the train DataModule's dataset_kwargs) to pass through.
        self.extra_kwargs = dict(kw)

        # Lazily-populated handles (opened on first access).
        self._ds: Any = None
        self._bt: Any = None
        self._times: Any = None
        self._n_t: int | None = None
        self._h: int | None = None
        self._w: int | None = None
        # Interior mid-frame indices (i with both i-1 and i+1 present) for this split.
        self._mid_indices: list[int] | None = None

    # -- lazy open ---------------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._bt is not None:
            return
        import numpy as np  # lazy
        import xarray as xr  # lazy

        ds = xr.open_zarr(self.cube_path, consolidated=False)
        self._ds = ds
        self._bt = ds[self.var]
        self._times = np.asarray(ds["time"].values)
        self._n_t = int(self._bt.sizes["time"])
        self._h = int(self._bt.sizes["y"])
        self._w = int(self._bt.sizes["x"])
        self._mid_indices = self._compute_split_indices(self._n_t)

    def _compute_split_indices(self, n_t: int) -> list[int]:
        """Contiguous time-based split into interior mid-frame indices (no leakage)."""
        interior = list(range(1, n_t - 1))  # need i-1 and i+1
        if not interior:
            return []
        if self.split == "all":
            return interior
        n = len(interior)
        n_train = int(round(n * self.train_frac))
        n_val = int(round(n * self.val_frac))
        # Guarantee non-empty train/val/test when there is enough data.
        n_train = max(n_train, 1) if n >= 1 else 0
        if n >= 3:
            n_val = max(n_val, 1)
            n_train = min(n_train, n - 2)  # leave >=1 each for val and test
        train = interior[:n_train]
        val = interior[n_train : n_train + n_val]
        test = interior[n_train + n_val :]
        return {"train": train, "val": val, "test": test}.get(self.split, interior)

    # -- dataset protocol --------------------------------------------------------------
    def __len__(self) -> int:
        """Number of triplets available for the configured split."""
        self._ensure_open()
        assert self._mid_indices is not None
        return len(self._mid_indices)

    def __getitem__(self, idx: int) -> Sample:
        """Return the :class:`~frameflow.contracts.Sample` triplet at position ``idx``.

        Args:
            idx: 0-based position within the split.

        Returns:
            A :class:`~frameflow.contracts.Sample` with ``I0``/``I1``/``It`` as ``(1, H, W)``
            float32 ``torch.Tensor`` (or numpy if torch is unavailable), ``t`` in ``(0, 1)``,
            and a ``meta`` dict (satellite, timestamps, normalization flag, validity mask,
            crop origin, mid index).

        Raises:
            IndexError: if ``idx`` is out of range for the split.
        """
        import numpy as np  # lazy

        self._ensure_open()
        assert self._mid_indices is not None
        if idx < 0:
            idx += len(self._mid_indices)
        if not (0 <= idx < len(self._mid_indices)):
            raise IndexError(f"TripletDataset index {idx} out of range (len={len(self._mid_indices)})")

        mid = self._mid_indices[idx]
        # Read the three frames as (H, W) float arrays (NaN where off-disk).
        f0 = self._read_frame(mid - 1)
        ft = self._read_frame(mid)
        f1 = self._read_frame(mid + 1)

        # Deterministic per-item RNG (independent of split position) for reproducibility.
        rng = np.random.default_rng(self.seed * 1_000_003 + mid)
        f0, ft, f1, origin = self._maybe_crop(f0, ft, f1, rng)

        # Validity mask (finite in ALL three frames) BEFORE filling NaNs.
        mask = np.isfinite(f0) & np.isfinite(ft) & np.isfinite(f1)

        if self.normalized:
            from ..preprocess import normalize

            f0 = normalize(f0, mode=self.norm_mode, dataset=self.dataset)
            ft = normalize(ft, mode=self.norm_mode, dataset=self.dataset)
            f1 = normalize(f1, mode=self.norm_mode, dataset=self.dataset)

        # NaN-safe: fill non-finite pixels with the neutral fill so the model never sees NaN.
        f0 = self._fill_nan(f0)
        ft = self._fill_nan(ft)
        f1 = self._fill_nan(f1)

        t = self._sample_t(rng)

        i0 = self._to_chw_tensor(f0)
        i1 = self._to_chw_tensor(f1)
        it = self._to_chw_tensor(ft)
        mask_t = self._to_chw_tensor(mask.astype("float32"))

        meta: dict[str, Any] = {
            "satellite": str(getattr(self._ds, "attrs", {}).get("satellite", "UNKNOWN")),
            "channel": str(getattr(self._ds, "attrs", {}).get("channel", "")),
            "normalized": self.normalized,
            "norm_mode": self.norm_mode if self.normalized else None,
            "dataset": self.dataset,
            "mid_index": int(mid),
            "bracket": [int(mid - 1), int(mid + 1)],
            "time_prev": str(self._times[mid - 1]),
            "time_mid": str(self._times[mid]),
            "time_next": str(self._times[mid + 1]),
            "crop_origin": [int(origin[0]), int(origin[1])],
            "mask": mask_t,
        }
        sample: Sample = {"I0": i0, "I1": i1, "It": it, "t": float(t), "meta": meta}
        return sample

    # -- internals ---------------------------------------------------------------------
    def _read_frame(self, i: int) -> "np.ndarray":
        import numpy as np  # lazy

        return np.asarray(self._bt.isel(time=i).values, dtype=np.float32)

    def _maybe_crop(
        self,
        f0: "np.ndarray",
        ft: "np.ndarray",
        f1: "np.ndarray",
        rng: "np.random.Generator",
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]":
        """Take the SAME random crop from all three frames (or full-frame if no crop)."""
        size = self.patch_size
        h, w = f0.shape[-2:]
        if size is None or size <= 0 or size >= h or size >= w:
            return f0, ft, f1, (0, 0)
        y0 = int(rng.integers(0, h - size + 1))
        x0 = int(rng.integers(0, w - size + 1))
        sl = (slice(y0, y0 + size), slice(x0, x0 + size))
        return f0[sl], ft[sl], f1[sl], (y0, x0)

    def _fill_nan(self, arr: "np.ndarray") -> "np.ndarray":
        import numpy as np  # lazy

        out = np.asarray(arr, dtype=np.float32)
        if np.isnan(out).any():
            out = np.where(np.isfinite(out), out, np.float32(self.nan_fill))
        return out

    def _sample_t(self, rng: "np.random.Generator") -> float:
        if self.t_mode == "random":
            # Strictly inside (0, 1): clamp away from the open-interval endpoints.
            return float(min(max(rng.uniform(0.05, 0.95), 1e-3), 1.0 - 1e-3))
        return 0.5

    def _to_chw_tensor(self, arr: "np.ndarray") -> Any:
        """Convert a 2-D ``(H, W)`` array to a channel-first ``(1, H, W)`` tensor/array."""
        import numpy as np  # lazy

        a = np.asarray(arr, dtype=np.float32)[np.newaxis, :, :]  # (1, H, W)
        try:
            import torch  # lazy

            return torch.from_numpy(np.ascontiguousarray(a))
        except Exception:  # pragma: no cover - torch is installed in this env
            return a

    def close(self) -> None:
        """Close the underlying Zarr dataset handle (safe to call multiple times)."""
        if self._ds is not None:
            try:
                self._ds.close()
            finally:
                self._ds = None
                self._bt = None
