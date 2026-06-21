"""Write VFI triplets to sharded sample files (WebDataset tar shards, or ``.npz`` fallback).

When dataloader throughput (not the model) is the bottleneck, baking the pre-extracted
``(I0, It, I1)`` triplets into sharded sample files lets the loader stream large, sequential
reads instead of many small random Zarr reads (research/03 §5.4). This module materializes a
:class:`~frameflow.data.datasets.triplet.TripletDataset` (or any sequence of
:class:`~frameflow.contracts.Sample`) into:

* **WebDataset tar shards** (default) — each sample is a group of files sharing a ``__key__``
  inside a ``.tar`` (``<key>.i0.npy``, ``<key>.i1.npy``, ``<key>.it.npy``, ``<key>.t.txt``,
  ``<key>.json``). Plain ``tarfile`` is used, so the artifact is readable by the
  ``webdataset`` library *and* by :func:`read_shard` here, with **no extra dependency**.
* **``.npz`` shards** (fallback / no-tar option) — each shard is a single compressed
  ``.npz`` holding stacked arrays ``i0``/``i1``/``it``/``t`` plus a JSON sidecar of metadata.

Both layouts are sharded (``maxcount`` samples per shard) so they parallelize across loader
workers. torch tensors are accepted and converted to numpy on write.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from ...contracts import Sample


__all__ = ["write_shards", "read_shard"]


def write_shards(
    samples: "Iterable[Sample] | Any",
    out_dir: str | Path,
    *,
    fmt: str = "tar",
    maxcount: int = 1000,
    prefix: str = "frameflow",
    compress: bool = True,
) -> list[Path]:
    """Write samples to sharded files and return the shard paths.

    Args:
        samples: an iterable of :class:`~frameflow.contracts.Sample` dicts (e.g. a
            :class:`~frameflow.data.datasets.triplet.TripletDataset`, which is iterable).
        out_dir: destination directory (created if missing).
        fmt: ``"tar"`` (WebDataset tar shards, default) or ``"npz"`` (compressed-array
            shards). Both shard at ``maxcount`` samples per file.
        maxcount: maximum number of samples per shard.
        prefix: shard filename prefix (``<prefix>-000000.tar`` / ``.npz``).
        compress: for ``fmt="npz"`` use ``np.savez_compressed`` (else ``np.savez``); ignored
            for tar shards.

    Returns:
        The list of written shard :class:`pathlib.Path` objects, in order.

    Raises:
        ValueError: on an unknown ``fmt`` or ``maxcount <= 0``.
    """
    if maxcount <= 0:
        raise ValueError("write_shards: maxcount must be positive")
    if fmt not in ("tar", "npz"):
        raise ValueError(f"write_shards: unknown fmt {fmt!r} (expected 'tar' or 'npz').")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if fmt == "tar":
        return _write_tar_shards(samples, out, maxcount, prefix)
    return _write_npz_shards(samples, out, maxcount, prefix, compress)


# ---------------------------------------------------------------------------
# tar (WebDataset) shards
# ---------------------------------------------------------------------------
def _write_tar_shards(
    samples: "Iterable[Sample]", out: Path, maxcount: int, prefix: str
) -> list[Path]:
    import numpy as np  # lazy

    shard_paths: list[Path] = []
    tar: tarfile.TarFile | None = None
    shard_idx = -1
    count_in_shard = maxcount  # force a new shard on the first sample

    def _open_new_shard() -> tarfile.TarFile:
        nonlocal shard_idx
        shard_idx += 1
        p = out / f"{prefix}-{shard_idx:06d}.tar"
        shard_paths.append(p)
        return tarfile.open(p, "w")

    try:
        for global_i, s in enumerate(samples):
            if count_in_shard >= maxcount:
                if tar is not None:
                    tar.close()
                tar = _open_new_shard()
                count_in_shard = 0
            key = f"{global_i:08d}"
            for field, arr_key in (("I0", "i0"), ("I1", "i1"), ("It", "it")):
                arr = _to_numpy(s[field]).astype(np.float32)
                _add_npy(tar, f"{key}.{arr_key}.npy", arr)
            _add_bytes(tar, f"{key}.t.txt", f"{float(s['t']):.6f}".encode("utf-8"))
            meta = _json_safe_meta(s.get("meta", {}))
            _add_bytes(tar, f"{key}.json", json.dumps(meta).encode("utf-8"))
            count_in_shard += 1
    finally:
        if tar is not None:
            tar.close()
    return shard_paths


def _add_npy(tar: "tarfile.TarFile", name: str, arr: "np.ndarray") -> None:
    import numpy as np  # lazy

    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    _add_bytes(tar, name, buf.getvalue())


def _add_bytes(tar: "tarfile.TarFile", name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


# ---------------------------------------------------------------------------
# npz shards
# ---------------------------------------------------------------------------
def _write_npz_shards(
    samples: "Iterable[Sample]", out: Path, maxcount: int, prefix: str, compress: bool
) -> list[Path]:
    import numpy as np  # lazy

    shard_paths: list[Path] = []
    buf_i0: list[np.ndarray] = []
    buf_i1: list[np.ndarray] = []
    buf_it: list[np.ndarray] = []
    buf_t: list[float] = []
    buf_meta: list[dict[str, Any]] = []
    shard_idx = 0

    def _flush() -> None:
        nonlocal shard_idx, buf_i0, buf_i1, buf_it, buf_t, buf_meta
        if not buf_i0:
            return
        p = out / f"{prefix}-{shard_idx:06d}.npz"
        saver = np.savez_compressed if compress else np.savez
        saver(
            p,
            i0=np.stack(buf_i0).astype(np.float32),
            i1=np.stack(buf_i1).astype(np.float32),
            it=np.stack(buf_it).astype(np.float32),
            t=np.asarray(buf_t, dtype=np.float32),
        )
        p.with_suffix(".json").write_text(json.dumps(buf_meta))
        shard_paths.append(p)
        shard_idx += 1
        buf_i0, buf_i1, buf_it, buf_t, buf_meta = [], [], [], [], []

    for s in samples:
        buf_i0.append(_to_numpy(s["I0"]).astype(np.float32))
        buf_i1.append(_to_numpy(s["I1"]).astype(np.float32))
        buf_it.append(_to_numpy(s["It"]).astype(np.float32))
        buf_t.append(float(s["t"]))
        buf_meta.append(_json_safe_meta(s.get("meta", {})))
        if len(buf_i0) >= maxcount:
            _flush()
    _flush()
    return shard_paths


def read_shard(path: str | Path) -> "list[dict[str, Any]]":
    """Read a shard written by :func:`write_shards` back into a list of sample dicts.

    Handy for tests/round-trips and for inspecting shards without the ``webdataset`` library.

    Args:
        path: a ``.tar`` or ``.npz`` shard path.

    Returns:
        A list of dicts with keys ``i0``/``i1``/``it`` (numpy arrays), ``t`` (float), and
        ``meta`` (dict).

    Raises:
        ValueError: if the file extension is neither ``.tar`` nor ``.npz``.
    """
    import numpy as np  # lazy

    p = Path(path)
    if p.suffix == ".tar":
        groups: dict[str, dict[str, Any]] = {}
        with tarfile.open(p, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                key, _, field = member.name.partition(".")
                data = tar.extractfile(member).read()  # type: ignore[union-attr]
                rec = groups.setdefault(key, {})
                if field == "i0.npy":
                    rec["i0"] = np.load(io.BytesIO(data), allow_pickle=False)
                elif field == "i1.npy":
                    rec["i1"] = np.load(io.BytesIO(data), allow_pickle=False)
                elif field == "it.npy":
                    rec["it"] = np.load(io.BytesIO(data), allow_pickle=False)
                elif field == "t.txt":
                    rec["t"] = float(data.decode("utf-8"))
                elif field == "json":
                    rec["meta"] = json.loads(data.decode("utf-8"))
        return [groups[k] for k in sorted(groups)]

    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            i0, i1, it, t = z["i0"], z["i1"], z["it"], z["t"]
        meta_path = p.with_suffix(".json")
        metas = json.loads(meta_path.read_text()) if meta_path.exists() else [{}] * len(t)
        return [
            {"i0": i0[k], "i1": i1[k], "it": it[k], "t": float(t[k]), "meta": metas[k]}
            for k in range(len(t))
        ]

    raise ValueError(f"read_shard: unsupported shard extension {p.suffix!r} (want .tar/.npz)")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_numpy(x: Any) -> "np.ndarray":
    """Convert a torch tensor / array-like to a numpy array (CPU, detached)."""
    import numpy as np  # lazy

    if hasattr(x, "detach") and hasattr(x, "cpu"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _json_safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Drop / coerce non-JSON-serializable meta values (e.g. tensor masks) for the sidecar."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            if all(isinstance(e, (str, int, float, bool)) for e in v):
                out[k] = list(v)
        # else: skip array/tensor-valued entries (e.g. the validity mask).
    return out
