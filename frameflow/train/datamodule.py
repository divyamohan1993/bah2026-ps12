"""FrameFlow training DataModule — a PyTorch Lightning wrapper over triplet datasets.

:class:`VFIDataModule` wraps the DATA team's :class:`frameflow.data.datasets.TripletDataset`
(lazy-imported, so this module imports cleanly before that team lands code) and exposes
train/val ``DataLoader``s to Lightning.

CRITICAL — TIME-BASED SPLIT (no temporal leakage)
-------------------------------------------------
Video-frame-interpolation triplets ``(I0, It, I1)`` are drawn from *consecutive* frames, so
adjacent triplets share frames. A naive random train/val split would put a triplet and its
temporal neighbour (which overlaps two of its three frames) in different splits, leaking the
answer (research/06 §6, §9). We therefore split **by timestamp**: the earliest fraction of
the timeline is train, the latest fraction is val (and optionally test). This guarantees the
validation frames are temporally disjoint from training frames.

The split is implemented two ways, in order of preference:
    1. If the underlying dataset supports a ``split=`` constructor argument (the
       :class:`TripletDataset` contract does), we instantiate one dataset per split and trust
       it to honour the time-based contract.
    2. Otherwise (e.g. an injected in-memory dataset for tests), we sort the dataset's
       samples by timestamp and take contiguous head/tail index ranges — never a random
       shuffle of the indices into different splits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset

from .. import config as _config

if TYPE_CHECKING:  # type-checkers only
    pass


__all__ = ["VFIDataModule", "time_based_split_indices"]


def time_based_split_indices(
    timestamps: Sequence[Any],
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    """Compute contiguous, time-ordered train/val/test index ranges (no leakage).

    Sorts sample indices by their timestamp and slices the sorted order into three
    contiguous blocks: the earliest ``train_frac`` for training, the next ``val_frac`` for
    validation, and the remainder for test. Because the blocks are contiguous in time, no
    timestamp appears in two splits and temporally-adjacent (overlapping) triplets stay
    together (research/06 §6).

    Args:
        timestamps: one sortable timestamp per dataset sample (any orderable type:
            ``datetime``, ``numpy.datetime64``, ISO strings, or floats).
        train_frac: fraction of the timeline used for training.
        val_frac: fraction used for validation (test gets ``1 - train - val``).

    Returns:
        ``(train_idx, val_idx, test_idx)`` lists of integer indices into the original
        (unsorted) sample order.
    """
    n = len(timestamps)
    if n == 0:
        return [], [], []
    order = sorted(range(n), key=lambda i: timestamps[i])
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    # Guarantee at least one sample in train and val when there is enough data.
    n_train = max(1, min(n_train, n - 1)) if n >= 2 else n
    n_val = max(0, min(n_val, n - n_train))
    if n >= 2 and n_val == 0:
        n_val = 1
        n_train = min(n_train, n - n_val)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _sample_timestamp(sample: Any, fallback_index: int) -> Any:
    """Best-effort extraction of a sortable timestamp from a Sample-like object.

    Looks in ``sample["meta"]`` for ``time`` / ``timestamp`` / ``t0`` keys (the
    :class:`frameflow.contracts.Sample` ``meta`` dict carries timestamps); falls back to the
    sample's position so ordering is still deterministic and contiguous.
    """
    meta = None
    if isinstance(sample, dict):
        meta = sample.get("meta")
    elif hasattr(sample, "get"):
        try:
            meta = sample.get("meta")  # type: ignore[call-arg]
        except Exception:
            meta = None
    if isinstance(meta, dict):
        for key in ("time", "timestamp", "t0", "start_time", "time0"):
            if key in meta and meta[key] is not None:
                return meta[key]
    return fallback_index


class VFIDataModule(pl.LightningDataModule):
    """LightningDataModule serving VFI triplets with a strict time-based train/val split.

    Either point it at a Zarr cube (it will lazy-import and build
    :class:`frameflow.data.datasets.TripletDataset` per split) or inject a ready dataset
    object (handy for tests and for decoupling from the DATA team). Injected datasets are
    split into contiguous time-ordered subsets so there is never temporal leakage.

    Args:
        cube_path: path to the analysis Zarr cube (used when no ``dataset`` is injected).
        dataset: an optional pre-built ``torch.utils.data.Dataset`` of
            :class:`~frameflow.contracts.Sample`-like items. If given, it is split by time
            into train/val subsets and ``cube_path`` is ignored.
        batch_size: dataloader batch size.
        num_workers: dataloader worker processes.
        patch_size: spatial patch size passed to :class:`TripletDataset`.
        normalized: whether the dataset should return normalized ``[0, 1]`` tensors.
        train_frac: fraction of the timeline used for training (time-based split).
        val_frac: fraction used for validation.
        pin_memory: pass-through to the dataloaders (auto-disabled on CPU by Lightning).
        persistent_workers: keep workers alive between epochs (only when ``num_workers>0``).
        dataset_kwargs: extra keyword args forwarded to the :class:`TripletDataset`
            constructor.
    """

    def __init__(
        self,
        cube_path: str | None = None,
        *,
        dataset: Dataset | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        patch_size: int = 256,
        normalized: bool = True,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        dataset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.cube_path = cube_path
        self._injected_dataset = dataset
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.patch_size = int(patch_size)
        self.normalized = bool(normalized)
        self.train_frac = float(train_frac)
        self.val_frac = float(val_frac)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.dataset_kwargs = dict(dataset_kwargs or {})

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        # ``save_hyperparameters`` would try to pickle the injected dataset; avoid it.

    # ------------------------------------------------------------------ factory helpers
    @classmethod
    def from_config(
        cls,
        data_cfg: "_config.DataConfig",
        *,
        dataset: Dataset | None = None,
        batch_size: int | None = None,
    ) -> "VFIDataModule":
        """Build a :class:`VFIDataModule` from a :class:`frameflow.config.DataConfig`.

        Args:
            data_cfg: the data configuration (cube path, split fractions, workers, ...).
            dataset: optional injected dataset (overrides ``cube_path``).
            batch_size: optional override of the dataloader batch size.

        Returns:
            A configured (not-yet-``setup``) :class:`VFIDataModule`.
        """
        return cls(
            cube_path=data_cfg.cube_path,
            dataset=dataset,
            batch_size=batch_size if batch_size is not None else 16,
            num_workers=data_cfg.num_workers,
            patch_size=data_cfg.patch_size,
            normalized=(data_cfg.norm_mode != "none"),
            train_frac=data_cfg.train_frac,
            val_frac=data_cfg.val_frac,
        )

    # ------------------------------------------------------------------ Lightning hooks
    def setup(self, stage: str | None = None) -> None:
        """Materialize the train/val datasets with a time-based split.

        Idempotent: re-running does not rebuild already-built splits. Uses the injected
        dataset if provided (contiguous time-ordered subsets), otherwise lazy-imports
        :class:`frameflow.data.datasets.TripletDataset` and instantiates one per split.
        """
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        if self._injected_dataset is not None:
            self._setup_from_injected(self._injected_dataset)
            return

        self._setup_from_cube()

    def _setup_from_injected(self, ds: Dataset) -> None:
        """Split an injected dataset into contiguous time-ordered train/val subsets."""
        n = len(ds)  # type: ignore[arg-type]
        # Pull a timestamp per sample for ordering. We read each sample once; for the tiny
        # in-memory datasets this is cheap. If samples are expensive, the DATA team's
        # split= path (below) avoids this entirely.
        timestamps: list[Any] = []
        for i in range(n):
            try:
                timestamps.append(_sample_timestamp(ds[i], i))
            except Exception:
                timestamps.append(i)
        train_idx, val_idx, _test_idx = time_based_split_indices(
            timestamps, train_frac=self.train_frac, val_frac=self.val_frac
        )
        # Ensure both splits are non-empty for a usable training run.
        if not val_idx and train_idx:
            val_idx = [train_idx[-1]]
            train_idx = train_idx[:-1] or [val_idx[0]]
        self.train_dataset = Subset(ds, train_idx)
        self.val_dataset = Subset(ds, val_idx)

    def _setup_from_cube(self) -> None:
        """Lazy-import :class:`TripletDataset` and build one dataset per split."""
        if not self.cube_path:
            raise ValueError(
                "VFIDataModule needs either an injected `dataset` or a `cube_path` "
                "pointing at a Zarr cube."
            )
        try:
            from ..data.datasets import TripletDataset  # lazy; owned by Team DATA
        except ImportError as exc:  # pragma: no cover - depends on sibling team
            raise ImportError(
                "frameflow.data.datasets.TripletDataset is not available yet (owned by "
                "Team DATA). Inject a `dataset=` object into VFIDataModule instead, or wait "
                "for the DATA module to land. See CONTRACTS.md §3.4."
            ) from exc

        common = dict(
            patch_size=self.patch_size,
            normalized=self.normalized,
            **self.dataset_kwargs,
        )
        # The TripletDataset contract takes split="train"/"val" and is required to split by
        # time internally (CONTRACTS.md §3.4 + research/06 §6). We trust that contract here.
        self.train_dataset = TripletDataset(self.cube_path, split="train", **common)
        self.val_dataset = TripletDataset(self.cube_path, split="val", **common)

    # ------------------------------------------------------------------ dataloaders
    def _make_loader(self, dataset: Dataset | None, *, shuffle: bool) -> DataLoader:
        """Construct a ``DataLoader`` with this module's worker/batch settings."""
        if dataset is None:
            raise RuntimeError("DataModule.setup() must run before requesting a dataloader.")
        persistent = self.persistent_workers and self.num_workers > 0
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=persistent,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training ``DataLoader`` (shuffled within the train-time block)."""
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation ``DataLoader`` (no shuffle)."""
        return self._make_loader(self.val_dataset, shuffle=False)
