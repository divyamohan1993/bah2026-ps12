"""Datasets: torch-compatible triplet dataset + sharded sample writers (CONTRACTS.md §3.3).

* :class:`~frameflow.data.datasets.triplet.TripletDataset` — a ``torch.utils.data.Dataset``
  that reads the canonical Zarr cube and yields :class:`~frameflow.contracts.Sample` triplets
  ``(I0, It, I1, t)`` as ``(1, H, W)`` tensors, normalized + NaN-safe, split by time.
* :func:`~frameflow.data.datasets.webdataset.write_shards` — bake a dataset (or any sample
  sequence) into WebDataset tar shards (or ``.npz`` shards) for high-throughput loading;
  :func:`~frameflow.data.datasets.webdataset.read_shard` reads them back.

torch is imported lazily inside :mod:`.triplet`, so importing this package never pulls torch.
"""

from __future__ import annotations

from .triplet import TripletDataset
from .webdataset import read_shard, write_shards

__all__ = ["TripletDataset", "write_shards", "read_shard"]
