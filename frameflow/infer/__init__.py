"""FrameFlow inference — synthesize intermediate TIR frames and export the model.

Team **INFER+VALIDATE** (see ``CONTRACTS.md`` §6). This subpackage turns a trained
:class:`~frameflow.models.base.VFIModel` into interpolated brightness-temperature frames
and the artefacts the rest of the system consumes:

Public modules:
    * :mod:`frameflow.infer.netcdf_io` — read/write the CF-style ``.nc`` output that
      PS-12 explicitly requires (:class:`frameflow.contracts.InferenceNetCDFSchema`):
      :func:`~frameflow.infer.netcdf_io.write_netcdf`,
      :func:`~frameflow.infer.netcdf_io.read_netcdf`.
    * :mod:`frameflow.infer.interpolate` — the core synthesis routines:
      :func:`~frameflow.infer.interpolate.interpolate_pair` (one intermediate frame at an
      arbitrary fraction ``t``) and
      :func:`~frameflow.infer.interpolate.interpolate_recursive` (binary subdivision to a
      2x/4x/8x densified sequence with observed-vs-interpolated provenance).
    * :mod:`frameflow.infer.batch` — :func:`~frameflow.infer.batch.interpolate_cube`,
      which densifies a whole Zarr cube and writes each frame as ``.nc``.
    * :mod:`frameflow.infer.export` — ONNX / TensorRT export
      (:func:`~frameflow.infer.export.to_onnx`), honouring the P2-ONNX rule that the
      timestep ``t`` carries a leading BATCH dimension so Triton dynamic batching works.

To keep imports light, heavy/optional dependencies (``onnx``, ``netCDF4`` engines) are
imported lazily inside the functions that need them; importing this package never pulls in
``frameflow.models`` (so it works even while the models package is mid-build).
"""

from __future__ import annotations

from .batch import interpolate_cube
from .export import to_onnx, to_tensorrt, triton_config_pbtxt
from .interpolate import (
    InterpolatedFrame,
    interpolate_pair,
    interpolate_pair_nc,
    interpolate_recursive,
)
from .netcdf_io import read_netcdf, write_frame_nc, write_netcdf

__all__ = [
    "write_netcdf",
    "write_frame_nc",
    "read_netcdf",
    "interpolate_pair",
    "interpolate_pair_nc",
    "interpolate_recursive",
    "interpolate_cube",
    "InterpolatedFrame",
    "to_onnx",
    "to_tensorrt",
    "triton_config_pbtxt",
]
