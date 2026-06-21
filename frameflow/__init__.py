"""FrameFlow — AI/ML optical-flow temporal interpolation of geostationary satellite imagery.

ISRO Bharatiya Antariksh Hackathon (BAH) 2026 — Problem Statement 12
"Fill in the Frames Seamlessly: Enhancing Temporal Resolution of Satellite Imagery
using AI/ML based on Optical Flow."

FrameFlow takes two consecutive geostationary Thermal-IR (~10 µm) brightness-temperature
frames (e.g. 00:00 and 00:20) and synthesizes the intermediate frame(s) (e.g. 00:10),
densifying cadence (30 min -> 15 min -> 7.5 min) without new satellite hardware. The
primary engine is RIFE / Practical-RIFE (IFNet), fine-tuned on single-channel
brightness temperature from GOES-19 ABI C13 / Himawari-9 AHI B13, and deployed on
INSAT-3DS/3DR TIR1.

This top-level package deliberately does NOT import its submodules at import time. Six
independent build teams land code into subpackages (``data``, ``models``, ``train``,
``infer``, ``validate``, ``serve``, ``viz``) and a top-level ``precompute`` module; the
authoritative interface contract lives in :mod:`frameflow.contracts`, shared physical
constants in :mod:`frameflow.constants`, and a runnable synthetic data generator in
:mod:`frameflow.synthetic`. Importing those three is always safe even before the teams'
modules exist.

See ``CONTRACTS.md`` for the per-module public API every team implements against, and
``ARCHITECTURE.md`` for the system design and citations to the ``research/`` reports.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
