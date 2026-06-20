"""FrameFlow typed configuration dataclasses (the Python side of the Hydra YAMLs).

These dataclasses give every stage a typed, defaulted config object. The Hydra YAML files
in ``configs/`` mirror this structure; Hydra/OmegaConf instantiates them at runtime. All
defaults are reasonable real values consistent with the research reports (BF16 mixed
precision, 256->512 patch curriculum, Charbonnier+census+MS-SSIM+gradient loss weights,
the FIXED Kelvin metric data_range, etc.).

Pure standard library — safe to import anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import constants as C


@dataclass
class DataConfig:
    """Data ingest / cube / dataloader configuration."""

    satellite: str = C.DEFAULT_SATELLITE
    channel: str = "C13"
    # Source access
    source: str = "synthetic"  # one of: synthetic | goes19 | himawari9 | gk2a | insat3ds
    cube_path: str = "data/cubes/synthetic.zarr"
    demo_nc_dir: str = "data/demo_nc"
    # Common analysis grid (lat/lon)
    bbox_west_south_east_north: list[float] = field(
        default_factory=lambda: list(C.DEFAULT_GRID_BBOX)
    )
    crs: str = C.DEFAULT_GRID_CRS
    resolution_deg: float = C.DEFAULT_GRID_RESOLUTION_DEG
    # Cube layout (R3 §8.1)
    chunks: list[int] = field(default_factory=lambda: list(C.CUBE_CHUNKS))
    shards: list[int] = field(default_factory=lambda: list(C.CUBE_SHARDS))
    compressor: str = C.CUBE_COMPRESSOR
    # Normalization (R3 §5.2). mode: "fixed_range" -> [vmin,vmax]->[0,1]; "zscore" -> mean/std.
    norm_mode: str = "fixed_range"
    norm_vmin_k: float = C.BT_NORM_VMIN_K
    norm_vmax_k: float = C.BT_NORM_VMAX_K
    stats_json: str = C.STATS_JSON_FILENAME
    # Training triplet sampling
    patch_size: int = 256
    cadence_min: int = C.DEFAULT_INPUT_CADENCE_MIN
    num_workers: int = 8
    # Time-based split fractions (NEVER random — avoid temporal leakage; R6 §6)
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1


@dataclass
class ModelConfig:
    """VFI model configuration (primary engine: RIFE / Practical-RIFE = IFNet; R1 §0)."""

    name: str = "rife"  # one of: rife | ifrnet | ifrnet_s | ema_vfi | super_slomo | film
    variant: str = "v4.25"
    pretrained: str = ""  # path/URL to pretrained weights (transfer learning; R6 §4.7)
    in_channels: int = 1  # single-channel brightness temperature
    # Single-channel adaptation strategy (R1 §7.1; R6 §4.7).
    # "replicate3" reuses RGB weights verbatim; "avg_conv1" averages RGB filters to 1ch.
    channel_adapt: str = "replicate3"
    arbitrary_t: bool = True  # native t in (0,1)
    params_m: float = 9.8  # approx parameter count in millions (RIFE ~10M)


@dataclass
class TrainConfig:
    """Training / fine-tuning configuration (R6 §0, §4, §5)."""

    epochs: int = 30
    batch_size: int = 16
    lr: float = 1.0e-4
    lr_min: float = 1.0e-5
    weight_decay: float = 1.0e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    precision: str = "bf16-mixed"  # BF16 on Ampere+; fall back to 16-mixed on T4/V100
    # Two-stage transfer (R6 §4.7): warm-up head, then full fine-tune.
    warmup_epochs: int = 3
    grad_accum: int = 1
    channels_last: bool = True
    compile: bool = False  # torch.compile (raises GPU mem ~50%); enable when stable
    seed: int = 42
    # Patch-size curriculum: start 256, move to 512 once stable (R6 §0).
    patch_size_start: int = 256
    patch_size_final: int = 512
    # Loss weights (R6 §5.3) — Charbonnier + census + MS-SSIM + gradient (+IFRNet privileged).
    loss_charbonnier: float = 1.0
    loss_census: float = 0.5
    loss_ms_ssim: float = 0.25
    loss_gradient: float = 0.1
    loss_flow_distill: float = 0.01  # IFRNet/RIFE privileged flow distillation
    loss_geo_consistency: float = 0.01  # IFRNet geometry consistency
    charbonnier_eps: float = 1.0e-3
    # Tracking
    tracker: str = "tensorboard"  # tensorboard | wandb | mlflow | none
    out_dir: str = "runs"
    ckpt_dir: str = "runs/ckpt"


@dataclass
class InferConfig:
    """Interpolation / inference configuration (R6 §1, §3)."""

    ckpt: str = "runs/ckpt/best_ssim.ckpt"
    model_version: str = "v0.1.0"
    interpolation_factor: int = 2  # 2 -> 30->15 min; 4 -> 7.5 min
    timesteps: list[float] = field(default_factory=lambda: [0.5])  # which t in (0,1) to emit
    precision: str = "fp16"  # fp16/bf16 only — INT8 ruins VFI quality (R6 §1.4)
    device: str = "cuda"
    # Export (R6 §1.3, §8.2). CODE-REVIEW CORRECTION P2 (ONNX): give t a leading BATCH dim
    # [B,1] and include it in dynamic_axes so Triton dynamic batching works.
    export_onnx: bool = False
    onnx_opset: int = 17
    onnx_path: str = "runs/export/vfi.onnx"
    trt_fp16: bool = True
    trt_engine: str = "runs/export/vfi_fp16.plan"
    # NetCDF output
    out_nc_dir: str = "out/interp_nc"


@dataclass
class ValidateConfig:
    """Validation / metrics / cross-validation configuration (R5)."""

    # CODE-REVIEW CORRECTION P1: full-reference metrics use the FIXED physical Kelvin range.
    bt_data_range_k: float = C.BT_DATA_RANGE_K
    bt_vmin_k: float = C.BT_METRIC_VMIN_K
    bt_vmax_k: float = C.BT_METRIC_VMAX_K
    use_per_image_minmax: bool = False  # MUST stay False (P1) — guards against regressions
    # Headline metric set (R5 §0)
    metrics: list[str] = field(
        default_factory=lambda: [
            "psnr", "ssim", "ms_ssim", "fsim", "bt_rmse_k", "csi_235k", "fss", "epe",
        ]
    )
    # Cold-cloud BT thresholds (K) for categorical skill (R5 §3.1)
    bt_thresholds_k: list[float] = field(default_factory=lambda: [235.0, 220.0, 210.0, 200.0])
    fss_scales_px: list[int] = field(default_factory=lambda: [1, 3, 9, 27, 81])
    # Which of the 40 cross-validation methods to run (R5 §4); empty == headline subset.
    crossval_methods: list[str] = field(default_factory=list)
    bootstrap_n: int = 1000  # bootstrap CIs (R5 §5.9)
    dual_impl_check: bool = True  # M40: compute key metrics with two libs and assert agree
    out_dir: str = "out/validation"


@dataclass
class ServeConfig:
    """Serving / precompute / web-artifact configuration (R3 §6, R4, R6 §3)."""

    # Precompute artifacts (the O(1) path)
    tile_size: int = C.DEFAULT_TILE_SIZE
    min_zoom: int = C.DEFAULT_MIN_ZOOM
    max_zoom: int = C.DEFAULT_MAX_ZOOM
    tile_format: str = C.DEFAULT_TILE_FORMAT
    colormap: str = C.DEFAULT_COLORMAP
    # CODE-REVIEW CORRECTION P2: per-frame tile sources. These templates are formatted per
    # frame index so each timestamp gets its OWN XYZ pyramid and PMTiles archive.
    tiles_url_template: str = "tiles/{index:03d}/{{z}}/{{x}}/{{y}}.webp"
    pmtiles_template: str = "pmtiles/{index:03d}.pmtiles"
    make_video: bool = True
    video_fps: int = 12
    out_dir: str = "out/web"
    manifest_name: str = "manifest.json"
    # Optional on-demand endpoint (R6 §2, §3.4)
    serve_host: str = "0.0.0.0"
    serve_port: int = 8000
    redis_url: str = "redis://localhost:6379/0"
    triton: bool = False


@dataclass
class FrameFlowConfig:
    """Top-level config aggregating every stage (the root Hydra config)."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    validate: ValidateConfig = field(default_factory=ValidateConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    seed: int = 42
    project: str = C.PROJECT_NAME


__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "InferConfig",
    "ValidateConfig",
    "ServeConfig",
    "FrameFlowConfig",
]
