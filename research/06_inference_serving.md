# 06 — Fastest Training Pipeline & O(1)/Near-Real-Time Inference & Serving
### ISRO BAH 2026 PS-12 — Satellite Frame Interpolation (TIR / Optical-Flow VFI)

**Author:** Deep-research worker (ML training + HPC inference specialist)
**Date:** 2026-06-20
**Scope:** How to train/fine-tune a RIFE/IFRNet/FILM-class video-frame-interpolation (VFI) model on single-channel thermal-infrared (TIR) geostationary satellite frames *fastest and cheapest*, and how to serve interpolated frames to a web dashboard with genuinely **O(1)** delivery (precompute + CDN) plus an optional near-real-time GPU endpoint.

> **Network status:** Live web research was available. 12+ WebSearch queries + WebFetch on primary sources (RIFE repo, IFRNet paper via ar5iv, FILM, Triton docs, serverless-GPU comparisons, FFCV/DALI, FSDP). All numbers below are cited. Where a claim is from Jan-2026 background knowledge rather than a fetched source, it is marked **[bg]**.

---

## 0. TL;DR — The Recommended Plan

**Model:** Fine-tune a **pretrained IFRNet-small/IFRNet** (or RIFE v4.x "Practical-RIFE") checkpoint. IFRNet gives the best quality/latency/param trade-off (IFRNet 5.0 M params, 25 ms, 35.80 dB Vimeo90K vs RIFE 9.8 M, 26 ms, 35.62 dB — IFRNet-small is 2.8 M / 19 ms / 35.59 dB). Source: IFRNet Table 1.

**Why fine-tune, not train from scratch:** RIFE/IFRNet were trained for **300 epochs** on Vimeo90K with **4× V100/L4 GPUs** — days of compute. Transfer-learning from the released weights and fine-tuning **20–40 epochs** on satellite triplets converges in **hours on one GPU**, because the optical-flow/warping backbone transfers; only the photometric statistics of TIR differ.

**Training recipe (one 16–24 GB GPU):**
- **Framework:** PyTorch + **PyTorch Lightning** (or Lightning Fabric) for AMP/checkpointing/logging boilerplate; keep the original RIFE/IFRNet model code.
- **Precision:** **BF16 mixed** (`bf16-mixed`) on Ampere+ (A100/A10/L4/RTX 30/40); FP16-mixed + `GradScaler` on T4/V100/older. ~2× throughput, ~40–50% memory cut. (PyTorch AMP docs; FSDP blog: bf16 ≈ 5× vs fp32 in their large-model setup.)
- **Resolution:** **256×256 patches** to start (matches RIFE/IFRNet's 224–256 native training crop), move to **512×512** patches once stable. Batch 16–32 at 256², batch 4–8 at 512² on 16–24 GB (math in §7).
- **Loss:** **Charbonnier + Census(ternary)** as the reconstruction core; add **Laplacian-pyramid / gradient loss** for edge sharpness; **keep IFRNet's flow-distillation + geometry-consistency terms if fine-tuning IFRNet**. **Skip VGG/LPIPS perceptual loss for single-channel TIR** unless you replicate the 1-channel→3-channel and accept domain mismatch (VGG is ImageNet-RGB). (§5.)
- **Data loader:** Convert `.nc/.h5` triplets to a **WebDataset tar** (or FFCV `.beton`) of pre-normalized uint16/float16 patches → eliminates the per-step netCDF decode bottleneck. NVIDIA **DALI** if you want GPU-side augmentation. (FFCV beats PyTorch DataLoader/WebDataset/DALI on raw throughput per CVPR'23.)
- **GPU target:** Free tier = **Kaggle T4×2 / Colab T4 or L4**; paid cheap = **RunPod / Vast.ai community RTX 4090 / A10 / L4** (~$0.2–0.5/hr); managed = **Modal** (per-second billing, sub-second cold starts via GPU snapshot).
- **MLOps:** **Weights & Biases** (or MLflow) for tracking, **Hydra/OmegaConf** for config, deterministic seeds, Docker for repro, a model card.

**Serving — the O(1) design (recommended hybrid):**
1. **PRIMARY (O(1) for the dashboard): PRECOMPUTE.** Run batch inference offline over the demo date-range, write interpolated frames as **PNG/WebP tiles + an MP4/WebM time-lapse + the required `.nc`**, push to **object storage behind a CDN** (Cloudflare R2/Pages, S3+CloudFront, or Vercel/Netlify static). The dashboard then does **zero GPU work per request** — it fetches a static, content-addressed URL. This is the genuine O(1) path the user wants and is the demo-safe option (no cold starts, no GPU bill during judging).
2. **OPTIONAL (on-demand): real-time endpoint.** Export the model to **ONNX (opset 17) → TensorRT FP16** and serve via **NVIDIA Triton** (or a thin FastAPI+GPU / Modal function) with **dynamic batching**. Use for "interpolate this new pair now." TensorRT FP16 gives ~**40–100% speedup vs ncnn/Vulkan** and RIFE already runs **30+ FPS at 720p on a 2080Ti**.
3. **Cache layer:** content-addressed keys = `hash(frameA)+hash(frameB)+t+model_ver`; **Redis** for hot results, **CDN** for the precomputed corpus. A repeat request is O(1) cache hit.

**Critical accuracy caveat — do NOT use naive INT8.** Quantizing RIFE/IFRNet to INT8 collapses quality: measured **−0.89 dB (RIFE flow)** to **−4.38 dB (IFRNet frame mode)** in the ANVIL study. **Use FP16/BF16, not INT8**, unless you do full QAT. (§1.4.)

---

## 1. Inference Acceleration

### 1.1 The toolbox and expected speedups

| Technique | What it does | Expected speedup (VFI/vision) | Accuracy impact | Notes for this project |
|---|---|---|---|---|
| **FP16 / BF16** | Half-precision compute on Tensor Cores | **~2×** vs FP32; "more than double on tensor-core GPUs" | negligible | First thing to do. BF16 safer (no loss scaling). |
| **torch.compile (Inductor)** | Graph capture → fused kernels | **1.0–2.5×** inference (RTX 4090; 2.53× for large@512); avg ~1.46× | none | Easiest PyTorch-native win; one line. Note GPU mem ↑ ~50%. |
| **TorchScript** | Static graph, no Python | modest (10–30%) **[bg]** | none | Legacy; prefer torch.compile or ONNX/TRT. |
| **ONNX Runtime** | Cross-platform graph engine | ~1.3–2× over eager CPU/GPU **[bg]** | none | Portable; CUDA/TensorRT/OpenVINO execution providers. |
| **TensorRT (FP16)** | NVIDIA-optimized engine | **+40–100% vs ncnn/Vulkan**; up to **6× vs PyTorch** (Torch-TensorRT blog) | negligible at FP16 | Best latency on NVIDIA. Concrete RIFE: ncnn 30.7 fps → TRT 45.9 fps @1080p FP16 RTX3050. |
| **OpenVINO** | Intel CPU/iGPU engine | 2–4× on Intel CPU **[bg]** | none | Only if you must serve CPU-only (no GPU budget). |
| **INT8 PTQ** | 8-bit weights/acts | ~2–4× + 4× compression | **BAD for VFI** (−0.89 to −4.38 dB) | **Avoid** for RIFE/IFRNet unless QAT. |
| **INT8 QAT** | Quantize-aware training | ~2–4× | recoverable | Only if INT8 is mandatory; costs a training run. |
| **Structured/channel pruning** | Remove channels/filters | 1.3–2× **[bg]** | small if light, then steep | Optional; IFRNet-small already a "pruned" design. |
| **Knowledge distillation** | Small student ← big teacher | depends | can match teacher | IFRNet *already* uses flow-distillation internally; you can distill IFRNet→IFRNet-small on your data. |

Sources: [RIFE/ncnn/TRT SVP wiki](https://www.svp-team.com/wiki/RIFE_AI_interpolation), [Torch-TensorRT NVIDIA blog](https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/), [collabora torch.compile vs TensorRT](https://www.collabora.com/news-and-blog/blog/2024/12/19/faster-inference-torch.compile-vs-tensorrt/), [HF torch.compile docs](https://huggingface.co/docs/transformers/en/perf_torch_compile), [ANVIL INT8 numbers](https://arxiv.org/pdf/2603.26835).

### 1.2 Recommended inference stack (decision tree)
- **GPU available (NVIDIA):** PyTorch → **ONNX (opset 17)** → **TensorRT FP16**. Fallback to **torch.compile + AMP autocast(bf16)** if TRT export is fiddly.
- **GPU available but want minimal effort:** `model.half()` + `torch.compile(model, mode="reduce-overhead")`.
- **CPU-only:** ONNX Runtime + OpenVINO EP, FP32/FP16.

### 1.3 The grid_sample / ONNX export caveat (IMPORTANT for RIFE/IFRNet)
RIFE and IFRNet both use **`F.grid_sample`** for backward warping. Historically this blocked ONNX export ("missing support of grid_sampler operator"). **Current status (use this):**
- **2D `grid_sample` is supported in ONNX opset ≥ 16** and exports cleanly from modern PyTorch.
- **TensorRT ≥ 8.5/8.6 natively imports 2D `GridSample`**; older TRT needed a graph-surgeon rename to a plugin (`GridSample3D`) — only relevant for 3D, which VFI doesn't use.
- **Practical path:** export with `opset_version=17`, fixed or dynamic H/W, then build the TRT engine with `trtexec --fp16`. If you hit a "No importer for GridSample" on an old TRT, upgrade TRT first; community RIFE ONNX/TRT engines already exist (e.g. ComfyUI-Rife-Tensorrt, TensorStack/RIFE on HF).

Sources: [ONNX grid_sample opset16 discussion](https://github.com/onnx/onnx/discussions/5825), [Practical-RIFE ONNX issue #50](https://github.com/hzwer/Practical-RIFE/issues/50), [ECCV-RIFE ONNX issue #33](https://github.com/hzwer/ECCV2022-RIFE/issues/33), [TensorRT GridSample issue #3400](https://github.com/NVIDIA/TensorRT/issues/3400), [grid-sample3d-trt-plugin](https://github.com/SeanWangJS/grid-sample3d-trt-plugin), [yuvraj108c ComfyUI-Rife-Tensorrt](https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt), [TensorStack/RIFE ONNX on HF](https://huggingface.co/TensorStack/RIFE/blob/main/model.onnx).

### 1.4 Quantization verdict
**Use FP16/BF16. Do not ship naive INT8 PTQ for VFI.** The ANVIL paper measured INT8 frame-interpolation degradation of **−0.19 dB (their robust method) up to −4.38 dB (IFRNet frame mode)** and **−0.89 dB (RIFE flow)** — catastrophic for a task graded on PSNR/SSIM. Super-resolution PTQ literature reaches "negligible drop at 8-bit" *only with advanced per-layer calibration* because activation ranges are highly dynamic and asymmetric. If you ever need INT8 (edge/CPU), do **QAT** (the NVIDIA TAO blog shows QAT recovers most INT8 accuracy). For this challenge, the cost/benefit says **stay FP16**.
Sources: [ANVIL](https://arxiv.org/pdf/2603.26835), [Toward Accurate PTQ for SR (CVPR'23)](https://openaccess.thecvf.com/content/CVPR2023/papers/Tu_Toward_Accurate_Post-Training_Quantization_for_Image_Super_Resolution_CVPR_2023_paper.pdf), [NVIDIA QAT+TAO](https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/), [PMQ-VE video quant](https://arxiv.org/pdf/2505.12266).

---

## 2. Serving Frameworks

| Framework | Batching | Multi-framework | Cold start | Best for | Verdict for PS-12 |
|---|---|---|---|---|---|
| **NVIDIA Triton** | **Server-side dynamic batching** (+85% throughput @bs16 vs unbatched in one benchmark; 420→780 req/s) | TensorRT/ONNX/PyTorch/TF, ensembles | container | max GPU throughput, prod | **Best for the optional on-demand endpoint** if you want a real server. |
| **TorchServe** | dynamic batching | PyTorch (.mar) | container | PyTorch-only shops | OK, simpler than Triton, fewer features. |
| **BentoML** | adaptive batching | wraps engines | container | packaging/deploy DX | Good DX; leans on an engine underneath. |
| **Ray Serve** | dynamic batching, autoscale | any Python | container | complex pipelines, autoscale | Overkill unless you need pipeline orchestration. |
| **FastAPI + GPU** | DIY (micro-batch) | any | process | smallest demo | **Simplest real-time endpoint**; fine for a hackathon demo. |
| **Modal** | you batch in fn | any | **sub-second** (GPU snapshot, 7B in seconds) | serverless, per-second billing | **Best cheap serverless** for bursty on-demand. |
| **Replicate** | platform | any | seconds | one-click public demo (~$0.005/req for big LLMs) | Easy public link; **Cloudflare acquired it Nov-2025**. |

Sources: [Triton vs TorchServe (Algoroq)](https://algoroq.io/compare-tech/triton-vs-torchserve-inference/), [Triton overview (Medium)](https://arks0001.medium.com/nvidia-triton-inference-server-d51e96df71f5), [batch-inference at scale](https://www.rohan-paul.com/p/batch-inference-at-scale-processing), [FastAPI vs Triton on K8s (arXiv)](https://arxiv.org/pdf/2602.00053), [serverless GPU comparison (Introl)](https://introl.com/blog/serverless-gpu-platforms-runpod-modal-beam-comparison-guide-2025), [RunPod serverless cold-start guide](https://www.runpod.io/articles/guides/top-serverless-gpu-clouds).

**Recommendation:** For PS-12 the dashboard itself should NOT call any of these per request (see §3). For the *optional* "interpolate now" button: **FastAPI+GPU (demo)** or **Modal (serverless, cheap)**; graduate to **Triton+TensorRT** only if you want to showcase production-grade dynamic batching.

---

## 3. The O(1) Serving Pattern (the heart of the user's ask)

### 3.1 Why O(1) = precompute + static CDN
On-demand GPU inference is O(model FLOPs) per request + cold-start + GPU cost. For a dashboard that replays **fixed** date-ranges (the demo, the INSAT-3DS deliverable), every interpolated frame is **deterministic** given (frameA, frameB, t, model). So compute them **once, offline, in a batch job**, and serve the artifacts as **static files**. Per-request work becomes a **CDN edge fetch = O(1)**, with **zero origin compute** and **zero GPU**. This is exactly the "static-site-generation / build-time render, edge-cached, zero origin compute per request" pattern.

> **Sound-bite for judges:** "Interpolation is precomputed and content-addressed; the dashboard serves frames from a CDN in O(1) — no model runs while you watch the animation."

### 3.2 Architecture
```
            OFFLINE (batch, GPU, once)                         ONLINE (per request, O(1))
  ┌───────────────────────────────────────┐        ┌──────────────────────────────────────┐
  │ .nc/.h5 frames (GOES-19 / INSAT-3DS)   │        │  Browser dashboard (timelapse player) │
  │   → preprocess (normalize TIR)         │        │        │                              │
  │   → TensorRT/ONNX VFI inference (batch) │        │        ▼                              │
  │   → write artifacts:                    │        │   CDN edge (Cloudflare/CloudFront)   │
  │      • PNG/WebP tiles per timestep      │  push  │   - /demo/goes19/2024-…/t075.webp    │
  │      • interp.mp4 / .webm timelapse     │ ─────► │   - /demo/…/timelapse.mp4            │
  │      • required .nc output             │ (R2/S3)│   - /demo/…/metrics.json (SSIM/PSNR) │
  │      • metrics.json (SSIM/PSNR/MSE/FSIM)│        │        ▲                              │
  └───────────────────────────────────────┘        │        │ cache miss (rare)            │
                                                     │   Origin object store (R2/S3)        │
   OPTIONAL on-demand path (new pair):               └──────────────────────────────────────┘
   Browser → FastAPI/Modal/Triton (GPU) → TensorRT infer → write to cache (Redis+CDN) → return URL
```

### 3.3 Caching (content-addressed)
- **Key:** `sha256(frameA_bytes) || sha256(frameB_bytes) || t || model_version`. Deterministic ⇒ same inputs always hit.
- **Hot tier:** **Redis** (or in-process LRU) for recently requested results / metrics JSON.
- **Cold tier / corpus:** object store (Cloudflare **R2** = no egress fees, or **S3+CloudFront**) behind CDN with long `Cache-Control: public, max-age=31536000, immutable` (safe because keys are content-addressed).
- **Invalidation:** new `model_version` ⇒ new keys ⇒ no stale reads; old corpus can be GC'd.

### 3.4 Hybrid recommendation (do this)
1. **Precompute** the full demo + INSAT-3DS deliverable → CDN. (Primary, O(1), demo-safe.)
2. **One small on-demand endpoint** (Modal serverless or FastAPI+GPU) for "interpolate an arbitrary new pair," writing results into the same content-addressed cache so the *second* request for the same pair is O(1).
3. Dashboard always reads from the cache/CDN; the GPU endpoint only ever runs on a genuine cache miss.

Sources: [Edge CDN + static caching (systemsarchitect.io)](https://www.systemsarchitect.io/blog/decision-checklist-edge-computing-cdn-caching-vs-origin-server-dc0005), [Edge CDN & AI inference (nearbycomputing)](https://www.nearbycomputing.com/edge-cdn-and-ai-inference/), [Moving ML inference cloud→edge (Bergum)](https://bergum.medium.com/moving-ml-inference-from-the-cloud-to-the-edge-d6f98dbdb2e3).

---

## 4. Training Infrastructure & Speed

### 4.1 Framework: Lightning vs raw PyTorch
- **PyTorch Lightning / Lightning Fabric:** free AMP, DDP/FSDP, checkpointing, grad-accum, logging hooks, deterministic flags. Recommended to wrap the *existing* RIFE/IFRNet `nn.Module` — you keep their model/warp code, Lightning handles the loop. (Lightning supports `16-mixed`/`bf16-mixed`/`bf16-true`.)
- **Raw PyTorch:** fine for a single GPU; you write the AMP+scaler+ckpt loop yourself (RIFE's repo already has one).
- **Verdict:** single GPU → either; multi-GPU or wanting clean MLOps → **Lightning**.

### 4.2 Speed levers (single GPU)
- **AMP (autocast + GradScaler)** → ~2× throughput, lower memory. Use `bf16` on Ampere+ (no scaler needed), `fp16`+scaler on T4/V100.
- **`torch.compile(model)`** for the training step too (avg ~1.3× train per TorchInductor).
- **Gradient checkpointing** only if memory-bound at 512² (trades ~20–30% compute for big activation savings; FSDP blog cites up to 10× headroom reinvested into batch size).
- **Larger batch via grad accumulation** if you can't fit it.
- **channels_last** memory format for conv nets (free ~1.2× on Tensor Cores) **[bg]**.

### 4.3 DDP vs FSDP
VFI models are **small (3–20 M params)** — they fit easily on one GPU. **You do NOT need FSDP** (FSDP is for models too big for one GPU). If you have 2× T4 (Kaggle) just use **DDP** for ~2× data throughput. FSDP's value (4–20× larger models) is irrelevant here. (FSDP blog.)

### 4.4 Data loading — kill the netCDF bottleneck
Per-step `.nc/.h5` reads + xarray decode will starve the GPU ("data loading/processing is the major bottleneck"). Fix:
1. **Offline ETL once:** read all `.nc`, extract the TIR channel, normalize (per-dataset min/max or brightness-temp scaling), cut overlapping triplets, store as **WebDataset `.tar` shards** of `float16`/`uint16` patches (or **FFCV `.beton`**).
2. **Training:** stream shards with many workers; optionally **NVIDIA DALI** for GPU-side crop/flip/rotate augmentation.
- **FFCV** is the throughput king (CVPR'23: beats DataLoader/WebDataset/DALI) but has a steeper setup; **WebDataset** is the pragmatic choice for `.tar` + cloud/Kaggle.
Sources: [FFCV CVPR'23](https://openaccess.thecvf.com/content/CVPR2023/papers/Leclerc_FFCV_Accelerating_Training_by_Removing_Data_Bottlenecks_CVPR_2023_paper.pdf), [FFCV benchmarks](https://docs.ffcv.io/benchmarks.html), [Data stalls in DNN training](https://arxiv.org/pdf/2007.06775).

### 4.5 Free/cheap GPU options
| Option | GPU | Cost | Notes |
|---|---|---|---|
| **Kaggle** | 2× T4 (16 GB ea) or P100 | **free**, 30 hr/wk | Best free for this; DDP across 2×T4. |
| **Colab** | T4 / L4 / (A100 on Pro+) | free–$10/mo | Session limits; good for prototyping. |
| **Lightning Studio** | free monthly credits | free tier | Integrates with Lightning. |
| **RunPod** | RTX 4090 / A10 / L4 / A100 | ~$0.2–0.6/hr community; serverless per-sec (90% cold starts <2 s) | Cheapest reliable paid; great for the on-demand endpoint too. |
| **Vast.ai** | marketplace 4090/3090/A100 | cheapest spot; Vast Serverless (Dec-2025) | Lowest $/hr, variable reliability. |
| **Modal** | A10/A100/H100 serverless | per-second; **sub-1 s cold start** | Best for serverless inference; $87 M Series-B Sep-2025. |
| **Replicate** | managed | per-run | One-click public demo; Cloudflare-owned. |

Sources: [Introl serverless GPU 2025](https://introl.com/blog/serverless-gpu-platforms-runpod-modal-beam-comparison-guide-2025), [RunPod top serverless](https://www.runpod.io/articles/guides/top-serverless-gpu-clouds), [RunPod vs Vast](https://www.runpod.io/articles/comparison/runpod-vs-vastai-training), [Modal/Replicate market notes](https://www.buildmvpfast.com/blog/serverless-gpu-ai-inference-platform-comparison-2026).

### 4.6 Expected fine-tune time (RIFE/IFRNet-class)
- **From scratch (reference):** 300 epochs, Vimeo90K (~64 k triplets), **4 GPUs** — order of **days**. RIFE/IFRNet both report this.
- **Fine-tune from pretrained on ~tens of thousands of satellite triplets:** **20–40 epochs**. A V100/L4/4090 does a 256² VFI epoch over ~30 k triplets in roughly **15–40 min** **[bg-estimate]**; so **~6–20 GPU-hours total**, i.e. **a few hours on a single 4090/A10**, or two Kaggle sessions on T4×2. (RIFE's own README/issues report fine-tuning "another 20 epochs" on ~11 k frames for the perceptual variant — confirms small fine-tune budgets work.)
Sources: [RIFE repo (30+FPS 720p 2080Ti; 4-GPU train cmd)](https://github.com/hzwer/ECCV2022-RIFE), [IFRNet 300ep/24bs/V100 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2205.14620), [RIFE fine-tune 20ep/11k frames](https://www.researchgate.net/publication/345788379_Real-Time_Intermediate_Flow_Estimation_for_Video_Frame_Interpolation).

### 4.7 Transfer learning strategy (concrete)
1. Load pretrained **IFRNet** (or RIFE v4.x) weights.
2. **Stage 1 (warm-up, 2–5 ep):** freeze the flow-estimation encoder, train only the synthesis/refine head on TIR at LR 1e-4 → adapts to single-channel statistics fast.
3. **Stage 2 (full FT, 15–30 ep):** unfreeze all, cosine LR 1e-4→1e-5 (IFRNet's schedule), full loss.
4. Single-channel input: either **replicate the 1 channel to 3** (cheapest, lets you reuse RGB-pretrained weights verbatim) or **edit conv1 to in_channels=1** and average the pretrained RGB filters → 1-channel kernel (cleaner, slightly better). Recommend **replicate-to-3 first** for zero surgery, then ablate.

---

## 5. Loss Functions for TIR VFI Fine-Tuning

### 5.1 Candidate terms
| Loss | Formula / idea | Good for | TIR caveat |
|---|---|---|---|
| **L1** | `|Î−I|` | baseline fidelity | a bit blurry |
| **Charbonnier** | `ρ(x)=(x²+ε²)^0.5`, ε=1e-3 | robust L1, stable | **default reconstruction** |
| **Census / ternary** | structural consistency of local patches under census transform | robust to illumination/photometric noise | **excellent for clouds** (brightness-temp varies) |
| **SSIM / MS-SSIM loss** | `1−SSIM` | structural similarity, matches eval metric | great — eval uses SSIM |
| **Gradient / Laplacian-pyramid** | match edges/high-freq | crisp cloud boundaries | recommended |
| **VGG perceptual / LPIPS** | distance in VGG-ImageNet feature space | perceptual sharpness | **VGG is RGB/ImageNet** — domain-mismatched for 1-ch TIR; needs 1→3 replication and accept mismatch. **Lower priority.** |
| **Gram-matrix / style (FILM)** | L2 of VGG Gram matrices | inpaint large-motion disocclusions, crispness | same RGB caveat; FILM's signature loss |
| **Flow-distillation (IFRNet/RIFE)** | privileged flow supervision | sharper, faster convergence | **keep if fine-tuning IFRNet/RIFE** |

### 5.2 IFRNet's exact loss (replicate when fine-tuning IFRNet)
From the paper:
- **Reconstruction:** `L_r = ρ(Î_t − I_t^gt) + L_cen(Î_t, I_t^gt)` with `ρ(x)=(x²+ε²)^0.5, α=0.5, ε=1e-3` — i.e. **Charbonnier + census**.
- **Task-oriented flow distillation:** `L_d = Σ_{k=1..3} Σ_{l=0,1} ρ(U_{2k}(F^k_{t→l}) − F^p_{t→l})`.
- **Geometry consistency:** `L_g = Σ_{k=1..3} L_cen(φ̂^k_t, φ^k_t)`.
- **Total:** `L = L_r + λ·L_d + η·L_g`, with **λ=0.01, η=0.01**.
Source: [IFRNet ar5iv](https://ar5iv.labs.arxiv.org/html/2205.14620).

### 5.3 Recommended loss combo for single-channel cloud TIR
```
L = 1.0 * Charbonnier(Î, I_gt)
  + 0.5 * Census(Î, I_gt)            # structural, illumination-robust → key for clouds
  + 0.25 * (1 - MS_SSIM(Î, I_gt))    # aligns with SSIM eval metric
  + 0.1  * GradientLoss(Î, I_gt)     # sharp cloud edges
  + (if IFRNet) 0.01*L_flow_distill + 0.01*L_geo   # keep IFRNet privileged losses
# Omit VGG/LPIPS initially (RGB/ImageNet mismatch on 1-channel TIR).
# If you later want perceptual sharpness, replicate 1→3 channels and add 0.05*LPIPS, ablate on SSIM/PSNR.
```
Rationale: graders use **MSE/PSNR/SSIM/FSIM**; Charbonnier+census+MS-SSIM directly optimize fidelity+structure, gradient loss preserves the cloud-edge detail FSIM rewards, and you avoid the RGB-perceptual domain trap.
Sources: [AceVFI survey (loss taxonomy)](https://arxiv.org/html/2506.01061v1), [census loss role](https://arxiv.org/pdf/2211.06024), [FILM Gram/style loss](https://film-net.github.io/).

---

## 6. Reproducibility / MLOps

- **Tracking:** **Weights & Biases** (best UX, free academic) or **MLflow** (self-host) or **TensorBoard** (offline-safe). Log: loss terms, val SSIM/PSNR/MSE/FSIM, sample interpolations, LR, GPU mem.
- **Config:** **Hydra + OmegaConf** — one YAML per experiment, CLI overrides, multirun sweeps.
- **Checkpointing:** save `model + optimizer + scaler + epoch + config-hash`; keep `best_ssim.ckpt` and `last.ckpt`.
- **Determinism:** `torch.manual_seed`, `numpy`, `random`, `cudnn.deterministic=True` (+ `cudnn.benchmark=False` for exact repro; flip benchmark on for speed once frozen), `PYTHONHASHSEED`. Lightning: `seed_everything(42, workers=True)`.
- **Docker:** pin CUDA/cuDNN/PyTorch/TensorRT; one image for train, one slim for serve.
- **Model registry:** W&B Artifacts or MLflow Registry; tag `model_version` (feeds the content-addressed cache key in §3.3).
- **Model card:** dataset (GOES-19 C13 / INSAT-3DS TIR1 / Himawari), preprocessing, train/val split (by **time**, not random, to avoid leakage), metrics, limitations (fast convection, terminator/day-night, sensor differences GOES↔INSAT), intended use.

---

## 7. Memory / Throughput Math (single 16–24 GB GPU)

**Rough activation-memory model** (VFI is activation-bound, conv-heavy, multi-scale + flow + warp):
```
peak_mem ≈ params_mem + optimizer_state + activations(batch, H, W) + overhead
```
- **Params + AdamW state (fp32 master + m,v):** IFRNet 5 M params → ~5M*4B*(1+2)=~60 MB; negligible. (Models are tiny; activations dominate.)
- **Activations** scale ~linearly with **batch × H × W × channels × #scales**. VFI nets process a feature pyramid + two warps, so the constant is larger than a plain CNN.

**Empirical sizing rules (BF16/FP16, AdamW, no grad-ckpt) — start here, then probe:**
| Resolution (patch) | Suggested batch on **16 GB** (T4/V100) | on **24 GB** (4090/A10/L4-24) |
|---|---|---|
| 256×256 | 16–24 | 32–48 |
| 384×384 | 8–12 | 16–24 |
| **512×512** | **4–8** | **8–16** |
| 720p full-frame (eval only) | 1–2 | 2–4 |

**How to size empirically (do this, don't trust the table blindly):**
1. Set AMP bf16/fp16, pick a resolution, set batch=2, run one step, read `torch.cuda.max_memory_allocated()`.
2. Activation memory ≈ linear in batch → `max_batch ≈ floor((GPU_free − fixed_overhead) / per_sample_mem)`; back off ~15% for fragmentation.
3. If 512² won't fit at batch≥4, enable **gradient checkpointing** (recompute activations) and/or **grad accumulation** to keep effective batch high.
4. **channels_last** + AMP for Tensor-Core utilization.

**Throughput intuition:** RIFE ≈ 26 ms, IFRNet ≈ 25 ms, IFRNet-small ≈ 19 ms per 720p frame (paper timings) → on a modern GPU you interpolate **hundreds of 512² frames/sec in batch** for the precompute job; the whole demo corpus precomputes in **minutes**.
Sources: [IFRNet timings (ar5iv)](https://ar5iv.labs.arxiv.org/html/2205.14620), [PyTorch AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html), [FSDP mem headroom blog](https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/).

---

## 8. Concrete Commands & Snippets

### 8.1 Fine-tune (Lightning Fabric wrapper, BF16, single/2-GPU)
```python
# train_vfi.py  — wraps existing IFRNet/RIFE nn.Module
import lightning as L
import torch, torch.nn.functional as F
from model import IFRNet            # existing repo code
from losses import charbonnier, census, ms_ssim_loss, gradient_loss
from data import make_webdataset    # streams .tar shards of TIR triplets

fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
fabric.launch(); fabric.seed_everything(42, workers=True)

net = IFRNet(); net.load_state_dict(torch.load("ifrnet_pretrained.pth"))  # transfer learning
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-5)
net, opt = fabric.setup(net, opt)
loader = fabric.setup_dataloaders(make_webdataset("shards/train-{000..099}.tar", bs=16, crop=256))

for epoch in range(30):
    for img0, gt, img1, t in loader:               # img: [B,1,H,W] (TIR); replicate→3 inside net if needed
        opt.zero_grad()
        pred, flow_pred, feats = net(img0, img1, t)
        loss = (charbonnier(pred, gt) + 0.5*census(pred, gt)
                + 0.25*ms_ssim_loss(pred, gt) + 0.1*gradient_loss(pred, gt)
                + 0.01*flow_distill(flow_pred) + 0.01*geo_consistency(feats))
        fabric.backward(loss); opt.step()
    sched.step()
    fabric.save("ckpt/last.ckpt", {"model": net, "opt": opt, "epoch": epoch})
```
Raw-PyTorch / DDP equivalent (the RIFE repo already ships this):
```bash
python3 -m torch.distributed.launch --nproc_per_node=2 train.py --world_size=2   # Kaggle T4x2
```
(Source: [RIFE repo train cmd](https://github.com/hzwer/ECCV2022-RIFE).)

### 8.2 Export ONNX → TensorRT FP16 (the real-time endpoint)
```python
import torch
net.eval().cuda()
dummy0 = torch.randn(1,3,512,512, device="cuda")   # or 1-channel if conv1 edited
torch.onnx.export(
    net, (dummy0, dummy0, torch.tensor([0.5], device="cuda")),
    "vfi.onnx", opset_version=17, input_names=["img0","img1","t"], output_names=["mid"],
    dynamic_axes={"img0":{0:"B",2:"H",3:"W"}, "img1":{0:"B",2:"H",3:"W"}, "mid":{0:"B",2:"H",3:"W"}},
)
```
```bash
# Build FP16 engine (2D grid_sample imports natively on TRT >= 8.5/8.6)
trtexec --onnx=vfi.onnx --fp16 --saveEngine=vfi_fp16.plan \
        --minShapes=img0:1x3x256x256,img1:1x3x256x256 \
        --optShapes=img0:4x3x512x512,img1:4x3x512x512 \
        --maxShapes=img0:8x3x512x512,img1:8x3x512x512
```

### 8.3 Triton dynamic-batching config (`config.pbtxt`)
```protobuf
name: "vfi"
platform: "tensorrt_plan"     # or backend: "onnxruntime"
max_batch_size: 16
input  [ { name: "img0" data_type: TYPE_FP16 dims: [3,-1,-1] },
         { name: "img1" data_type: TYPE_FP16 dims: [3,-1,-1] },
         { name: "t"    data_type: TYPE_FP32 dims: [1] } ]
output [ { name: "mid"  data_type: TYPE_FP16 dims: [3,-1,-1] } ]
instance_group [ { count: 1 kind: KIND_GPU } ]
dynamic_batching { preferred_batch_size: [4, 8] max_queue_delay_microseconds: 2000 }
```
(Source: [Triton model_configuration docs](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md).)

### 8.4 O(1) precompute → CDN (offline batch job)
```bash
# 1) batch-interpolate the demo range with the TRT engine, write artifacts
python precompute.py --engine vfi_fp16.plan --src goes19_2024xxxx.nc \
       --out out/demo --emit png,webp,mp4,nc,metrics --model_ver v1.2

# 2) publish to Cloudflare R2 (no egress fees) behind a CDN; immutable cache
rclone copy out/demo r2:vfi-demo/v1.2 --header-upload "Cache-Control: public,max-age=31536000,immutable"
# dashboard now fetches https://cdn.example/vfi-demo/v1.2/goes19/2024xxxx/t075.webp  → O(1)
```

### 8.5 Optional on-demand endpoint with content-addressed cache (FastAPI / Modal)
```python
@app.post("/interpolate")
def interpolate(a: bytes, b: bytes, t: float = 0.5):
    key = sha256(a).hexdigest()[:16] + sha256(b).hexdigest()[:16] + f"_{t}_{MODEL_VER}"
    if (url := redis.get(key)): return {"url": url, "cached": True}      # O(1) hit
    mid = trt_infer(decode(a), decode(b), t)                            # GPU only on miss
    url = put_object_cdn(f"ondemand/{key}.webp", encode(mid))
    redis.set(key, url); return {"url": url, "cached": False}
```

---

## 9. Risks, Pitfalls, and Mitigations
- **INT8 ruins VFI quality** (−0.9 to −4.4 dB). → FP16/BF16 only.
- **grid_sample export** on *old* TensorRT (<8.5) fails. → upgrade TRT, opset 17, 2D only; community engines exist as fallback.
- **netCDF decode starves GPU.** → offline ETL to WebDataset/FFCV.
- **VGG/LPIPS on 1-channel TIR** is domain-mismatched. → census + MS-SSIM + gradient instead; add LPIPS only via 1→3 replication + ablation.
- **Random train/val split leaks temporal neighbors.** → split by **time/date**; never put adjacent timesteps in both sets.
- **Day/night terminator & GOES↔INSAT sensor gap** hurt transfer. → normalize per-sensor (brightness temperature), fine-tune on INSAT for Step-4.
- **Serverless cold starts** during a live demo. → precompute (O(1)) is the demo path; keep on-demand as a "nice-to-have."
- **torch.compile/Inductor raises GPU memory** (~+50% inference). → budget for it or skip compile on the smallest GPUs.

---

## 10. Source List (all fetched/searched this session)
1. RIFE (ECCV2022) repo — train cmd, 30+FPS 720p 2080Ti, model versions: https://github.com/hzwer/ECCV2022-RIFE
2. IFRNet (CVPR2022) full text (ar5iv) — losses, 300ep/bs24/V100, Table 1 numbers: https://ar5iv.labs.arxiv.org/html/2205.14620 ; PDF: https://arxiv.org/pdf/2205.14620
3. FILM (ECCV2022) — Gram/style loss, large motion: https://film-net.github.io/ ; repo: https://github.com/google-research/frame-interpolation
4. RIFE+TensorRT FP16 speedups (SVP wiki): https://www.svp-team.com/wiki/RIFE_AI_interpolation
5. ComfyUI-Rife-Tensorrt (community TRT engines): https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt
6. Torch-TensorRT up-to-6× (NVIDIA): https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
7. torch.compile vs TensorRT (Collabora): https://www.collabora.com/news-and-blog/blog/2024/12/19/faster-inference-torch.compile-vs-tensorrt/
8. torch.compile FP16 1.0–2.5× (HF docs): https://huggingface.co/docs/transformers/en/perf_torch_compile
9. ANVIL — INT8 VFI degradation numbers: https://arxiv.org/pdf/2603.26835
10. Toward Accurate PTQ for SR (CVPR'23): https://openaccess.thecvf.com/content/CVPR2023/papers/Tu_Toward_Accurate_Post-Training_Quantization_for_Image_Super_Resolution_CVPR_2023_paper.pdf
11. NVIDIA QAT + TAO (INT8 recovery): https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/
12. ONNX grid_sample opset16 discussion: https://github.com/onnx/onnx/discussions/5825
13. Practical-RIFE ONNX issue #50: https://github.com/hzwer/Practical-RIFE/issues/50 ; ECCV-RIFE ONNX #33: https://github.com/hzwer/ECCV2022-RIFE/issues/33
14. TensorRT GridSample issue #3400: https://github.com/NVIDIA/TensorRT/issues/3400 ; grid-sample3d plugin: https://github.com/SeanWangJS/grid-sample3d-trt-plugin
15. Triton model configuration (dynamic batching, instance_group): https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md
16. Triton vs TorchServe (Algoroq): https://algoroq.io/compare-tech/triton-vs-torchserve-inference/
17. FastAPI vs Triton on K8s (arXiv): https://arxiv.org/pdf/2602.00053
18. Serverless GPU comparison (Introl 2025): https://introl.com/blog/serverless-gpu-platforms-runpod-modal-beam-comparison-guide-2025
19. RunPod top serverless / cold starts: https://www.runpod.io/articles/guides/top-serverless-gpu-clouds
20. RunPod vs Vast.ai: https://www.runpod.io/articles/comparison/runpod-vs-vastai-training
21. FFCV (CVPR'23) + benchmarks: https://openaccess.thecvf.com/content/CVPR2023/papers/Leclerc_FFCV_Accelerating_Training_by_Removing_Data_Bottlenecks_CVPR_2023_paper.pdf ; https://docs.ffcv.io/benchmarks.html
22. Data stalls in DNN training: https://arxiv.org/pdf/2007.06775
23. PyTorch FSDP efficient large-scale training: https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/
24. PyTorch AMP recipe: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
25. AceVFI survey (loss taxonomy, census/Charbonnier): https://arxiv.org/html/2506.01061v1
26. Progressive Motion Context (census+Charbonnier combo): https://arxiv.org/pdf/2211.06024
27. Optical Flow for Intermediate Frame Interpolation of Multispectral Geostationary Satellite Data (GOES-16 15min→1min, Super SloMo per-band) — domain precedent: https://ntrs.nasa.gov/citations/20190033878
28. Edge CDN + static caching (systemsarchitect.io): https://www.systemsarchitect.io/blog/decision-checklist-edge-computing-cdn-caching-vs-origin-server-dc0005
29. Moving ML inference cloud→edge (Bergum): https://bergum.medium.com/moving-ml-inference-from-the-cloud-to-the-edge-d6f98dbdb2e3

*End of report.*
