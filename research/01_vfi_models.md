# Deep Research: State-of-the-Art Optical-Flow Video Frame Interpolation (VFI) for Satellite/Weather Imagery (2025-2026)

**Prepared for:** ISRO BAH 2026 — Problem Statement 12, "Fill in the Frames Seamlessly: Enhancing Temporal Resolution of Satellite Imagery using AI/ML based on Optical Flow."

**Task recap:** Given two consecutive geostationary Thermal-IR (~10 µm) frames (e.g. 00:00 and 00:20), synthesize the intermediate frame(s) (e.g. 00:10). Train/validate on dense GOES-19 ABI Channel 13 (10-min, 10.3 µm clean longwave IR window) and Himawari-8/9 AHI; deploy on INSAT-3DS/3DR TIR1. I/O in `.nc`/`.h5`. Optimize jointly for **speed (near-real-time)** and **quality**, with a **commercial/government-friendly license** and **easy fine-tuning on single-channel brightness-temperature data**.

**Method note:** This report is based on ~16 live web searches and several full-text fetches (June 2026), supplemented by training knowledge current to Jan 2026. Where a number could not be independently re-verified live, it is flagged. All key claims carry URLs in the Sources section.

---

## 0. TL;DR — Ranked Recommendation

| Role | Model | Why |
|---|---|---|
| **PRIMARY (implement this)** | **RIFE / Practical-RIFE v4.25–4.26 (IFNet)** | Fastest real-time flow-based VFI (~10 ms @ 512², ~30–60+ FPS @ 720p on one GPU); MIT license (commercial/ISRO-deployment safe); native **arbitrary-time** t∈(0,1) via temporal encoding; tiny (~10 M params) → trivial to fine-tune on single-channel TIR; the PS itself names RIFE. Best speed/quality/license/fine-tune balance. |
| **QUALITY CHALLENGER #1** | **EMA-VFI** (CVPR 2023, Apache-2.0) | Top-tier accuracy (Vimeo90K PSNR 36.64); hybrid CNN+Transformer; supports fixed **and** arbitrary timestep; small (25 ms) and base (132 ms) variants let you trade speed↔quality. Strong, honest "high-quality" baseline. |
| **QUALITY CHALLENGER #2 (large/non-linear motion)** | **GIMM-VFI** (NeurIPS 2024) | SOTA on large-motion/arbitrary-time benchmarks (X-Test, SNU-FILM-arb); decouples a tiny implicit motion module (0.25 M params) on top of a pretrained flow backbone (RAFT or FlowFormer). Best when cloud motion is large/fast. Heavier but excellent for the "non-linear cloud dynamics" the PS calls out. |
| **MANDATORY BASELINE (the PS literally names it + is the closest prior art)** | **Super SloMo**, ideally the **task-specific GOES variant** (Vandal & Nemani, IEEE TNNLS 2021) | The PS names Super SloMo; and the single most relevant prior work is a Super-SloMo adaptation to GOES-R ABI brightness temperature. Reimplement as the scientific baseline you beat. |
| **CLASSICAL / NON-AI BASELINE** | **Linear blend + classical optical flow (Farnebäck / TV-L1 warp)** | Required to demonstrate "AI beats traditional optical flow" — exactly the PS's motivating claim. Cheap to add, makes your metrics story compelling. |
| **OPTIONAL "large-motion" baseline** | **FILM** (Google, ECCV 2022, Apache-2.0) | Designed for large motion, robust, well-engineered, easy to run. Slower (~0.4 s @ 720p) and **fixed t=0.5 only** out of the box (recurse for others), so not primary — but a credible large-motion comparator. |
| **OPTICAL-FLOW BACKBONE (if a model needs an external flow net)** | **RAFT** (default, robust) → upgrade to **SEA-RAFT** (ECCV 2024, faster+accurate) or **GMFlow** (best on very large displacement) | For cloud motion (large, non-rigid, non-linear). RIFE/EMA-VFI estimate flow internally and need no external net; GIMM-VFI/AMT-style methods benefit from RAFT/GMFlow-class flow. |

**One-line answer:** Implement **RIFE (Practical-RIFE v4.x)** as the primary deliverable, fine-tuned on single-channel GOES-19 Ch13 + Himawari TIR, and benchmark it against **EMA-VFI**, **GIMM-VFI** (for large motion), **Super SloMo** (named baseline / closest prior art), **FILM** (large-motion comparator), and a **classical optical-flow + linear** baseline. Use SSIM/PSNR/MSE/RMSE-in-Kelvin/LPIPS, and report per-regime (calm vs. convective/cyclone). Deploy the trained RIFE on INSAT-3DS/3DR.

---

## 1. The single most important domain insight

**This exact problem has been solved before in the literature, and it tells you what works.**

> Vandal, T. & Nemani, R., **"Temporal Interpolation of Geostationary Satellite Imagery with Task-Specific Optical Flow"** (arXiv 1907.12013; published IEEE Transactions on Neural Networks and Learning Systems, 2021; NASA NTRS 20210020625).

What they did and found (full-text extracted):
- **Base model:** adapted **Super SloMo (SSM)** — two U-Nets: one for optical flow, one for interpolation + visibility (occlusion) maps, trained **end-to-end**.
- **Data:** **GOES-R ABI all 16 bands**, mesoscale, native **1-min** cadence; training pairs are **10 min apart with a random intermediate label** `I_t` (i.e. arbitrary-time supervision); 256×256 crops; train 2018 / test 2019.
- **Satellite-specific changes:** process **one channel at a time** (C=1 per model); **standardized (per-channel) normalization** of brightness temperature; visibility maps absorb intensity changes from convection.
- **Three variants:** SSM-G (one global model, all bands), SSM-T (task-specific, one model per band), SSM-TMS (task-specific + multi-scale 3/5/7 kernels).
- **Results (t=0.5, 10-min gap), Band 13 (the clean-IR band = exactly your channel):**
  - Linear baseline PSNR **38.667** → **SSM-T PSNR 45.439** (+6.8 dB).
  - RMSE **2.286 K → 0.991 K** (sub-Kelvin error).
  - SSIM **0.782 → 0.933**.
- **Conclusions:** task-specific flow + multi-scale blocks beat both bilinear and *global* optical flow, especially for **high-frequency severe weather**; learned flows are physically realistic; visibility maps handle occlusion and intensity change during convection.

**Implications for your project (directly actionable):**
1. Per-channel (single-channel) training is the right design; you don't need RGB.
2. Standardize brightness temperature per channel; keep raw Kelvin for reporting RMSE-in-Kelvin (a more physically meaningful metric than PSNR for TIR).
3. **Task-specific (fine-tuned on satellite) beats generic** — so *fine-tune*, don't just run pretrained weights.
4. Report a **linear baseline** and a **global/classical optical-flow baseline** — the paper shows AI's gain is largest exactly there (and the PS asks you to show this).
5. Sub-Kelvin RMSE and SSIM ≥ 0.93 on Ch13 at 10-min is a realistic, citable quality bar to target/beat.

Super SloMo is older/slower than modern VFI, so you should **modernize the backbone to RIFE** while keeping these satellite-specific recipes — and keep Super SloMo itself as a named baseline.

---

## 2. Taxonomy of VFI methods (and where each family fits this PS)

The June-2025 survey **AceVFI** (arXiv 2506.01061) groups VFI into:
- **Flow-based (motion-explicit):** estimate optical flow → warp → blend. *Best fit for this PS* (the PS mandates optical flow; physically interpretable motion vectors; fast). RIFE, IFRNet, EMA-VFI, AMT, FILM, Super SloMo, XVFI, M2M, GIMM-VFI, BiM-VFI, Super SloMo.
- **Kernel-based:** learn per-pixel adaptive convolution kernels (no explicit flow). SepConv/SepConv++, CAIN, AdaCoF. Handles small motion well, struggles with large displacement — **not ideal** for fast cloud motion, and doesn't satisfy the "optical flow" requirement.
- **Hybrid CNN+Transformer:** EMA-VFI, VFIformer, BiFormer. Strong accuracy, moderate speed.
- **State-space (Mamba):** VFIMamba (NeurIPS 2024) — linear-complexity inter-frame modeling, SOTA-class on large motion.
- **Generative / diffusion / hallucination-based:** LDMVFI, MCVD, VIDIM, MoMo, TLB-VFI, PerVFI, Framer. Best *perceptual* quality on textured natural video, but **slow** and can **hallucinate** structures — risky for a scientific product where pixels are physical measurements (brightness temperature). Useful only as a perceptual-quality upper-bound comparator, not as the deliverable.

**For a near-real-time, physically-faithful, optical-flow satellite product → flow-based is the correct family.** Diffusion is a research-comparison, not the primary engine.

---

## 3. Flow-based VFI models — detailed comparison

### 3.1 Comparison table (the core decision matrix)

| Model | Year/Venue | Flow mechanism | Speed (approx.) | Params | Vimeo90K PSNR | Arbitrary t? | Multi-frame | License | Repo | TIR fine-tune ease |
|---|---|---|---|---|---|---|---|---|---|---|
| **RIFE / Practical-RIFE v4.x** | 2020→2024 (ECCV'22) | **IFNet**: directly estimates *intermediate* flow, coarse-to-fine IFBlocks, no external flow net | **~10 ms @512²**; ~30–60+ FPS @720p on 1 modern GPU; 4–27× faster than SuperSloMo/DAIN | **~10 M** | ~35.6 (v3.x); v4.x higher | **Yes** (temporal encoding; v4.x designed for any t) | Yes (recursive + native t) | **MIT** | github.com/hzwer/Practical-RIFE ; ECCV2022-RIFE | **Excellent** (small, clean training code) |
| **IFRNet** | CVPR 2022 | Single encoder-decoder; jointly refines bilateral flow + features | Very fast; **11.5× faster than ABME** at similar params | ~5–20 M (S/L) | ~35.8 (L) | Yes (variant) | Yes | **MIT** | github.com/ltkong218/IFRNet | Excellent |
| **EMA-VFI** | CVPR 2023 | Inter-frame **attention** reused for motion + appearance; hybrid CNN+Transformer | small **25 ms**, base **132 ms** @512² (V100) | small 14.5 M / base 65.7 M | **36.64** (base) | **Yes** (fixed + arbitrary) | Yes | **Apache-2.0** | github.com/MCG-NJU/EMA-VFI | Very good |
| **AMT** | CVPR 2023 | **All-pairs** bidirectional correlation volumes; multi-field bilateral flows | Efficient (CNN, competitive w/ transformers) | ~3–25 M (S/L) | ~36.5 (L) | Yes | Yes | (check repo) | github.com/MCG-NKU/AMT | Good |
| **VFIMamba** | NeurIPS 2024 | SSM/S6 Mixed-SSM Block, interleaved tokens, linear complexity | Efficient (linear) | moderate | **36.64**, SSIM 0.9819 | Yes | Yes | (check repo) | github.com/MCG-NJU/VFIMamba | Good (newer code) |
| **GIMM-VFI** | NeurIPS 2024 | **Implicit motion** (INR) on top of pretrained **RAFT/FlowFormer** flow; predicts flow at any t | Heavier (RAFT/FlowFormer backbone) + tiny 0.25 M INR head | flow backbone + **0.25 M** | high | **Yes (native, by design)** | Yes | (check repo) | github.com/GSeanCDAT/GIMM-VFI | Moderate (two-stage) |
| **FILM** | ECCV 2022 (Google) | Scale-agnostic shared-weight motion; no external flow net | **~0.393 s @720p (V100)**; 3.95× faster than ABME | ~types vary | strong (large motion) | **No (t=0.5 only; recurse)** | Recursive only | **Apache-2.0** | github.com/google-research/frame-interpolation | Good (TF; PyTorch ports exist) |
| **XVFI** | ICCV 2021 | Recursive multi-scale BiOF-I/BiOF-T; complementary flow reversal | Scalable (start at any scale) | moderate | — (X-Test focus) | Yes | Yes (recursive) | (check repo) | github.com/JihyongOh/XVFI | Moderate |
| **BiM-VFI** | CVPR 2025 | **Bidirectional Motion field** for *non-uniform* motion; KD flow supervision | moderate | moderate | high | **Yes (arbitrary t)** | Yes | (check repo) | (CVPR2025) | Newer — fewer integrations |
| **M2M-VFI** | CVPR 2022 | Many-to-many splatting; PWC-Net flow | Fast | small | ~35.4 | Yes | Yes | (check) | github.com/feinanshan/M2M_VFI | Good |
| **Super SloMo** | CVPR 2018 | Two U-Nets: flow + visibility; arbitrary-time warp+blend | Moderate (older) | ~40 M | ~34.0 | **Yes** (designed for it) | Yes | (various reimpl.) | many | **Excellent** (simple; satellite-proven) |
| **FLAVR** | 2020 | **Flow-agnostic** 3D CNN (no explicit flow) | Fast | ~40 M | ~36.3 | limited (fixed multiples) | Yes | (check) | github.com/tarun005/FLAVR | N/A flow (doesn't meet "optical flow" req cleanly) |
| **DAIN** | CVPR 2019 | Depth-aware flow + PWC-Net | **Slow** | ~24 M | ~34.7 | Yes | Yes | (research) | github.com/baowenbo/DAIN | Poor (heavy, dated) |
| **SepConv/++** | ICCV'17 / 2021 | Adaptive separable **kernels** (no flow) | Fast | ~21 M | ~33.8 | No (t=0.5) | No | (research, often NC) | — | N/A flow |
| **CAIN** | AAAI 2020 | Channel-attention, **no flow** | Fast | ~42 M | ~34.7 | No | No | (check) | — | N/A flow |

Notes: Vimeo90K PSNR figures are the widely-cited literature values for the 2× (t=0.5) setting; exact numbers vary by training recipe/version. "Speed" entries cite the most directly comparable published measurement found (mixed GPUs/resolutions — see Sources). License "(check repo)" = not independently re-verified live in this pass; verify the LICENSE file before shipping.

### 3.2 Per-model notes most relevant to this PS

**RIFE / Practical-RIFE (PRIMARY)**
- IFNet directly estimates the *intermediate* flow (from t to both inputs) end-to-end — no separate, slow flow network — which is exactly why it is fast and why it naturally outputs the intermediate frame the PS wants.
- v4.x line is explicitly built for **arbitrary-timestep** interpolation via temporal encoding, so 00:05 / 00:10 / 00:15 (t=0.25/0.5/0.75) and "30→15→7.5 min" come for free; recursion gives ×4/×8.
- **MIT license** (confirmed on both `hzwer/ECCV2022-RIFE` and `hzwer/Practical-RIFE` LICENSE files) — clean for ISRO deployment and any commercialization.
- ~10 M params + simple training scripts → fastest path to fine-tune on single-channel TIR. v4.17 added FILM's Gram/style loss; v4.25 added more flow blocks. Practical-RIFE also ships Lite variants for edge/real-time.
- Caveat: official training code is "for reference"; expect to clean it up. The architecture is small enough that this is easy.

**EMA-VFI (QUALITY CHALLENGER #1)**
- Reuses one inter-frame attention map for both motion extraction and appearance enhancement → strong accuracy at good efficiency. **Apache-2.0.** Vimeo90K 36.64. small/base variants give a speed↔quality dial. Supports fixed and arbitrary timestep. Excellent honest "best-quality-still-fast" comparator and a viable alternative primary if RIFE quality is insufficient on convective scenes.

**GIMM-VFI (QUALITY CHALLENGER #2 — large/non-linear motion)**
- Models motion as a continuous implicit neural representation on top of pretrained **RAFT** (GIMM-VFI-R) or **FlowFormer** (GIMM-VFI-F) flow, predicting flow at *any* t. SOTA-class on large-motion/arbitrary-time benchmarks: XTest-2K 32.7–32.9 dB, SNU-FILM Hard(8×) ~32.6 dB, Extreme(16×) ~28.0 dB; lowest-LPIPS on SNU-FILM-arb Extreme (0.058). The tiny 0.25 M motion head is appealing, but you inherit a heavy flow backbone → slower. Best choice if cyclone/fast-cloud cases show RIFE blurring; otherwise overkill for near-real-time.

**FILM (large-motion comparator)**
- Google's large-motion specialist; robust, well-engineered, Apache-2.0, easy demos. But default operates at **t=0.5 only** (recurse for other times) and ~0.4 s/720p → not real-time and not natively arbitrary-time. Keep as a large-motion quality comparator, not the deliverable.

**XVFI / BiM-VFI / VFIMamba (large-/non-uniform-motion frontier)**
- XVFI: recursive multi-scale, built for 4K *extreme* motion (relevant analogy to fast convective tops). BiM-VFI (CVPR 2025): explicitly targets **non-uniform** motion (accel/decel/direction change) — conceptually ideal for cloud growth/dissipation; +26%/45% LPIPS/STLPIPS over prior SOTA; newer, fewer integrations. VFIMamba: linear-complexity SSM, SOTA-class. All are good optional comparators if you want a "frontier" entry.

**Super SloMo (NAMED BASELINE / closest prior art)**
- Named in the PS and the basis of the GOES task-specific paper (§1). Reimplement as the scientific baseline; it is *the* apples-to-apples comparison for "did modern VFI improve on the established satellite approach."

**Avoid as primary:** kernel-only (SepConv, CAIN) and flow-agnostic (FLAVR) — they don't cleanly satisfy the "based on optical flow" requirement and handle large cloud motion worse. DAIN is heavy/dated. These can appear only as ablation comparators if at all.

---

## 4. Optical-flow estimators (for models needing an external flow net)

RIFE, IFRNet, EMA-VFI, AMT estimate flow **internally** — you don't need a separate flow net for the primary path. You need an external estimator only for GIMM-VFI-style methods or if you build a custom "flow → warp → fuse" pipeline (or to produce the explicit motion-vector visualization the PS dashboard may want).

| Estimator | Strength for cloud motion | Speed | Notes |
|---|---|---|---|
| **RAFT** | Robust, accurate iterative refinement; great general default; handles non-rigid motion well | Moderate (iteration count scales time) | The de-facto baseline; GIMM-VFI-R uses it. Safe choice. |
| **SEA-RAFT** | RAFT accuracy, **≥2.3× faster**; SOTA on Spring (EPE 3.69); better generalization | **Fast** | ECCV 2024 Oral. Best speed/accuracy RAFT-family pick for near-real-time. Repo: princeton-vl/SEA-RAFT. |
| **GMFlow** | **Best on very large displacement** (global attention/matching at 1/8 scale) | Slower (~10× GMFlow vs efficient methods) | Use when motion is very large (fast cyclone bands). Global matching suits big cloud displacement. |
| **FlowFormer / FlowFormer++** | Highest accuracy | **Very slow** (~70× slower than SEA-RAFT on RTX 2080) | Quality ceiling; GIMM-VFI-F uses it. Not for real-time. |
| **GMA** | RAFT + global motion aggregation; better in occlusion | Moderate | Good occlusion handling (cloud edges). |
| **PWC-Net / SpyNet** | Light, fast, classic | Fast | Lower accuracy on large/non-rigid motion; used in older M2M/DAIN. Fine as a cheap baseline. |
| **NeuFlow / NeuFlow v2** | Real-time on edge devices | **Very fast** | If INSAT deployment must run on constrained hardware. |

**Recommendation:** For cloud motion (large, non-rigid, non-linear): **RAFT** as the robust default; **SEA-RAFT** to hit near-real-time with minimal accuracy loss; **GMFlow** reserved for very-large-displacement (fast cyclone) cases. But note your **primary RIFE path needs none of these** — they matter mainly for the GIMM-VFI comparator and for motion-vector visualization.

---

## 5. Diffusion / generative VFI (research comparators only)

| Model | Idea | Quality | Speed | Risk for this PS |
|---|---|---|---|---|
| **LDMVFI** (AAAI 2024) | First latent-diffusion VFI; conditional generation | Strong **perceptual** quality, esp. high-res | Slow (multi-step sampling) | Hallucination of structure; pixels aren't physical measurements |
| **MoMo** (AAAI 2025) | Diffusion over **optical flow** (not pixels); two-stage | Superior perceptual at **lower cost** than pixel-diffusion | Faster than pixel-space diffusion, still > flow-only | Better than LDMVFI for science (diffuses motion, warps real pixels) |
| **VIDIM** (2024) | Cascaded diffusion, high fidelity | High fidelity, large motion | Slow | Heavy |
| **TLB-VFI** (2025) | Temporal-aware latent Brownian-bridge diffusion | SOTA perceptual | Slow | Research-grade |
| **PerVFI / Framer** | Perception-oriented / controllable interpolation | High perceptual | Slow | Research-grade |

**Verdict:** Diffusion VFI is the **perceptual-quality frontier** but is slow and can **fabricate** plausible-but-wrong cloud structures — unacceptable as the primary engine for a measurement product judged on MSE/PSNR/SSIM vs. ground truth. If you want one generative comparator, pick **MoMo** (it diffuses *flow* then warps real pixels, so it stays closer to physical content) and present it purely as a perceptual upper bound. **Do not ship diffusion as the deliverable.**

---

## 6. Weather/satellite-specific temporal models — and the crucial interpolation-vs-forecasting distinction

**Critical scoping point:** Most "nowcasting" models **forecast/extrapolate** the *future* from past frames. Your PS is **interpolation** — you have *both* bracketing frames (00:00 and 00:20) and must fill the middle (00:10). Interpolation is an easier, better-posed problem and is what optical-flow VFI is built for. Don't mistakenly build a forecaster.

| Model | Task | Relevance |
|---|---|---|
| **Vandal & Nemani — task-specific SSM on GOES-R** (IEEE TNNLS 2021) | **Interpolation** (TIR) | **THE closest prior art.** Adapt directly (see §1). |
| **Afzali Gorooh, Delle Monache et al. — 3D U-Net diffusion nowcasting of IR brightness temperature** (Scientific Reports, Jan 2026) | **Nowcasting** (forecast, 6 h hist → 6 h @15-min) | Not interpolation, but proves diffusion > ConvLSTM/3D-U-Net/optical-flow extrapolation for IR Tb on SSIM/CRPS; great for your "why not just optical flow" / metrics framing. |
| **"Temporal Frame Interpolation for robust precipitation forecaster"** (arXiv 2311.18341) | VFI **as data augmentation** for nowcasting | Evidence VFI improves weather DL; supports using VFI-densified frames. |
| **DGMR** (DeepMind, Nature 2021) | Nowcasting (radar, GAN) | Context: generative nowcasting; ensemble/CRPS evaluation ideas. |
| **NowcastNet** (Nature 2023) | Nowcasting (physics + GAN, semi-Lagrangian advection) | Context: physically-constrained advection; advection ≈ optical-flow warp. |
| **MetNet-1/2/3** (Google) | Nowcasting (large-context NN) | Context only. |
| **Earthformer** (NeurIPS 2022) | Spatiotemporal forecasting (cuboid attention) | Context; backbone option for DiffCast. |
| **DiffCast** (CVPR 2024) | Nowcasting (residual diffusion + deterministic backbone) | Context: deterministic backbone + diffusion residual idea. |
| **PreDiff, DiffCast, CasCast** | Nowcasting (diffusion) | Context. |
| **Pangu-Weather / FengWu / GraphCast** | Global NWP (medium-range) | Out of scope (synoptic forecasting), context only. |

**Takeaways:** (1) Frame your work as **interpolation**, citing Vandal & Nemani as the direct precedent and target to beat. (2) Borrow the **classical optical-flow advection baseline** from the nowcasting literature to show AI's advantage. (3) Borrow **CRPS / ensemble** evaluation ideas only if you add a generative comparator; otherwise SSIM/PSNR/MSE/RMSE-K/FSIM/LPIPS suffice.

---

## 7. Adaptations needed for satellite TIR (engineering checklist)

1. **Single-channel input (not RGB).** Replace the 3-channel stem with 1-channel (or replicate the gray channel to 3 and fine-tune). For pretrained RGB weights, a standard trick: **average the first-conv RGB weights across channels** to initialize the 1-channel conv, then fine-tune. RIFE/EMA-VFI/IFRNet stems are tiny → cheap to re-init and fine-tune.
2. **Brightness-temperature dynamic range.** Don't use 8-bit [0,255]. Work in float. **Per-channel standardization** (subtract mean, divide std) as in Vandal & Nemani, or min–max to [0,1] over a fixed physically-meaningful Kelvin range (e.g. ~180–330 K for Ch13). Keep raw Kelvin to report **RMSE/MAE in Kelvin** (more physical than PSNR).
3. **Large, non-linear cloud motion.** Convective tops grow/dissipate (not pure translation) and cyclone bands move fast. Mitigations: train at native resolution, use **multi-scale** flow (RIFE coarse-to-fine already does this; multi-scale kernels per Vandal & Nemani helped), and benchmark a large-motion model (GIMM-VFI/FILM/XVFI/BiM-VFI) on the hardest convective/cyclone cases. Stratify your test set by motion magnitude.
4. **Geostationary projection & registration.** Resample to a common grid (e.g. fixed-grid/geostationary projection or a lat-lon crop); ensure GOES/Himawari/INSAT frames are co-registered and parallax/navigation-consistent. Train and test on consistent grids. For INSAT deployment, regrid INSAT TIR1 to the training grid resolution.
5. **Missing/NaN pixels (space-look, off-disk, bad scans).** VFI nets can't ingest NaN. Options: (a) mask + fill (nearest/inpaint) before the net and re-apply the mask after; (b) add a validity-mask channel; (c) exclude NaN pixels from the loss. Always exclude off-disk/NaN from metric computation.
6. **`.nc`/`.h5` I/O.** Use `xarray`/`netCDF4`/`h5py`. Wrap the model with a loader that reads Ch13/TIR1, normalizes, runs inference at arbitrary t, denormalizes back to Kelvin, and writes `.nc` with proper coordinates/attrs (CF conventions). The PS explicitly requires `.nc` in and out.
7. **Domain gap GOES/Himawari → INSAT.** INSAT-3DS/3DR TIR1 is ~4 km, 30-min native (vs GOES 2 km/10-min). Train on dense GOES/Himawari; **fine-tune or at least validate** on INSAT; account for resolution/cadence differences (you may downsample GOES to ~4 km to better match INSAT, or fine-tune on INSAT pairs where a denser reference exists). Expect a domain gap — report it honestly.
8. **Temporal supervision for arbitrary t.** Train with random intermediate labels (as Vandal & Nemani did) so the model generalizes to t=0.25/0.5/0.75, enabling 30→15→7.5 min.

---

## 8. Recommended experimental design (to satisfy "compare methods" + win on metrics)

- **Primary model:** RIFE (Practical-RIFE v4.25/4.26), 1-channel, fine-tuned on GOES-19 Ch13 + Himawari TIR; arbitrary-t enabled.
- **Comparators (the PS requires comparison):**
  1. Linear/bilinear blend (trivial baseline).
  2. Classical optical flow warp (Farnebäck or TV-L1) + blend (the "traditional optical flow" the PS says fails).
  3. **Super SloMo** (named baseline; reimplement task-specific per §1).
  4. **EMA-VFI** (best-quality-still-fast).
  5. **GIMM-VFI** (large/non-linear motion; uses RAFT/FlowFormer flow).
  6. (Optional) **FILM** and/or **BiM-VFI**/**VFIMamba** for large-/non-uniform-motion frontier.
  7. (Optional, perceptual upper bound) **MoMo** (flow-diffusion).
- **Metrics:** PSNR, SSIM, MSE, **RMSE/MAE in Kelvin**, FSIM, **LPIPS** (perceptual), plus motion-aware checks (e.g. error stratified by optical-flow magnitude; structure/edge metrics on cloud boundaries). Report **per-regime** (calm vs. convective vs. cyclone), since gains concentrate in high-motion cases.
- **Ablations:** generic-pretrained vs. fine-tuned (show fine-tuning wins, per §1); t=0.5 vs. arbitrary-t; effect of normalization; effect of large-motion backbone on hard cases.
- **Deployment:** apply the winning model to INSAT-3DS/3DR TIR1 to produce 15-min (and 7.5-min) animations; build the original-vs-interpolated dashboard with SSIM/PSNR/MSE plots (PS Step 2/3).

---

## 9. Why RIFE as primary (justification on the four axes)

- **Speed:** Fastest flow-based VFI (~10 ms @512²; 4–27× faster than Super SloMo/DAIN; real-time at 720p on one GPU). Meets "near real-time." Lite variants for edge/INSAT ops.
- **Quality:** Competitive Vimeo90K PSNR; v4.x added FILM-style/Gram losses + more flow blocks; sufficient for smooth-to-moderate cloud motion, which dominates 10-min IR. For the hardest convective/cyclone cases, the report's plan adds GIMM-VFI/large-motion comparators so you're covered and can quantify any gap.
- **License:** **MIT** (verified) — unambiguous for ISRO/government deployment and any downstream commercialization. (EMA-VFI Apache-2.0 and FILM Apache-2.0 are also safe; some research repos are non-commercial — verify before shipping.)
- **Fine-tunability on single-channel TIR:** ~10 M params, simple architecture, available training scripts → the easiest model to retrain on brightness-temperature data; tiny stem makes the 3→1 channel change trivial. The PS even names RIFE, signaling evaluator familiarity.

**Risk & mitigation:** RIFE may blur very fast/non-linear convective growth. Mitigate by (a) fine-tuning on satellite data (proven to help), (b) using arbitrary-t multi-scale, and (c) reporting GIMM-VFI/BiM-VFI on the hardest cases — if they materially win there, present a two-tier system (fast RIFE default + heavy model for severe events).

---

## 10. Concrete first implementation (what to build first)

1. Stand up the **data pipeline**: pull GOES-19 ABI Ch13 (NOAA AWS Open Data), Himawari AHI; regrid/normalize to a common grid; build (I0, I1, I_t) triplets 20 min apart with random t (and fixed t=0.5 for the headline 00:00/00:20→00:10 demo); `.nc` I/O via xarray.
2. **Fine-tune Practical-RIFE v4.25** (1-channel) on the triplets; standardize Tb; mask NaN/off-disk in the loss.
3. Implement **baselines**: linear, classical-OF warp, Super SloMo (task-specific).
4. Add **EMA-VFI** and **GIMM-VFI** comparators.
5. **Evaluate**: PSNR/SSIM/MSE/RMSE-K/FSIM/LPIPS, per-regime; produce comparison plots.
6. **Deploy** the winner on INSAT-3DS/3DR TIR1; build the dashboard (original vs. interpolated animations + metric report).

---

## Sources

**Satellite/weather VFI & nowcasting (most relevant):**
- Temporal Interpolation of Geostationary Satellite Imagery with Task-Specific Optical Flow (Vandal & Nemani): https://arxiv.org/abs/1907.12013 · IEEE: https://ieeexplore.ieee.org/document/9511282/ · NASA NTRS: https://ntrs.nasa.gov/citations/20210020625 · full text: https://ar5iv.labs.arxiv.org/html/1907.12013
- Optical Flow for Intermediate Frame Interpolation of Multispectral Geostationary Satellite Data (NASA NTRS): https://ntrs.nasa.gov/citations/20190033878
- Deterministic nowcasting of geostationary satellite IR brightness temperature using 3D U-Net diffusion (Scientific Reports, 2026): https://www.nature.com/articles/s41598-025-34207-9 · PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12859110/
- Learning Robust Precipitation Forecaster by Temporal Frame Interpolation: https://arxiv.org/pdf/2311.18341
- Deep Temporal Interpolation of Radar-based Precipitation: https://arxiv.org/pdf/2203.01277
- How to use score-based diffusion in earth system science: a satellite nowcasting example: https://arxiv.org/pdf/2505.10432
- Precipitation nowcasting of satellite data using physically-aligned neural networks: https://arxiv.org/html/2511.05471v1
- NowcastNet (Nature 2023): https://www.nature.com/articles/s41586-023-06184-4 · DGMR/Skillful nowcasting: https://arxiv.org/pdf/2104.00954
- DiffCast (CVPR 2024): https://arxiv.org/abs/2312.06734 · repo: https://github.com/DeminYu98/DiffCast
- GOES-19 ABI Band 13 quick guide (CIMSS): https://cimss.ssec.wisc.edu/goes/OCLOFactSheetPDFs/ABIQuickGuide_Band13.pdf · NOAA STAR Band 13: https://www.star.nesdis.noaa.gov/GOES/conus_band.php?sat=G16&band=13
- INSAT-3D payloads (MOSDAC): https://www.mosdac.gov.in/insat-3d-payloads · INSAT-3DR (eoPortal): https://www.eoportal.org/satellite-missions/insat-3dr

**VFI surveys & leaderboards:**
- AceVFI survey (2025): https://arxiv.org/pdf/2506.01061
- VFI rankings leaderboard (AIVFI): https://github.com/AIVFI/Video-Frame-Interpolation-Rankings-and-Video-Deblurring-Rankings

**Flow-based VFI models:**
- RIFE: https://arxiv.org/abs/2011.06294 · ECCV2022-RIFE: https://github.com/hzwer/ECCV2022-RIFE · Practical-RIFE: https://github.com/hzwer/Practical-RIFE (both LICENSE files = MIT)
- IFRNet (CVPR 2022): https://github.com/ltkong218/IFRNet · paper: https://arxiv.org/pdf/2205.14620
- EMA-VFI (CVPR 2023): https://github.com/MCG-NJU/EMA-VFI · paper: https://arxiv.org/pdf/2303.00440
- AMT (CVPR 2023): https://github.com/MCG-NKU/AMT · paper: https://arxiv.org/abs/2304.09790
- VFIMamba (NeurIPS 2024): https://github.com/MCG-NJU/VFIMamba · paper: https://arxiv.org/abs/2407.02315
- GIMM-VFI (NeurIPS 2024): https://github.com/GSeanCDAT/GIMM-VFI · paper: https://arxiv.org/html/2407.08680v4
- BiM-VFI (CVPR 2025): https://arxiv.org/abs/2412.11365
- FILM (ECCV 2022, Google): https://github.com/google-research/frame-interpolation · paper: https://arxiv.org/pdf/2202.04901
- XVFI (ICCV 2021): https://arxiv.org/abs/2103.16206
- Super SloMo (CVPR 2018): https://arxiv.org/pdf/1712.00080
- FLAVR: https://arxiv.org/pdf/2012.08512
- MoMo (AAAI 2025, flow-diffusion): https://github.com/JHLew/MoMo · paper: https://arxiv.org/abs/2406.17256
- LDMVFI (AAAI 2024): https://github.com/danier97/LDMVFI · paper: https://arxiv.org/abs/2303.09508
- TLB-VFI (2025): https://arxiv.org/pdf/2507.04984

**Optical-flow estimators:**
- RAFT — see survey refs; SEA-RAFT (ECCV 2024): https://github.com/princeton-vl/SEA-RAFT · paper: https://arxiv.org/pdf/2405.14793
- GMFlow: https://arxiv.org/pdf/2111.13680
- NeuFlow v2: https://arxiv.org/pdf/2408.10161

**Adaptation (grayscale fine-tuning):**
- Transfer learning on greyscale images (first-conv weight averaging): https://towardsdatascience.com/transfer-learning-on-greyscale-images-how-to-fine-tune-pretrained-models-on-black-and-white-9a5150755c7a/
