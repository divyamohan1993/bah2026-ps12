# 05 — Rigorous Validation, Metrics & a 30+ Method Cross-Validation Framework

**Project:** ISRO BAH 2026 PS-12 — "Fill in the Frames Seamlessly" (AI/Optical-Flow temporal interpolation of geostationary TIR satellite imagery; INSAT-3DS/3DR TIR1 ~10.8 µm, validated against GOES-19 ABI Ch-13 and Himawari AHI).
**Scope of this doc:** the validation half of the PS — *how to prove the interpolated frames are correct*. Metric suite, motion/nowcasting metrics, TIR domain metrics, an explicit enumerated **30+ method** multi-satellite cross-validation framework, reporting structure, and the exact Python libraries + reproducibility recipe.
**Date:** 2026-06-20. Network research performed (12+ WebSearch + several WebFetch). Where a number/claim is from a specific source, the URL is inline; a consolidated Sources list is at the end.

---

## 0. TL;DR — Recommended metric suite (the "core 14" + extensions)

The PS asks for "SSIM, MSE, PSNR, FSIM, etc." plus "any other suitable metrics… suitable to capture the cloud movements." We go far beyond the literal ask. The suite is layered:

**Layer A — Full-reference pixel/structure fidelity (per interpolated frame vs withheld ground-truth frame):**
1. MSE / 2. RMSE / 3. MAE (a.k.a. Interpolation Error, IE) — raw error, reported in **Kelvin** for TIR, not just normalized DN.
4. PSNR — headline number, easy to compare to VFI literature.
5. SSIM — headline structural metric (PS-named).
6. MS-SSIM — multi-scale SSIM; better for large cloud fields at multiple scales.
7. FSIM / FSIMc — feature-similarity (phase congruency + gradient); PS-named; strong on edges/cloud boundaries.
8. GMSD — gradient-magnitude similarity deviation; cheap, edge-sensitive, low score = good.
9. VIF / VIFp — visual information fidelity; sensitive to information loss (blur).
10. UQI / Q-index — universal quality index (SSIM precursor).
11. ERGAS — global relative dimensionless error (remote-sensing standard).
12. SAM — spectral angle mapper (multi-band only; use across TIR1/TIR2/WV if available).
13. SCC — spatial correlation coefficient (edge/high-pass correlation).
14. LPIPS (+ optionally DISTS) — learned perceptual similarity (caveat: trained on RGB; for single-channel TIR replicate channel ×3 — see §3.6; report as secondary).

**Layer B — Motion / temporal / video metrics (capture "cloud movement"):**
15. Optical-flow End-Point-Error (EPE) of estimated vs reference flow.
16. Warping error / Relational Warping Error (RWE).
17. Temporal optical-flow consistency (tOF) and temporal-LPIPS (tLPIPS) on the triplet.
18. Flicker/temporal-stability metric over the up-sampled sequence.
19. VMAF (video, treats the sequence as video; secondary, RGB-trained caveat).

**Layer C — Nowcasting / meteorological skill (treat cloud/cold-cloud as the "event"):**
20. CSI / POD / FAR / Bias score at brightness-temperature (BT) thresholds (e.g. 235 K, 220 K, 200 K cold-cloud tops).
21. Fractions Skill Score (FSS) at multiple neighborhood sizes (scale-aware).
22. SAL (Structure–Amplitude–Location) object-based score on cold-cloud objects.
23. CRPS (if/when producing probabilistic/ensemble interpolation).
24. Object/feature tracking error (centroid displacement, area/intensity error of tracked storms).

**Layer D — TIR / brightness-temperature domain fidelity:**
25. BT bias & BT-RMSE in **Kelvin**; histogram/CDF matching (e.g. earth-mover / KS distance of the BT distribution).
26. Cloud-top-temperature (CTT) error on cold-cloud pixels.
27. Gradient/edge-preservation ratio (does the model blur fronts?).
28. Radially-averaged Power-Spectral-Density (PSD) ratio — fine-scale structure preservation.
29. Blurriness metric (e.g. variance-of-Laplacian, high-frequency energy ratio) vs ground truth.

**Layer E — Cross-checks / statistics / baselines (the robustness engine — see §4):**
30+. Triple collocation, cross-satellite agreement, baseline deltas, ablations, stratified breakdowns, significance tests. (Enumerated explicitly in §4 as the **30+ methods** list.)

> **Headline reporting set** (put these on the dashboard / front of report): PSNR, SSIM, MS-SSIM, FSIM, BT-RMSE (K), CSI@235K, FSS@scale, EPE. Everything else lives in the appendix tables and per-case studies.

---

## 1. Full-reference image-quality metrics (Layer A) — what each measures, libraries, TIR sensitivity

> Convention: TIR Ch-13 / TIR1 are **single-channel** brightness-temperature fields. Pixel-fidelity metrics should be computed **both** on (i) physical BT in Kelvin (for MSE/RMSE/MAE/PSNR with an explicit, fixed data range) and (ii) min–max or fixed-range normalized arrays (for SSIM-family, which assume a known dynamic range). Always state the `data_range` used — SSIM/PSNR are meaningless without it.

| # | Metric | What it measures | Range / better | Sensitivity to cloud structure/motion | Pros for TIR single-channel | Cons / cautions |
|---|--------|------------------|----------------|----------------------------------------|-----------------------------|------------------|
| 1 | **MSE** | Mean squared pixel error | ≥0, ↓ | Penalizes any displacement heavily (double penalty for shifted clouds) | Direct, in K² if BT | Not perceptual; a 1-px cloud shift can dominate |
| 2 | **RMSE** | √MSE, same units as data | ≥0, ↓ | As MSE | Reportable in **K** | Same double-penalty issue |
| 3 | **MAE / IE** | Mean abs error (a.k.a. Interpolation Error in VFI) | ≥0, ↓ | More robust to outliers than MSE | In **K**; VFI-standard | Still pixel-aligned |
| 4 | **PSNR** | Log of peak²/MSE | dB, ↑ | Inherits MSE behavior | Headline, comparable to VFI papers | Needs fixed peak/data_range; poor perceptual correlation |
| 5 | **SSIM** | Local luminance/contrast/structure similarity (windowed) | [-1,1], ↑ | Good at structural match of cloud texture; still penalizes misalignment | PS-named; standard | Single-scale; window size matters; assumes known range |
| 6 | **MS-SSIM** | SSIM over a Gaussian pyramid (multi-scale) | [0,1], ↑ | Better for multi-scale cloud fields (synoptic + meso) | Robust headline structural metric | Needs image ≥ ~160 px per side for 5 scales |
| 7 | **FSIM/FSIMc** | Phase-congruency + gradient feature similarity | [0,1], ↑ | Excellent on cloud **edges/boundaries**, fronts | PS-named; perceptually strong | Heavier compute; FSIMc needs color → use FSIM for TIR |
| 8 | **GMSD** | Std-dev of gradient-magnitude similarity map | ≥0, ↓ | Very edge/structure sensitive (front sharpness) | Cheap, sharp-edge detector | Only gradients; ignores absolute level |
| 9 | **VIF/VIFp** | Mutual information vs natural-scene statistics | ≥0 (1=ref), ↑ | Strongly drops when interpolation **blurs** detail | Great blur detector for over-smoothing | NSS prior is RGB-natural; pixel-aligned |
| 10 | **UQI** | Universal quality index (corr×lum×contrast) | [-1,1], ↑ | SSIM-like, coarser | Simple baseline | Superseded by SSIM |
| 11 | **ERGAS** | Global relative dimensionless synthesis error | ≥0, ↓ | Aggregate radiometric fidelity (RS standard) | Designed for satellite fusion | Needs ratio of resolutions; band-wise |
| 12 | **SAM** | Spectral angle between band vectors | rad/deg, ↓ | Multi-band spectral consistency (TIR1 vs TIR2 vs WV) | Detects spectral distortion across bands | **Needs ≥2 bands**; N/A for single channel |
| 13 | **SCC** | Spatial correlation of high-pass images | [-1,1], ↑ | Fine-texture correlation | Good "did we keep texture?" check | Sensitive to filter choice |
| 14 | **LPIPS** | Distance in deep-net feature space | ≥0, ↓ | Captures perceptual cloud-texture realism | Catches GAN/blur artifacts humans see | Trained on **RGB ImageNet**; for TIR replicate ch×3 (§3.6) — secondary only |
| (+) | **DISTS** | Deep structure+texture similarity | ≥0, ↓ | Texture-aware, tolerant of small shifts | Good for "looks right" texture | Same RGB-domain caveat |

**Key cross-cutting fact (why we need Layers B–D):** in the VFI literature, "PSNR, SSIM and LPIPS do not provide satisfactory correlation with perceptual quality" for interpolated content (arXiv 1901.05362 / ECCV'22 136750231), and "interpolated frames may not score highly in terms of PSNR or SSIM" even when motion is correct (arXiv 2508.09078). So pixel metrics alone are **necessary but not sufficient** — they over-penalize physically-plausible motion and under-penalize subtle blur. This is the core scientific justification for the multi-method approach the user demands.

### Library map for Layer A
- **scikit-image** (`skimage.metrics`): `mean_squared_error`, `peak_signal_noise_ratio`, `structural_similarity` (with `data_range=`). Reliable references for MSE/PSNR/SSIM.
- **sewar** (`sewar.full_ref`): `mse, rmse, psnr, ssim, uqi, msssim, ergas, scc, rase, sam, vifp, psnrb, q2n, d_lambda, d_s, qnr`. NumPy `HxWxC`. (PyPI confirms list.) Great one-stop for ERGAS/UQI/SAM/RASE/VIFp/SCC.
- **piq** (PyTorch Image Quality): `psnr, ssim, multi_scale_ssim, fsim, gmsd, multi_scale_gmsd, vif_p, vsi, haarpsi, mdsi, dss, brisque, dists, lpips, srsim, content/style`. Tensor `NCHW` in [0,1]. **Primary source for FSIM, GMSD, DISTS, HaarPSI.**
- **torchmetrics** (`torchmetrics.image`): `StructuralSimilarityIndexMeasure`, `MultiScaleStructuralSimilarityIndexMeasure`, `PeakSignalNoiseRatio`, `SpectralAngleMapper`, `ErrorRelativeGlobalDimensionlessSynthesis (ERGAS)`, `UniversalImageQualityIndex`, `LearnedPerceptualImagePatchSimilarity (LPIPS)`, `SpectralDistortionIndex`, `VisualInformationFidelity`. Good for batched GPU eval + logging.
- **lpips** (Zhang et al. official): `lpips.LPIPS(net='alex'|'vgg')`. For grayscale replicate to 3 channels.
- **image-similarity-measures**: CLI/py for `rmse, psnr, ssim, fsim, sam, sre, uiq, issm` — convenient cross-check implementation (use to *verify* sewar/piq agree).
- **DISTS-pytorch** if not using piq's DISTS.

> **Reproducibility rule:** compute every metric with **two independent implementations** where possible (e.g. SSIM via skimage *and* piq; ERGAS via sewar *and* torchmetrics) and assert they agree to tolerance. Disagreement = bug or a `data_range`/layout mismatch. This is itself one of the 30+ cross-checks (method #29).

---

## 2. Motion / temporal / video metrics (Layer B) — "capture the cloud movement"

The PS explicitly wants metrics "suitable to capture the cloud movements." Pixel metrics don't; these do.

**15. Optical-flow End-Point-Error (EPE).** Average Euclidean distance between predicted flow vector and reference flow vector per pixel: `EPE = (1/N) Σ ||u_pred − u_gt||₂`, in pixels (the standard Sintel/KITTI metric; arXiv 2401.00833, Middlebury flowEval). Variants: **EPE-all / EPE-noc / EPE-occ** (all / non-occluded / occluded), and **KITTI F1-all** (% of pixels with EPE > 3 px *and* > 5% of |gt|). **Average Angular Error (AAE)** = mean `arccos(û·û_gt)` measures flow *direction* error. *How to get "reference" flow for satellite (no GT flow exists):* use a strong, independent flow estimator (e.g. RAFT or TV-L1) on the **real** frame triplet (frame0→frame2 and the true middle) as a pseudo-reference, and compare to the flow your model implicitly/explicitly produced. This validates the *motion field*, not just the pixels.

**16. Warping error / Relational Warping Error (RWE).** Warp frame *t-1* to *t* with estimated flow and measure masked L1: `WE = mean |I_t − warp(I_{t-1}, flow)|`. Because WE is non-zero even for ground truth (intensity changes + flow error), use **RWE = WE(prediction) − WE(ground-truth-frames)** to subtract the floor (arXiv survey on temporal consistency). Lower RWE = motion-consistent interpolation.

**17. Temporal optical-flow consistency (tOF) & tLPIPS.** tOF compares the optical flow *of the predicted sequence* to the flow *of the GT sequence* across the triplet (lower = smoother, more correct motion). tLPIPS = LPIPS computed on temporal differences (frame-to-frame), capturing temporal flicker perceptually. Both are standard in video-SR / VFI evaluation.

**18. Flicker / temporal-stability.** Over the densified sequence (e.g. 30→7.5 min, 3 inserted frames), measure short-time temporal variance / "jerk" of pixel intensities along *t* and compare to the smoothness of a real dense GOES sequence. A good interpolation should have temporal-derivative statistics matching real dense data, not introduce step jumps at insertion points.

**19. VMAF (Video Multimethod Assessment Fusion).** ML-fused perceptual *video* metric (Netflix); ST-VMAF adds temporal features. Treat (GT-dense vs predicted-dense) as reference/test video. **Secondary** for us (RGB/consumer-video trained), but useful as one more independent vote and for the dashboard "video quality" number.

**Library map for Layer B:** flow via **`torchvision.models.optical_flow.raft_large`** (or OpenCV `cv2.optflow.DualTVL1OpticalFlow_create` / `calcOpticalFlowFarneback`); EPE/warp are a few lines of NumPy/Torch; tLPIPS via the `lpips` package on temporal diffs; VMAF via `ffmpeg-libvmaf` / `vmaf` python bindings.

---

## 3. Nowcasting/meteorological skill + TIR-domain metrics (Layers C & D)

These are the metrics that make this a *scientific remote-sensing* validation rather than a generic VFI benchmark — exactly the "robust, collaborated findings" the user wants. Cold cloud tops (low BT) are the "events."

### 3.1 Categorical skill at BT thresholds (#20)
Binarize each frame at cold-cloud BT thresholds (e.g. **235 K, 220 K, 210 K, 200 K**). Build a 2×2 contingency table (hits a, false alarms b, misses c, correct-negatives d) between predicted and true frame:
- **POD** = a/(a+c) (probability of detection)
- **FAR** = b/(a+b) (false-alarm ratio)
- **CSI/Threat** = a/(a+b+c)
- **Bias score** = (a+b)/(a+c)
- (optional **HSS/ETS** for chance-corrected skill)
This answers: *does the interpolated frame put the cold convective cloud in the right place with the right extent?* (Standard nowcasting verification; arXiv 2406.04867 survey.)

### 3.2 Fractions Skill Score, FSS (#21)
Neighborhood/scale-aware: compute fractional coverage of "event" in n×n windows for prediction and truth, then `FSS = 1 − MSE_fractions / MSE_fractions_ref`, swept over neighborhood sizes (e.g. 1,3,9,27,81 px). FSS rises with scale and tells you the **smallest scale at which the interpolation is skillful** (FSS ≥ 0.5). Crucial because pixel metrics "penalize even slight spatial deviations" while FSS rewards getting the right *amount* at the right *scale* (arXiv 2406.04867).

### 3.3 SAL — Structure / Amplitude / Location (#22)
Object-based (Wernli et al. 2008; AMS WAF). Identify cold-cloud "objects" above a threshold, then:
- **A** = normalized domain-total error (amplitude/intensity),
- **L** = error in center-of-mass location (+ object scatter),
- **S** = error in object size/shape (volume-scaled).
SAL "avoids double penalisation for timing and locational errors inherent in RMSE" (journals.ametsoc.org WAF-D-17-0162) — perfect for moving clouds where RMSE is unfair.

### 3.4 CRPS (#23)
If we ever output a **probabilistic/ensemble** interpolation (e.g. flow-ensemble or dropout-MC), CRPS measures how well the predictive BT distribution matches the observed BT: `CRPS = ∫ (F_pred(x) − 1{x ≥ obs})² dx`. For deterministic output it reduces to MAE, so only meaningful with an ensemble. (arXiv 2104.00954 DGMR uses pooled-CRPS.)

### 3.5 Object/feature tracking error (#24)
Detect & track storm cells (e.g. simple thresholded blob tracking, or `tobac`/`pysteps` feature tracking) in the true dense sequence and in the predicted dense sequence; compare **centroid displacement error, area error, min-BT (intensity) error, lifetime**. Directly measures whether interpolation preserves *physical motion of features*.

### 3.6 BT fidelity in Kelvin + histogram matching (#25)
- **BT bias** = mean(pred − truth) in K; **BT-RMSE** in K. Report alongside known sensor inter-cal references for context: e.g. INSAT-3D CTT validation gave bias −0.31 K, RMSE 10.30 K vs radiosonde (MDPI rs11232811); GOES-16 ABI vs Himawari-8 AHI IR differences are within **0.3 K** for bands 10–12/14 (GSICS, data.jma.go.jp). So our *interpolation* error budget should be judged against the **~0.3 K sensor-to-sensor floor** and ~1–2 K sounding accuracy — anything within a few K is excellent; large K errors flag real problems.
- **Histogram/CDF matching:** compare BT histograms (pred vs truth) via **KS distance, Earth-Mover/Wasserstein distance, χ², histogram-intersection**. Detects systematic warm/cold drift or contrast compression even when SSIM looks fine.

### 3.7 Cloud-top-temperature error on cold pixels (#26)
Restrict BT-RMSE/bias to the coldest X% (cloud) pixels — interpolation often does well on clear sky but smears cold convective tops; this isolates the hard, important regime.

### 3.8 Gradient / edge preservation (#27)
Edge/gradient maps (Sobel/Scharr) of pred vs truth → correlation + ratio of gradient energy. Quantifies front/boundary sharpness loss (the classic optical-flow "blur" failure the PS calls out).

### 3.9 Radially-averaged Power-Spectral-Density ratio (#28)
2-D FFT → |F|² → radial average → 1-D PSD(k). Plot PSD_pred/PSD_truth vs spatial wavenumber. Interpolation that blurs shows a **deficit at high k** (fine scales lost); a faithful method keeps the ratio ≈1 across scales. This is the single most diagnostic metric for "did we preserve fine-scale structure?" (radially-averaged PSD; ResearchGate / super-res practice).

### 3.10 Blurriness metric (#29)
**Variance-of-Laplacian**, **Tenengrad**, or **high-frequency energy ratio** of pred vs truth. A scalar "sharpness" that should match the truth's sharpness (not just be high or low).

**Library map for Layers C/D:** **`pysteps`** (verification module: FSS, CSI/POD/FAR/ETS, ROC, reliability, CRPS, spectral & FSS plots — purpose-built for precip/nowcasting; works on any 2-D field incl. BT), **`xskillscore`** (CRPS, Brier, rank histograms, correlation, RMSE on xarray), **`scipy.stats`** (KS, Wasserstein, χ²), **`scipy.ndimage`/`skimage`** (objects, labeling, Sobel), **`numpy.fft`** (PSD), **`tobac`** (cloud feature tracking). SAL: `pysteps`-style or custom (small).

---

## 4. THE 30+ METHOD ROBUST CROSS-VALIDATION FRAMEWORK (the core ask)

The user's mandate: *"Use at least 30 different methods for extremely collaborated robust findings. Use all the broad acquired data to verify and fill the gaps of the other datasets. Use multiple satellites/datasets to verify."*

Below are **40 concrete, distinct validation / cross-check methods**, grouped. Each is independently runnable and produces evidence; together they triangulate truth from many directions and let each dataset cover another's blind spots. (Numbered M1…M40.)

### A. Withheld-ground-truth interpolation tests (the primary, exact-match evaluations)
**M1 — Leave-the-middle-frame-out on dense GOES-19 (PRIMARY).** Use 00:00 + 00:20 → predict 00:10; the *real* 00:10 ABI Ch-13 frame is withheld ground truth. Compute the entire Layer-A/B/C/D suite. This is the gold-standard exact validation enabled by GOES-19's 10-min (and 5-min CONUS / 1-min meso) cadence.
**M2 — Leave-the-middle-frame-out on Himawari (independent dense source).** Same protocol on Himawari AHI Band 13/14 (10-min full-disk) → second independent exact-match testbed in a different geography/sensor. Fills GOES's geographic gap (Asia/India-facing).
**M3 — Multi-step recursion vs direct (2× vs 4× vs 8×).** Validate 30→15 (1 frame), 30→7.5 (3 frames), 30→3.75 (7 frames) by holding out the corresponding real GOES frames. Tests error growth with up-sampling factor (PS explicitly wants 30→15→7.5).
**M4 — Recursion-depth / error-accumulation test.** Compare recursive halving (interp the interp) vs single-shot multi-frame; quantify how metrics degrade per recursion level. Detects compounding blur.
**M5 — Asymmetric-interval test.** Predict off-center times (e.g. 00:05, 00:15) not just the midpoint, using GOES 5-/1-min frames as truth — checks the model isn't only good at t=0.5.
**M6 — Long-gap stress test.** Feed 60-min-apart frames, predict the 3 intervening real 15-min frames — simulates INSAT's coarser cadence and tests extrapolation of motion over big gaps.

### B. Cross-satellite verification (multi-satellite, fills geographic/temporal gaps)
**M7 — GOES + Himawari overlap over the Pacific.** In the longitudinal overlap (~western Pacific / dateline), the *same* cloud scene is seen by both. Interpolate one satellite's sequence and validate against the *other satellite's* near-simultaneous observation (after GSICS inter-cal + reprojection). Independent-sensor truth. (ABI–AHI agree within ~0.3 K in IR — data.jma.go.jp — so residual beyond that is interpolation error.)
**M8 — INSAT + Himawari overlap over the Indian Ocean.** INSAT-3DS/3DR (≈82°E) and Himawari (≈140°E) overlap over the Indian Ocean / SE Asia. Validate INSAT interpolation against Himawari's denser (10-min) real frames at the in-between times — this is how we get *real* higher-temporal-resolution truth for INSAT, whose own cadence is 30 min. **This is the key "fill the gap" cross-check for the final INSAT deliverable.**
**M9 — INSAT vs GOES (where geometry allows) / three-way scene.** Where any two-of-three see common scenes, do pairwise interpolation-vs-other-observation checks.
**M10 — Cross-satellite consistency of *derived motion*.** Compare optical-flow fields estimated independently from GOES and Himawari over the overlap — the true cloud motion is one physical field; agreement validates the flow estimator independent of interpolation.
**M11 — Inter-calibration normalization (GSICS) as a pre-step + sanity check.** Apply/verify GSICS-style BT inter-cal between sensors before cross-comparison; the *known* ~0.3 K inter-sensor bias is a built-in reference floor for interpreting cross-satellite RMSE.

### C. Independent polar-orbiter "truth at a time" overpasses
**M12 — MODIS (Terra/Aqua) overpass collocation.** At each MODIS overpass time over the domain, MODIS Band 31 (~11 µm) BT is an independent, higher-spatial-res truth *at that instant*. Compare the interpolated geostationary frame nearest that exact time → independent-sensor, independent-orbit verification.
**M13 — VIIRS (SNPP/NOAA-20/21) overpass collocation.** Same idea with VIIRS M15/I5 TIR — more overpasses, fills MODIS temporal gaps.
**M14 — Polar-orbiter as truth specifically at *interpolated* timestamps.** Where a LEO overpass lands between two geostationary acquisitions (i.e. exactly when we synthesized a frame), it directly validates the *synthetic* frame against a real measurement — the strongest possible independent check of interpolation.

### D. Statistical triangulation
**M15 — Triple collocation (TC).** With three independent estimates of the same BT (e.g. interpolated-GOES, real-Himawari, polar-orbiter), TC estimates the **random error variance of each** without assuming any one is perfect (assumes mutually uncorrelated errors). Yields an *absolute* error bar on the interpolation independent of a "truth." (Loew 2017 Rev. Geophys.; MDPI rs17223751.) **This is the rigorous way to assign uncertainty when no single ground truth is perfect.**
**M16 — N-way / quadruple collocation extension.** Add a 4th source (e.g. NWP cloud field or a second LEO) to relax TC assumptions and cross-check error-correlation (MDPI quadruple-collocation).
**M17 — Error-variance budget check.** Verify the TC-derived interpolation error variance is consistent across regions/seasons (internal consistency of the uncertainty estimate).

### E. Baseline & method comparisons (is the AI actually helping?)
**M18 — Naive frame-copy baseline (persistence).** Just repeat frame0 as the "interpolated" frame. Any real method must beat this on every metric.
**M19 — Linear-blend baseline.** `0.5*(frame0+frame2)`; classic, produces ghosting — must be beaten, especially on FSIM/PSD/blur.
**M20 — Classical optical-flow baselines: Farnebäck & TV-L1.** OpenCV warps; the PS itself contrasts "traditional optical-flow." Show the AI beats these, particularly on extreme motion and edge sharpness.
**M21 — DL VFI backbones head-to-head: RIFE vs Super-SloMo vs (FILM/IFRNet/AMT).** Same data, same metrics → pick the best per the PS's "best model" instruction; reports a ranked table.
**M22 — Ablation: with vs without explicit optical flow.** Does the flow module help vs a pure CNN/transformer interpolator? Quantify the contribution the PS centers on.
**M23 — Ablation: flow backbone swap (RAFT vs PWC-Net vs TV-L1 inside the pipeline).** Sensitivity of results to the flow estimator.
**M24 — Ablation: loss-function / warping (forward vs backward warp, with/without refinement net, with/without perceptual loss).** Links training choices to validation metrics.
**M25 — Ablation: input normalization / radiometric pre-processing.** With/without histogram normalization, K vs DN input — shows robustness.

### F. Stratified / conditional evaluation (where does it work, where does it fail?)
**M26 — Extreme-motion stratification (cyclones / deep convection).** Bin test cases by motion magnitude (from flow); report metrics for fast vs slow scenes. The PS's hardest, most valuable regime. Use named cyclone case studies.
**M27 — Phenomenon stratification: cyclone vs thunderstorm vs fire vs flood vs clear.** Per-phenomenon metric tables (PS lists these phenomena explicitly).
**M28 — Per-region stratification.** Tropics vs mid-latitude, land vs ocean, Indian subcontinent vs Pacific — exposes geographic bias and is where multi-satellite data fills gaps.
**M29 — Per-season / diurnal stratification.** Monsoon vs dry; day vs night (TIR works at night — verify no diurnal bias); convective afternoon vs calm night.
**M30 — Cloud-regime stratification by BT bands.** Separate metrics for warm/clear, mid, and cold (convective) BT ranges (ties to §3.7).

### G. Correlation, spectral & significance analyses
**M31 — Error-vs-cloud-speed correlation.** Regress per-frame error (RMSE/SSIM/EPE) against estimated cloud speed; quantify the (expected) degradation with motion and report R². Directly characterizes the model's motion limits.
**M32 — Error-vs-lead-fraction / vs up-sampling-factor correlation.** Error as a function of distance from the nearest real frame (t=0.5 hardest) and of 2×/4×/8×.
**M33 — Fourier/spectral fidelity (radially-averaged PSD ratio).** §3.9 applied as a *cross-method*: compare PSD ratios of AI vs each baseline to prove fine-scale superiority quantitatively.
**M34 — Wavelet multi-resolution fidelity.** 2-D DWT (e.g. `pywt`) energy per sub-band, pred vs truth — scale-localized structure check complementary to PSD.
**M35 — Statistical-significance testing.** Paired tests across the test set: **paired t-test / Wilcoxon signed-rank** on per-frame metric differences (AI vs baseline), with **bootstrap confidence intervals** on mean PSNR/SSIM/BT-RMSE. Proves improvements aren't noise. Apply Holm/Benjamini–Hochberg correction across the many metrics.
**M36 — Per-metric agreement / rank-correlation matrix.** Compute Spearman/Kendall correlation *between metrics* across all test frames; identifies redundant metrics and confirms multi-metric consensus (when 14 metrics all agree a method is better, that's the "collaborated robust finding").

### H. Physical-consistency & implementation cross-checks
**M37 — Multi-band consistency.** Interpolate TIR1 and (if available) TIR2 / WV / mid-IR; check inter-band relationships are preserved (e.g. split-window BT difference, WV–IR relationship) via SAM/correlation — physical sanity the single-channel metrics can't see.
**M38 — Mass/feature conservation & physical-plausibility checks.** Total cold-cloud area and domain-mean BT should evolve smoothly/monotonically through inserted frames; flag non-physical creation/destruction of cloud. (Energy/area conservation between bracketing frames.)
**M39 — Temporal-consistency at the seams.** Specifically test the boundary between a real frame and an inserted frame for discontinuities (flicker metric §B/#18 applied at insertion points).
**M40 — Dual-implementation metric verification + fixed-seed reproducibility.** Every metric computed by two libraries and asserted equal (skimage vs piq vs torchmetrics vs sewar); fixed random seeds, pinned versions, hashed data manifest. Guards against metric-implementation artifacts contaminating the "robust" conclusions.

> **That's 40 distinct methods (≥30 as required).** A compact "minimum viable robust" subset if time-constrained: M1, M2, M3/M4, M7, M8, M12/M13, M15, M18–M22, M26–M30, M31, M33, M35–M36 — already ~22 and spanning every category.

### How broad multi-satellite data "fills the gaps" (the user's central point)
- **GOES-19** gives *dense exact-match truth* (10/5/1-min) but only over the Americas/Pacific → primary training/validation, weak over India.
- **Himawari** gives a *second independent dense source* over Asia–Pacific and **overlaps INSAT over the Indian Ocean** → supplies the *real higher-temporal-resolution truth* for INSAT that INSAT itself lacks (M8). This is the linchpin: INSAT (30-min) can be validated to 10-min truth using Himawari.
- **INSAT-3DS/3DR** is the *deployment target*; cross-checks (M8/M9) let us trust INSAT results despite no native dense truth.
- **MODIS/VIIRS (LEO)** provide *instantaneous independent-sensor truth at specific times* anywhere, including over India and exactly at synthesized timestamps (M12–M14) — covering the geostationary blind spots and breaking any single-sensor calibration dependence.
- **Triple collocation (M15)** turns these three independent views into *absolute uncertainty estimates* without trusting any one as perfect.
Each dataset's weakness (GOES geography, INSAT cadence, LEO temporal sparsity) is covered by another's strength → genuinely "collaborated, robust" validation.

---

## 5. Reporting structure, plots & uncertainty quantification

The PS requires a "report comparing results with ground truth… include plots." Recommended structure:

**5.1 Headline scorecard (page 1):** table of the headline set (PSNR, SSIM, MS-SSIM, FSIM, BT-RMSE[K], CSI@235K, FSS@~50km, EPE) for AI vs each baseline, with bootstrap 95% CIs and significance stars. One bar chart per metric.

**5.2 Metric-vs-time line plots:** each metric over the test period (diurnal/seasonal trends; spot monsoon/convective dips). Overlay AI vs baselines.

**5.3 Error maps (spatial):** per-case 2-D maps of (pred − truth) in K, |error|, and SSIM-map / GMSD-map. Reveals *where* (cloud edges, fast cells) errors concentrate.

**5.4 Spatial heatmaps (aggregate):** mean error / mean SSIM accumulated over many cases per grid cell → systematic geographic bias map (ties to M28).

**5.5 Distribution plots:** histograms/violin/box of per-frame metrics (AI vs baselines); BT histogram overlays (pred vs truth) with KS/Wasserstein annotated (M25).

**5.6 Motion diagnostics:** flow-field quiver overlays; EPE map; error-vs-cloud-speed scatter + regression (M31); PSD-ratio curve and DWT sub-band bars (M33/M34).

**5.7 Nowcasting-skill panels:** FSS-vs-scale curves (M21/#21), performance/Roebber diagram (POD vs SR) at BT thresholds, SAL S–A–L scatter (M22).

**5.8 Case studies (the persuasive core):** for ≥3 events — a **cyclone** (extreme motion), **deep convection/thunderstorm**, and a **fire or flood** scene — show side-by-side ground-truth vs interpolated vs baseline animations/film-strips, with the per-case metric table and error map. These directly address the PS's named phenomena.

**5.9 Uncertainty quantification:** (a) bootstrap CIs on every aggregate metric; (b) **triple-collocation error bars** (M15) as the headline absolute uncertainty; (c) sensitivity bands from ablations (M22–M25); (d) explicit statement of the **sensor inter-cal floor (~0.3 K ABI–AHI; ~1–2 K sounding)** as the reference against which interpolation error is judged. State that errors within a few K and FSS-skillful down to meso-scale = success.

**5.10 Reproducibility appendix:** library versions, `data_range` per metric, exact thresholds (BT bins, FSS scales), data manifest hashes, seeds, and the dual-implementation agreement table (M40).

---

## 6. Concrete Python implementation plan (reproducible)

**Environment (pin versions):**
```
numpy, scipy, pandas, matplotlib, xarray, netCDF4, h5py      # IO (.nc/.h5 per PS) + analysis
scikit-image            # MSE, PSNR, SSIM (reference impls)
sewar                   # ERGAS, UQI, SAM, RASE, SCC, VIFp, MS-SSIM, PSNR-B, Q2n
piq                     # FSIM, GMSD, MS-SSIM, VIFp, DISTS, HaarPSI, VSI (PyTorch, NCHW [0,1])
torchmetrics            # batched GPU SSIM/MS-SSIM/PSNR/ERGAS/SAM/UIQ/LPIPS/VIF/SpectralDistortion (logging)
lpips                   # official LPIPS (alex/vgg) for grayscale ch×3
image-similarity-measures   # independent cross-check of rmse/psnr/ssim/fsim/sam/uiq/issm
torch, torchvision      # RAFT optical flow (raft_large) for EPE/warp/tOF
opencv-python           # Farnebäck & TV-L1 baselines, warping, image ops
pysteps                 # FSS, CSI/POD/FAR/ETS, ROC, reliability, CRPS, spectral verification (nowcasting)
xskillscore             # CRPS, Brier, rank histograms (xarray)
pywt                    # wavelet sub-band fidelity (M34)
satpy + pyresample      # read GOES ABI / Himawari AHI / INSAT, reproject to common grid, BT in K
tobac                   # cloud-feature tracking (M5/M24 object metrics)
ffmpeg + vmaf/libvmaf   # VMAF (secondary)
```

**Skeleton (compute the full per-frame suite):**
```python
import numpy as np, torch
from skimage.metrics import (mean_squared_error as sk_mse,
                             peak_signal_noise_ratio as sk_psnr,
                             structural_similarity as sk_ssim)
from sewar.full_ref import ergas, uqi, sam, rase, scc, vifp, msssim
import piq

def to_t(x):  # HxW float -> 1x1xHxW torch in [0,1]
    return torch.from_numpy(x[None,None]).float()

def per_frame_metrics(pred, true, data_range, bt_pred_K, bt_true_K):
    p01 = (pred - pred.min())/(np.ptp(pred)+1e-9)      # for SSIM-family
    t01 = (true - true.min())/(np.ptp(true)+1e-9)
    tp, tt = to_t(p01), to_t(t01)
    return {
      # Layer A pixel/structure
      "mse":   sk_mse(pred, true),
      "rmse":  float(np.sqrt(sk_mse(pred, true))),
      "mae":   float(np.mean(np.abs(pred-true))),
      "psnr":  sk_psnr(true, pred, data_range=data_range),
      "ssim":  sk_ssim(true, pred, data_range=data_range),
      "ms_ssim": float(piq.multi_scale_ssim(tp, tt, data_range=1.0)),
      "fsim":  float(piq.fsim(tp.repeat(1,3,1,1), tt.repeat(1,3,1,1), data_range=1.0)),
      "gmsd":  float(piq.gmsd(tp, tt, data_range=1.0)),
      "vif":   float(piq.vif_p(tp, tt, data_range=1.0)),
      "uqi":   float(uqi(true, pred)),
      "ergas": float(ergas(true, pred)),
      "scc":   float(scc(true, pred)),
      # Layer D BT domain (Kelvin!)
      "bt_bias_K": float(np.mean(bt_pred_K - bt_true_K)),
      "bt_rmse_K": float(np.sqrt(np.mean((bt_pred_K - bt_true_K)**2))),
    }
# + Layer B (RAFT EPE/warp), Layer C (pysteps FSS/CSI on BT thresholds), PSD/wavelet, etc.
```
- **`data_range`:** for Kelvin metrics use the physical span (e.g. 330−180 = 150 K) consistently; for SSIM-family use the normalized [0,1] range = 1.0. **Never let `data_range` drift between methods.**
- **Aggregation:** stack per-frame dicts into a DataFrame; compute means + **bootstrap CIs**; run **Wilcoxon** AI-vs-baseline per metric (M35); build the **inter-metric Spearman matrix** (M36).
- **Cross-checks:** assert skimage-SSIM ≈ piq-SSIM, sewar-ERGAS ≈ torchmetrics-ERGAS (M40).

---

## 7. Pitfalls / scientific cautions (so the "robust findings" hold up)
- **State `data_range` everywhere** — silent default mismatches make PSNR/SSIM incomparable. (#1 source of bogus numbers.)
- **Pixel metrics over-penalize correct motion & under-penalize blur** — never rank methods on PSNR/SSIM alone; require Layer-B/C/D consensus (arXiv 1901.05362, 2508.09078).
- **LPIPS/DISTS/VMAF are RGB-trained** — for TIR they're *secondary votes*; replicate channels ×3, never the sole verdict.
- **Reproject + inter-calibrate before any cross-satellite comparison** (parallax, viewing geometry, GSICS BT offsets) — else cross-checks measure geometry, not interpolation.
- **TC assumes uncorrelated errors** — pick maximally independent triplets (GEO-interp / other-GEO / LEO) and report when the assumption is shaky (MDPI rs17223751).
- **Judge K-errors against the ~0.3 K inter-sensor / ~1–2 K sounding floor** — don't over-claim sub-floor accuracy.
- **Avoid train/test leakage in time** — withheld frames and their neighbors must not appear in training; stratify splits by event and date.

---

## Sources
**Full-reference IQA & comparisons**
- Ding et al., *Comparison of Full-Reference IQA Models* — https://pmc.ncbi.nlm.nih.gov/articles/PMC7817470/ , https://www.cns.nyu.edu/pub/lcv/ding20c-reprint.pdf
- IQA via FSIM/SSIM/MSE/PSNR comparative study — https://www.scirp.org/journal/paperinformation?paperid=90911
- Adequacy of common IQA for medical images — https://arxiv.org/html/2405.19224v1
- HaarPSI — https://arxiv.org/pdf/1607.06140

**VFI / video / motion metrics**
- Efficient motion-based metrics for VFI — https://arxiv.org/pdf/2508.09078
- Technical Report on Visual Quality Assessment for Frame Interpolation — https://arxiv.org/pdf/1901.05362
- A Perceptual Quality Metric for VFI (ECCV'22) — https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750231.pdf
- A Subjective Quality Study for VFI — https://arxiv.org/html/2202.07727
- LAVIB benchmark — https://arxiv.org/html/2406.09754v1
- AceVFI survey (temporal-consistency metrics tOF/tLPIPS/FWE) — https://github.com/CMLab-Korea/Awesome-Video-Frame-Interpolation
- Blind Video Temporal Consistency (Deep Video Prior) — https://arxiv.org/pdf/2010.11838
- VMAF — https://cloudinary.com/glossary/vmaf , https://arxiv.org/pdf/1901.06279

**Optical-flow EPE/AAE**
- Rethinking RAFT — https://arxiv.org/pdf/2401.00833
- Middlebury flow evaluation methodology — https://vision.middlebury.edu/flow/flowEval-iccv07.pdf

**Nowcasting verification (FSS/CSI/POD/FAR/CRPS/SAL)**
- DL for precipitation nowcasting survey — https://arxiv.org/html/2406.04867v1 , https://arxiv.org/pdf/2406.04867
- Skillful Nowcasting (DGMR; pooled-CRPS) — https://arxiv.org/pdf/2104.00954
- SAL spatial verification (Wernli; AMS) — https://journals.ametsoc.org/view/journals/wefo/33/4/waf-d-17-0162_1.xml ; SpatialVx/Intercomparison — https://www.researchgate.net/publication/249612862
- Hybrid physics-AI nowcasting (FSS/CSI usage) — https://www.nature.com/articles/s41612-024-00834-8

**Triple/N-way collocation & EO validation**
- Non-ideal error stats in TC validation — https://doi.org/10.3390/rs17223751
- Extended quadruple collocation — https://www.researchgate.net/publication/318072361
- TC for two error-correlated datasets (L-band BT) — https://www.mdpi.com/2072-4292/12/20/3381
- Validation practices for satellite EO (Loew 2017, Rev. Geophys.) — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017RG000562

**TIR / brightness-temperature / inter-calibration (GOES/Himawari/INSAT)**
- INSAT-3D CTT retrieval & validation (bias −0.31 K, RMSE 10.30 K) — https://www.mdpi.com/2072-4292/11/23/2811 ; NRSC CTT doc — https://nices.nrsc.gov.in/docs/CTT50.pdf
- GSICS GEO-LEO IR inter-cal (ABI–AHI within ~0.3 K) — https://www.data.jma.go.jp/mscweb/data/monitoring/gsics/ir/techinfo_geoleoir.html
- Himawari-8 cloud-top height vs CloudSat — https://www.mdpi.com/2073-4433/12/2/173
- Himawari-8/MTSAT-2 SST accuracy (W. Pacific overlap) — https://www.mdpi.com/2072-4292/10/2/212

**Spectral / PSD / blur**
- Radially-averaged PSD discussion — https://www.researchgate.net/post/Radially-averaged-power-spectral-density-PSD-in-3D-space
- IR blur-kernel / super-res (PSNR/SSIM/SFR) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8309741/

**Libraries**
- piq docs — https://piq.readthedocs.io/en/latest/functions.html , https://piq.readthedocs.io/en/latest/modules.html ; examples — https://github.com/photosynthesis-team/piq/blob/master/examples/image_metrics.py
- sewar (PyPI, metric list) — https://pypi.org/project/sewar/
- torchmetrics MS-SSIM — https://lightning.ai/docs/torchmetrics/stable/image/multi_scale_structural_similarity.html
- LPIPS on grayscale (replicate ch×3) — https://zea.readthedocs.io/en/stable/notebooks/metrics/lpips_example.html ; R-LPIPS — https://arxiv.org/abs/2307.15157
- Image-similarity tutorial — https://medium.com/data-science/measuring-similarity-in-two-images-using-python-b72233eb53c6
