# 3D Improved U-Net for Prostate Radiotherapy Contouring (HipMRI 3D)

**Author:** Yogesh Sevda (s49116889) · COMP3710 Pattern Analysis Project · **Difficulty: Hard** (3D Improved U-Net, Section 2.2)

---

## 1. Problem and algorithm

### The problem
External-beam radiotherapy for prostate cancer needs the prostate (the treatment target) and nearby
organs at risk — bladder, rectum, bone and body outline — contoured on every slice of a pelvic MRI.
Manual contouring is slow and varies between clinicians. An automatic **draft contour** that a
planner reviews and edits would save time, but only if it is reliable where it matters most: at the
**base and apex of the prostate**, where the gland is small and its border with the bladder and
rectum is hard to see. Over-contouring there irradiates healthy rectum/bladder wall; under-contouring
risks leaving tumour untreated. This project segments 6 classes (background, body, bone, bladder,
rectum, prostate) in 3D MRI volumes from the HipMRI prostate study (38 patients, 211 volumes).

### The engineering dilemma
A 2D network segments each slice independently: cheap and fast, but blind to the slices above and
below, so contours can jump between slices — especially at the prostate's ends. A 3D network sees
the whole volume, but costs far more memory and compute. **Does true 3D context actually fix the
base/apex errors of a 2D baseline, or does it mostly polish the easy central slices while costing
much more?** This repository answers that by training both on identical data and comparing them
globally, per class, and specifically on prostate extremity slices.

### How the 3D Improved U-Net works
![Architecture](figures/architecture.png)

The model (Isensee et al., 2018) is an encoder–decoder ("U") network with 3D convolutions:

* **Encoder (left side).** Five levels with 16 → 256 feature maps. Each level halves the volume size
  with a stride-2 3×3×3 convolution and then applies a **context module**, a residual block
  (instance norm → LeakyReLU → conv → dropout → instance norm → LeakyReLU → conv, plus the input).
  Going deeper, each voxel's feature summarises a larger 3D neighbourhood, including several slices
  above and below — the through-plane context a 2D model never sees.

* **Decoder (right side).** Each level upsamples ×2, **concatenates the matching encoder features**
  (skip connection, which restores fine boundary detail lost when downsampling), and recombines them
  with a **localisation module** (3×3×3 conv then 1×1×1 conv).

* **Deep supervision.** 1×1×1 segmentation layers on three decoder levels are upsampled and summed
  into the final output, so the coarse levels get a direct training signal and learn faster.

* **Output.** 6-channel logits per voxel; softmax/argmax gives the label map.

**Baseline:** a standard 2D U-Net (Ronneberger et al., 2015), 4 pooling levels, 32 → 512 feature maps,
applied to each axial slice and restacked into a volume. With 7.76 M parameters it is close in size to
the 3D model (9.46 M), so differences come from 3D context, not model capacity.

**Loss (both models):** 0.5 × soft Dice + 0.5 × cross-entropy. The prostate is only **0.08 % of all
voxels** (Section 4), so plain cross-entropy would be dominated by background; the per-class Dice term
weights every class equally.

---

## 2. Feasibility review (one page)

**3D Improved U-Net for Prostate Radiotherapy Contouring (HipMRI 3D) — Difficulty: Hard**

**1. User need, scope and acceptance criteria**
*User:* a radiotherapy planner who reviews and edits a draft contour of the prostate and nearby organs
at risk (bladder, rectum).
*Purpose:* test the open dilemma — does true 3D volumetric context fix the boundary and
slice-transition errors at the prostate base and apex that a 2D slice-wise U-Net makes, and does that
gain justify the extra GPU memory and training time?
*Acceptance criteria:*
(a) Test-set Dice ≥ 0.7 on every label, including prostate, on a held-out patient-level split.
(b) Per-label Dice and IoU for both the 3D Improved U-Net and the 2D baseline on identical test patients.
(c) Prostate Dice reported separately for base/apex slices vs central slices, plus 3–5 diagnosed failure cases.
(d) Both models train and run on a single Rangpur A100; peak VRAM, parameter count and per-volume
inference time reported for each.

**2. Model choice and course concepts**
*Data:* HipMRI 3D (`semantic_MRs` + `semantic_labels_only`), 38 patients / 211 volumes, 6 classes.
Downsampled, z-score normalised; labels one-hot encoded with nearest-neighbour resampling.
*Proposed model:* 3D Improved U-Net (Isensee et al., 2018) — residual context modules, localisation
modules and deep supervision.
*Baseline:* standard 2D U-Net (Ronneberger et al., 2015) applied slice-by-slice to the same volumes and
restacked, so both models are compared on exactly the same test patients.
*Loss:* soft Dice + cross-entropy, because the prostate is under 2 % of the volume.
*Course link:* convolutional receptive fields — 3D kernels see neighbouring slices, giving through-plane
context a 2D network cannot use; skip connections recover boundary detail lost in downsampling.

**3. Preliminary feasibility evidence**
Data audit on Rangpur (25 Oct): 211 MR volumes in `semantic_MRs` and 211 label volumes in
`semantic_labels_only` — a complete one-to-one image/label set. Rangpur access and data paths confirmed.

**4. Risks, budget and fallback**
*Data leakage:* split by patient ID, so all volumes from one patient stay in one split.
*GPU memory:* downsampled volumes, batch size 1–2 and mixed precision; patch-based training if needed.
*Prostate Dice below 0.7:* Dice loss, augmentation (flips, small rotations, intensity shifts), deep supervision.
*Budget:* about 10 A100 GPU-hours on Rangpur. *Next experiment:* train both models on the same split and
compare. *Fallback:* standard 3D U-Net (Normal difficulty) with the same pipeline.

---

## 3. Repository structure and usage

| File | Purpose |
|---|---|
| `modules.py` | `ImprovedUNet3D`, `UNet2D` baseline, `DiceCELoss` (PyTorch only) |
| `dataset.py` | NIfTI loading, preprocessing + cache, patient-level split, augmentation, 3D/2D `Dataset`s |
| `train.py` | training, validation, checkpointing, testing, loss/Dice plots, metrics JSON |
| `utils.py` | volumetric Dice/IoU, prostate extremity-vs-central Dice, inference helpers |
| `predict.py` | loads trained weights, runs the test set, writes figures and results tables |
| `slurm/*.slurm` | Rangpur batch scripts for training and prediction |

### Dependencies
Exact versions used for all reported results (Rangpur `torch` conda environment):

| Package | Version | Used for |
|---|---|---|
| Python | 3.11.16 | |
| PyTorch | 2.14.0 (CUDA 13.0 build) | models, training, augmentation, metrics |
| nibabel | 5.4.2 | reading NIfTI volumes (`dataset.py` only) |
| NumPy | 2.4.6 | NIfTI → tensor conversion, plotting (not used in `modules.py`) |
| matplotlib | 3.11.1 | training curves and figures |

Install into a fresh environment with `pip install -r requirements.txt` (pins the versions above).

Hardware: Rangpur, 1 × NVIDIA A100-PCIE-40GB per job.

### Reproducing the results
```bash
# on Rangpur, inside recognition/ImprovedUNet3D_s49116889 with the conda env active
python dataset.py                 # one-off: preprocess 211 volumes to ~/hipmri_cache (≈0.8 GB)
sbatch slurm/train_3d.slurm       # 3D Improved U-Net -> runs/3d/
sbatch slurm/train_2d.slurm       # 2D U-Net baseline -> runs/2d/
sbatch slurm/predict.slurm        # test-set inference + figures -> figures/
```
Trained weights (`runs/<model>/best.pt`, ≈40 MB each) are deliberately **not committed**, as the brief
requires; the commands above regenerate them in about 50 minutes of A100 time.
`python train.py --model 3d --quick` runs a 2-epoch smoke test. All randomness is seeded (`--seed 42`):
the patient split is fully deterministic; training results may vary slightly between runs because of
non-deterministic GPU kernels.

---

## 4. Data, preprocessing and splits

**Raw data.** `B006_Week0_LFOV.nii.gz` (MRI) pairs with `B006_Week0_SEMANTIC.nii.gz` (labels): patient
`B006`, scanned at week 0. Volumes are 256 × 256 × 128 or 256 × 256 × 144 voxels at ≈1.56 mm isotropic.

**Preprocessing** (`dataset.preprocess_case`, run once and cached):

1. **z-score normalisation** per volume — removes scanner-dependent intensity scale.
2. **Pad the slice axis to 160** (with the volume's minimum intensity / background label) so both volume
   sizes share one shape **without cropping any anatomy**.
3. **Downsample ×2 in every axis → 80 × 128 × 128 voxels (≈3.1 mm)**: average pooling for images
   (anti-aliased), nearest-neighbour for labels (class values are never blended). This keeps a whole
   volume plus activations comfortably within GPU memory and makes all dimensions divisible by 16 for
   the four downsampling steps.
4. **Augmentation (training only):** random left–right flip, random in-plane rotation (±10°) and
   scaling (0.9–1.1) for 3D volumes, random intensity scale/shift (±10 %).

**Class balance after preprocessing** (all 211 volumes):

| Class | background | body | bone | bladder | rectum | prostate |
|---|---|---|---|---|---|---|
| % of voxels | 68.48 | 28.18 | 2.68 | 0.46 | 0.115 | **0.080** |

**Patient-level split** (`dataset.patient_split`, seed 42):

| Split | Patients | Volumes |
|---|---|---|
| Train | 26 | 136 |
| Validation | 6 | 48 |
| Test | 6 | 27 |

Each patient was scanned several times (different weeks), and those scans are nearly identical
anatomically. Splitting **by patient** guarantees the test set contains only people the model has never
seen; splitting by volume or slice would leak the same anatomy into training and inflate test scores.
The split is checked with assertions in code and saved to `runs/<model>/split.json`. Validation (mean
foreground Dice) selects the checkpoint; the test set is used once, at the end.

---

## 5. Training configuration

| | 3D Improved U-Net | 2D U-Net baseline |
|---|---|---|
| Input | whole volume 1 × 80 × 128 × 128 | axial slice 1 × 128 × 128 (all 80 per volume) |
| Batch size | 2 volumes | 32 slices |
| Epochs | 200 (validate every 5) | 40 (validate every epoch) |
| Optimiser | Adam, lr 5e-4, weight decay 1e-5, cosine decay | Adam, lr 1e-3, weight decay 1e-5, cosine decay |
| Precision | bfloat16 mixed precision | bfloat16 mixed precision |
| Model selection | best validation mean foreground Dice → epoch **120** (val 0.918) | same → epoch **33** (val 0.896) |

![3D training curves](figures/curves_3d.png)
![2D training curves](figures/curves_2d.png)

Both models converge smoothly with no sign of over-fitting: validation Dice plateaus rather than falling
while training loss keeps decreasing slowly. The 3D model's validation prostate Dice settles around
0.85–0.86; the 2D model's around 0.80–0.81 with visibly noisier epoch-to-epoch swings, an early hint
that slice-by-slice prediction is less stable on the prostate.

---

## 6. Results

All numbers are on the **held-out test set: 6 patients never seen in training or model selection
(27 volumes)**. Dice/IoU are computed on whole 3D volumes for both models.

### 6.1 Segmentation accuracy per class

| Class | 3D Improved U-Net Dice | 3D IoU | 2D U-Net Dice | 2D IoU | 3D better in |
|---|---|---|---|---|---|
| background | 0.999 | 0.998 | 0.998 | 0.995 | – |
| body | 0.988 | 0.977 | 0.985 | 0.970 | 22 / 27 volumes |
| bone | 0.929 | 0.868 | 0.925 | 0.861 | 26 / 27 |
| bladder | 0.952 | 0.910 | 0.950 | 0.906 | 18 / 27 |
| rectum | 0.864 | 0.764 | 0.855 | 0.753 | 15 / 27 |
| **prostate** | **0.870** | **0.773** | 0.822 | 0.700 | **26 / 27** |
| foreground mean | **0.921** | | 0.907 | | |

* **Every label meets the ≥ 0.7 Dice target** for both models (acceptance criterion a).
* Large, high-contrast structures (body, bone, bladder) are essentially solved by either model; the
  differences there are ≤ 0.004 Dice.

* The gap is concentrated on the **prostate (+0.048 Dice, +0.073 IoU)**, the smallest class
  (0.08 % of voxels). The 3D model is better on 26 of 27 test volumes and on **every one of the 6 test
  patients** (per-patient mean prostate Dice 3D vs 2D: J026 0.883 vs 0.827, K008 0.826 vs 0.815,
  S028 0.887 vs 0.818, T009 0.906 vs 0.884, T014 0.828 vs 0.781, V027 0.899 vs 0.853).

* Worst-case volumes: the 3D model's lowest prostate Dice is 0.783 (2D: 0.734). For the rectum the 3D
  model's worst volume (0.699) is still better than the 2D model's (0.607). The bladder is the one
  class where the 2D model has the better worst case (0.836 vs 0.776).

### 6.2 The dilemma: base/apex vs central prostate slices

| Prostate Dice on… | 3D Improved U-Net | 2D U-Net | 3D gain |
|---|---|---|---|
| extremity slices (outer 25 % at each end) | **0.840** | 0.778 | **+0.062** |
| central slices (middle 50 %) | 0.907 | 0.889 | +0.018 |

![Prostate Dice along the gland](figures/slice_profile.png)

The plot shows mean per-slice prostate Dice as a function of position along the true prostate
(0 = first slice, 1 = last). Slice index increases towards the bladder (the coronal views in 6.4 show the
bladder above the prostate), so **0 ≈ apex** (inferior, pelvic floor) and **1 ≈ base** (superior,
against the bladder).

* In the **central half** both models are at 0.86–0.92 and the curves almost touch: here 3D context
  adds little.

* At the **apex** the 2D model collapses to **0.40** in the outermost tenth of the gland, while the 3D
  model holds **0.68** — the single largest difference anywhere in the evaluation.

* At the **base** both models drop (0.76 vs 0.72), and the 3D advantage is small.

So the answer to the brief's core question is: **3D context does genuinely reduce the hazardous
extremity errors, rather than just polishing easy central slices — its gain at the extremities is
about 3.5× its gain in the centre — but it does not eliminate them.** Both models remain clearly worse at the
ends of the gland than in the middle (Section 7 shows why).

### 6.3 Computational cost

| Resource (1 × A100-40GB) | 3D Improved U-Net | 2D U-Net | Ratio |
|---|---|---|---|
| Parameters | 9,462,722 | 7,762,630 | 1.2× |
| Peak training VRAM | 2.77 GB | 0.92 GB | 3.0× |
| Peak inference VRAM | 0.76 GB | 0.45 GB | 1.7× |
| Inference time per volume | 269 ms | 260 ms | 1.0× |
| Training time | 32 min | 15 min | 2.1× |

The brief anticipated 3D models costing "5–10× the GPU memory and training time". At this
resolution (80 × 128 × 128, ≈3.1 mm voxels) the real overhead is **3× training memory and 2× training
time**, and **inference is no slower**: one 3D forward pass over the volume costs about the same as 80
2D forward passes. Both fit easily on a single consumer-grade GPU.

### 6.4 Example predictions

Top row: axial slice through the centre of the prostate. Bottom row: coronal view through the same point
(superior is up), which shows how consistent each model is **across** slices.
Colours: body grey, bone white, bladder yellow, rectum brown, prostate red.

![Example T014_Week2](figures/comparison_T014_Week2.png)
![Example J026_Week4](figures/comparison_J026_Week4.png)
![Example V027_Week7](figures/comparison_V027_Week7.png)

These are the 3D model's worst (T014_Week2, prostate Dice 0.783), median (J026_Week4, 0.881) and best
(V027_Week7, 0.925) test volumes by prostate Dice. In the central axial slices the two models are nearly indistinguishable, consistent with
6.2; differences show up at the top and bottom of the gland in the coronal views.

---

## 7. Failure case analysis

`predict.py` ranks every axial slice that contains prostate (in the ground truth or the prediction) by
per-slice prostate Dice for the 3D model and keeps **one slice per test patient**, mixing slices **inside**
the true prostate extent (under-segmentation) with slices **outside** it (over-extension). Each figure
shows MRI, ground truth, the 3D prediction, the 2D prediction on the same slice, and the 3D model's
errors (**red = false positive, blue = false negative**).

| # | Slice | Position | 3D error | 3D slice Dice | 2D slice Dice | Type |
|---|---|---|---|---|---|---|
| 1 | K008_Week0 z=36 | first prostate slice (apex) | 32 voxels missed → body | 0.00 | 0.00 | A: apex under-segmentation |
| 2 | S028_Week0 z=49 | last prostate slice (base) | 38 voxels missed → **bladder** 33, body 5 | 0.00 | 0.10 | B: base–bladder confusion |
| 3 | T014_Week1 z=27 | first prostate slice (apex) | 19 voxels missed → body | 0.00 | 0.00 | A: apex under-segmentation |
| 4 | J026_Week0 z=30 | 1 slice below the true apex | 18 voxels of false prostate | 0.00 | – (none predicted) | C: apex over-extension |
| 5 | V027_Week1 z=35 | 1 slice below the true apex | 12 voxels of false prostate | 0.00 | – (none predicted) | C: apex over-extension |

**All five worst slices are at the very first or last slice of the gland, or one slice beyond it** —
none is in the central region. This directly confirms the dilemma described in the brief.

**Type A — apex under-segmentation (failures 1 and 3).** At its apex the prostate tapers to a tiny cross-
section: 19–32 voxels, i.e. roughly 2–3 cm² at our 3.1 mm resolution, wedged against the rectum
(brown, immediately adjacent in the ground truth) and the pelvic-floor muscles, which have very similar
T2 intensity. Both models label it as body. Root triggers: (i) **partial-volume blurring** from the ×2
downsampling, which leaves only a few voxels across the structure; (ii) **low soft-tissue contrast**
against surrounding muscle; (iii) the Dice term rewards getting the bulk of the gland right, so a few
apex voxels contribute little to the loss. 3D context does not rescue these slices, because the
neighbouring slice below contains no prostate at all.

![Failure 1](figures/failure_1_K008_Week0.png)
![Failure 3](figures/failure_3_T014_Week1.png)

**Type B — base/bladder confusion (failure 2).** On the last prostate slice the base pushes up into the
bladder wall. The 3D model assigns 33 of the 38 missed voxels to **bladder**, the 2D model gets only a
sliver (0.10). The trigger is the **ambiguous prostate–bladder interface** at the base: at 3.1 mm slice
thickness, this slice averages prostate with the bright, fluid-filled bladder directly above it, and
the 3D model's context — which sees a lot of bladder in the slices above — pulls the decision towards
bladder. Clinically this is **under-contouring the target**, the error the brief associates with risk of
tumour recurrence.

![Failure 2](figures/failure_2_S028_Week0.png)

**Type C — apex over-extension (failures 4 and 5): an error the 3D model introduces.** Here the 3D model
continues the prostate **one slice past the true apex** with a small 12–18-voxel blob directly in front of
the rectum, where the 2D model correctly predicts no prostate. This is the flip side of 3D context:
the model has learnt that the gland is continuous across slices and **extrapolates its shape one slice
too far**. One 3.1 mm slice is a small geometric error, and the apex location is itself hard to define
precisely in MRI, but the location matters clinically: false prostate next to the rectum means
**over-contouring towards the rectal wall**, which the brief links to radiation proctitis.

![Failure 4](figures/failure_4_J026_Week0.png)
![Failure 5](figures/failure_5_V027_Week1.png)

**Which cases are unsuitable for automatic drafting?** The first and last 1–2 prostate slices at both the
apex and the base, the prostate–bladder interface, and any small prostate island adjacent to the rectum
below the main gland.

---

## 8. Recommendation to the project manager

**Recommendation: advance the 3D Improved U-Net as a draft-contouring assistant, with mandatory human
review of the prostate apex and base. Do not use it for unreviewed contours.**

**Why the 3D model is worth its cost**

* It is more accurate on the clinical target: prostate Dice 0.870 vs 0.822, better on 26 / 27 volumes and on
  all 6 unseen patients, with no class meaningfully worse.

* Its advantage is largest exactly where 2D fails: +0.062 Dice on the apex/base slices vs +0.018 centrally,
  and 0.68 vs 0.40 on the outermost apex slices.

* The cost is modest: 3× training memory (2.8 GB), 2× training time (32 min), and **the same inference
  time (≈0.27 s per volume)**, so there is no deployment-speed penalty.

**Why it is not ready to run unsupervised**

* The extremity slices are still its weakest region (0.84 vs 0.91 centrally; 0.68 at the very apex), and
  all five worst slices are there.

* It has a characteristic over-extension of the apex towards the rectum (Type C), and it can absorb the
  prostate base into the bladder (Type B). Both affect organs at risk or target coverage.

**Proposed operating rule for the contouring pipeline**

1. Accept the automatic body, bone and bladder contours with a light check (Dice ≥ 0.92 on unseen
   patients; the bladder can still swallow the prostate base, so check that interface).
2. For the prostate, auto-accept the central slices but **flag the first and last ~25 % of slices of the
   predicted prostate, plus one slice beyond each end, for mandatory review** — this covers every failure
   observed.
3. Highlight any prostate component touching the rectum below the main gland for explicit sign-off.

**Next experiments (in priority order)**

1. Train at full resolution (1.56 mm) using 3D patches, to target the partial-volume apex failures (Type A).
2. Add a boundary-aware loss term (e.g. a surface/boundary loss) or higher weight on extremity slices.
3. Repeat both models with 3+ random seeds to put confidence intervals on the gains.
4. Evaluate on more patients: 6 test patients (27 volumes, with repeated scans of the same patient) is
   enough to see a consistent direction, but not to estimate rare failure rates.

**Acceptance criteria from the feasibility review**

| Criterion | Result |
|---|---|
| (a) Test Dice ≥ 0.7 on every label incl. prostate | ✅ lowest class mean: rectum 0.864, prostate 0.870 |
| (b) Per-label Dice + IoU for both models on identical test patients | ✅ Section 6.1 |
| (c) Extremity vs central prostate Dice + 3–5 diagnosed failures | ✅ Sections 6.2 and 7 |
| (d) Single A100, VRAM / parameters / inference time reported | ✅ Section 6.3 (max 2.8 GB) |

**Limitations.** One training run per model (no seed variance); 6 test patients; images downsampled to
≈3.1 mm voxels; the ground-truth apex/base boundaries are themselves uncertain, so 1-slice disagreements
there partly reflect annotation ambiguity.

---

## 9. Artificial Intelligence Usage Disclosure

> _[Yogesh: this section must be accurate and in your own words. Edit the draft below so that it
> describes exactly what you did. Delete anything that is not true and add anything missing.]_

**Tools used.** Anthropic Claude (via the Claude app), used throughout the project.

**What AI assisted with.**

* Choosing the project and model from the brief, and drafting the feasibility review.
* Writing the first versions of all code: `modules.py`, `dataset.py`, `utils.py`, `train.py`,
  `predict.py` and the Slurm scripts.

* Debugging Rangpur job failures (missing `matplotlib` on compute nodes, conda activation inside jobs,
  checking the CUDA driver) and the Git workflow.

* Drafting this README, including the architecture figure and the wording of the analysis.

**How I verified the AI output.**

* Inspected the real data on Rangpur myself (file names, 211 image/label pairs, 38 patients, volume
  shapes, voxel size, label values 0–5) and checked the loader against it.

* Smoke-tested every script before full runs; checked that the patient split assertions pass and that no
  patient appears in two splits (`figures/split.json`).

* Caught and fixed issues found during review, for example the 2D baseline originally skipping padding
  slices (unfair comparison), and the first failure-case selection picking five slices from one patient.

* Checked every reported number against the training logs (`slurm-train_*.out`) and `test_metrics_*.json`,
  and every figure by eye.

* _[Add: which parts of the code you read through line by line and can explain.]_

All results in this report come from my own runs on Rangpur.

---

## References

1. O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," *MICCAI*, 2015.
2. F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, K. H. Maier-Hein, "Brain tumor segmentation and radiomics survival prediction: Contribution to the BRATS 2017 challenge," *Brainlesion (BrainLes) Workshop, MICCAI*, 2018.
3. Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, O. Ronneberger, "3D U-Net: Learning dense volumetric segmentation from sparse annotation," *MICCAI*, 2016.
4. F. Milletari, N. Navab, S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," *3DV*, 2016 (soft Dice loss).
5. HipMRI Study for prostate cancer radiotherapy (open release), provided via the COMP3710 Rangpur share; use governed by its `data_use_agreement.txt`.
