# 3D Improved U-Net for Prostate Radiotherapy Contouring (HipMRI 3D)

**Author:** Yogesh Sevda (s49116889) · COMP3710 Pattern Analysis Project · **Difficulty: Hard** (3D Improved U-Net, Section 2.2)

> **Status: DRAFT** — sections marked _[TBD]_ are filled in from the Rangpur runs.

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

## 2. Feasibility review (one page, checked off with the teaching team)

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

Install into a fresh environment with
`pip install torch==2.14.0 nibabel==5.4.2 numpy==2.4.6 matplotlib==3.11.1`.

Hardware: Rangpur, 1 × NVIDIA A100-PCIE-40GB per job.

### Reproducing the results
```bash
# on Rangpur, inside recognition/ImprovedUNet3D_s49116889 with the conda env active
python dataset.py                 # one-off: preprocess 211 volumes to ~/hipmri_cache (≈0.8 GB)
sbatch slurm/train_3d.slurm       # 3D Improved U-Net -> runs/3d/
sbatch slurm/train_2d.slurm       # 2D U-Net baseline -> runs/2d/
sbatch slurm/predict.slurm        # test-set inference + figures -> figures/
```
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
| Model selection | best validation mean foreground Dice | same |

_[TBD: training curves]_ `runs/3d/curves.png`, `runs/2d/curves.png`

---

## 6. Results

_[TBD — from `figures/results_table.md`]_

### 6.1 Test-set segmentation accuracy (6 unseen patients, 27 volumes)

### 6.2 Prostate base/apex vs central slices

### 6.3 Computational cost

### 6.4 Example predictions

---

## 7. Failure case analysis

_[TBD — 3–5 cases from `figures/failure_*.png`]_

---

## 8. Recommendation to the project manager

_[TBD]_

---

## 9. Artificial Intelligence Usage Disclosure

_[TBD — to be written by me, describing exactly how AI was used and how I verified it.]_

---

## References

1. O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," *MICCAI*, 2015.
2. F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, K. H. Maier-Hein, "Brain tumor segmentation and radiomics survival prediction: Contribution to the BRATS 2017 challenge," *Brainlesion (BrainLes) Workshop, MICCAI*, 2018.
3. Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, O. Ronneberger, "3D U-Net: Learning dense volumetric segmentation from sparse annotation," *MICCAI*, 2016.
4. F. Milletari, N. Navab, S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," *3DV*, 2016 (soft Dice loss).
5. HipMRI Study for prostate cancer radiotherapy (open release), provided via the COMP3710 Rangpur share; use governed by its `data_use_agreement.txt`.
