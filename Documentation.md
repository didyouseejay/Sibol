# SIBOL Project Documentation

**SIBOL: System for Image-Based Observation and Learning**
*A Comparative Study of Classical Machine Learning and Deep Learning Approaches for Flower Image Classification*

This file is the running record of all work on the project: what changed, why, which files were affected, how the change was validated, the results, known issues and next steps. It is **rewritten after every project change** so it always describes the current state. Earlier changes are summarised in the [Change history](#12-change-history).

**Last updated:** 2026-11-14 (Milestone 2 implementation)

---

## 1. Project at a glance

| Item | Detail |
|---|---|
| Goal | Compare three approaches to classifying Oxford Flowers-102 images |
| Approach 1 | Classical: HOG features + linear SVM, core algorithms written in NumPy/SciPy |
| Approach 2 | Custom CNN trained from scratch in PyTorch |
| Approach 3 | ImageNet-pretrained ResNet-18 adapted to 102 classes (transfer learning) |
| Dataset | Oxford Flowers-102: 8,189 images, 102 classes, official train/val/test splits |
| Tools | VS Code + Claude Code, Jupyter notebooks, Google Colab (T4 GPU), Google Drive, Gradio, Git/GitHub |

## 2. Current status

| Phase | Status |
|---|---|
| 1. Project scaffolding | Done (2026-11-11) |
| 2. Dataset acquisition and exploration | Done (2026-11-11); all 10 split checks passed |
| 2b. Infrastructure stabilisation | Done (2026-11-14); pushed to `origin/main` |
| 2c. Fresh-session Colab confirmation run | Done (2026-11-14); 10/10 checks passed, all Phase 2 results reproduced |
| 3. Milestone 2: classical pipeline (HOG + linear SVM) | **Implemented and unit-tested locally** (2026-11-14). **Training/evaluation on Flowers-102 not yet run** (needs Colab) |
| 4. Custom CNN | Not started |
| 5. ResNet-18 transfer learning | Not started |
| 6. Evaluation and comparison | Not started |
| 7. Gradio web app | Not started |
| 8. Final report | Not started |

The HOG + linear SVM model is implemented but has **not been trained on Flowers-102**. **No classification results exist yet.**

---

## 3. Phase 1: Project scaffolding (2026-11-11)

**What changed:** created the project structure (`notebooks/`, `src/{data,classical,models,evaluation,app}/`, `tests/`, `docs/`, `outputs/{figures,metrics}/`), `README.md`, `requirements.txt`, `.gitignore`, dataset helpers in `src/data/flowers102.py`, unit tests, `notebooks/00_environment_setup.ipynb` and a dataset exploration scaffold.

**Why:** to give all three approaches one shared, tested way of loading the official dataset splits, and to make the Colab/Drive workflow reliable from the start.

**Key design decisions**

| Decision | Reason |
|---|---|
| Reusable code in `src/`, experiments in notebooks | Notebooks stay short; code can be tested and reused by the app |
| Use torchvision's `Flowers102` | It reads the official `setid.mat` splits, so results are comparable with the literature |
| Minimum versions only in `requirements.txt` | Avoids replacing Colab's GPU builds of PyTorch |
| Notebook 00 never reinstalls PyTorch/Torchvision | Reinstalling can break Colab's CUDA setup |
| scikit-learn / scikit-image only as references and for metrics | The study is about our own HOG and SVM implementations |

**Validation:** both notebooks valid (nbformat v4) and executed without errors locally; 9 unit tests passed. Notebook 00 found that the local Jupyter kernel's PyTorch 2.3.1 cannot exchange data with NumPy 2.5.2, so a PyTorch↔NumPy check was added to its summary.

---

## 4. Phase 2: Dataset acquisition and exploration (2026-11-11)

**What changed:** the dataset was prepared and explored in Google Colab. The run is preserved as [`notebooks/archive/01_dataset_exploration_sibol_run_2026-11-11.ipynb`](notebooks/archive/01_dataset_exploration_sibol_run_2026-11-11.ipynb) and written up in [`docs/dataset_exploration_report.md`](docs/dataset_exploration_report.md).

**How the run worked** (workarounds later fixed in Phase 2b):
- The GitHub repository was empty, so a setup cell wrote a simplified copy of `src/data/flowers102.py` into `/content/sibol`.
- The archive was copied from Drive to `/content/data`. When the metadata files were missing, the loader fell back to `download=True` and downloaded `imagelabels.mat` and `setid.mat` from the official site.

**Results**

| Check | Observed |
|---|---|
| Split sizes | train 1020, val 1020, test 6149 |
| Total | 8189 |
| Classes per split | 102 in each |
| Images per class (train, val) | exactly 10 |
| Files shared between splits | 0 |
| **Integrity checks** | **10 / 10 passed** |

| Split | Min / class | Max / class | Mean / class |
|---|---:|---:|---:|
| Train | 10 | 10 | 10.0 |
| Val | 10 | 10 | 10.0 |
| Test | 20 | 238 | 60.3 |

Image sizes (300 random training images): width 500–857 px (median 667), height 500–861 px (median 500), aspect ratio 0.58–1.71 (median 1.33). The shorter side was exactly 500 px in 94% of sampled images (corrected in Phase 2c; this file previously said "every").

**Takeaways for later phases:** use the official splits unchanged; report macro-averaged and per-class metrics because the test set is imbalanced; every pipeline needs a resize/crop step; expect cluttered backgrounds, variable scale and colour variation within classes.

---

## 5. Phase 2b: Infrastructure stabilisation (2026-11-14)

Fixes the infrastructure problems recorded after Phase 2, before Milestone 2 starts.

### 5.1 Task 1: Git repository

**What changed**
- Initialised a Git repository and created the **initial commit** (`20f73a4`) after reviewing every staged file. It is now on branch `main` and pushed to `https://github.com/didyouseejay/sibol.git`.
- Rewrote `.gitignore`:
  - **excluded:** datasets and archives (`/data/`, `*.tgz`, `*.mat`, `flowers-102/`, `jpg/`), model checkpoints and weights (`/checkpoints/`, `*.pt`, `*.pth`, `*.ckpt`, …), cached features (`/features/`, `/cache/`, `*.npy`, `*.npz`, `*.h5`, …), temporary files, secrets and credentials (`.env`, `kaggle.json`, `credentials*.json`, …), Python caches, virtual environments and personal editor settings (`.vscode/`);
  - **tracked:** source code, tests, docs, notebooks, lightweight figures (`outputs/figures/*.png|svg|pdf`) and small metric files (`outputs/metrics/*.csv|json`).
- Added `.gitattributes` so line endings are consistent between Windows and Colab (Linux).
- **Notebook outputs:** the 3.3 MB executed notebook was inspected. Its four embedded figures are byte-identical to the PNGs in `outputs/figures/`, which was verified by comparing bytes. It was archived as a 34 KB record, `notebooks/archive/01_dataset_exploration_sibol_run_2026-11-11.ipynb`, which keeps all text outputs (the 10 PASS lines, statistics, class names); each image was replaced by a pointer to its PNG. A "do not run" banner was added. No figures or observations were deleted.
- Added `tests/test_notebooks.py`, which fails if a working notebook embeds images, exceeds 1 MB, or contains `download=True`.

**Why:** the project had no version control, the GitHub repository was empty (so Colab could not get the real code), and heavy notebook outputs would bloat the repository.

**Issue found and fixed during staged-file review:** the old pattern `data/` also matched `src/data/`, which would have **left the dataset helper code out of the commit**. All dataset, cache and temp folder patterns are now anchored to the repository root (`/data/`, …).

### 5.2 Task 2: No unintended internet downloads

**What changed** in `src/data/flowers102.py` (rewritten):
- `load_split()` **never downloads**. It no longer has a `download` parameter, always calls torchvision with `download=False`, and raises `FileNotFoundError` if the prepared data is incomplete.
- New `prepare_dataset(source_dir, work_root, download=False)` is the **only** path that can download, and only when `download=True` is passed. It then fetches only the *missing* files, from the official site, with MD5 verification.
- With `download=False`, missing source files raise a `FileNotFoundError` that lists each missing file, the folder where it should be placed, the official download page, and the alternatives (change the source-folder setting, or set `DOWNLOAD = True`).
- New `verify_source_files()` checks the official MD5 checksums of the three source files.
- The notebook's old fallback (`except RuntimeError: … download=True`) was removed.

**Why:** the 2026-11-11 run silently downloaded two metadata files from the internet even though `DOWNLOAD = False`.

### 5.3 Task 4: Fresh Colab session reproducibility

**What changed:** `notebooks/01_dataset_exploration_sibol.ipynb` was rewritten (no outputs) to run with **Runtime → Run all** in a brand-new Colab session:

| Requirement | How it is met |
|---|---|
| Mount Google Drive explicitly | Section 2 calls `drive.mount()` and checks `MyDrive/sibol` exists |
| All paths from configuration | Section 1 holds every path and option (`REPO_URL`, `CODE_DIR_COLAB`, `SOURCE_DIR_COLAB`, `WORK_ROOT_COLAB`, `FIGURES_DIR_COLAB`, local equivalents, `DOWNLOAD`, `VERIFY_MD5`, `SEED`, …) |
| Verify the three source files | Section 5 checks `102flowers.tgz`, `imagelabels.mat` and `setid.mat` exist in `MyDrive/sibol/data/` and match the official MD5 checksums |
| Extract images to temporary storage | Section 6 extracts to `/content/flowers102/flowers-102/jpg` (Colab's fast local disk) unless a complete copy (8,189 images) already exists in that runtime |
| Load labels and official splits | Section 7 loads all three splits with torchvision; nothing is downloaded |
| Run all 10 integrity checks | Section 8 uses `f102.integrity_checks()` and **stops the notebook** if any check fails. The disjointness check now fails instead of being skipped when file names are unavailable |
| No leftovers from earlier sessions | Clones the code if absent; refuses a non-empty folder without the helper (e.g. an old empty clone); clears previously imported `src` modules; rejects an outdated helper that lacks `prepare_dataset` |

Other improvements:
- Figures are saved to `MyDrive/sibol/outputs/figures` and persist across sessions. Previously they were saved in the Colab VM and lost.
- The single-class figure is now saved too.
- Extraction is safe: only `jpg/*.jpg` entries are written, into a temporary folder that is renamed only when extraction completes.
- Every class tied for the smallest test count is now listed. Previously only 5 tied classes were shown, and which 5 depended on sort order.
- The recorded observations were kept in Section 13 and reworded to reflect the tie.

**Drive check (read-only):** a Google Drive search on 2026-11-14 showed all three source files already in `MyDrive/sibol/data/`: `102flowers.tgz` (344,862,509 bytes), `imagelabels.mat` (502 bytes) and `setid.mat` (14,989 bytes). The archive is no longer under `src/`. **No Drive files were moved, deleted or modified.**

### 5.4 Task 5: Validation performed and results

| Validation | Result |
|---|---|
| nbformat v4 schema validation of all 3 notebooks (warnings treated as errors) | Pass |
| Every code cell of every notebook parsed with Python's `ast` | Pass (7 + 13 + 11 code cells) |
| Unit tests `python -m pytest tests/` (Python 3.12) | **30 passed**: 22 dataset-helper tests + 8 notebook hygiene tests |
| `DOWNLOAD=False` with missing files, unit level (network blocked at socket level) | `FileNotFoundError` naming the missing files and folder; no network call; nothing created |
| `DOWNLOAD=False`, whole notebook executed locally with network blocked and no dataset present | Stopped at Section 5 with the expected `FileNotFoundError`; no network attempt; no `data/` folder created |
| `download=True` is the only path that calls the downloader | Pass (unit test) |
| Safe extraction (fake archive with `../evil.jpg`, nested and non-jpg entries) | Only the 3 valid images extracted; nothing written outside the work folder |
| Incomplete extraction is detected and redone; a complete one is reused | Pass |
| The 10 integrity checks on synthetic official-shaped data; overlap, imbalance and missing file names detected | Pass |
| `.gitignore` probe: 23 paths that must be ignored, 15 that must be tracked | All correct |
| Secrets scan of staged content (API keys, tokens, passwords, private keys, emails) and notebook metadata | None found |
| No dataset, archive, `.mat`, image or checkpoint staged | Confirmed |
| Embedded figures vs `outputs/figures/*.png` | Byte-identical (all 4) |

**Positive path:** validated by the fresh-session Colab run in Phase 2c (Section 6).

### 5.5 Files affected

| File | Change |
|---|---|
| `src/data/flowers102.py` | Rewritten: prepare/verify/extract functions, no-download loading, the 10 integrity checks |
| `tests/test_flowers102_helpers.py` | Rewritten: 22 tests (no network, extraction safety, checks) |
| `tests/test_notebooks.py` | New: notebook hygiene tests |
| `notebooks/01_dataset_exploration_sibol.ipynb` | Rewritten for fresh Colab sessions; no outputs |
| `notebooks/archive/01_dataset_exploration_sibol_run_2026-11-11.ipynb` | New: slim archived record of the 2026-11-11 run |
| `notebooks/01_dataset_exploration.ipynb` | **Removed**: an outputless scaffold fully superseded by the notebook above |
| `.gitignore` | Rewritten and root-anchored |
| `.gitattributes` | New: line-ending normalisation |
| `README.md` | Notebook list, dataset-file instructions, network policy, versioning policy |
| `docs/dataset_exploration_report.md` | Links to the run record, notes on the fix, tie wording, figures now tracked |
| `Documentation.md` | Rewritten (this file) |

---

## 6. Phase 2c: Fresh-session Colab confirmation run (2026-11-14)

**What happened:** `notebooks/01_dataset_exploration_sibol.ipynb` was run with **Runtime → Run all** in a fresh Colab session. The notebook cloned the repository (commit `20f73a4`), mounted Drive, verified the three source files (**MD5 OK** for all three), extracted 8,189 images to `/content/flowers102` in 5 s, loaded the splits without downloading anything, passed **10/10** integrity checks and saved the four figures to `MyDrive/sibol/outputs/figures`.

The executed copy is kept locally as `notebooks/01_dataset_exploration_sibol_result.ipynb` and is **not tracked**. It embeds images and Drive paths, so `.gitignore` now excludes `notebooks/*_result.ipynb`. A scan found no secrets, tokens or email addresses in it.

**Results compared with the 2026-11-11 run**

| Item | 2026-11-14 run | Same as Phase 2? |
|---|---|---|
| Split sizes, total, classes, 10 per class, 0 shared files | train 1020, val 1020, test 6149; 8189; 102 each; 10; 0 | Yes |
| Test images per class | min 20, max 238, mean 60.3 | Yes |
| Largest test classes | petunia 238, passion flower 231, wallflower 176, water lily 174, watercress 164 | Yes |
| Smallest test classes (20 each) | **7 tied:** pink primrose, canterbury bells, moon orchid, fire lily, prince of wales feathers, mexican aster, bolero deep blue | Yes (Phase 2 printed only 5 of the 7) |
| Image sizes (300 train, seed 42) | width 500–857 (median 667), height 500–861 (median 500), aspect 0.58–1.71 (median 1.33) | Yes |
| Shorter side = 500 px | **94%** of sampled images | **No:** Phase 2 said "every" image, judged from the scatter plot |
| Random grid and pink primrose figure | Same 16 images (same seed); the visual observations hold | Yes |

**Corrections made**

| File | Change |
|---|---|
| `notebooks/01_dataset_exploration_sibol.ipynb` | Section 13 only: provenance now cites this run; all 7 tied smallest classes listed; "every image has a 500 px shorter side" replaced by "94%" (the other ~18 images are above 500 px by an amount too small to see in the plot; not measured) |
| `docs/dataset_exploration_report.md` | The same two corrections; the run date notes the re-confirmation |
| `.gitignore` | Ignores `notebooks/*_result.ipynb` |
| `tests/test_notebooks.py` | The no-images / size / no-download rules skip git-ignored `*_result.ipynb` copies; those copies are still checked for valid JSON |
| `Documentation.md` | Rewritten (this file) |

**Validation:** `py -3.12 -m pytest tests/`: 31 passed (the 30 tracked tests plus a JSON-validity case for the local result copy). The notebook diff is the 3 changed lines in Section 13. `git check-ignore` confirms the result copy is ignored. **Nothing was committed or pushed** in this phase.

---

## 7. Milestone 2: HOG + linear SVM implementation (2026-11-14)

> **Status:** implemented and tested locally. **The model has not been trained or evaluated on Flowers-102**; that requires the Colab run (Section 11). No experimental results are reported here.

### 7.1 What changed and why

The classical pipeline (approach 1) was implemented **from scratch in NumPy**, following the approved plan and these decisions:

| Decision | Choice |
|---|---|
| Resizing | Direct resize to 128 × 128 (no centre crop). Aspect-ratio distortion is a documented limitation |
| Final model | Trained on the official **train** split only (1,020); hyperparameters chosen on **val** (1,020); evaluated **once** on **test** (6,149). No refit on train + val |
| Selection metric | Validation **macro-F1** (ties: validation top-1, then grid order); top-1, macro precision, macro recall and macro-F1 all reported |
| Class imbalance | Optional positive-class weighting (`None`, a number, or `"balanced"` = n_neg / n_pos = 101); unweighted and balanced are both in the validation grid and compared, with no assumption that weighting helps |
| scikit-image | Only an optional development test (loose agreement); never used to compute features |

No ready-made HOG or SVM is used. scikit-learn is used only for metrics. No new dependencies.

### 7.2 Implementation details

**Preprocessing** (`src/classical/preprocessing.py`): RGB → Pillow bilinear resize to 128 × 128 → grayscale with BT.601 weights (0.299, 0.587, 0.114) in NumPy → float64 in [0, 1]. `FeatureStandardizer` (NumPy mean/std, chunked transform) is fitted on the training features only; zero-variance features keep scale 1.

**HOG** (`src/classical/hog.py`), one function per step:

| Step | Function | Detail |
|---|---|---|
| 1 | `compute_gradients` | centred [-1, 0, 1] differences; replicated borders (one-sided difference at the edge) |
| 2 | `magnitude_orientation` | `hypot(gx, gy)`; unsigned angle `atan2(gy, gx) mod 180` in [0, 180) |
| 3 | `cell_histograms` | 8 × 8 cells; magnitude-weighted votes split linearly between the two nearest of 9 bin centres (10°, 30°, …, 170°), wrapping at 0/180° |
| 4 | `build_blocks` | 2 × 2 cells, stride 1 → 15 × 15 blocks of 36 values |
| 5 | `l2_hys` | L2 (ε = 1e-5), clip at 0.2, L2 again; all-zero blocks stay zero (no NaN) |
| 6 | `hog_features` | flatten → **8,100** values (same layout as scikit-image's `feature_vector=True`) |

**Linear SVM** (`src/classical/linear_svm.py`), `LinearSVMOvR`: 102 one-vs-rest binary SVMs trained together as `W` (8,100 × 102) and `b` (102). Per class c:

`J_c(w, b) = (λ/2)·||w||² + (1/N)·Σᵢ s_ic · max(0, 1 − y_ic (w·xᵢ + b))`

- Hinge loss, L2 on `w` only (**the bias is not regularised**).
- Sample weight `s_ic` = positive weight for the class's own images, 1 for the rest. The average stays over N, so λ means the same with and without weighting.
- Mini-batch subgradient descent, seeded shuffling, step `η_t = η₀ / (1 + decay·epoch)`, zero initialisation.
- Prediction = argmax of the 102 scores. Parameters = 8,100 × 102 + 102 = **826,302**.

**Feature cache** (`src/classical/features.py`): one `hog_<split>_<config-hash>.npz` per split (float32 features, image IDs such as `image_00001`, labels, JSON metadata). A cache is reused only if the preprocessing config, HOG config (including an implementation version), cache-format version, split name, sample count, feature shape and a SHA-256 of the ordered (image ID, label) list all match, and every value is finite. Otherwise it is re-extracted. Writes are atomic (temporary file, then rename). During extraction, each label returned by the loader is checked against the split's label list. The metadata keeps the original extraction time, so timings are still reported when features come from the cache. `assert_disjoint_splits` rejects shared or duplicate image IDs and wrong split sizes.

**Evaluation** (`src/evaluation/`): `metrics.py` (top-1, macro P/R/F1 over all 102 labels with `zero_division=0`, per-class metrics, confusion matrix, JSON/CSV writers, environment info) and `plots.py` (HOG illustration, training curve, validation grid, row-normalised confusion matrix, lowest/highest per-class F1). Both are model-agnostic so the CNN and ResNet can reuse them.

**Notebook** `notebooks/02_hog_svm.ipynb` (committed without outputs) has sections 1–12 as requested. It reuses the notebook-01 conventions: config cell, explicit Drive mount, clone, purge of old `src` modules, MD5-verified source files, extraction to `/content`, the 10 integrity checks, and a stop on failure. It also refuses an old code checkout that lacks the M2 modules, and the test section raises if it is run a second time in the same session.

| Output (Colab) | Location |
|---|---|
| HOG cache (~265 MB: train 33 MB, val 33 MB, test 200 MB) | `MyDrive/sibol/features/hog/` |
| `hog_svm_best.npz`, `hog_svm_standardizer.npz` | `MyDrive/sibol/checkpoints/` |
| `hog_svm_results.json`, `hog_svm_validation_grid.csv`, `hog_svm_test_per_class.csv`, `hog_svm_test_confusion_matrix.csv` | `MyDrive/sibol/outputs/metrics/` |
| `hog_demo.png`, `hog_svm_training_curve.png`, `hog_svm_validation_grid.png`, `hog_svm_confusion_matrix.png`, `hog_svm_per_class_f1.png` | `MyDrive/sibol/outputs/figures/` |

**Validation grid:** λ ∈ {1e-4, 1e-3, 1e-2} × η₀ ∈ {1e-4, 1e-3, 1e-2} × positive weight ∈ {None, balanced} = 18 configurations, 50 epochs, batch 128, decay 0.05, seed 42. Locally, 30 epochs at full size (1,020 × 8,100 × 102) took about 10 s and stayed numerically stable for η₀ up to 0.1, so the grid should take roughly 5–10 minutes on a Colab CPU. This is an estimate, not a Colab measurement.

**Timings recorded separately:** HOG extraction per split (decode + resize + grayscale + HOG), HOG computation alone, SVM training time of the selected configuration, total grid time, SVM inference per image (standardise + score + argmax), and an end-to-end per-image estimate.

### 7.3 Validation performed (local: Windows, Python 3.12, NumPy 2.5.2, scikit-learn 1.5.1, scikit-image 0.26.0)

| Validation | Result |
|---|---|
| `py -3.12 -m pytest tests/` | **108 passed**, 0 skipped: HOG 20, SVM 19, preprocessing 14, features/cache/pipeline 13, evaluation 5, dataset helpers 22, notebooks 15 |
| HOG hand-checked values | horizontal edge → only the 90° bin; vertical edge → split 50/50 between bins 0 and 8; 45° ramp → 25 % / 75 % into bins 1 and 2; votes conserve magnitude; ramp gradients exact; L2-Hys matches the formula; zero image → zero vector |
| HOG validity | length 8,100; finite and non-negative on zero, constant, noise and checkerboard images; block norms ≤ 1; invariant to brightness offset and contrast scale |
| scikit-image reference (development only) | correlation **0.961** with `skimage.feature.hog` on a smooth synthetic image (expected to be below 1: different binning and border handling) |
| SVM | analytic gradient matches finite differences (weighted case); bias not regularised (two-epoch test); positive weighting scales positive hinge terms exactly; balanced = n_neg / n_pos; seeded runs identical; objective decreases; separable blobs learned; score and prediction shapes; 826,302 parameters for 8,100 × 102 |
| Cache | reloads without touching images; rebuilt after a config change, changed labels/IDs, a renamed file from another split, a corrupted file, or `force=True` |
| Splits | overlapping, duplicate or wrongly sized splits are rejected |
| Synthetic end-to-end pipeline | oriented-stripe images: HOG → train-only standardisation → SVM → metrics, ≥ 90 % accuracy on val and test |
| **Whole notebook executed locally** against a synthetic stand-in dataset with the official split sizes (1,020 / 1,020 / 6,149) and 3-epoch training | all 19 code cells ran; a second run loaded all three caches in 0.7 s; all output files were written; figures checked visually. The metrics from this run come from fake data and are **not results** |
| Bug found by that run | standardising the 6,149 test vectors created a 380 MB float64 temporary and failed on this machine (MemoryError). `transform` now works in 1,024-row chunks and writes directly to the target dtype |
| Measured locally | HOG alone ≈ 4–6 ms per 128 × 128 image |

**Not verified locally:** loading the real images (no torchvision and no dataset on this machine), real feature extraction, training, validation and test metrics, and Colab timings. **These need the Colab run.**

### 7.4 Files affected

| File | Change |
|---|---|
| `src/classical/preprocessing.py`, `hog.py`, `linear_svm.py`, `features.py` | New |
| `src/classical/__init__.py`, `src/evaluation/__init__.py` | Docstrings now list the modules |
| `src/evaluation/metrics.py`, `plots.py` | New |
| `tests/test_hog.py`, `test_linear_svm.py`, `test_preprocessing.py`, `test_features.py`, `test_evaluation.py` | New |
| `tests/test_notebooks.py` | Added a check that every code cell of every working notebook parses |
| `notebooks/02_hog_svm.ipynb` | New, without outputs |
| `README.md` | Status, structure, Drive layout, roadmap |
| `Documentation.md` | This section, status, inventory, issues, next steps, history |

`src/data/flowers102.py`, the official splits, notebooks 00/01, `.gitignore` and `requirements.txt` were **not changed**. Nothing was committed or pushed.

---

## 8. File inventory

| File / folder | Purpose | Tracked |
|---|---|---|
| `README.md` | Project overview and setup guide | Yes |
| `Documentation.md` | This running log | Yes |
| `requirements.txt` | Python dependencies (minimum versions) | Yes |
| `.gitignore`, `.gitattributes` | Versioning rules and line endings | Yes |
| `notebooks/00_environment_setup.ipynb` | Environment and GPU diagnostics, Drive mount | Yes |
| `notebooks/01_dataset_exploration_sibol.ipynb` | Prepares, verifies (10 checks) and explores the dataset | Yes |
| `notebooks/archive/01_dataset_exploration_sibol_run_2026-11-11.ipynb` | Record of the 2026-11-11 run (do not run) | Yes |
| `notebooks/01_dataset_exploration_sibol_result.ipynb` | Executed copy of the 2026-11-14 Colab run, with figures | **No** (git-ignored, local only) |
| `src/data/flowers102.py` | Dataset preparation, loading and integrity checks | Yes |
| `notebooks/02_hog_svm.ipynb` | Milestone 2: HOG + linear SVM, train / validate / test | Yes |
| `src/classical/` | Preprocessing, HOG, linear SVM, feature cache | Yes |
| `src/evaluation/` | Metrics, result files and figures (shared by all models) | Yes |
| `src/models/`, `src/app/` | Empty packages for later milestones | Yes |
| `tests/` | 108 unit, pipeline and hygiene tests | Yes |
| `docs/dataset_exploration_report.md` | Written findings from Phase 2 | Yes |
| `outputs/figures/*.png` | 4 exploration figures (2.47 MB total; largest `sample_images.png`, 1.72 MB) | Yes |
| `outputs/metrics/` | Empty, for future results | Yes (`.gitkeep`) |
| `.vscode/` | Personal editor settings | No |

**Data locations (not in git)**

| Location | Contents | Lifetime |
|---|---|---|
| `MyDrive/sibol/data/` | `102flowers.tgz`, `imagelabels.mat`, `setid.mat` | Persistent |
| `/content/flowers102/flowers-102/` (Colab) | Extracted `jpg/` + copies of the two `.mat` files | Deleted when the runtime ends |
| `MyDrive/sibol/outputs/figures/` | Figures from Colab runs | Persistent |
| `MyDrive/sibol/outputs/metrics/` | Metric files from Colab runs | Persistent |
| `MyDrive/sibol/features/hog/` | HOG feature cache, one `.npz` per split (~265 MB) | Persistent |
| `MyDrive/sibol/checkpoints/` | `hog_svm_best.npz`, `hog_svm_standardizer.npz` | Persistent |
| `<project>/features/`, `<project>/checkpoints/` (local) | Same, for local runs | Persistent, git-ignored |
| `<project>/data/` (local, optional) | Same three source files; `data/extracted/` for images | Persistent, git-ignored |

---

## 9. Environment notes

### Local machine (Windows 11)

| Python | Used by | PyTorch | Torchvision | NumPy | SciPy | Problems |
|---|---|---|---|---|---|---|
| 3.13 | `python` command | 2.15 (CPU) | 0.29 | present | **missing** | no SciPy, no pytest |
| 3.12 | Jupyter kernel, tests | 2.3.1 (CPU) | **missing** | 2.5.2 | 1.17.1 | PyTorch cannot exchange data with NumPy 2.x; no torchvision |

No local CUDA GPU. Nothing was installed. Git 2.51 is configured with user `didyouseejay`; the repository uses branch `main`, tracking `origin/main`.

### Google Colab

Used for the dataset runs on 2026-11-11 and 2026-11-14. Packages are preinstalled and not reinstalled.

---

## 10. Known issues and open items

| # | Issue | Impact | Suggested fix |
|---|---|---|---|
| 1 | Local Python environments are incomplete (Section 9) | The dataset cannot be loaded locally; local tests run on Python 3.12 only | Create a project virtual environment (commands in `README.md`) |
| 2 | Working notebooks must be committed without embedded images | `tests/test_notebooks.py` fails if a Colab-saved notebook with images is committed | Clear outputs before committing (Edit → Clear all outputs), or save the executed copy as `notebooks/<name>_result.ipynb`, which is git-ignored |
| 3 | Helpers read torchvision's private `_labels` / `_image_files` | A future torchvision could rename them | Label reading has a fallback; if file names disappear, the disjointness check fails loudly instead of passing silently |
| 4 | `sample_images.png` (1.72 MB) is the largest tracked file | Small repository growth | Acceptable; future figures should stay under about 2 MB |
| 5 | 6% of sampled images have a shorter side above 500 px; their exact sizes were not measured | None for the plan: every pipeline resizes anyway | Optional: print the non-500 sizes, or measure all 8,189 images |
| 6 | **M2 not yet trained or evaluated on Flowers-102** | No HOG + SVM results exist | Run `notebooks/02_hog_svm.ipynb` in Colab (Section 11) |
| 7 | Colab clones from GitHub, but the M2 code is not pushed yet | Notebook 02 stops at Section 1.3 with a message | Commit and push first, or copy the project to Drive and set `CODE_DIR_COLAB` |
| 8 | Direct 128 × 128 resize distorts aspect ratios (0.58–1.71) | Shapes are stretched, which may weaken HOG features | Accepted for M2; could be revisited in a later ablation |
| 9 | HOG ignores colour; the validation-grid ranges were chosen from a synthetic stability check, not from flower data | The best setting may sit on the edge of the grid | If the selected λ or learning rate is at a grid edge, extend the grid (validation only) and record it |
| 10 | Outputs written to Drive do not reach the repository automatically | Tracked `outputs/metrics` and `outputs/figures` stay without M2 files until copied | After the run, copy the small JSON/CSV/PNG files into the repository if they should be versioned |

**Resolved in Phase 2c:** repository pushed (`origin/main`); the rewritten notebook was confirmed by a fresh-session Colab run.

**Resolved in Phase 2b** (issue numbers from the previous version of this file): 1 git repository initialised (push pending, see issue 1 above), 2 Colab now uses the real project code, 3 source data persists on Drive and extraction to temporary disk is by design, 4 archive already in `MyDrive/sibol/data/`, 5 silent download removed, 6 Drive mounted explicitly, 7 figures saved to Drive, 8 heavy notebook archived as a slim record.

---

## 11. Next steps

1. **You:** review the Milestone 2 changes (Section 7.4), then commit and push them (Claude does not commit or push). Colab needs the pushed code.
2. **Run `notebooks/02_hog_svm.ipynb` in Colab:** use a CPU runtime (no GPU needed), then **Runtime → Run all**. If `/content/sibol` exists from an earlier session in the same runtime, run `!rm -rf /content/sibol` first. The first run extracts HOG features (several minutes) and caches them on Drive; later runs reuse them.
3. Save the executed copy as `notebooks/02_hog_svm_result.ipynb` (git-ignored), and share the printed outputs so the notebook's Section 12 ("Observations") and this file can be filled in with the **actual** numbers.
4. If the selected hyperparameters lie on a grid edge, extend the grid and re-run (validation only; the test section runs once per session).
5. Then Milestone 3: custom CNN.

---

## 12. Change history

| Date | Change |
|---|---|
| 2026-11-11 | Phase 1: project scaffolding (structure, README, requirements, .gitignore, data helpers, 9 tests, notebooks 00 and 01). |
| 2026-11-11 | Added a PyTorch↔NumPy compatibility check to notebook 00 after it found a NumPy 2.x / PyTorch conflict in the local Jupyter kernel. |
| 2026-11-11 | Phase 2: dataset acquired and explored in Colab; all 10 split checks passed. Observations, `docs/dataset_exploration_report.md`, figures in `outputs/figures/`, README update; created this file. |
| 2026-11-14 | Phase 2b: infrastructure stabilisation. Git initialised with a reviewed initial commit (not pushed); `.gitignore` rewritten and root-anchored (fixed `src/data/` being ignored); `.gitattributes` added; dataset helper rewritten so loading never downloads and `DOWNLOAD=True` is the only download path; notebook rewritten for fresh Colab sessions with explicit Drive mount, config-driven paths, MD5-verified source files, extraction to temporary disk and a stop on failed checks; 2026-11-11 run archived as a 34 KB record; superseded scaffold notebook removed; tests expanded from 9 to 30. |
| 2026-11-14 | Phase 2c: fresh-session Colab run confirmed 10/10 checks and all Phase 2 numbers. Corrected "shorter side 500 px in every image" to 94% and listed all 7 tied smallest test classes (notebook Section 13, report, this file). `.gitignore` now excludes `notebooks/*_result.ipynb`; notebook hygiene tests skip those copies. Not committed. |
| 2026-11-14 | Milestone 2 implementation: from-scratch NumPy HOG (8,100 features, L2-Hys) and one-vs-rest hinge-loss linear SVM (L2 on weights only, optional positive weighting, seeded mini-batch SGD); train-only standardiser; validated split-aware HOG cache; shared evaluation metrics/plots; notebook 02; tests 31 → 108. Notebook executed end to end locally on a synthetic stand-in dataset (fixed a test-split memory spike). **No Flowers-102 training or results yet.** Not committed. |
