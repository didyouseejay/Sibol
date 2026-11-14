# Dataset Exploration Report: Oxford Flowers-102

**Project:** SIBOL — A Comparative Study of Classical Machine Learning and Deep Learning Approaches for Flower Image Classification
**Notebook (run record):** [`notebooks/archive/01_dataset_exploration_sibol_run_2026-11-11.ipynb`](../notebooks/archive/01_dataset_exploration_sibol_run_2026-11-11.ipynb). The current, fixed notebook is [`notebooks/01_dataset_exploration_sibol.ipynb`](../notebooks/01_dataset_exploration_sibol.ipynb).
**Run date:** 2026-11-11; all results re-confirmed by a fresh-session Colab run of the fixed notebook on 2026-11-14 (code commit `20f73a4`)
**Environment:** Google Colab, dataset extracted to `/content/data/flowers-102`, loaded with `torchvision.datasets.Flowers102`

All numbers in this report come directly from that run's outputs. No models were trained. The run used an earlier version of the notebook. Its two workarounds (a hand-written helper and a silent metadata download) were removed on 2026-11-14; see Section 3 and `Documentation.md`.

---

## 1. Purpose

Before building the three classifiers (HOG + linear SVM, custom CNN, ResNet-18), we need to know:

1. that the dataset loaded correctly and matches the official splits,
2. how the classes are distributed in each split,
3. what the images look like, and
4. how large the images are, since this affects preprocessing.

## 2. Dataset

| Property | Value |
|---|---|
| Name | Oxford Flowers-102 (Nilsback & Zisserman, ICVGIP 2008) |
| Classes | 102 flower categories |
| Images | 8,189 |
| Official split file | `setid.mat` |
| Label file | `imagelabels.mat` |
| Image archive | `102flowers.tgz` |
| Class names | From torchvision's `Flowers102.classes` (index 0 = pink primrose) |

## 3. How the data was obtained

1. The image archive `102flowers.tgz` was already on Google Drive at `MyDrive/sibol/src/102flowers.tgz`.
2. The notebook copied it to `/content/data/flowers-102/102flowers.tgz` on the Colab VM.
3. On the first load attempt, torchvision raised an error because the two metadata files were missing. The notebook then called the loader with `download=True`. torchvision reused the local archive (it was not downloaded again), extracted it, and downloaded the two small metadata files (`imagelabels.mat`, `setid.mat`) from the official source.

> **Fixed on 2026-11-14.** That automatic download was unintended. The current notebook reads all three source files from `MyDrive/sibol/data/` (where they now are), verifies their official MD5 checksums, and never downloads unless `DOWNLOAD = True`.

## 4. Split verification

| Check | Expected | Observed | Result |
|---|---|---|---|
| Train size | 1,020 | 1,020 | PASS |
| Validation size | 1,020 | 1,020 | PASS |
| Test size | 6,149 | 6,149 | PASS |
| Total images | 8,189 | 8,189 | PASS |
| Classes in train / val / test | 102 / 102 / 102 | 102 / 102 / 102 | PASS |
| Images per class (train) | 10 | min 10, max 10 | PASS |
| Images per class (val) | 10 | min 10, max 10 | PASS |
| Files shared between splits | 0 | 0 | PASS |

**Conclusion:** the loaded data matches the official splits exactly. All three approaches will use these splits unchanged: train for fitting, val for model and hyperparameter selection, test only for the final comparison.

## 5. Class distribution

Figure: `outputs/figures/class_distribution.png`

| Split | Images | Min / class | Max / class | Mean / class |
|---|---:|---:|---:|---:|
| Train | 1,020 | 10 | 10 | 10.0 |
| Val | 1,020 | 10 | 10 | 10.0 |
| Test | 6,149 | 20 | 238 | 60.3 |

Smallest test classes: seven classes are tied at 20 images each: pink primrose, canterbury bells, moon orchid, fire lily, prince of wales feathers, mexican aster and bolero deep blue. (The 2026-11-11 run printed only 5 of them; the 2026-11-14 run lists every tied class.)
Largest test classes: petunia (238), passion flower (231), wallflower (176), water lily (174), watercress (164).

**What this means for the study:**
- Training data is very small (10 images per class). This favours methods that need little data or reuse prior knowledge, which is the question the study is designed to test.
- The test set is imbalanced (about 12x between the smallest and largest class), so overall accuracy is dominated by the large classes. Each model should therefore be reported with **macro-averaged** metrics and per-class accuracy as well as overall accuracy.

## 6. Visual characteristics

Figures: `outputs/figures/sample_images.png` (16 random training images, seed 42) and `outputs/figures/class_000_examples.png` (all 10 pink primrose training images).

Observed in these images:
- **Cluttered natural backgrounds**: leaves, grass, soil and fences rather than plain backgrounds.
- **Multiple objects**: some images show several flowers, or flowers with buds (artichoke, pink primrose).
- **Colour varies within a class**: the three morning glory samples are pink, white/lilac and blue; pink primrose ranges from near-white to deep pink.
- **Scale and framing vary**: some flowers fill the frame (corn poppy, colt's foot), others are small in a larger scene (cautleya spicata, fritillary).
- **Watermarks**: several images contain photographer or copyright text (corn poppy, yellow iris, tree poppy, one pink primrose image).

These observations come from 26 images and illustrate the dataset's variety. They are not a systematic measurement.

## 7. Image sizes

Figure: `outputs/figures/image_sizes.png` (300 random training images, seed 42)

| Measure | Min | Median | Max |
|---|---:|---:|---:|
| Width (px) | 500 | 667 | 857 |
| Height (px) | 500 | 500 | 861 |
| Aspect ratio (w/h) | 0.58 | 1.33 | 1.71 |

The **shorter side is exactly 500 px in 94%** of sampled images (printed by the 2026-11-14 run): landscape images are 500 px tall and portrait images are 500 px wide. The other 6% (about 18 images) have a shorter side above 500 px. In the scatter plot they cannot be told apart from the 500 px lines, so the excess appears small, but it was not measured. (An earlier version of this report said *every* sampled image had a 500 px shorter side; that was read from the plot and is corrected here.) Most sampled images are landscape (median aspect ratio 1.33).

**Preprocessing considerations** (decisions are still to be made):
- Images vary in size, but HOG needs a fixed input size to produce fixed-length feature vectors, and the CNNs are trained on fixed-size batches. Every pipeline therefore needs a resize step.
- A common approach is to resize the shorter side and then centre-crop a square. With aspect ratios up to 1.71, a centre crop removes part of the outer image area; flowers are usually centred in the samples, but this has not been measured.
- ResNet-18 was pretrained on ImageNet at 224 × 224 with ImageNet normalisation. The transfer learning pipeline will need the same input conventions.
- Colour appears to be an important cue, but it also varies within classes (Section 6). Standard HOG is computed on gradient orientation and ignores colour; this is a known property of the classical approach that the comparison should discuss.

## 8. Limitations of this exploration

- Image sizes were measured on 300 training images, not the full dataset.
- Visual observations are qualitative and based on 26 images.
- The figures were saved inside the Colab VM (`/content/sibol/outputs/figures`), which is deleted when the session ends. The copies in the local `outputs/figures/` folder were extracted from the outputs embedded in the notebook.

## 9. Figures

| File | Content |
|---|---|
| `outputs/figures/class_distribution.png` | Images per class for train, val and test |
| `outputs/figures/sample_images.png` | 16 random training images with class names |
| `outputs/figures/class_000_examples.png` | All 10 training images of class 0 (pink primrose) |
| `outputs/figures/image_sizes.png` | Width vs height of 300 random training images |

These figures are tracked in git as lightweight outputs. Running the current notebook recreates them in `MyDrive/sibol/outputs/figures/`.
