# SIBOL — System for Image-Based Observation and Learning

**A Comparative Study of Classical Machine Learning and Deep Learning Approaches for Flower Image Classification**

SIBOL is a computer vision project that compares three ways of classifying flower images from the **Oxford Flowers-102** dataset: a classical hand-engineered pipeline, a convolutional neural network trained from scratch, and a transfer-learned ImageNet model. All three are trained and evaluated on the same official data splits, so the comparison is fair.

> **Project status:** dataset verified; Milestone 2 (from-scratch HOG + linear SVM) trained and evaluated: **test top-1 0.1197, macro-F1 0.1024**. Milestone 2b (feature improvements: colour, square-root normalisation, flip augmentation, centre crop) is implemented and tested locally; its Colab run (`notebooks/03_classical_improvements.ipynb`) is pending. Details in [`Documentation.md`](Documentation.md).

---

## Objectives

1. Implement a **classical** image classifier (HOG features + linear SVM) with the core algorithms written in NumPy/SciPy, so every step is visible and understood.
2. Design and train a **custom CNN from scratch** in PyTorch.
3. Adapt an **ImageNet-pretrained ResNet-18** to the 102 flower classes (transfer learning).
4. **Compare** the three approaches on the same official test split using accuracy, macro-averaged metrics, per-class performance, confusion matrices, training cost and inference speed.
5. Deploy the best model(s) in an interactive **Gradio** web application.

## Methods

| # | Approach | Features | Classifier | Main libraries |
|---|----------|----------|------------|----------------|
| 1 | Classical | Histogram of Oriented Gradients (HOG), implemented by us | Linear SVM, implemented by us | NumPy, SciPy |
| 2 | Deep learning from scratch | Learned by a custom CNN | Softmax output layer | PyTorch |
| 3 | Transfer learning | ResNet-18 pretrained on ImageNet | New 102-class output layer (fine-tuned) | PyTorch, Torchvision |

- **Approach 1** shows what hand-crafted gradient features can do with very little training data.
- **Approach 2** shows how a network learns features with no prior knowledge, using only ~2,000 labelled training images.
- **Approach 3** tests how much features learned on a large, general dataset help on a small, specialised one.

scikit-learn and scikit-image are used **only** as references to check our own implementations and for evaluation metrics. They are not the classifiers being studied.

## Dataset: Oxford Flowers-102

- 8,189 images of flowers common in the United Kingdom, in **102 categories**.
- **Official splits** (from the dataset's `setid.mat`):

| Split | Images | Per class |
|-------|-------:|-----------|
| Train | 1,020 | 10 |
| Validation | 1,020 | 10 |
| Test | 6,149 | 20–238 (imbalanced) |

- Loaded with `torchvision.datasets.Flowers102`, which reads the official split file.
- Source: M.-E. Nilsback and A. Zisserman, *Automated Flower Classification over a Large Number of Classes*, ICVGIP 2008. <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>

The dataset is **never committed** to this repository (see `.gitignore`).

## Repository structure

```
sibol/
├── notebooks/                  # Jupyter notebooks for experiments (run in order)
│   ├── 00_environment_setup.ipynb
│   ├── 01_dataset_exploration_sibol.ipynb   # prepares, verifies and explores the dataset
│   ├── 02_hog_svm.ipynb                     # Milestone 2: HOG + linear SVM (train / validate / test)
│   ├── 03_classical_improvements.ipynb      # Milestone 2b: colour, sqrt, flip, centre-crop ablation
│   └── archive/                             # records of past runs (do not run)
├── src/                        # Reusable Python code imported by the notebooks
│   ├── data/                   # Dataset loading and split helpers
│   ├── classical/              # HOG, colour histograms, linear SVM, feature cache, ablation (NumPy)
│   ├── models/                 # Custom CNN and ResNet-18              [planned]
│   ├── evaluation/             # Metrics, confusion matrices, plots
│   └── app/                    # Gradio web application                [planned]
├── tests/                      # Unit tests (pytest)
├── docs/                       # Reports, notes and diagrams
│   └── dataset_exploration_report.md
├── outputs/
│   ├── figures/                # Generated plots (git-ignored)
│   └── metrics/                # Generated metric files (git-ignored)
├── requirements.txt
├── Documentation.md            # Running log of all project work
└── README.md
```

## Development workflow

| Stage | Tool |
|-------|------|
| Writing code and notebooks | VS Code + Claude Code (local) |
| Experiments | Jupyter notebooks (`.ipynb`) |
| Training | Google Colab with an NVIDIA T4 GPU |
| Storage for data, checkpoints and outputs | Google Drive (`MyDrive/sibol/`) |
| Web demo | Gradio |

Typical loop:

1. Write and test code locally in `src/` and `tests/`.
2. Sync the repository to Google Drive (or clone it in Colab).
3. In Colab, run `notebooks/00_environment_setup.ipynb` first. It mounts Drive and checks the GPU.
4. Run the experiment notebooks. Source data and checkpoints stay on Drive; images are extracted to Colab's fast temporary disk (`/content/flowers102`) each session; figures are saved to `MyDrive/sibol/outputs/figures`.

### Dataset files

Put the three official files in `MyDrive/sibol/data/` (Colab) or `<project>/data/` (local):
`102flowers.tgz`, `imagelabels.mat`, `setid.mat` (from <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>).

Notebooks **never download data silently**. With `DOWNLOAD = False` (default), a missing file stops the notebook with a `FileNotFoundError` explaining what to put where. Only `DOWNLOAD = True` allows downloading.

Google Drive layout created by notebook 00:

```
MyDrive/sibol/
├── data/            # 102flowers.tgz, imagelabels.mat, setid.mat (source files)
├── checkpoints/     # Saved model weights (e.g. hog_svm_best.npz)
├── features/hog/    # Cached HOG features, one .npz per split (~265 MB)
└── outputs/
    ├── figures/
    └── metrics/
```

## Setup

### Local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name sibol --display-name "Python (sibol)"
```

For a local NVIDIA GPU, install `torch`/`torchvision` first using the command from <https://pytorch.org/get-started/locally/>.

Then run the tests:

```bash
python -m pytest tests/
```

### Google Colab

1. Choose **Runtime → Change runtime type → T4 GPU**.
2. Open and run `notebooks/00_environment_setup.ipynb`.
3. **Do not** reinstall PyTorch or Torchvision. Colab's preinstalled CUDA builds are what we want. The notebook lists any other missing packages and installs them only if you opt in.

## Roadmap

- [x] Project scaffolding, environment diagnostics, dataset exploration scaffold
- [x] Dataset acquisition and split verification (Colab, 2026-11-11)
- [x] Classical pipeline: HOG + linear SVM (NumPy): test top-1 0.1197, macro-F1 0.1024 (M2 run 2)
- [ ] Classical improvements (M2b): colour, sqrt, flip, centre-crop ablation: implemented and tested locally; Colab run pending
- [ ] Custom CNN from scratch (PyTorch)
- [ ] ResNet-18 transfer learning
- [ ] Unified evaluation and comparison
- [ ] Gradio web application
- [ ] Final report

## Data and artifact policy

Datasets, extracted images, model checkpoints, cached features, temporary files and secrets are **not** version-controlled (see `.gitignore`). Lightweight figures in `outputs/figures/` and small metric files (`.csv`, `.json`) in `outputs/metrics/` **are** tracked. Working notebooks are committed without embedded images; `tests/test_notebooks.py` enforces this.
