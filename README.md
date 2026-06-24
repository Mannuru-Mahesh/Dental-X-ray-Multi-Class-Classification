# Dental X-ray Multi-Class Classification

An end-to-end medical image classification pipeline built on a set of 120
dental periapical X-ray images: ETL, exploratory data analysis, feature
engineering, and a comparison of six classical machine learning algorithms
plus a convolutional neural network.

## Important disclaimer

The source dataset (a Kaggle-style export named `Dataset/`) contains only
120 raw `.jpg` X-ray images with **no class labels, no annotation file, and
no folder structure by class**. There is no ground truth.

To still demonstrate a complete classification pipeline, this project
generates **weak / heuristic labels** with classic image-processing rules
(see [Auto-labeling methodology](#auto-labeling-methodology) below):

- **Filling**: a compact, well-defined region noticeably brighter
  (more radiopaque) than the surrounding tooth structure, consistent with
  restorative material.
- **Caries**: a compact region noticeably darker (more radiolucent) than
  the surrounding tooth structure, consistent with demineralized tissue.
- **Normal**: neither pattern is detected above a data-driven threshold.

**These labels have not been verified by a dentist or radiologist.** They
will contain mislabeled examples (root canals, pulp chambers, and normal
anatomical shadows can be picked up as false "caries"; small or thin
fillings can be missed). Every result in this README should be read as a
demonstration of pipeline engineering and ML methodology on a very small
dataset, not as a clinically validated diagnostic tool. See
[Limitations](#limitations) for the full picture.

## Project pipeline

```
Raw JPGs  ->  Extract  ->  Transform  ->  Auto-label  ->  EDA
                                                            |
                          Feature extraction  <-------------
                                  |
                 +----------------+----------------+
                 |                                 |
         Classical ML (6 algos)              CNN (from scratch)
                 |                                 |
                 +----------------+----------------+
                                  |
                       Final comparison + metrics
```

Run the whole thing with:

```bash
pip install -r requirements.txt
python main.py
```

This regenerates every file under `data/processed/`, `data/*.csv`,
`results/figures/`, `results/metrics/`, and `models/` from the 120 raw
images in `data/raw/`. Total runtime is roughly 1-2 minutes on CPU.

To classify a new X-ray image with the trained models:

```bash
python predict.py path/to/image.jpg --model classical
python predict.py path/to/image.jpg --model cnn
```

## Repository structure

```
dental-xray-classification/
├── data/
│   ├── raw/                  120 source X-ray images
│   ├── processed/            preprocessed images + tooth masks (generated)
│   ├── manifest.csv          per-image file metadata (generated)
│   ├── labels.csv            heuristic weak labels + anomaly features (generated)
│   └── features.csv          handcrafted ML feature matrix (generated)
├── src/
│   ├── etl/
│   │   ├── extract.py        scan raw images, build manifest
│   │   ├── transform.py      denoise, CLAHE, tooth segmentation
│   │   └── auto_label.py     heuristic weak-labeling (Normal/Caries/Filling)
│   ├── features/
│   │   └── feature_extraction.py   GLCM, LBP, intensity, edge density
│   ├── models/
│   │   ├── classical_models.py     6-algorithm training + cross-validation
│   │   └── cnn_model.py            CNN trained from scratch (PyTorch)
│   ├── evaluation/
│   │   ├── metrics.py              shared scoring utilities
│   │   └── final_comparison.py     cross-model comparison + plots
│   └── visualization/
│       └── eda_plots.py            exploratory graphs
├── results/
│   ├── figures/               all 12 generated graphs
│   └── metrics/                per-model JSON metrics + summary CSVs
├── models/                    saved model artifacts (.joblib, .pt)
├── main.py                    runs the full pipeline end to end
├── predict.py                 inference on a new image
├── requirements.txt
└── LICENSE
```

## Dataset

| Property | Value |
|---|---|
| Source images | 120 dental periapical X-rays |
| Original format | JPG, 748x512, grayscale/RGB mixed |
| Processed resolution | 256x256 (classical pipeline), 128x128 (CNN input) |
| Classes | Normal, Caries, Filling (heuristically assigned) |
| Class balance | Normal: 47, Caries: 42, Filling: 31 |

## ETL process

**Extract** (`src/etl/extract.py`): scans `data/raw/`, validates every
image opens correctly, and records width, height, color mode, and file
size into `data/manifest.csv`.

**Transform** (`src/etl/transform.py`): for every image —
1. Convert to grayscale and resize to 256x256 (aspect-preserving center crop).
2. Denoise with non-local means filtering (preserves fine edges better
   than Gaussian blur, important for not washing out thin lesions).
3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
   normalize exposure differences across source images.
4. Segment the tooth structure from the dark radiographic background via
   Otsu thresholding and morphological cleanup, keeping only connected
   components large enough to be real anatomy.

**Load / auto-label** (`src/etl/auto_label.py`): computes a local
intensity "trend" per image with a large median blur, then flags pixels
that deviate strongly brighter or darker than that trend within the tooth
mask. Connected-component filtering removes single-pixel noise. The
resulting bright-area-ratio and dark-area-ratio are converted to discrete
class labels with thresholds chosen by Otsu's method on the actual
*distribution* of those ratios across the dataset (a data-driven cut
point rather than a hand-picked constant). Output: `data/labels.csv`.

## Auto-labeling methodology

The decision boundary the auto-labeler ends up using is visualized below
— each point is one X-ray, plotted by how much localized darkening
(x-axis) vs. localized brightening (y-axis) was detected inside its tooth
region. The dashed lines are the Otsu-derived thresholds.

![Auto-labeling decision space](results/figures/05_anomaly_feature_scatter.png)

Sample processed images per resulting class:

![Sample grid](results/figures/02_sample_grid.png)

## Exploratory data analysis

Class distribution after auto-labeling:

![Class distribution](results/figures/01_class_distribution.png)

Pixel intensity distribution inside the tooth region, by class — Filling
cases skew brighter, Caries cases show a wider/darker spread:

![Intensity histograms](results/figures/03_intensity_histograms.png)

Raw file size distribution and segmented tooth-area ratio by class:

![Image properties](results/figures/04_image_properties.png)

## Feature engineering

`src/features/feature_extraction.py` computes, inside the segmented tooth
region only:

- First-order intensity statistics: mean, std, skewness, kurtosis, 10th/90th percentile
- GLCM (Gray-Level Co-occurrence Matrix) texture: contrast, dissimilarity, homogeneity, energy, correlation, ASM, averaged over 2 distances x 4 angles
- Local Binary Pattern (LBP) histogram (uniform, radius 2, 16 points)
- Canny edge density
- Segmented tooth-area ratio

**Ablation finding**: an earlier version also included HOG (Histogram of
Oriented Gradients) descriptors, which added roughly 1,568 dimensions on
top of the 32 features above. With only 120 samples, that high-dimensional
feature space overwhelmed the model and cross-validated macro-F1 of the
best classifier dropped from **0.726 to 0.398**. HOG was therefore
dropped; the final feature set used everywhere below has 32 dimensions.

| Feature set | CV accuracy | CV macro-F1 |
|---|---|---|
| Full (GLCM+LBP+Intensity+Edge+HOG, 1600 dims) | 0.417 | 0.398 |
| No HOG (GLCM+LBP+Intensity+Edge, 32 dims) | 0.742 | 0.726 |

Every classical model is trained inside a
`Pipeline(StandardScaler -> PCA(95% variance) -> Classifier)` so all
algorithms see the same standardized, decorrelated feature space.

## Algorithms compared

**Classical ML** (`src/models/classical_models.py`), each evaluated with
stratified 5-fold cross-validation (more reliable than a single split at
n=120) and then refit on an 80/20 stratified train/test split for final
held-out metrics:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine (RBF kernel)
- Random Forest
- Gradient Boosting
- XGBoost

**Deep learning** (`src/models/cnn_model.py`): a compact CNN trained from
scratch in PyTorch (4 conv blocks, batch norm, global average pooling,
dropout-regularized head), with heavy augmentation (random flip, rotation,
color jitter, translation) given the very small dataset. Class-weighted
cross-entropy loss compensates for the mild class imbalance.

A note on transfer learning: a standard approach for small medical
imaging datasets is to fine-tune a network pretrained on ImageNet (e.g.
ResNet18). This sandbox's network access is restricted to package
registries, and torchvision's pretrained weights are hosted elsewhere and
were not reachable, so that path was not available here. Training from
scratch with strong augmentation was used as the runnable alternative;
see [Limitations](#limitations).

## Results

### Cross-validated comparison (classical ML, 5-fold, more robust at this sample size)

| Model | CV Accuracy | CV Macro-F1 |
|---|---|---|
| Logistic Regression | 0.742 +/- 0.049 | 0.726 +/- 0.043 |
| Gradient Boosting | 0.725 +/- 0.104 | 0.698 +/- 0.117 |
| XGBoost | 0.708 +/- 0.118 | 0.692 +/- 0.133 |
| Random Forest | 0.700 +/- 0.107 | 0.670 +/- 0.118 |
| SVM (RBF) | 0.675 +/- 0.113 | 0.653 +/- 0.111 |
| KNN | 0.633 +/- 0.072 | 0.617 +/- 0.082 |

![Classical model comparison](results/figures/06_classical_model_comparison.png)

![Classical confusion matrices](results/figures/07_classical_confusion_matrices.png)

![Feature importance](results/figures/08_feature_importance.png)

### CNN training behavior

![CNN training curves](results/figures/09_cnn_training_curves.png)

![CNN confusion matrix](results/figures/10_cnn_confusion_matrix.png)

### Final comparison — all 7 algorithms on the same held-out test split

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC AUC (OvR macro) |
|---|---|---|---|---|---|
| CNN (from scratch) | 0.722 | 0.795 | 0.726 | 0.727 | 0.927 |
| Random Forest | 0.667 | 0.756 | 0.630 | 0.624 | 0.843 |
| Logistic Regression | 0.583 | 0.572 | 0.574 | 0.552 | 0.877 |
| KNN | 0.542 | 0.557 | 0.537 | 0.539 | 0.773 |
| SVM (RBF) | 0.542 | 0.606 | 0.537 | 0.530 | 0.782 |
| XGBoost | 0.500 | 0.500 | 0.481 | 0.484 | 0.790 |
| Gradient Boosting | 0.500 | 0.497 | 0.481 | 0.481 | 0.770 |

![Final model comparison](results/figures/11_final_model_comparison.png)

![Best model per-class metrics](results/figures/12_best_model_per_class_metrics.png)

### Reading the two tables together

Logistic Regression has the best *cross-validated* macro-F1 among the
classical models (0.726, averaged over 5 folds), which is the more
reliable estimate at this sample size. On the *single* held-out test
split used for the final 7-way comparison, the CNN comes out ahead
instead (0.727 macro-F1, 0.927 ROC AUC). With only 120 images and a
12-24 image test split, single-split rankings carry real sampling
variance — both results are reported rather than picking whichever looks
better, and the recommended takeaway is: **Logistic Regression and the
CNN are roughly tied as the strongest models here; both clearly
outperform KNN, SVM, and the boosting methods on this dataset.**

## Limitations

- **No ground truth.** Every metric in this README measures agreement
  with a heuristic, unvalidated labeling rule, not real clinical
  diagnoses. High accuracy here means "the model learned to reproduce the
  image-processing rule," not "the model detects real caries."
- **Tiny dataset.** 120 images split three ways leaves single-digit test
  counts per class; all metrics have wide uncertainty (see the CV
  standard deviations above).
- **No pretrained-weight transfer learning.** The CNN was trained from
  scratch rather than fine-tuned from ImageNet weights, due to sandboxed
  network access; a pretrained backbone would likely improve and
  stabilize CNN performance considerably.
- **Single-label simplification.** Real X-rays can show both a filling
  and caries in the same image; the auto-labeler assigns one dominant
  label per image rather than multi-label tags.
- **Not for clinical use.** This repository is a methodology and
  engineering demonstration only.

## What would make this production-grade

- Real radiologist-annotated labels (ideally with inter-rater agreement)
- A larger dataset, ideally thousands of images across imaging sources
- Transfer learning from a medical-imaging-pretrained backbone
- Multi-label / object-detection formulation (localizing the lesion or
  restoration, not just classifying the whole image)
- External validation on a held-out clinic's data to check generalization

## License

MIT License. See `LICENSE`.
