# DLA 1 — Transfer Learning and Retrieval on GTSRB

## Overview

This laboratory studies traffic-sign recognition through two complementary uses of pretrained convolutional networks. The first is supervised classification: fixed ImageNet features with a linear SVM are compared with progressively less constrained ResNet-18 fine-tuning. The second is training-free classification: pretrained embeddings are used for nearest-neighbour retrieval and a Nearest-Mean Classifier (NMC).

The report is based on preserved notebook outputs and lightweight evidence under [`results/`](results/). It does not rerun training or replace unsuccessful experiments with hypothetical results.

**Final index:** [`DLA_1.ipynb`](DLA_1.ipynb)

**Official assignment:** [`ASSIGNMENT.md`](ASSIGNMENT.md)

**Notebook guide:** [`notebooks/README.md`](notebooks/README.md)

## Objectives and assignment coverage

| Requirement | Implementation | Evidence | Status |
| --- | --- | --- | ---: |
| Exercise 1.1: GTSRB EDA | Dataset inspection, geometry, class distribution, examples | [`01_eda_and_feature_baseline.ipynb`](notebooks/01_eda_and_feature_baseline.ipynb) | Completed |
| Exercise 1.2: stable baseline | Pretrained ResNet-18 feature extraction + linear SVM | Same notebook, test accuracy `0.6412` | Completed |
| Exercise 1.3: fine-tuning baseline | Frozen backbone, new 43-class head, stratified validation | [`02_finetuning_pipeline.ipynb`](notebooks/02_finetuning_pipeline.ipynb) | Completed |
| Exercise 2: pipeline consolidation | YAML configuration, reusable modules, artifacts, optional W&B | [`02b_pipeline_consolidation.ipynb`](notebooks/02b_pipeline_consolidation.ipynb) | Completed |
| Exercise 3.1: improved fine-tuning | Layer 4 unfreezing, augmentation and regularization study | [`03_improvements_and_retrieval.ipynb`](notebooks/03_improvements_and_retrieval.ipynb) | Completed |
| Exercise 3.2: retrieval and NMC | Similarity comparison, Precision@K, mAP and class centroids | [`03b_retrieval_training_free_classification.ipynb`](notebooks/03b_retrieval_training_free_classification.ipynb) | Completed |

## Theoretical background

A pretrained CNN can be used either as a fixed map \(f(x)\) or adapted to the target task. The stable baseline trains a linear SVM on the penultimate ResNet representation. This isolates the quality of the pretrained representation from neural-network optimization. Fine-tuning instead minimizes cross-entropy on GTSRB and can adapt higher-level features, but it is more sensitive to the learning rate, trainable layers, class imbalance, augmentation, and checkpoint selection.

For retrieval, query and gallery features are compared using cosine similarity, Euclidean distance, or a dot product. Precision@K measures the fraction of the first K retrieved images that share the query label. Average Precision summarizes the full ranking. NMC replaces individual gallery examples with one centroid per class and predicts the nearest centroid.

## Dataset and exploratory analysis

The executed Torchvision dataset contained `26,640` training images and `12,630` test images across `43` classes. These are the sizes observed by the submitted code and are therefore the values used in this report. The training-class counts ranged from `150` to `1,500`, a 10:1 ratio. Width ranged from `25` to `243` pixels and height from `25` to `225`; both means were approximately `50` pixels. Full values and provenance are in [`dataset_summary.json`](results/dataset_summary.json).

### Raw examples

![GTSRB examples](figures/gtsrb_examples.png)

The samples confirm substantial variability in scale, illumination, framing, and sign appearance. This supports resizing all samples to a fixed tensor shape and using augmentation conservatively: transformations that change sign geometry or colour semantics can damage the label.

### Image geometry

![GTSRB image geometry](figures/gtsrb_image_geometry.png)

The native image dimensions and aspect ratios are heterogeneous. The pipeline resizes images to `64 × 64`, which makes batching possible but introduces a geometric approximation. This is a practical compromise for the available GPU rather than a claim that native resolution is irrelevant.

### Class imbalance

![GTSRB class distribution](figures/gtsrb_class_distribution.png)

The imbalance means aggregate accuracy can hide weak minority-class recall. Stratification preserves every class in train and validation; the baseline split contained `20,820` train and `5,820` validation samples, with no missing class. Weighted cross-entropy and focal loss were included in the experiment registry, although the best final result came from selective fine-tuning with label smoothing rather than from reporting a weighted-loss run as the winner.

## Preprocessing and stable baseline

Images are resized to `64 × 64`, converted to tensors, and normalized using the pretrained backbone convention. A ResNet-18 with default ImageNet weights is converted to a feature extractor by removing its final classifier. The resulting fixed embeddings feed a linear `sklearn.svm.SVC` with `C=1.0`.

| Parameter | Value | Verified in |
| --- | ---: | --- |
| Backbone | ResNet-18, default pretrained weights | `config/config.yaml` |
| Input size | 64 × 64 | `config/config.yaml` |
| Feature-extraction batch size | 128 | `config/config.yaml` |
| Classifier | Linear SVM | `config/config.yaml` |
| SVM C | 1.0 | `config/config.yaml` |
| Training samples | 26,640 | baseline notebook output |
| Test samples | 12,630 | baseline notebook output |
| Fit time | 114.97 s | baseline notebook output |
| Test accuracy | **0.6412** | baseline notebook output |

This baseline is stable because only the convex classifier is optimized after deterministic feature extraction. Its limitation is equally clear: ImageNet features are not adapted to small traffic signs, and the linear decision surface cannot recover information absent from those embeddings.

## Fine-tuning baseline

The first neural baseline freezes the ResNet-18 backbone and trains only a new 43-class fully connected head. The split is deterministic and stratified. The notebook reports only `22,059` trainable parameters out of `11,198,571` total.

| Parameter | Value |
| --- | ---: |
| Seed | 42 |
| Train / validation | 20,820 / 5,820 |
| Trainable layers | Final classifier only |
| Loss | Cross-entropy |
| Optimizer | Adam |
| Initial learning rate | 1e-3 |
| Batch size | 128 |
| Epochs | 10 |
| Scheduler | Step decay |
| Best validation accuracy | 0.4667 at epoch 6 |
| Test accuracy | **0.5038** |

The training accuracy reached `0.8274`, while validation remained near `0.4655`. The classifier head therefore fitted the training representation without generalizing sufficiently. This explains why the fine-tuning baseline underperformed the SVM: despite its name, almost no convolutional feature was actually fine-tuned, and the nonlinear neural head did not compensate for the frozen domain mismatch.

## Pipeline consolidation and W&B

The second exercise separates responsibilities into configuration, data, models, losses, training, evaluation, tracking, and visualization modules. Each local run can save the effective YAML configuration, epoch history, and summary. The final lightweight exports are versioned under [`results/`](results/); model checkpoints and full run folders remain ignored.

Weights & Biases (W&B) was used during development to compare training/validation accuracy, loss, learning rate, and run metadata. Integration is implemented in [`tracking.py`](src/dla_lab1/tracking.py). It is optional, disabled by default, and never performs an automatic login. Local W&B directories are not versioned because they can be large and environment-specific; the report uses exported CSV/JSON evidence instead.

## Improved fine-tuning

The improvement study unfreezes ResNet-18 `layer4` and the classifier, then compares no augmentation, spatial augmentation, aggressive augmentation, conservative augmentation, label smoothing, and discriminative learning rates. The selected documented configuration used conservative augmentation with label smoothing.

| Parameter | Improved configuration |
| --- | ---: |
| Trainable layers | `layer4` + `fc` |
| Trainable share observed in notebook | 75.15% |
| Loss | Cross-entropy, label smoothing 0.05 |
| Optimizer | AdamW |
| Initial learning rate | 5e-4 |
| Weight decay | 0.02 |
| Batch size | 128 |
| Maximum epochs | 20 |
| Scheduler | Step decay, factor 0.5 every 5 epochs |
| Gradient clipping | 1.0 |
| AMP | Enabled when CUDA is available |
| Final notebook test accuracy | **0.8025** |

### Validation comparison

![Validation comparison](figures/gtsrb_validation_accuracy_summary.svg)

Unfreezing `layer4` consistently improved validation accuracy over the head-only variants. Aggressive augmentation did not dominate conservative augmentation, which is plausible for traffic signs because strong geometric or colour changes can alter class-defining details. The plotted values come from [`run_validation_summary.csv`](results/run_validation_summary.csv).

### Training curves

![Fine-tuning curves](figures/gtsrb_training_curves.svg)

The improved run rapidly approaches perfect training accuracy while validation saturates below `0.77`, a visible generalization gap. Label smoothing prevents the training loss from being directly comparable with the unsmoothed baseline loss, so the report compares validation accuracy rather than claiming a lower loss.

There is an important provenance limitation: the preserved `0.8025` test output and the exported named validation histories were created at different stages of the experiment notebook. The test metric remains verifiable in the notebook, but this repository does not claim that every exported validation row was evaluated on the test set. No result has been recomputed to remove this discrepancy.

### Final classification comparison

![GTSRB test accuracy comparison](figures/gtsrb_test_accuracy_comparison.svg)

The improved model gains `0.1613` absolute accuracy over the SVM and `0.2987` over the head-only baseline. The negative head-only result remains in the report because it motivates selective unfreezing. Source: [`test_metrics.csv`](results/test_metrics.csv).

No confusion-matrix array or per-example prediction file was preserved in the final run. The notebooks include class-wise classification reports, but this report does not manufacture a confusion matrix from aggregate precision/recall values.

## Retrieval and Nearest-Mean Classification

The training split acts as the gallery (`26,640` images) and the test split as the query set (`12,630` images). ResNet-50 features were also inspected for the retrieval/NMC exercise. Cosine similarity performed slightly better than Euclidean distance and clearly better than an unnormalized dot product.

| Method | P@1 | P@5 | P@10 |
| --- | ---: | ---: | ---: |
| Cosine similarity | **0.4812** | **0.4594** | **0.4394** |
| Euclidean distance | 0.4763 | 0.4505 | 0.4308 |
| Dot product | 0.3611 | 0.3465 | 0.3375 |

Additional observed results were mAP `0.1696`, macro class mAP `0.1301`, and NMC test accuracy `0.4185`.

![Retrieval metrics](figures/gtsrb_retrieval_metrics.svg)

Precision decreases as K grows because more visually similar but class-inconsistent neighbours enter the ranking. NMC is cheaper at inference because it stores one centroid per class, but it loses multimodal within-class structure. Both retrieval approaches underperform supervised fine-tuning; their value is training-free reuse and interpretability, not maximum classification accuracy.

## What worked, what did not, and why

**Worked:** fixed pretrained features established a reproducible baseline; stratified splitting avoided missing classes; unfreezing the final residual block enabled target adaptation; conservative augmentation and label smoothing improved validation behaviour; cosine similarity was the strongest retrieval metric.

**Did not work as well:** the head-only neural baseline generalized worse than the SVM; aggressive augmentation reduced validation performance; no-augmentation runs reached perfect training accuracy but a substantially lower validation plateau; NMC compressed the gallery too aggressively.

**Problems resolved:** reusable local modules replaced repeated notebook code; paths are relative to the laboratory root; heavy cells are controlled by flags; W&B login is manual; histories and metrics are now independently inspectable on GitHub.

## Limitations

- Only one main seed is reported; variability across independent training seeds was not measured.
- The native 26,640-image training split observed by Torchvision is the submitted experimental basis; no attempt is made to merge other GTSRB distributions.
- Hardware constraints favoured `64 × 64` inputs and limited the experiment budget.
- The final confusion matrix and per-sample predictions were not saved.
- W&B raw logs and checkpoints are local, so the versioned summaries and notebook outputs are the public evidence.

## Reproducibility

Run notebooks in order. Quick inspection requires no training. To retrain, install the root requirements, download GTSRB through Torchvision, set `RUN_TRAINING = True` only in the desired notebook, and optionally enable W&B after an explicit `wandb login`. Feature extraction and final test evaluation have separate flags to prevent accidental recomputation.

Regenerate report assets with:

```bash
python tools/build_report_assets.py
```

The command reads versioned evidence and notebook outputs; it does not train a model.

## File structure

| Path | Purpose |
| --- | --- |
| `DLA_1.ipynb` | GitHub-readable laboratory index |
| `notebooks/` | Final executed notebooks in exercise order |
| `exploratory/` | Historical trial notebook, retained but excluded from the final path |
| `src/dla_lab1/` | Reusable implementation |
| `scripts/` | Environment, training, extraction, and checkpoint evaluation entry points |
| `config/config.yaml` | Central experiment registry and defaults |
| `results/` | Lightweight public evidence |
| `figures/` | Figures extracted or generated from evidence |

### Exploratory-notebook decision

| File | Unique content | Referenced by final report | Required for submission | Decision |
| --- | ---: | ---: | ---: | --- |
| `exploratory/Esperimenti_di_prova.ipynb` | Yes: historical trials and W&B outputs | Only as provenance | No | Retained outside `notebooks/`; not a source for headline results |

## Main dependencies

| Library | Purpose | Used in |
| --- | --- | --- |
| PyTorch | Tensor computation, training, checkpoints | Fine-tuning and evaluation |
| Torchvision | GTSRB and pretrained ResNet models | All computer-vision experiments |
| scikit-learn | SVM, reports, metrics | Stable baseline and evaluation |
| pandas / NumPy | Tables, metadata, evidence processing | EDA and reports |
| Matplotlib | EDA, curves, visual retrieval | Notebooks |
| PyYAML | Experiment configuration | Pipeline |
| Weights & Biases | Optional run tracking | Pipeline consolidation |

## Local modules and main functions

| Function/class | Source file | Purpose | Inputs | Outputs | Used in |
| --- | --- | --- | --- | --- | --- |
| `build_dataloaders` | [`data.py`](src/dla_lab1/data.py) | Create stratified train/validation/test loaders | Config, transforms, root | Loaders and split metadata | Fine-tuning notebooks |
| `build_feature_extractor` | [`models.py`](src/dla_lab1/models.py) | Remove the classifier from a pretrained ResNet | Model name, weights | CNN feature extractor | SVM and retrieval |
| `run_finetuning` | [`experiments.py`](src/dla_lab1/experiments.py) | Execute one registered experiment | Config, experiment name, root | Model, history, metrics, artifacts | Exercises 1.3 and 3.1 |
| `retrieval_precision_at_k` | [`features.py`](src/dla_lab1/features.py) | Compute label agreement in top-K retrieval | Similarities, labels, K | Precision@K | Exercise 3.2 |
| `retrieval_mean_average_precision_chunked` | [`features.py`](src/dla_lab1/features.py) | Compute mAP without one full similarity allocation | Feature tensors, labels, chunk size | mAP metrics | Exercise 3.2 |
| `nearest_mean_classifier` | [`features.py`](src/dla_lab1/features.py) | Classify by nearest class centroid | Gallery/query features and labels | Predictions | Exercise 3.2 |
| `save_run_artifacts` | [`tracking.py`](src/dla_lab1/tracking.py) | Save config, history, summary, and optional tracking data | Run state and output path | Local evidence files | Pipeline consolidation |

## AI use

AI assistance supported requirement interpretation, debugging, module organization, documentation, and review of result provenance. It did not provide experimental observations. See [`../AI_USAGE.md`](../AI_USAGE.md).

## Sources

- [GTSRB benchmark website](https://benchmark.ini.rub.de/gtsrb_dataset.html)
- [Torchvision GTSRB API](https://pytorch.org/vision/main/generated/torchvision.datasets.GTSRB.html)
- [Torchvision ResNet API](https://pytorch.org/vision/stable/models/resnet.html)
- [Scikit-learn SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Weights & Biases documentation](https://docs.wandb.ai/)

## Conclusion

The laboratory demonstrates why a stable pretrained-feature baseline is a useful control and why selective adaptation matters. The strongest result, `0.8025` test accuracy, came only after unfreezing high-level features and controlling augmentation/regularization. The retrieval experiments provide a different, lower-performing but training-free use of the representation. Negative results and remaining provenance limits are kept visible rather than smoothed over.
