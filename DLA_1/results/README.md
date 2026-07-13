# Lab 1 result evidence

These lightweight files make the reported metrics inspectable without publishing datasets, W&B logs or model checkpoints.

| File | Content | Primary source |
| --- | --- | --- |
| `dataset_summary.json` | GTSRB cardinality, imbalance and image dimensions | Output of `01_eda_and_feature_baseline.ipynb` |
| `test_metrics.csv` | Final classification and retrieval metrics | Executed notebook outputs |
| `run_validation_summary.csv` | Validation comparison across the saved pipeline runs | Local `artifacts/runs/*/summary.json` files |
| `baseline_history.csv` | Head-only fine-tuning history | Local pipeline artifact and notebook output |
| `improved_history.csv` | Selected improved-run history | Local pipeline artifact and notebook output |

The test accuracy `0.8025` is preserved from the final test output of the improved notebook. The validation files describe the named pipeline runs and must not be interpreted as a new test evaluation.
