# Lab 1 Notebook Structure

The original notebooks remain in the project root for traceability. The clean notebooks in this folder should be used as the final report structure:

1. `01_eda_and_feature_baseline.ipynb`
   - dataset loading;
   - exploratory data analysis;
   - preprocessing rationale;
   - ResNet-18 feature extractor plus SVM baseline.

2. `02_finetuning_pipeline.ipynb`
   - reproducible train/validation split;
   - model factory;
   - training configuration from YAML;
   - baseline fine-tuning results.

3. `03_improvements_and_retrieval.ipynb`
   - targeted improvements;
   - layer4 unfreezing;
   - conservative augmentation.

4. `03b_retrieval_training_free_classification.ipynb`
   - retrieval as training-free classification;
   - gallery/query feature extraction;
   - similarity metrics and Precision@K;
   - Nearest-Mean Classifier and mAP summary.

Keep code cells short and delegate reusable logic to `src/dla_lab1`.
