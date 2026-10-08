# Ensemble Learning for Windows PE Malware Detection (EMBER)

This project asks one question: does an ensemble of LightGBM, Random Forest, XGBoost and a neural network detect Windows PE malware better than any single classifier? It is tested on the EMBER 2018 benchmark, with simple baselines for comparison, a temporal drift experiment, and SHAP interpretation.

## Headline results

Evaluated on the EMBER 2018 test set (200,000 samples, 50/50 malware and benign).

| Model | AUC | TPR @ 1% FPR |
|---|---|---|
| Logistic regression (baseline) | 0.9185 | 0.0000 |
| Decision tree (baseline) | 0.9279 | 0.1192 |
| LightGBM (best single model) | 0.9864 | 0.8422 |
| Ensemble, fitted weights | 0.9911 | 0.8916 |
| Ensemble, rank average | 0.9905 | 0.8989 |

- By AUC the ensemble gains about 0.005 over the best single model. At a 1% false-positive budget it gains about 5.7 points of detection.
- Logistic regression has a respectable AUC but catches nothing at a 1% false-positive budget, so AUC alone is a poor yardstick for security work.
- XGBoost received a fitted weight of zero (0.993 correlation with LightGBM). The neural network went from 10% in the proposal to about 53%, because diversity matters more than individual strength.
- Trained on 2017 and tested on 2018, AUC falls by about 3 points but detection at 1% FPR falls by 26 to 48 points, and the ensemble falls behind LightGBM alone.
- SHAP shows the model leans on build-pipeline markers (code-signing certificate, resource table, debug directory). These are forgeable, which is an evasion risk and a plausible cause of the drift.

## Dataset

EMBER 2018 and EMBER 2017 (v2), using the full published 2381-dimensional feature vector.

- 2018 after filtering unlabelled rows: 600,000 train (420,000 train / 180,000 validation, stratified, seed 42) and 200,000 test.
- The 2017 set is used only for the drift experiment. 50,000 hashes overlap with 2018, none reach the 2018 test set (checked by `check_overlap.py`).
- Raw JSONL files are not included. Download them from https://github.com/elastic/ember and place them under `data/raw/ember_2018/` and `data/raw/ember_2017_v2/`.

## Project structure

```
src/
  extract_and_save.py    JSONL to 2381-dim float32 arrays (chunked, resumable)
  data_processing.py     stratified split (as indices) and chunked StandardScaler
  model_training.py      LightGBM, XGBoost, Random Forest, neural network
  baselines.py           logistic regression, decision tree, chance
  ensemble.py            weight grid search and rank averaging
  evaluation.py          master table, operating points, figures
  check_overlap.py       2017/2018 sha256 leakage check
  recover_metadata.py    recovers appeared dates and hashes for the test set
  drift_experiment.py    train on 2017, evaluate on 2018
  interpretation.py      SHAP analysis on LightGBM
  data_loader.py         superseded pandas loader, kept as a failed approach
notebooks/
  results_summary.ipynb  assembles all results into tables and figures
data/    raw/, processed/, processed_2017/   (not tracked)
models/  trained models, scaler, saved probabilities
results/ metrics JSON files and figures
```

## Setup

Tested on Windows with Python 3.11, an AMD Ryzen 7 5800H and 15 GB of usable RAM, no GPU.

```
numpy 1.24.3   pandas 2.0.2   scikit-learn 1.2.2   lightgbm 3.3.5
xgboost 1.7.5  tensorflow 2.12.0   shap 0.42.1   scipy   matplotlib
```

## Running the pipeline

Run from the project root, in this order. Use PowerShell rather than an IDE terminal, because an IDE holding 2 to 3 GB alongside the arrays caused an out-of-memory crash during development.

```
python src/extract_and_save.py
python src/data_processing.py
python src/model_training.py all
python src/baselines.py all
python src/ensemble.py
python src/evaluation.py
python src/check_overlap.py
python src/recover_metadata.py
python src/drift_experiment.py
python src/interpretation.py
```

`extract_and_save.py` reads its input folder and output folder from `main()`. Run it once pointed at `ember_2018` (output `data/processed`) and once at `ember_2017_v2` (output `data/processed_2017`). The notebook is then run top to bottom to display everything.

## Design notes

- Memory: a first pandas version (`data_loader.py`) ran out of memory because nested JSON sits in memory as Python objects, roughly ten times larger than float32 arrays, and `pd.concat` held two copies. The final extractor streams line by line, converts 25,000-record chunks to float32, saves one part per file, and stitches parts with a memory-mapped array.
- No scaled copy of the data is stored. Tree models do not need scaling, and the neural network applies the saved scaler per batch. The scaler is fitted on training rows only.
- Ensemble weights and thresholds are fitted on validation and reported on test. Drift weights are carried over from 2018, not refitted.
- Rank averaging is included because the neural network's probabilities are poorly calibrated (validation loss rose while AUC improved).

## Limitations

- Random Forest was trained on a 150,000-row stratified subsample.
- No hyperparameter tuning and no cross-validation (single stratified split, proposal values used as given).
- Single random seed, so no variance estimates.
- EMBER has binary labels only, so the per-family analysis in the proposal was replaced by the drift experiment.
- The test set spans two months (2018-11 and 2018-12).
- SHAP is computed on LightGBM only, and hashed-bucket reverse lookup is suggestive, not exact.

## Ethics

EMBER ships extracted features and one-way hashes, not binaries, so no executable content or personal data is handled. Publishing a malware classifier and its feature importances can help attackers, which is a real tension. Deployment also implies regular retraining, and a model that relies on absent legitimacy markers may unfairly flag unsigned legitimate software.

## Reference

Anderson, H. S., and Roth, P. (2018). EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models. arXiv:1804.04637.