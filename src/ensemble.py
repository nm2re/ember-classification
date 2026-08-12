import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)

ROOT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

MODEL_NAMES = ["lgb", "rf", "xgb", "nn"]
PROPOSAL_WEIGHTS = {"lgb": 0.40, "rf": 0.30, "xgb": 0.20, "nn": 0.10}
PROPOSAL_THRESHOLD = 0.60


def compute_metrics(y_true, proba, threshold=0.5):
    """
    Computes AUC, F1, precision, TPR, and FPR at a given threshold.
    """
    predicted_labels = (proba >= threshold).astype(int)
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(
        y_true, predicted_labels
    ).ravel()

    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "f1": float(f1_score(y_true, predicted_labels)),
        "precision": float(precision_score(y_true, predicted_labels)),
        "tpr": float(recall_score(y_true, predicted_labels)),
        "fpr": float(false_pos / (false_pos + true_neg)),
        "threshold": threshold,
        "tp": int(true_pos), "fp": int(false_pos),
        "tn": int(true_neg), "fn": int(false_neg),
    }


def print_metrics(label, m):
    """
    Prints one row of AUC/F1/TPR/FPR/precision for a model.
    """
    print(
        f"  {label:28s} AUC {m['auc']:.4f}  F1 {m['f1']:.4f}  "
        f"TPR {m['tpr']:.4f}  FPR {m['fpr']:.4f}  P {m['precision']:.4f}"
    )


def weighted_average_proba(probas_by_model, weights):
    """
    Combines model probabilities using a weighted average.
    """
    total_weight = sum(weights.values())
    return sum(
        probas_by_model[name] * (weight / total_weight)
        for name, weight in weights.items()
    )


def weighted_average_rank(probas_by_model, weights):
    """
    Combines models by averaging normalised ranks instead of raw probabilities.
    """
    total_weight = sum(weights.values())
    n_samples = len(next(iter(probas_by_model.values())))

    combined_rank = np.zeros(n_samples)
    for name, weight in weights.items():
        normalised_rank = rankdata(probas_by_model[name]) / n_samples
        combined_rank += normalised_rank * (weight / total_weight)

    return combined_rank


def find_best_threshold(y_true, proba):
    """
    Searches thresholds 0.05-0.95 for the one that maximises F1.
    """
    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in np.arange(0.05, 0.96, 0.01):
        predicted_labels = (proba >= threshold).astype(int)
        f1 = f1_score(y_true, predicted_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold


def fit_ensemble_weights(y_val, val_probas):
    """
    Grid searches weight combinations (steps of 0.1) scored by validation AUC.
    """
    weight_options = np.arange(0, 11)

    best_auc = 0.0
    best_weights = None

    for combo in itertools.product(weight_options, repeat=4):
        if sum(combo) == 0:
            continue

        candidate_weights = dict(zip(MODEL_NAMES, combo))
        combined_proba = weighted_average_proba(val_probas, candidate_weights)
        auc = roc_auc_score(y_val, combined_proba)

        if auc > best_auc:
            best_auc = auc
            best_weights = candidate_weights

    total = sum(best_weights.values())
    normalised_weights = {name: w / total for name, w in best_weights.items()}

    return normalised_weights, best_auc


def main():
    """
    Builds and evaluates the ensemble: compares individual models, proposal
    weights, fitted weights, and rank averaging on the test set.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    missing_models = [
        name for name in MODEL_NAMES
        if not (MODELS_DIR / f"{name}_val_proba.npy").exists()
    ]
    if missing_models:
        print(f"ERROR: missing probability arrays for {missing_models}. "
              f"Run model_training.py for those first.")
        return

    y_train_full = np.load(PROCESSED_DIR / "y_train.npy")
    val_idx = np.load(PROCESSED_DIR / "val_idx.npy")
    y_val = y_train_full[val_idx]
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    val_probas = {
        name: np.load(MODELS_DIR / f"{name}_val_proba.npy") for name in MODEL_NAMES
    }
    test_probas = {
        name: np.load(MODELS_DIR / f"{name}_test_proba.npy") for name in MODEL_NAMES
    }

    report = {}

    print("\nIndividual models (test set, threshold 0.5)")
    for name in MODEL_NAMES:
        m = compute_metrics(y_test, test_probas[name], threshold=0.5)
        report[name] = m
        print_metrics(name, m)

    best_single_model = max(MODEL_NAMES, key=lambda name: report[name]["auc"])
    print(f"\n  strongest single model: {best_single_model} "
          f"(AUC {report[best_single_model]['auc']:.4f})")

    # correlation shows how much diversity there is to actually exploit
    print("\nPairwise correlation of test predictions")
    for model_a, model_b in itertools.combinations(MODEL_NAMES, 2):
        correlation = float(np.corrcoef(test_probas[model_a], test_probas[model_b])[0, 1])
        print(f"  {model_a}-{model_b}: {correlation:.4f}")

    print("\nEnsemble: proposal weights (40 LGB / 30 RF / 20 XGB / 10 NN)")
    proposal_test_proba = weighted_average_proba(test_probas, PROPOSAL_WEIGHTS)

    m_proposal_threshold = compute_metrics(y_test, proposal_test_proba, PROPOSAL_THRESHOLD)
    report["ensemble_proposal"] = m_proposal_threshold
    print_metrics(f"threshold {PROPOSAL_THRESHOLD}", m_proposal_threshold)

    m_proposal_05 = compute_metrics(y_test, proposal_test_proba, threshold=0.5)
    report["ensemble_proposal_t05"] = m_proposal_05
    print_metrics("threshold 0.5", m_proposal_05)

    print("\nFitting weights on validation")
    fitted_weights, val_auc = fit_ensemble_weights(y_val, val_probas)
    weights_str = "  ".join(f"{name} {w:.2f}" for name, w in fitted_weights.items())
    print(f"  {weights_str}")
    print(f"  validation AUC {val_auc:.4f}")

    fitted_val_proba = weighted_average_proba(val_probas, fitted_weights)
    fitted_threshold = find_best_threshold(y_val, fitted_val_proba)
    print(f"  threshold chosen on validation: {fitted_threshold:.2f}")

    fitted_test_proba = weighted_average_proba(test_probas, fitted_weights)
    m_fitted = compute_metrics(y_test, fitted_test_proba, fitted_threshold)
    report["ensemble_fitted"] = m_fitted
    report["fitted_weights"] = fitted_weights
    print_metrics("fitted weights", m_fitted)

    # rank averaging uses the same weights, just applied to ranks instead of raw probabilities
    print("\nEnsemble: rank averaging (calibration-independent)")
    rank_val_proba = weighted_average_rank(val_probas, fitted_weights)
    rank_threshold = find_best_threshold(y_val, rank_val_proba)

    rank_test_proba = weighted_average_rank(test_probas, fitted_weights)
    m_rank = compute_metrics(y_test, rank_test_proba, rank_threshold)
    report["ensemble_rank"] = m_rank
    print_metrics(f"threshold {rank_threshold:.2f}", m_rank)

    ensemble_aucs = {
        key: report[key]["auc"]
        for key in ["ensemble_proposal", "ensemble_fitted", "ensemble_rank"]
    }
    best_ensemble_name = max(ensemble_aucs, key=ensemble_aucs.get)
    auc_gain = ensemble_aucs[best_ensemble_name] - report[best_single_model]["auc"]

    print("\n" + "-" * 62)
    print(f"Best single model : {best_single_model} AUC {report[best_single_model]['auc']:.4f}")
    print(f"Best ensemble     : {best_ensemble_name} AUC {ensemble_aucs[best_ensemble_name]:.4f}")
    print(f"Difference        : {auc_gain:+.4f} AUC")

    if auc_gain <= 0:
        print("\nThe ensemble does not beat its strongest member. This is a")
        print("legitimate finding, not a failure - see the correlation figures above.")
    print("-" * 62)

    np.save(RESULTS_DIR / "ensemble_test_proba.npy", fitted_test_proba.astype(np.float32))
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved results/metrics.json and results/ensemble_test_proba.npy")


if __name__ == "__main__":
    main()