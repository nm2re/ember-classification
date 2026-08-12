import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix
)

ROOT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

BASELINE_NAMES = ["chance", "logreg", "dtree"]
ENSEMBLE_MEMBER_NAMES = ["lgb", "rf", "xgb", "nn"]
PROPOSAL_WEIGHTS = {"lgb": 0.40, "rf": 0.30, "xgb": 0.20, "nn": 0.10}

DISPLAY_NAMES = {
    "chance": "Chance (constant)",
    "logreg": "Logistic regression",
    "dtree": "Decision tree (d=10)",
    "lgb": "LightGBM",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "nn": "Neural Network",
    "ens_proposal": "Ensemble (proposal weights)",
    "ens_fitted": "Ensemble (fitted weights)",
    "ens_rank": "Ensemble (rank average)",
}


def compute_metrics(y_true, proba, threshold):
    """
    Computes AUC, F1, precision, TPR, and FPR at a given threshold.
    """
    predicted_labels = (proba >= threshold).astype(int)
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(
        y_true, predicted_labels
    ).ravel()

    auc = float(roc_auc_score(y_true, proba)) if len(set(proba)) > 1 else 0.5
    fpr = float(false_pos / (false_pos + true_neg)) if (false_pos + true_neg) else 0.0

    return {
        "auc": auc,
        "f1": float(f1_score(y_true, predicted_labels, zero_division=0)),
        "precision": float(precision_score(y_true, predicted_labels, zero_division=0)),
        "tpr": float(recall_score(y_true, predicted_labels, zero_division=0)),
        "fpr": fpr,
        "threshold": float(threshold),
        "tp": int(true_pos), "fp": int(false_pos),
        "tn": int(true_neg), "fn": int(false_neg),
    }


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


def tpr_at_fixed_fpr(y_true, proba, target_fpr):
    """
    Finds the detection rate achievable at a fixed false-positive budget.
    """
    fpr_values, tpr_values, thresholds = roc_curve(y_true, proba)
    idx = np.searchsorted(fpr_values, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr_values[idx]), float(thresholds[idx])


def load_test_predictions():
    """
    Loads every available model's saved test-set predictions.
    """
    probas = {}
    for name in BASELINE_NAMES + ENSEMBLE_MEMBER_NAMES:
        proba_path = MODELS_DIR / f"{name}_test_proba.npy"
        if proba_path.exists():
            probas[name] = np.load(proba_path)
        else:
            print(f"  missing {name}_test_proba.npy, skipping")
    return probas


def load_ensemble_weights_and_threshold():
    """
    Loads fitted ensemble weights and threshold, falling back to proposal weights.
    """
    weights = PROPOSAL_WEIGHTS
    threshold = 0.5

    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        previous_results = json.load(open(metrics_path))
        weights = previous_results.get("fitted_weights", PROPOSAL_WEIGHTS)
        threshold = previous_results.get("ensemble_fitted", {}).get("threshold", 0.5)

    return weights, threshold


def build_all_predictions():
    """
    Loads baseline/model predictions and builds the three ensemble variants.
    """
    probas = load_test_predictions()
    member_probas = {
        name: probas[name] for name in ENSEMBLE_MEMBER_NAMES if name in probas
    }

    fitted_weights, fitted_threshold = load_ensemble_weights_and_threshold()

    probas["ens_proposal"] = weighted_average_proba(member_probas, PROPOSAL_WEIGHTS)
    probas["ens_fitted"] = weighted_average_proba(member_probas, fitted_weights)
    probas["ens_rank"] = weighted_average_rank(member_probas, fitted_weights)

    thresholds = {name: 0.5 for name in probas}
    thresholds["ens_proposal"] = 0.60
    thresholds["ens_fitted"] = fitted_threshold

    return probas, thresholds


def build_master_table(y_true, probas, thresholds, model_order):
    """
    Computes and prints the main comparison table.
    """
    report = {}

    print(f"\n{'Model':<30}{'AUC':>8}{'F1':>8}{'TPR':>8}{'FPR':>8}{'Prec':>8}")
    print("-" * 70)

    for name in model_order:
        m = compute_metrics(y_true, probas[name], thresholds[name])
        report[name] = m
        print(
            f"{DISPLAY_NAMES[name]:<30}{m['auc']:>8.4f}{m['f1']:>8.4f}"
            f"{m['tpr']:>8.4f}{m['fpr']:>8.4f}{m['precision']:>8.4f}"
        )

    return report


def add_operating_point_metrics(y_true, probas, report, model_order):
    """
    Adds TPR at 1%/0.1% FPR budgets to the report.
    """
    print(f"\n{'Model':<30}{'TPR@1%FPR':>12}{'TPR@0.1%FPR':>14}")
    print("-" * 56)

    for name in model_order:
        if name == "chance":
            continue

        tpr_1pct, _ = tpr_at_fixed_fpr(y_true, probas[name], 0.01)
        tpr_0_1pct, _ = tpr_at_fixed_fpr(y_true, probas[name], 0.001)

        report[name]["tpr_at_1pct_fpr"] = tpr_1pct
        report[name]["tpr_at_0.1pct_fpr"] = tpr_0_1pct

        print(f"{DISPLAY_NAMES[name]:<30}{tpr_1pct:>12.4f}{tpr_0_1pct:>14.4f}")


def plot_roc_curves(y_true, probas, report, model_order):
    """
    Plots full ROC curves for every model on one axis.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for name in model_order:
        if name == "chance":
            continue
        fpr, tpr, _ = roc_curve(y_true, probas[name])

        is_ensemble = name.startswith("ens")
        line_style = "-" if is_ensemble else "--"
        line_width = 2.0 if is_ensemble else 1.2

        ax.plot(
            fpr, tpr, line_style, linewidth=line_width,
            label=f"{DISPLAY_NAMES[name]} ({report[name]['auc']:.4f})"
        )

    ax.plot([0, 1], [0, 1], ":", color="grey", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, EMBER 2018 test set (200,000 samples)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_roc_curves_zoomed(y_true, probas, model_order):
    """
    Plots the same curves zoomed into the low-FPR region.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for name in model_order:
        if name == "chance":
            continue
        fpr, tpr, _ = roc_curve(y_true, probas[name])

        is_ensemble = name.startswith("ens")
        line_style = "-" if is_ensemble else "--"
        line_width = 2.0 if is_ensemble else 1.2

        ax.plot(fpr, tpr, line_style, linewidth=line_width, label=DISPLAY_NAMES[name])

    ax.set_xlim(0, 0.05)
    ax.set_ylim(0.80, 1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC detail, low false positive region")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves_zoom.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(report, probas):
    """
    Plots confusion matrices for a subset of key models.
    """
    models_to_show = [
        name for name in ["logreg", "dtree", "lgb", "nn", "ens_fitted", "ens_rank"]
        if name in probas
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))

    for ax, name in zip(axes.ravel(), models_to_show):
        m = report[name]
        confusion = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])

        ax.imshow(confusion, cmap="Blues")
        for (row, col), value in np.ndenumerate(confusion):
            text_color = "white" if value > confusion.max() / 2 else "black"
            ax.text(col, row, f"{value:,}", ha="center", va="center", color=text_color)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred benign", "Pred malware"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Benign", "Malware"])
        ax.set_title(f"{DISPLAY_NAMES[name]}\nF1 {m['f1']:.4f}  FPR {m['fpr']:.4f}", fontsize=10)

    for ax in axes.ravel()[len(models_to_show):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_threshold_sweep(y_true, probas, thresholds, report, model_order):
    """
    Plots how TPR/FPR/F1/precision shift with threshold, for the best ensemble.
    """
    ensemble_names = [name for name in model_order if name.startswith("ens")]
    best_ensemble = max(ensemble_names, key=lambda name: report[name]["auc"])

    threshold_range = np.arange(0.05, 0.96, 0.01)
    sweep_results = [
        compute_metrics(y_true, probas[best_ensemble], t) for t in threshold_range
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(threshold_range, [r["tpr"] for r in sweep_results], label="TPR (malware caught)")
    ax.plot(threshold_range, [r["fpr"] for r in sweep_results], label="FPR (benign flagged)")
    ax.plot(threshold_range, [r["f1"] for r in sweep_results], label="F1")
    ax.plot(threshold_range, [r["precision"] for r in sweep_results], label="Precision")
    ax.axvline(
        thresholds.get(best_ensemble, 0.5), color="grey", linestyle=":",
        label="Selected threshold"
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(f"Threshold sensitivity, {DISPLAY_NAMES[best_ensemble]}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "threshold_sweep.png", dpi=150)
    plt.close(fig)

    return best_ensemble


def print_final_verdict(report, probas, best_ensemble):
    """
    Prints the headline comparison: best baseline vs best single model vs best ensemble.
    """
    best_baseline = max(
        [name for name in BASELINE_NAMES if name in probas and name != "chance"],
        key=lambda name: report[name]["auc"]
    )
    best_member = max(
        [name for name in ENSEMBLE_MEMBER_NAMES if name in probas],
        key=lambda name: report[name]["auc"]
    )

    print("\n" + "=" * 70)
    print(f"Best baseline      {DISPLAY_NAMES[best_baseline]:<28}AUC {report[best_baseline]['auc']:.4f}")
    print(f"Best single model  {DISPLAY_NAMES[best_member]:<28}AUC {report[best_member]['auc']:.4f}")
    print(f"Best ensemble      {DISPLAY_NAMES[best_ensemble]:<28}AUC {report[best_ensemble]['auc']:.4f}")
    print(f"\nEnsemble over baseline      {report[best_ensemble]['auc'] - report[best_baseline]['auc']:+.4f}")
    print(f"Ensemble over best member   {report[best_ensemble]['auc'] - report[best_member]['auc']:+.4f}")
    print("=" * 70)


def main():
    """
    Pulls every model's saved test predictions into one comparison, produces
    the report figures, and writes results/final_metrics.json.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    probas, thresholds = build_all_predictions()

    model_order = [
        name for name in
        BASELINE_NAMES + ENSEMBLE_MEMBER_NAMES + ["ens_proposal", "ens_fitted", "ens_rank"]
        if name in probas
    ]

    report = build_master_table(y_test, probas, thresholds, model_order)
    add_operating_point_metrics(y_test, probas, report, model_order)

    plot_roc_curves(y_test, probas, report, model_order)
    plot_roc_curves_zoomed(y_test, probas, model_order)
    plot_confusion_matrices(report, probas)
    best_ensemble = plot_threshold_sweep(y_test, probas, thresholds, report, model_order)

    print_final_verdict(report, probas, best_ensemble)

    with open(RESULTS_DIR / "final_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nWritten to results/:")
    for filename in [
        "final_metrics.json", "roc_curves.png", "roc_curves_zoom.png",
        "confusion_matrices.png", "threshold_sweep.png"
    ]:
        print(f"  {filename}")


if __name__ == "__main__":
    main()