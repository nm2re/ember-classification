import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score, precision_score, recall_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.30
BATCH_SIZE = 50000
RF_SUBSAMPLE_SIZE = 150000  # matches the 2018 dataset version

ROOT_DIR = Path(__file__).parent.parent
PROCESSED_2017_DIR = ROOT_DIR / "data" / "processed_2017"
PROCESSED_2018_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models_2017"
RESULTS_DIR = ROOT_DIR / "results"

ENSEMBLE_MEMBER_NAMES = ["lgb", "rf", "xgb", "nn"]


def compute_metrics(y_true, proba, threshold):
    """
    Computes AUC, F1, precision, TPR, FPR, and TPR at a 1% FPR budget.
    """
    predicted_labels = (proba >= threshold).astype(int)
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_true, predicted_labels).ravel()

    fpr_curve, tpr_curve, _ = roc_curve(y_true, proba)
    idx_at_1pct = max(np.searchsorted(fpr_curve, 0.01, side="right") - 1, 0)

    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "f1": float(f1_score(y_true, predicted_labels)),
        "precision": float(precision_score(y_true, predicted_labels, zero_division=0)),
        "tpr": float(recall_score(y_true, predicted_labels)),
        "fpr": float(false_pos / (false_pos + true_neg)),
        "tpr_at_1pct_fpr": float(tpr_curve[idx_at_1pct]),
        "threshold": float(threshold),
    }


def predict_in_batches(predict_fn, X, batch_size=BATCH_SIZE):
    """
    Runs predictions in batches to avoid loading the whole array into memory.
    """
    predictions = np.empty(X.shape[0], dtype=np.float32)
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        predictions[start:end] = predict_fn(np.asarray(X[start:end]))
    return predictions


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


def prepare_data():
    """
    Loads 2017 training data and 2018 test data, splits 2017 into
    train/val, and fits a scaler on the 2017 training rows.
    """
    X_train_full = np.load(PROCESSED_2017_DIR / "X_train.npy", mmap_mode="r")
    y_train_full = np.load(PROCESSED_2017_DIR / "y_train.npy")
    print(f"2017 training data: {X_train_full.shape}")

    X_test = np.load(PROCESSED_2018_DIR / "X_test.npy", mmap_mode="r")
    y_test = np.load(PROCESSED_2018_DIR / "y_test.npy")
    print(f"2018 test data:     {X_test.shape}")

    if X_train_full.shape[1] != X_test.shape[1]:
        raise SystemExit(
            f"ERROR: feature count mismatch, 2017 has {X_train_full.shape[1]}, "
            f"2018 has {X_test.shape[1]}."
        )

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train_full)),
        test_size=VALIDATION_FRACTION,
        stratify=y_train_full,
        random_state=RANDOM_SEED,
    )
    train_idx.sort()
    val_idx.sort()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_DIR / "scaler.pkl"

    if scaler_path.exists():
        scaler = pickle.load(open(scaler_path, "rb"))
        print("  reusing fitted scaler")
    else:
        print("  fitting scaler on 2017 training rows")
        scaler = StandardScaler()
        for start in range(0, len(train_idx), BATCH_SIZE):
            batch_idx = train_idx[start:start + BATCH_SIZE]
            scaler.partial_fit(np.asarray(X_train_full[batch_idx], dtype=np.float64))
        pickle.dump(scaler, open(scaler_path, "wb"))

    return X_train_full, y_train_full, train_idx, val_idx, X_test, y_test, scaler


def train_lightgbm(X, y, train_idx):
    import lightgbm as lgb

    print("\nLightGBM")
    X_train, y_train = np.asarray(X[train_idx]), y[train_idx]

    start_time = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=7, learning_rate=0.1, num_leaves=64,
        n_jobs=-1, random_state=RANDOM_SEED, verbose=-1,
    )
    model.fit(X_train, y_train)
    print(f"  trained in {time.time() - start_time:.0f}s")

    return model


def train_xgboost(X_train, y_train):
    import xgboost as xgb

    print("\nXGBoost")
    start_time = time.time()
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist",
        n_jobs=-1, random_state=RANDOM_SEED, eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    print(f"  trained in {time.time() - start_time:.0f}s")

    return model


def train_random_forest(X, y, train_idx):
    from sklearn.ensemble import RandomForestClassifier

    print("\nRandom Forest")

    rows_to_use = train_idx
    if RF_SUBSAMPLE_SIZE and RF_SUBSAMPLE_SIZE < len(train_idx):
        # match the 2018 run's subsample so the comparison stays fair
        rng = np.random.default_rng(RANDOM_SEED)
        y_train_labels = y[train_idx]
        n_per_class = RF_SUBSAMPLE_SIZE // 2

        malware_rows = rng.choice(train_idx[y_train_labels == 1], n_per_class, replace=False)
        benign_rows = rng.choice(train_idx[y_train_labels == 0], n_per_class, replace=False)

        rows_to_use = np.concatenate([malware_rows, benign_rows])
        rows_to_use.sort()
        print(f"  subsampled to {len(rows_to_use)} rows, matching the 2018 run")

    start_time = time.time()
    model = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        n_jobs=-1, random_state=RANDOM_SEED,
    )
    model.fit(np.asarray(X[rows_to_use]), y[rows_to_use])
    print(f"  trained in {time.time() - start_time:.0f}s")

    return model


class BatchGenerator:
    """
    Feeds shuffled, scaled batches to the neural network without holding
    the whole training set in memory at once.
    """

    def __init__(self, source_array, indices, labels, scaler, batch_size=1024, shuffle=True):
        self.source_array = source_array
        self.indices = np.array(indices)
        self.labels = labels
        self.scaler = scaler
        self.batch_size = batch_size

        self.batch_order = np.arange(len(self.indices))
        if shuffle:
            np.random.default_rng(RANDOM_SEED).shuffle(self.batch_order)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_number):
        selected = self.batch_order[
            batch_number * self.batch_size: (batch_number + 1) * self.batch_size
        ]
        rows = np.sort(self.indices[selected])

        X_batch = self.scaler.transform(np.asarray(self.source_array[rows])).astype(np.float32)
        y_batch = self.labels[rows]
        return X_batch, y_batch


def train_neural_network(X, y, train_idx, val_idx, scaler):
    from tensorflow import keras

    print("\nNeural Network")

    # local subclass so the TensorFlow import stays inside this function
    class KerasBatchGenerator(keras.utils.Sequence, BatchGenerator):
        pass

    train_generator = KerasBatchGenerator(X, train_idx, y, scaler, shuffle=True)
    val_generator = KerasBatchGenerator(X, val_idx, y, scaler, shuffle=False)

    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    start_time = time.time()
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=3, restore_best_weights=True
            )
        ],
    )
    print(f"  trained in {time.time() - start_time:.0f}s")

    return model


def train_all_members(X, y, train_idx, val_idx, X_test, scaler):
    """
    Trains all four ensemble members on 2017 data and collects their predictions.
    """
    val_probas = {}
    test_probas = {}
    X_val = np.asarray(X[val_idx])

    def record_predictions(name, predict_fn):
        val_probas[name] = predict_in_batches(predict_fn, X_val)
        test_probas[name] = predict_in_batches(predict_fn, X_test)
        val_auc = roc_auc_score(y[val_idx], val_probas[name])
        print(f"  {name} 2017-validation AUC {val_auc:.4f}")

    lgb_model = train_lightgbm(X, y, train_idx)
    record_predictions("lgb", lambda batch: lgb_model.predict_proba(batch)[:, 1])
    pickle.dump(lgb_model, open(MODELS_DIR / "lgb_model.pkl", "wb"))

    X_train, y_train = np.asarray(X[train_idx]), y[train_idx]
    xgb_model = train_xgboost(X_train, y_train)
    record_predictions("xgb", lambda batch: xgb_model.predict_proba(batch)[:, 1])
    xgb_model.save_model(str(MODELS_DIR / "xgb_model.json"))
    del X_train, y_train

    rf_model = train_random_forest(X, y, train_idx)
    record_predictions("rf", lambda batch: rf_model.predict_proba(batch)[:, 1])
    pickle.dump(rf_model, open(MODELS_DIR / "rf_model.pkl", "wb"))

    nn_model = train_neural_network(X, y, train_idx, val_idx, scaler)
    record_predictions("nn",lambda batch: nn_model.predict(scaler.transform(batch).astype(np.float32), verbose=0).ravel(),)

    nn_model.save(MODELS_DIR / "nn_model.keras")

    return val_probas, test_probas


def load_2018_ensemble_weights():
    """
    Loads the ensemble weights fitted during the 2018 run.
    """
    weights_source = json.load(open(RESULTS_DIR / "metrics.json"))
    return weights_source.get(
        "fitted_weights", {"lgb": 0.4, "rf": 0.3, "xgb": 0.2, "nn": 0.1}
    )


def print_comparison_table(results_2018, results_2017):
    """
    Prints the 2018-trained vs 2017-trained comparison for each model.
    """
    print("\n" + "=" * 78)
    print("Evaluated on the same 2018 test set (200,000 samples)")
    print("=" * 78)
    print(
        f"{'Model':<22}{'AUC 2018':>10}{'AUC 2017':>10}{'delta':>9}"
        f"{'T@1% 2018':>11}{'T@1% 2017':>11}{'delta':>9}"
    )
    print("-" * 78)

    for name in ENSEMBLE_MEMBER_NAMES + ["ens_fitted", "ens_rank"]:
        if name not in results_2018:
            continue

        auc_2018 = results_2018[name]["auc"]
        auc_2017 = results_2017[name]["auc"]
        tpr1_2018 = results_2018[name].get("tpr_at_1pct_fpr", float("nan"))
        tpr1_2017 = results_2017[name]["tpr_at_1pct_fpr"]

        print(
            f"{name:<22}{auc_2018:>10.4f}{auc_2017:>10.4f}{auc_2017 - auc_2018:>+9.4f}"
            f"{tpr1_2018:>11.4f}{tpr1_2017:>11.4f}{tpr1_2017 - tpr1_2018:>+9.4f}"
        )

    print("-" * 78)
    print("AUC 2018 = trained on 2018.  AUC 2017 = trained on 2017.")
    print("Both are scored on the same 2018 test set, so the difference")
    print("is attributable to the training data being a year older.")
    print("T@1% is detection rate at a 1% false positive budget.")
    print("=" * 78)


def main():
    """
    Trains the full pipeline on EMBER 2017 and evaluates it on the EMBER 2018
    test set, then compares against the 2018-trained results.
    """

    if not (PROCESSED_2017_DIR / "X_train.npy").exists():
        raise SystemExit(f"ERROR: {PROCESSED_2017_DIR / 'X_train.npy'} not found. Run "f"extract_and_save.py against ember_2017_v2 first.")

    X, y, train_idx, val_idx, X_test, y_test, scaler = prepare_data()
    print(f"  train {len(train_idx)} | val {len(val_idx)} | test {len(y_test)}")

    val_probas, test_probas = train_all_members(X, y, train_idx, val_idx, X_test, scaler)

    # reuse 2018's weights so only the training data changes, not the combination rule
    weights = load_2018_ensemble_weights()
    weights_str = "  ".join(f"{name} {w:.2f}" for name, w in weights.items())
    print(f"\nEnsemble weights (carried over from the 2018 run): {weights_str}")

    ensemble_fitted_proba = weighted_average_proba(test_probas, weights)
    ensemble_rank_proba = weighted_average_rank(test_probas, weights)

    results_2018 = json.load(open(RESULTS_DIR / "final_metrics.json"))

    results_2017 = {
        name: compute_metrics(y_test, test_probas[name], threshold=0.5)
        for name in ENSEMBLE_MEMBER_NAMES
    }


    results_2017["ens_fitted"] = compute_metrics(y_test, ensemble_fitted_proba,threshold=results_2018.get("ens_fitted", {}).get("threshold", 0.5),)

    results_2017["ens_rank"] = compute_metrics(y_test, ensemble_rank_proba, threshold=0.5)

    print_comparison_table(results_2018, results_2017)

    with open(RESULTS_DIR / "drift_metrics.json", "w") as f:
        json.dump({"trained_2017": results_2017, "weights": weights}, f, indent=2)

    print("\nSaved results/drift_metrics.json")


if __name__ == "__main__":
    main()