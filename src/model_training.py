import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

RANDOM_SEED = 42
RF_SUBSAMPLE = 150000  # set to None to use all 420k rows

ROOT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"


def load_splits():
    """
    Loads the train/validation/test feature arrays and index splits.
    """
    X_train_full = np.load(PROCESSED_DIR / "X_train.npy", mmap_mode="r")
    y_train_full = np.load(PROCESSED_DIR / "y_train.npy")
    train_idx = np.load(PROCESSED_DIR / "train_idx.npy")
    val_idx = np.load(PROCESSED_DIR / "val_idx.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy", mmap_mode="r")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    return X_train_full, y_train_full, train_idx, val_idx, X_test, y_test


def print_validation_report(name, y_val, val_proba):
    """
    Prints AUC and F1 for a model's validation predictions.
    """
    auc = roc_auc_score(y_val, val_proba)
    f1 = f1_score(y_val, (val_proba >= 0.5).astype(int))
    print(f"  {name} validation AUC {auc:.4f} | F1 {f1:.4f}")
    return auc, f1


def save_probability_outputs(name, val_proba, test_proba):
    """
    Saves a model's validation and test predictions to disk.
    """
    np.save(MODELS_DIR / f"{name}_val_proba.npy", val_proba.astype(np.float32))
    np.save(MODELS_DIR / f"{name}_test_proba.npy", test_proba.astype(np.float32))


def predict_in_batches(predict_fn, X, batch_size=50000):
    """
    Runs predictions in batches so a large memmapped array is never fully loaded at once.
    """
    predictions = np.empty(X.shape[0], dtype=np.float32)
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        predictions[start:end] = predict_fn(np.asarray(X[start:end]))
    return predictions


def train_lightgbm(X, y, train_idx, val_idx, X_test):
    import lightgbm as lgb

    print("LightGBM")
    X_train = np.asarray(X[train_idx])
    y_train = y[train_idx]

    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=7, learning_rate=0.1, num_leaves=64,
        n_jobs=-1, random_state=RANDOM_SEED,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"  trained in {time.time() - start_time:.0f}s")

    del X_train, y_train

    with open(MODELS_DIR / "lgb_model.pkl", "wb") as f:
        pickle.dump(model, f)

    predict = lambda batch: model.predict_proba(batch)[:, 1]
    val_proba = predict_in_batches(predict, np.asarray(X[val_idx]))
    test_proba = predict_in_batches(predict, X_test)
    return val_proba, test_proba


def train_xgboost(X, y, train_idx, val_idx, X_test):
    import xgboost as xgb

    print("XGBoost")
    X_train = np.asarray(X[train_idx])
    y_train = y[train_idx]

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist",
        n_jobs=-1, random_state=RANDOM_SEED, eval_metric="logloss",
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"  trained in {time.time() - start_time:.0f}s")

    del X_train, y_train

    model.save_model(str(MODELS_DIR / "xgb_model.json"))

    predict = lambda batch: model.predict_proba(batch)[:, 1]
    val_proba = predict_in_batches(predict, np.asarray(X[val_idx]))
    test_proba = predict_in_batches(predict, X_test)
    return val_proba, test_proba


def select_rf_training_rows(y, train_idx):
    """
    Picks a stratified subsample for Random Forest if RF_SUBSAMPLE is set.
    """
    if RF_SUBSAMPLE is None or RF_SUBSAMPLE >= len(train_idx):
        return train_idx

    rng = np.random.default_rng(RANDOM_SEED)
    y_train_labels = y[train_idx]
    n_per_class = RF_SUBSAMPLE // 2

    malware_rows = train_idx[y_train_labels == 1]
    benign_rows = train_idx[y_train_labels == 0]

    sampled_rows = np.concatenate([
        rng.choice(malware_rows, n_per_class, replace=False),
        rng.choice(benign_rows, n_per_class, replace=False),
    ])
    sampled_rows.sort()

    print(f"  subsampled to {len(sampled_rows)} stratified rows (of {len(train_idx)})")
    return sampled_rows


def train_random_forest(X, y, train_idx, val_idx, X_test):
    from sklearn.ensemble import RandomForestClassifier

    print("Random Forest")

    rows_to_use = select_rf_training_rows(y, train_idx)
    X_train = np.asarray(X[rows_to_use])
    y_train = y[rows_to_use]

    model = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        n_jobs=-1, random_state=RANDOM_SEED,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"  trained in {time.time() - start_time:.0f}s")

    del X_train, y_train

    with open(MODELS_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump(model, f)

    predict = lambda batch: model.predict_proba(batch)[:, 1]
    val_proba = predict_in_batches(predict, np.asarray(X[val_idx]))
    test_proba = predict_in_batches(predict, X_test)
    return val_proba, test_proba


class ScaledSequence:
    """
    Feeds shuffled, scaled batches to the neural network without holding
    a scaled copy of the whole training matrix in memory.
    """

    def __init__(self, source, indices, labels, scaler, batch_size=1024, shuffle=True):
        self.source = source
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

        X_batch = self.scaler.transform(np.asarray(self.source[rows])).astype(np.float32)
        y_batch = self.labels[rows]
        return X_batch, y_batch


def build_neural_network(n_features):
    """
    Builds and compiles the neural network architecture.
    """
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
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
    return model


def train_neural_net(X, y, train_idx, val_idx, X_test):
    from tensorflow import keras

    print("Neural Network")

    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # local subclass keeps the TensorFlow import inside this function
    class KerasScaledSequence(keras.utils.Sequence, ScaledSequence):
        pass

    train_generator = KerasScaledSequence(X, train_idx, y, scaler, shuffle=True)
    val_generator = KerasScaledSequence(X, val_idx, y, scaler, shuffle=False)

    model = build_neural_network(n_features=X.shape[1])

    start_time = time.time()
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=3, restore_best_weights=True
            )
        ],
        verbose=1,
    )
    print(f"  trained in {time.time() - start_time:.0f}s")

    model.save(MODELS_DIR / "nn_model.keras")

    predict = lambda batch: model.predict(scaler.transform(batch).astype(np.float32), verbose=0).ravel()
    val_proba = predict_in_batches(predict, np.asarray(X[val_idx]))
    test_proba = predict_in_batches(predict, X_test)
    return val_proba, test_proba


TRAINERS = {
    "lgb": train_lightgbm,
    "xgb": train_xgboost,
    "rf": train_random_forest,
    "nn": train_neural_net,
}


def main():
    """
    Trains the four ensemble members: LightGBM, XGBoost, Random Forest, and a
    Neural Network. Run one at a time, or "all" for all four in sequence:
    """
    valid_args = list(TRAINERS) + ["all"]

    if len(sys.argv) < 2 or sys.argv[1] not in valid_args:
        print("Usage: python src/model_training.py [lgb|xgb|rf|nn|all]")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models_to_run = list(TRAINERS) if sys.argv[1] == "all" else [sys.argv[1]]

    X, y, train_idx, val_idx, X_test, y_test = load_splits()
    y_val = y[val_idx]
    print(f"train {len(train_idx)} | val {len(val_idx)} | test {len(y_test)}\n")

    for name in models_to_run:
        train_fn = TRAINERS[name]
        val_proba, test_proba = train_fn(X, y, train_idx, val_idx, X_test)

        print_validation_report(name, y_val, val_proba)
        save_probability_outputs(name, val_proba, test_proba)

        print(f"  saved {name} model and probability arrays\n")


if __name__ == "__main__":
    main()