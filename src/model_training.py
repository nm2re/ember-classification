"""
Train the four ensemble members.

Run one at a time so a failure in the last model does not cost the earlier ones:

    python src/model_training.py lgb
    python src/model_training.py xgb
    python src/model_training.py nn
    python src/model_training.py rf     (slowest, see RF_SUBSAMPLE below)

Or all four in sequence:

    python src/model_training.py all

Each run writes three things to models/:
    <name>_model.<ext>       the fitted model
    <name>_val_proba.npy     P(malware) on the 180k validation rows
    <name>_test_proba.npy    P(malware) on the 200k test rows

Saving the probability arrays means ensemble.py and evaluation.py never have to
re-run inference. Tuning ensemble weights becomes an operation on four small
arrays instead of four model loads plus 200k predictions each.

Memory notes:
  * Tree models read the raw (unscaled) features. Materialising the 420k
    training rows costs about 4.0 GB; LightGBM and XGBoost then build their own
    binned copy on top, so expect 6-8 GB peak. Close other applications.
  * The neural network scales batches on the fly via the saved scaler rather
    than holding a scaled copy of the whole matrix.
  * Random Forest is the memory and time risk. RF_SUBSAMPLE caps its training
    rows; set to None to use all 420k, but expect a long run.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

RANDOM_SEED = 42

# Random Forest trains on this many stratified rows. Set to None for all 420k.
# A subsample here is a defensible, documentable limitation; silently running
# out of memory three hours in is not.
RF_SUBSAMPLE = 150000

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def load_splits():
    X = np.load(PROCESSED / "X_train.npy", mmap_mode="r")
    y = np.load(PROCESSED / "y_train.npy")
    train_idx = np.load(PROCESSED / "train_idx.npy")
    val_idx = np.load(PROCESSED / "val_idx.npy")
    X_test = np.load(PROCESSED / "X_test.npy", mmap_mode="r")
    y_test = np.load(PROCESSED / "y_test.npy")
    return X, y, train_idx, val_idx, X_test, y_test


def report(name, y_val, val_proba):
    auc = roc_auc_score(y_val, val_proba)
    f1 = f1_score(y_val, (val_proba >= 0.5).astype(int))
    print(f"  {name} validation AUC {auc:.4f} | F1 {f1:.4f}")
    return auc, f1


def save_outputs(name, val_proba, test_proba):
    np.save(MODELS / f"{name}_val_proba.npy", val_proba.astype(np.float32))
    np.save(MODELS / f"{name}_test_proba.npy", test_proba.astype(np.float32))


def predict_in_chunks(predict_fn, X, chunk=50000):
    """Predict over a memmapped array without materialising all of it."""
    out = np.empty(X.shape[0], dtype=np.float32)
    for i in range(0, X.shape[0], chunk):
        out[i:i + chunk] = predict_fn(np.asarray(X[i:i + chunk]))
    return out


# ---------------------------------------------------------------------------

def train_lightgbm(X, y, train_idx, val_idx, X_test):
    import lightgbm as lgb

    print("LightGBM")
    X_tr = np.asarray(X[train_idx])
    y_tr = y[train_idx]

    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        num_leaves=64,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    start = time.time()
    model.fit(X_tr, y_tr)
    print(f"  trained in {time.time() - start:.0f}s")

    del X_tr, y_tr

    with open(MODELS / "lgb_model.pkl", "wb") as fh:
        pickle.dump(model, fh)

    fn = lambda a: model.predict_proba(a)[:, 1]
    val_proba = predict_in_chunks(fn, np.asarray(X[val_idx]))
    test_proba = predict_in_chunks(fn, X_test)
    return val_proba, test_proba


def train_xgboost(X, y, train_idx, val_idx, X_test):
    import xgboost as xgb

    print("XGBoost")
    X_tr = np.asarray(X[train_idx])
    y_tr = y[train_idx]

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
    )
    start = time.time()
    model.fit(X_tr, y_tr)
    print(f"  trained in {time.time() - start:.0f}s")

    del X_tr, y_tr

    model.save_model(str(MODELS / "xgb_model.json"))

    fn = lambda a: model.predict_proba(a)[:, 1]
    val_proba = predict_in_chunks(fn, np.asarray(X[val_idx]))
    test_proba = predict_in_chunks(fn, X_test)
    return val_proba, test_proba


def train_random_forest(X, y, train_idx, val_idx, X_test):
    from sklearn.ensemble import RandomForestClassifier

    print("Random Forest")

    idx = train_idx
    if RF_SUBSAMPLE is not None and RF_SUBSAMPLE < len(train_idx):
        rng = np.random.default_rng(RANDOM_SEED)
        y_tr_all = y[train_idx]
        per_class = RF_SUBSAMPLE // 2
        pos = train_idx[y_tr_all == 1]
        neg = train_idx[y_tr_all == 0]
        idx = np.concatenate([
            rng.choice(pos, per_class, replace=False),
            rng.choice(neg, per_class, replace=False),
        ])
        idx.sort()
        print(f"  subsampled to {len(idx)} stratified rows "
              f"(of {len(train_idx)}) - document this as a limitation")

    X_tr = np.asarray(X[idx])
    y_tr = y[idx]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    start = time.time()
    model.fit(X_tr, y_tr)
    print(f"  trained in {time.time() - start:.0f}s")

    del X_tr, y_tr

    with open(MODELS / "rf_model.pkl", "wb") as fh:
        pickle.dump(model, fh)

    fn = lambda a: model.predict_proba(a)[:, 1]
    val_proba = predict_in_chunks(fn, np.asarray(X[val_idx]))
    test_proba = predict_in_chunks(fn, X_test)
    return val_proba, test_proba


def train_neural_net(X, y, train_idx, val_idx, X_test):
    import tensorflow as tf
    from tensorflow import keras

    print("Neural Network")

    with open(MODELS / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)

    class ScaledSequence(keras.utils.Sequence):
        """Applies the fitted scaler per batch so no scaled copy of the full
        matrix is ever held in memory."""

        def __init__(self, source, indices, labels, batch_size=1024, shuffle=True):
            self.source = source
            self.indices = np.array(indices)
            self.labels = labels
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.order = np.arange(len(self.indices))
            if shuffle:
                np.random.default_rng(RANDOM_SEED).shuffle(self.order)

        def __len__(self):
            return int(np.ceil(len(self.indices) / self.batch_size))

        def __getitem__(self, i):
            sel = self.order[i * self.batch_size:(i + 1) * self.batch_size]
            rows = np.sort(self.indices[sel])
            batch = scaler.transform(np.asarray(self.source[rows]))
            return batch.astype(np.float32), self.labels[rows]

    n_features = X.shape[1]
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

    train_seq = ScaledSequence(X, train_idx, y, shuffle=True)
    val_seq = ScaledSequence(X, val_idx, y, shuffle=False)

    start = time.time()
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=10,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3, restore_best_weights=True)],
        verbose=1,
    )
    print(f"  trained in {time.time() - start:.0f}s")

    model.save(MODELS / "nn_model.keras")

    def nn_predict(arr):
        return model.predict(scaler.transform(arr).astype(np.float32),
                             verbose=0).ravel()

    val_proba = predict_in_chunks(nn_predict, np.asarray(X[val_idx]))
    test_proba = predict_in_chunks(nn_predict, X_test)
    return val_proba, test_proba


# ---------------------------------------------------------------------------

TRAINERS = {
    "lgb": train_lightgbm,
    "xgb": train_xgboost,
    "rf": train_random_forest,
    "nn": train_neural_net,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in list(TRAINERS) + ["all"]:
        print("Usage: python src/model_training.py [lgb|xgb|rf|nn|all]")
        return

    MODELS.mkdir(parents=True, exist_ok=True)
    which = list(TRAINERS) if sys.argv[1] == "all" else [sys.argv[1]]

    X, y, train_idx, val_idx, X_test, y_test = load_splits()
    y_val = y[val_idx]
    print(f"train {len(train_idx)} | val {len(val_idx)} | test {len(y_test)}\n")

    for name in which:
        val_proba, test_proba = TRAINERS[name](X, y, train_idx, val_idx, X_test)
        report(name, y_val, val_proba)
        save_outputs(name, val_proba, test_proba)
        print(f"  saved {name} model and probability arrays\n")


if __name__ == "__main__":
    main()