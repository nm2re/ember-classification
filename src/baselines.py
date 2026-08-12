import pickle
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# baseline models for comparison against the ensemble

RANDOM_SEED = 42
CHUNK = 25000
LOGREG_EPOCHS = 3
DTREE_DEPTH = 10

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def load():
    """
    Loads the processed arrays and index splits.
    """
    X = np.load(PROCESSED / "X_train.npy", mmap_mode="r")
    y = np.load(PROCESSED / "y_train.npy")
    train_idx = np.load(PROCESSED / "train_idx.npy")
    val_idx = np.load(PROCESSED / "val_idx.npy")
    X_test = np.load(PROCESSED / "X_test.npy", mmap_mode="r")
    y_test = np.load(PROCESSED / "y_test.npy")
    return X, y, train_idx, val_idx, X_test, y_test


def predict_in_chunks(fn, X, chunk=CHUNK):
    """
    Runs predictions in batches so the full memmapped array is never loaded at once.
    """
    out = np.empty(X.shape[0], dtype=np.float32)
    for i in range(0, X.shape[0], chunk):
        out[i:i + chunk] = fn(np.asarray(X[i:i + chunk]))
    return out


def report(name, y_val, val_proba):
    """
    Prints model's validation predictions.
    """
    pred = (val_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    auc = roc_auc_score(y_val, val_proba)
    f1 = f1_score(y_val, pred)
    print(f"  {name} validation AUC {auc:.4f} | F1 {f1:.4f} | FPR {fp / (fp + tn):.4f}")


def save(name, model, val_proba, test_proba):
    """
    Saves the fitted model if there are any and its predicted probabilities to disk.
    """
    if model is not None:
        with open(MODELS / f"{name}_model.pkl", "wb") as f:
            pickle.dump(model, f)
    np.save(MODELS / f"{name}_val_proba.npy", val_proba.astype(np.float32))
    np.save(MODELS / f"{name}_test_proba.npy", test_proba.astype(np.float32))


def train_logreg(X, y, train_idx, val_idx, X_test):
    """
    Trains logistic regression in shuffled batches since 420k rows can't fit in memory at once (16gb is not enough)
    """
    print("Logistic regression")

    with open(MODELS / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                          random_state=RANDOM_SEED, learning_rate="optimal")

    rng = np.random.default_rng(RANDOM_SEED)
    start = time.time()

    for epoch in range(LOGREG_EPOCHS):
        order = train_idx.copy()
        rng.shuffle(order)
        for i in range(0, len(order), CHUNK):
            rows = np.sort(order[i:i + CHUNK])
            batch = scaler.transform(np.asarray(X[rows]))
            model.partial_fit(batch, y[rows], classes=np.array([0, 1]))
        print(f"  epoch {epoch + 1}/{LOGREG_EPOCHS} done", end="\r")

    print(f"\n  trained in {time.time() - start:.0f}s")

    predict = lambda a: model.predict_proba(scaler.transform(a))[:, 1]
    val_proba = predict_in_chunks(predict, np.asarray(X[val_idx]))
    test_proba = predict_in_chunks(predict, X_test)
    return model, val_proba, test_proba


def train_dtree(X, y, train_idx, val_idx, X_test):
    """
    Trains a single depth-limited decision tree, small enough to fit in one shot
    """
    print(f"Decision tree (depth {DTREE_DEPTH})")

    X_tr = np.asarray(X[train_idx])
    y_tr = y[train_idx]

    model = DecisionTreeClassifier(max_depth=DTREE_DEPTH, min_samples_split=5,
                                   random_state=RANDOM_SEED)
    start = time.time()
    model.fit(X_tr, y_tr)
    print(f"  trained in {time.time() - start:.0f}s")
    print(f"  leaves: {model.get_n_leaves()}")

    del X_tr, y_tr

    predict = lambda a: model.predict_proba(a)[:, 1]
    val_proba = predict_in_chunks(predict, np.asarray(X[val_idx]))
    test_proba = predict_in_chunks(predict, X_test)
    return model, val_proba, test_proba


def train_chance(X, y, train_idx, val_idx, X_test):
    """
    Constant 0.5 predictor, used as a random-guessing floor to compare against.
    """
    print("Chance (constant majority-class predictor)")
    val_proba = np.full(len(val_idx), 0.5, dtype=np.float32)
    test_proba = np.full(X_test.shape[0], 0.5, dtype=np.float32)
    return None, val_proba, test_proba


TRAINERS = {
    "logreg": train_logreg,
    "dtree": train_dtree,
    "chance": train_chance,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in list(TRAINERS) + ["all"]:
        print("Usage: python src/baselines.py [logreg|dtree|chance|all]")
        return

    MODELS.mkdir(parents=True, exist_ok=True)
    which = list(TRAINERS) if sys.argv[1] == "all" else [sys.argv[1]]

    X, y, train_idx, val_idx, X_test, y_test = load()
    y_val = y[val_idx]
    print(f"train {len(train_idx)} | val {len(val_idx)} | test {len(y_test)}\n")

    for name in which:
        model, val_proba, test_proba = TRAINERS[name](X, y, train_idx, val_idx, X_test)
        if name != "chance":
            report(name, y_val, val_proba)
        save(name, model, val_proba, test_proba)
        print(f"  saved {name}\n")


if __name__ == "__main__":
    main()