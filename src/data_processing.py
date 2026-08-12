import pickle
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

VAL_FRACTION = 0.30
RANDOM_SEED = 42
CHUNK = 50000

def summarise(name, y):
    """
    Prints row count and malware/benign split for a label array.
    """
    n_mal = int(y.sum())
    print(f"  {name}: {len(y)} rows | malware {n_mal} | benign {len(y) - n_mal}")


def main():
    """
    Prepares extracted EMBER features for model training
    """
    root = Path(__file__).parent.parent
    processed = root / "data" / "processed"
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)

    x_train_path = processed / "X_train.npy"
    if not x_train_path.exists():
        print(f"ERROR: {x_train_path} not found. Run extract_and_save.py first.")
        return

    # mmap_mode='r' means only the slice we read actually loads into memory
    X = np.load(x_train_path, mmap_mode="r")
    y = np.load(processed / "y_train.npy")

    print(f"X_train: {X.shape} {X.dtype}")
    print(f"y_train: {y.shape} {y.dtype}")

    unlabelled = int((y == -1).sum())
    if unlabelled:
        print(f"WARNING: {unlabelled} rows still carry label -1. "
              f"They should have been filtered during extraction.")
    if X.shape[0] != y.shape[0]:
        print(f"ERROR: row mismatch, X has {X.shape[0]}, y has {y.shape[0]}")
        return

    # --- stratified split, stored as indices instead of copying the arrays ---
    print("\nStratified split")
    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=VAL_FRACTION,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    train_idx.sort()
    val_idx.sort()

    summarise("train", y[train_idx])
    summarise("val  ", y[val_idx])

    np.save(processed / "train_idx.npy", train_idx)
    np.save(processed / "val_idx.npy", val_idx)
    print(f"  saved train_idx.npy / val_idx.npy")

    # --- fit scaler on training rows only, chunked so it never holds it all in memory ---
    print("\nFitting StandardScaler on training rows (chunked)")
    start = time.time()
    scaler = StandardScaler()

    for i in range(0, len(train_idx), CHUNK):
        block = train_idx[i:i + CHUNK]
        scaler.partial_fit(np.asarray(X[block], dtype=np.float64))
        done = min(i + CHUNK, len(train_idx))
        print(f"  {done}/{len(train_idx)} rows", end="\r")

    print(f"\n  fitted in {time.time() - start:.0f}s")

    zero_var = int((scaler.var_ == 0).sum())
    print(f"  zero-variance features: {zero_var} of {X.shape[1]}")

    with open(models / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  saved models/scaler.pkl")

    # test set is just loaded and summarised here, not touched otherwise
    print("\nTest set")
    X_test = np.load(processed / "X_test.npy", mmap_mode="r")
    y_test = np.load(processed / "y_test.npy")
    print(f"  X_test: {X_test.shape}")
    summarise("test ", y_test)

    print("\nDone")


if __name__ == "__main__":
    main()