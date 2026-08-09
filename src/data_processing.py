"""
Prepare the extracted EMBER features for model training.

Design notes:

  * X_train is 600000 x 2381 float32, about 5.7 GB. Anything that copies the
    whole array doubles that, so this script avoids copies entirely.

  * No scaled version of the data is written to disk. Tree models (LightGBM,
    Random Forest, XGBoost) are invariant to feature scaling, so scaling all
    2381 columns for their benefit would cost 5.7 GB of disk and buy nothing.
    Only the neural network needs scaling, and it can apply the saved scaler
    to batches at training time.

  * The train/validation split is stored as index arrays rather than as two
    new arrays. Indices for 600k rows are ~2.4 MB; the arrays would be 5.7 GB.

  * StandardScaler is fitted with partial_fit over chunks, so the fit never
    holds more than one chunk in memory.

Outputs (all small):
  data/processed/train_idx.npy
  data/processed/val_idx.npy
  models/scaler.pkl
"""

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
    n_mal = int(y.sum())
    print(f"  {name}: {len(y)} rows | malware {n_mal} | benign {len(y) - n_mal}")


def main():
    root = Path(__file__).parent.parent
    processed = root / "data" / "processed"
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)

    x_train_path = processed / "X_train.npy"
    if not x_train_path.exists():
        print(f"ERROR: {x_train_path} not found. Run extract_and_save.py first.")
        return

    # mmap_mode='r' opens a view backed by the file. Reading a slice pulls only
    # that slice into memory; the full 5.7 GB is never resident.
    X = np.load(x_train_path, mmap_mode="r")
    y = np.load(processed / "y_train.npy")

    print(f"X_train: {X.shape} {X.dtype}")
    print(f"y_train: {y.shape} {y.dtype}")

    # Sanity checks
    unlabelled = int((y == -1).sum())
    if unlabelled:
        print(f"WARNING: {unlabelled} rows still carry label -1. "
              f"They should have been filtered during extraction.")
    if X.shape[0] != y.shape[0]:
        print(f"ERROR: row mismatch, X has {X.shape[0]}, y has {y.shape[0]}")
        return

    # ------------------------------------------------------------------
    # Stratified split, stored as indices
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Incremental scaler fit, training rows only
    # ------------------------------------------------------------------
    # Fitting on training rows only matters: if validation or test statistics
    # leak into the mean and variance, the validation score is optimistic.
    print("\nFitting StandardScaler on training rows (chunked)")
    start = time.time()
    scaler = StandardScaler()

    for i in range(0, len(train_idx), CHUNK):
        block = train_idx[i:i + CHUNK]
        scaler.partial_fit(np.asarray(X[block], dtype=np.float64))
        done = min(i + CHUNK, len(train_idx))
        print(f"  {done}/{len(train_idx)} rows", end="\r")

    print(f"\n  fitted in {time.time() - start:.0f}s")

    # A constant column has zero variance. StandardScaler sets those scales to
    # 1.0 rather than dividing by zero, but it is worth knowing how many there
    # are: hashed feature blocks often leave many buckets permanently empty.
    zero_var = int((scaler.var_ == 0).sum())
    print(f"  zero-variance features: {zero_var} of {X.shape[1]}")

    with open(models / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"  saved models/scaler.pkl")

    # ------------------------------------------------------------------
    print("\nTest set")
    X_test = np.load(processed / "X_test.npy", mmap_mode="r")
    y_test = np.load(processed / "y_test.npy")
    print(f"  X_test: {X_test.shape}")
    summarise("test ", y_test)

    print("\nDone. Nothing large was written; splits are indices, scaling is "
          "applied at training time by whichever model needs it.")


if __name__ == "__main__":
    main()