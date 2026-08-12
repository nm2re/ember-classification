import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw" / "ember_2018"
PROCESSED = ROOT / "data" / "processed"


def main():
    """
    Needed for the temporal drift analysis, which substitutes for the per-family analysis EMBER can't support
    """
    test_file = RAW / "test_features.jsonl"
    if not test_file.exists():
        print(f"ERROR: {test_file} not found")
        return

    appeared, sha256, labels = [], [], []

    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("label", -1) == -1:
                continue
            appeared.append(r.get("appeared", ""))
            sha256.append(r.get("sha256", ""))
            labels.append(r["label"])

    appeared = np.array(appeared)
    sha256 = np.array(sha256)
    labels = np.array(labels, dtype=np.int32)

    # check row order matches y_test.npy before trusting this metadata
    y_test = np.load(PROCESSED / "y_test.npy")
    if len(labels) != len(y_test) or not np.array_equal(labels, y_test):
        print("ERROR: recovered rows do not match y_test.npy. Do not use.")
        print(f"  recovered {len(labels)}, expected {len(y_test)}")
        return

    np.save(PROCESSED / "test_appeared.npy", appeared)
    np.save(PROCESSED / "test_sha256.npy", sha256)

    print(f"Verified alignment against y_test.npy ({len(labels)} rows)")
    print("\nDistribution by month:")
    months, counts = np.unique(appeared, return_counts=True)
    for m, c in zip(months, counts):
        mal = int(labels[appeared == m].sum())
        print(f"  {m}: {c:>7} samples ({mal} malware, {c - mal} benign)")

    print("\nSaved test_appeared.npy and test_sha256.npy")

if __name__ == "__main__":
    main()