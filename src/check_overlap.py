import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"


def collect_hashes(folder, labelled_only=True):
    """
    Reads every .jsonl file in a folder and returns {sha256: label}.
    """
    out = {}
    files = sorted(folder.glob("*.jsonl"))
    if not files:
        print(f"  no .jsonl files in {folder}")
        return out

    for path in files:
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                label = r.get("label", -1)
                if labelled_only and label == -1:
                    continue
                out[r["sha256"]] = label
                n += 1
        print(f"    {path.name}: {n}")
    return out


def report_overlap(h2018, h2017, shared):
    """
    Prints overlap stats and flags samples with conflicting labels.
    """
    pct_2017 = 100 * len(shared) / len(h2017)
    pct_2018 = 100 * len(shared) / len(h2018)
    print(f"  {pct_2017:.2f}% of 2017, {pct_2018:.2f}% of 2018")

    disagree = [h for h in shared if h2018[h] != h2017[h]]
    if disagree:
        print(f"  {len(disagree)} shared samples carry CONFLICTING labels")
        print("  (ground truth was revised between collections)")

    print("\nOverlap found. Options, in order of preference:")
    print("1. Exclude shared hashes from the evaluation set and report how many were removed. Clean, and easy to justify in the writeup")
    print("2. Drop the cross-dataset experiment and rely on the existing within-2018 temporal split, which is already a valid result.")
    print("3. Proceed without exclusion only if the percentage is tiny, and state the figure explicitly as a limitation.")


def save_shared_hashes(shared):
    """
    Writes the shared hashes to disk so they can be excluded later if needed.
    """
    out = ROOT / "data" / "processed" / "shared_hashes.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(sorted(shared)))
    print(f"\nShared hashes written to {out}")


def main():
    f2018 = RAW / "ember_2018"
    f2017 = RAW / "ember_2017_v2"

    for f in (f2018, f2017):
        if not f.exists():
            print(f"ERROR: {f} not found")
            return

    print("EMBER 2018:")
    h2018 = collect_hashes(f2018)
    print(f"  total labelled: {len(h2018)}")

    print("\nEMBER 2017 v2:")
    h2017 = collect_hashes(f2017)
    print(f"  total labelled: {len(h2017)}")

    shared = set(h2018) & set(h2017)

    print("\n" + "-" * 20)
    print(f"Shared sha256 values: {len(shared)}")

    if not shared:
        print("\nNo overlap. A cross-dataset drift experiment is valid:")
        print("-" * 20)
        return

    report_overlap(h2018, h2017, shared)
    print("-" * 20)

    save_shared_hashes(shared)


if __name__ == "__main__":
    main()