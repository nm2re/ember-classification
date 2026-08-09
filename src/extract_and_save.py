"""
EMBER feature extraction.

Reimplements EMBER's official feature vector (2381 dims with datadirectories,
2351 without) using batched FeatureHasher calls.

Two phases:
  1. Each JSONL file is loaded, vectorised, and written to data/processed/parts/
     as its own .npy. Raw data is freed before the next file is read.
  2. Parts are stitched into X_train.npy / y_train.npy via a memory-mapped
     array, so the full matrix is never held in RAM at once.

Safe to re-run: phase 1 skips parts that already exist on disk.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher


# ---------------------------------------------------------------------------
# Hashers, created once and reused. Sizes match EMBER's published layout.
# ---------------------------------------------------------------------------

H_MACHINE = FeatureHasher(10, input_type="string")
H_COFF_CHAR = FeatureHasher(10, input_type="string")
H_SUBSYSTEM = FeatureHasher(10, input_type="string")
H_DLL_CHAR = FeatureHasher(10, input_type="string")
H_MAGIC = FeatureHasher(10, input_type="string")

H_SEC_SIZE = FeatureHasher(50, input_type="pair")
H_SEC_ENTROPY = FeatureHasher(50, input_type="pair")
H_SEC_VSIZE = FeatureHasher(50, input_type="pair")
H_SEC_ENTRY = FeatureHasher(50, input_type="string")
H_SEC_CHAR = FeatureHasher(50, input_type="string")

H_LIBRARIES = FeatureHasher(256, input_type="string")
H_IMPORTS = FeatureHasher(1024, input_type="string")
H_EXPORTS = FeatureHasher(128, input_type="string")


def _as_str_list(value):
    """Coerce a field into a list of strings. Guards against the
    'Samples can not be a single string' hasher error."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


# ---------------------------------------------------------------------------
# Per-field extraction. Each returns a (n_records, width) float32 array.
# ---------------------------------------------------------------------------

def f_byte_histogram(records):
    """256 dims. Normalised byte-value counts."""
    arr = np.array([r["histogram"] for r in records], dtype=np.float32)
    totals = arr.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return arr / totals


def f_byte_entropy(records):
    """256 dims. Normalised byte-entropy histogram."""
    arr = np.array([r["byteentropy"] for r in records], dtype=np.float32)
    totals = arr.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return arr / totals


def f_strings(records):
    """104 dims. String statistics plus normalised printable distribution."""
    out = np.zeros((len(records), 104), dtype=np.float32)
    for i, r in enumerate(records):
        s = r.get("strings") or {}
        printables = s.get("printables", 0) or 0
        divisor = float(printables) if printables > 0 else 1.0
        dist = np.asarray(s.get("printabledist", [0] * 96), dtype=np.float32)
        if dist.size != 96:
            dist = np.zeros(96, dtype=np.float32)
        out[i] = np.hstack([
            s.get("numstrings", 0),
            s.get("avlength", 0),
            printables,
            dist / divisor,
            s.get("entropy", 0),
            s.get("paths", 0),
            s.get("urls", 0),
            s.get("registry", 0),
            s.get("MZ", 0),
            ]).astype(np.float32)
    return out


def f_general(records):
    """10 dims. File-level metadata."""
    keys = ["size", "vsize", "has_debug", "exports", "imports",
            "has_relocations", "has_resources", "has_signature",
            "has_tls", "symbols"]
    out = np.zeros((len(records), 10), dtype=np.float32)
    for i, r in enumerate(records):
        g = r.get("general") or {}
        out[i] = [float(g.get(k, 0) or 0) for k in keys]
    return out


def f_header(records):
    """62 dims. COFF and optional header, string fields hashed."""
    n = len(records)

    machine, coff_char = [], []
    subsystem, dll_char, magic = [], [], []
    numeric = np.zeros((n, 12), dtype=np.float32)

    num_keys = ["major_image_version", "minor_image_version",
                "major_linker_version", "minor_linker_version",
                "major_operating_system_version", "minor_operating_system_version",
                "major_subsystem_version", "minor_subsystem_version",
                "sizeof_code", "sizeof_headers", "sizeof_heap_commit"]

    for i, r in enumerate(records):
        h = r.get("header") or {}
        coff = h.get("coff") or {}
        opt = h.get("optional") or {}

        machine.append(_as_str_list(coff.get("machine", "")))
        coff_char.append(_as_str_list(coff.get("characteristics", [])))
        subsystem.append(_as_str_list(opt.get("subsystem", "")))
        dll_char.append(_as_str_list(opt.get("dll_characteristics", [])))
        magic.append(_as_str_list(opt.get("magic", "")))

        numeric[i, 0] = float(coff.get("timestamp", 0) or 0)
        for j, k in enumerate(num_keys):
            numeric[i, j + 1] = float(opt.get(k, 0) or 0)

    return np.hstack([
        numeric[:, :1],
        H_MACHINE.transform(machine).toarray(),
        H_COFF_CHAR.transform(coff_char).toarray(),
        H_SUBSYSTEM.transform(subsystem).toarray(),
        H_DLL_CHAR.transform(dll_char).toarray(),
        H_MAGIC.transform(magic).toarray(),
        numeric[:, 1:],
    ]).astype(np.float32)


def f_sections(records):
    """255 dims. Section counts plus hashed name/size/entropy/vsize pairs."""
    n = len(records)
    general = np.zeros((n, 5), dtype=np.float32)
    sizes, entropies, vsizes, entries, chars = [], [], [], [], []

    for i, r in enumerate(records):
        sec = r.get("section") or {}
        sections = sec.get("sections") or []
        entry = sec.get("entry", "") or ""

        general[i] = [
            len(sections),
            sum(1 for s in sections if (s.get("size", 0) or 0) == 0),
            sum(1 for s in sections if not s.get("name")),
            sum(1 for s in sections
                if "MEM_READ" in (s.get("props") or [])
                and "MEM_EXECUTE" in (s.get("props") or [])),
            sum(1 for s in sections if "MEM_WRITE" in (s.get("props") or [])),
        ]

        sizes.append([(str(s.get("name", "")), float(s.get("size", 0) or 0))
                      for s in sections])
        entropies.append([(str(s.get("name", "")), float(s.get("entropy", 0) or 0))
                          for s in sections])
        vsizes.append([(str(s.get("name", "")), float(s.get("vsize", 0) or 0))
                       for s in sections])
        entries.append(_as_str_list(entry))
        chars.append([p for s in sections
                      if s.get("name") == entry
                      for p in (s.get("props") or [])])

    return np.hstack([
        general,
        H_SEC_SIZE.transform(sizes).toarray(),
        H_SEC_ENTROPY.transform(entropies).toarray(),
        H_SEC_VSIZE.transform(vsizes).toarray(),
        H_SEC_ENTRY.transform(entries).toarray(),
        H_SEC_CHAR.transform(chars).toarray(),
    ]).astype(np.float32)


def f_imports(records):
    """1280 dims. 256 for DLL names, 1024 for lib:function pairs."""
    libs, funcs = [], []
    for r in records:
        imp = r.get("imports") or {}
        if not isinstance(imp, dict):
            imp = {}
        libs.append(sorted({str(k).lower() for k in imp.keys()}))
        funcs.append([f"{str(lib).lower()}:{fn}"
                      for lib, flist in imp.items()
                      for fn in (flist or [])])
    return np.hstack([
        H_LIBRARIES.transform(libs).toarray(),
        H_IMPORTS.transform(funcs).toarray(),
    ]).astype(np.float32)


def f_exports(records):
    """128 dims. Hashed exported function names."""
    rows = [_as_str_list(r.get("exports")) for r in records]
    return H_EXPORTS.transform(rows).toarray().astype(np.float32)


def f_datadirectories(records):
    """30 dims. Size and virtual address for 15 data directories."""
    out = np.zeros((len(records), 30), dtype=np.float32)
    for i, r in enumerate(records):
        dd = r.get("datadirectories") or []
        for j in range(min(15, len(dd))):
            entry = dd[j] or {}
            out[i, 2 * j] = float(entry.get("size", 0) or 0)
            out[i, 2 * j + 1] = float(entry.get("virtual_address", 0) or 0)
    return out


# ---------------------------------------------------------------------------
# Phase 1: one file in, one .npy pair out.
# ---------------------------------------------------------------------------

def vectorise_file(path, chunk_size=25000):
    """Read a JSONL file and return (X, y). Processes in chunks so peak
    memory stays bounded regardless of file size."""
    X_chunks, y_chunks = [], []
    buffer = []

    def flush(buf):
        if not buf:
            return
        labels = np.array([r["label"] for r in buf], dtype=np.int32)
        blocks = [
            f_byte_histogram(buf),
            f_byte_entropy(buf),
            f_strings(buf),
            f_general(buf),
            f_header(buf),
            f_sections(buf),
            f_imports(buf),
            f_exports(buf),
        ]
        if any("datadirectories" in r for r in buf[:1]):
            blocks.append(f_datadirectories(buf))
        X_chunks.append(np.hstack(blocks).astype(np.float32))
        y_chunks.append(labels)

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("label", -1) == -1:
                continue          # EMBER's deliberately unlabelled samples
            buffer.append(record)
            if len(buffer) >= chunk_size:
                flush(buffer)
                buffer = []
                gc.collect()
    flush(buffer)

    if not X_chunks:
        return None, None
    return np.vstack(X_chunks), np.concatenate(y_chunks)


def run_phase1(data_path, parts_path):
    parts_path.mkdir(parents=True, exist_ok=True)
    files = sorted(data_path.glob("train_features_*.jsonl"))
    files.append(data_path / "test_features.jsonl")

    for path in files:
        if not path.exists():
            print(f"  missing, skipping: {path.name}")
            continue

        stem = path.stem
        x_out = parts_path / f"X_{stem}.npy"
        y_out = parts_path / f"y_{stem}.npy"

        if x_out.exists() and y_out.exists():
            print(f"  {path.name}: already done, skipping")
            continue

        start = time.time()
        X, y = vectorise_file(path)
        if X is None:
            print(f"  {path.name}: no labelled rows")
            continue

        np.save(x_out, X)
        np.save(y_out, y)
        print(f"  {path.name}: {X.shape[0]} rows x {X.shape[1]} features "
              f"in {time.time() - start:.0f}s")

        del X, y
        gc.collect()


# ---------------------------------------------------------------------------
# Phase 2: stitch parts without loading everything into RAM.
# ---------------------------------------------------------------------------

def stitch(parts_path, out_path, part_stems, x_name, y_name):
    x_files = [parts_path / f"X_{s}.npy" for s in part_stems]
    y_files = [parts_path / f"y_{s}.npy" for s in part_stems]
    x_files = [p for p in x_files if p.exists()]
    y_files = [p for p in y_files if p.exists()]

    if not x_files:
        print(f"  nothing to stitch for {x_name}")
        return

    total_rows = 0
    n_features = None
    for p in x_files:
        arr = np.load(p, mmap_mode="r")
        total_rows += arr.shape[0]
        n_features = arr.shape[1]
        del arr

    final = np.lib.format.open_memmap(
        out_path / x_name, mode="w+", dtype=np.float32,
        shape=(total_rows, n_features))

    cursor = 0
    for p in x_files:
        arr = np.load(p, mmap_mode="r")
        final[cursor:cursor + arr.shape[0]] = arr
        cursor += arr.shape[0]
        del arr
    final.flush()
    del final

    labels = np.concatenate([np.load(p) for p in y_files])
    np.save(out_path / y_name, labels)

    n_mal = int(labels.sum())
    print(f"  {x_name}: {total_rows} x {n_features} "
          f"(malware {n_mal}, benign {len(labels) - n_mal})")


# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).parent.parent
    data_path = root / "data" / "raw" / "ember_2018"
    out_path = root / "data" / "processed"
    parts_path = out_path / "parts"

    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    print("Phase 1: vectorising each file")
    run_phase1(data_path, parts_path)

    print("\nPhase 2: stitching")
    train_stems = [p.stem for p in sorted(data_path.glob("train_features_*.jsonl"))]
    stitch(parts_path, out_path, train_stems, "X_train.npy", "y_train.npy")
    stitch(parts_path, out_path, ["test_features"], "X_test.npy", "y_test.npy")

    print(f"\nDone. Output in {out_path}")
    print("Part files kept in data/processed/parts/ - delete once verified.")


if __name__ == "__main__":
    main()