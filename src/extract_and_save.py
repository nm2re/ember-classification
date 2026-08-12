import gc
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import FeatureHasher

# hashers created once and reused; sizes match EMBER's published layout
HASH_MACHINE = FeatureHasher(10, input_type="string")
HASH_COFF_CHARACTERISTICS = FeatureHasher(10, input_type="string")
HASH_SUBSYSTEM = FeatureHasher(10, input_type="string")
HASH_DLL_CHARACTERISTICS = FeatureHasher(10, input_type="string")
HASH_MAGIC = FeatureHasher(10, input_type="string")

HASH_SECTION_SIZE = FeatureHasher(50, input_type="pair")
HASH_SECTION_ENTROPY = FeatureHasher(50, input_type="pair")
HASH_SECTION_VSIZE = FeatureHasher(50, input_type="pair")
HASH_SECTION_ENTRY_NAME = FeatureHasher(50, input_type="string")
HASH_SECTION_ENTRY_PROPS = FeatureHasher(50, input_type="string")

HASH_IMPORTED_LIBRARIES = FeatureHasher(256, input_type="string")
HASH_IMPORTED_FUNCTIONS = FeatureHasher(1024, input_type="string")
HASH_EXPORTED_FUNCTIONS = FeatureHasher(128, input_type="string")


def as_string_list(value):
    """
    Coerces a field into a list of strings, since FeatureHasher needs an
    iterable of strings per record.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def extract_byte_histogram(records):
    """
    256 dims: normalised counts of each byte value in the file.
    """
    histograms = np.array([r["histogram"] for r in records], dtype=np.float32)
    row_totals = histograms.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1.0
    return histograms / row_totals


def extract_byte_entropy(records):
    """
    256 dims: normalised byte-entropy histogram.
    """
    entropy_hists = np.array([r["byteentropy"] for r in records], dtype=np.float32)
    row_totals = entropy_hists.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1.0
    return entropy_hists / row_totals


def extract_strings(records):
    """
    104 dims: string statistics plus a 96-bin printable-character distribution.
    """
    output = np.zeros((len(records), 104), dtype=np.float32)

    for i, record in enumerate(records):
        strings_info = record.get("strings") or {}

        printable_count = strings_info.get("printables", 0) or 0
        divisor = float(printable_count) if printable_count > 0 else 1.0

        printable_dist = np.asarray(
            strings_info.get("printabledist", [0] * 96), dtype=np.float32
        )
        if printable_dist.size != 96:
            printable_dist = np.zeros(96, dtype=np.float32)

        output[i] = np.hstack([
            strings_info.get("numstrings", 0),
            strings_info.get("avlength", 0),
            printable_count,
            printable_dist / divisor,
            strings_info.get("entropy", 0),
            strings_info.get("paths", 0),
            strings_info.get("urls", 0),
            strings_info.get("registry", 0),
            strings_info.get("MZ", 0),
            ]).astype(np.float32)

    return output


def extract_general_info(records):
    """
    10 dims: basic file-level metadata.
    """
    field_names = [
        "size", "vsize", "has_debug", "exports", "imports",
        "has_relocations", "has_resources", "has_signature",
        "has_tls", "symbols",
    ]
    output = np.zeros((len(records), 10), dtype=np.float32)

    for i, record in enumerate(records):
        general_info = record.get("general") or {}
        output[i] = [float(general_info.get(field, 0) or 0) for field in field_names]

    return output


def extract_header(records):
    """
    62 dims: COFF and optional PE header fields, string fields hashed.
    """
    n_records = len(records)

    machine_values = []
    coff_characteristics_values = []
    subsystem_values = []
    dll_characteristics_values = []
    magic_values = []

    numeric_fields = np.zeros((n_records, 12), dtype=np.float32)
    numeric_field_names = [
        "major_image_version", "minor_image_version",
        "major_linker_version", "minor_linker_version",
        "major_operating_system_version", "minor_operating_system_version",
        "major_subsystem_version", "minor_subsystem_version",
        "sizeof_code", "sizeof_headers", "sizeof_heap_commit",
    ]

    for i, record in enumerate(records):
        header = record.get("header") or {}
        coff = header.get("coff") or {}
        optional = header.get("optional") or {}

        machine_values.append(as_string_list(coff.get("machine", "")))
        coff_characteristics_values.append(as_string_list(coff.get("characteristics", [])))
        subsystem_values.append(as_string_list(optional.get("subsystem", "")))
        dll_characteristics_values.append(as_string_list(optional.get("dll_characteristics", [])))
        magic_values.append(as_string_list(optional.get("magic", "")))

        numeric_fields[i, 0] = float(coff.get("timestamp", 0) or 0)
        for j, field_name in enumerate(numeric_field_names):
            numeric_fields[i, j + 1] = float(optional.get(field_name, 0) or 0)

    return np.hstack([
        numeric_fields[:, :1],
        HASH_MACHINE.transform(machine_values).toarray(),
        HASH_COFF_CHARACTERISTICS.transform(coff_characteristics_values).toarray(),
        HASH_SUBSYSTEM.transform(subsystem_values).toarray(),
        HASH_DLL_CHARACTERISTICS.transform(dll_characteristics_values).toarray(),
        HASH_MAGIC.transform(magic_values).toarray(),
        numeric_fields[:, 1:],
    ]).astype(np.float32)


def extract_sections(records):
    """
    255 dims: section counts plus hashed name/size/entropy/vsize pairs.
    """
    n_records = len(records)

    section_counts = np.zeros((n_records, 5), dtype=np.float32)
    size_pairs, entropy_pairs, vsize_pairs = [], [], []
    entry_point_names, entry_point_props = [], []

    for i, record in enumerate(records):
        section_info = record.get("section") or {}
        sections = section_info.get("sections") or []
        entry_point_name = section_info.get("entry", "") or ""

        section_counts[i] = [
            len(sections),
            sum(1 for s in sections if (s.get("size", 0) or 0) == 0),
            sum(1 for s in sections if not s.get("name")),
            sum(
                1 for s in sections
                if "MEM_READ" in (s.get("props") or [])
                and "MEM_EXECUTE" in (s.get("props") or [])
            ),
            sum(1 for s in sections if "MEM_WRITE" in (s.get("props") or [])),
        ]

        size_pairs.append(
            [(str(s.get("name", "")), float(s.get("size", 0) or 0)) for s in sections]
        )
        entropy_pairs.append(
            [(str(s.get("name", "")), float(s.get("entropy", 0) or 0)) for s in sections]
        )
        vsize_pairs.append(
            [(str(s.get("name", "")), float(s.get("vsize", 0) or 0)) for s in sections]
        )
        entry_point_names.append(as_string_list(entry_point_name))
        entry_point_props.append([
            prop for s in sections
            if s.get("name") == entry_point_name
            for prop in (s.get("props") or [])
        ])

    return np.hstack([
        section_counts,
        HASH_SECTION_SIZE.transform(size_pairs).toarray(),
        HASH_SECTION_ENTROPY.transform(entropy_pairs).toarray(),
        HASH_SECTION_VSIZE.transform(vsize_pairs).toarray(),
        HASH_SECTION_ENTRY_NAME.transform(entry_point_names).toarray(),
        HASH_SECTION_ENTRY_PROPS.transform(entry_point_props).toarray(),
    ]).astype(np.float32)


def extract_imports(records):
    """
    1280 dims: 256 buckets for DLL names, 1024 for library:function pairs.
    """
    imported_libraries = []
    imported_functions = []

    for record in records:
        imports = record.get("imports") or {}
        if not isinstance(imports, dict):
            imports = {}

        imported_libraries.append(sorted({str(lib).lower() for lib in imports.keys()}))
        imported_functions.append([
            f"{str(lib).lower()}:{function_name}"
            for lib, function_list in imports.items()
            for function_name in (function_list or [])
        ])

    return np.hstack([
        HASH_IMPORTED_LIBRARIES.transform(imported_libraries).toarray(),
        HASH_IMPORTED_FUNCTIONS.transform(imported_functions).toarray(),
    ]).astype(np.float32)


def extract_exports(records):
    """
    128 dims: hashed exported function names.
    """
    exported_names_per_record = [as_string_list(r.get("exports")) for r in records]
    return HASH_EXPORTED_FUNCTIONS.transform(exported_names_per_record).toarray().astype(np.float32)


def extract_data_directories(records):
    """
    30 dims: size and virtual address for each of PE's 15 data directories.
    """
    output = np.zeros((len(records), 30), dtype=np.float32)

    for i, record in enumerate(records):
        directories = record.get("datadirectories") or []
        for j in range(min(15, len(directories))):
            entry = directories[j] or {}
            output[i, 2 * j] = float(entry.get("size", 0) or 0)
            output[i, 2 * j + 1] = float(entry.get("virtual_address", 0) or 0)

    return output


def vectorise_file(path, chunk_size=25000):
    """
    Reads a JSONL file in chunks and converts it into feature vectors,
    keeping peak memory bounded by chunk size instead of file size.
    """
    completed_chunks_X = []
    completed_chunks_y = []
    buffer = []

    def process_buffer(records_buffer):
        if not records_buffer:
            return

        labels = np.array([r["label"] for r in records_buffer], dtype=np.int32)

        feature_blocks = [
            extract_byte_histogram(records_buffer),
            extract_byte_entropy(records_buffer),
            extract_strings(records_buffer),
            extract_general_info(records_buffer),
            extract_header(records_buffer),
            extract_sections(records_buffer),
            extract_imports(records_buffer),
            extract_exports(records_buffer),
        ]

        # only v2 records carry datadirectories; check the first record as a sample
        if any("datadirectories" in r for r in records_buffer[:1]):
            feature_blocks.append(extract_data_directories(records_buffer))

        completed_chunks_X.append(np.hstack(feature_blocks).astype(np.float32))
        completed_chunks_y.append(labels)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if record.get("label", -1) == -1:
                continue  # EMBER's deliberately unlabelled samples

            buffer.append(record)
            if len(buffer) >= chunk_size:
                process_buffer(buffer)
                buffer = []
                gc.collect()

    process_buffer(buffer)

    if not completed_chunks_X:
        return None, None

    return np.vstack(completed_chunks_X), np.concatenate(completed_chunks_y)


def run_phase1(data_path, parts_path):
    """
    Vectorises every raw JSONL file one at a time, saving each as its own .npy pair.
    """
    parts_path.mkdir(parents=True, exist_ok=True)

    files_to_process = sorted(data_path.glob("train_features_*.jsonl"))
    files_to_process.append(data_path / "test_features.jsonl")

    for file_path in files_to_process:
        if not file_path.exists():
            print(f"  missing, skipping: {file_path.name}")
            continue

        file_stem = file_path.stem
        output_X_path = parts_path / f"X_{file_stem}.npy"
        output_y_path = parts_path / f"y_{file_stem}.npy"

        if output_X_path.exists() and output_y_path.exists():
            print(f"  {file_path.name}: already done, skipping")
            continue

        start_time = time.time()
        X, y = vectorise_file(file_path)

        if X is None:
            print(f"  {file_path.name}: no labelled rows")
            continue

        np.save(output_X_path, X)
        np.save(output_y_path, y)

        elapsed = time.time() - start_time
        print(f"  {file_path.name}: {X.shape[0]} rows x {X.shape[1]} features in {elapsed:.0f}s")

        del X, y
        gc.collect()


def stitch_parts_into_single_array(parts_path, out_path, part_file_stems, output_X_name, output_y_name):
    """
    Combines per-file .npy parts into one final array pair without loading
    all parts into RAM at once.
    """
    X_part_paths = [parts_path / f"X_{stem}.npy" for stem in part_file_stems]
    y_part_paths = [parts_path / f"y_{stem}.npy" for stem in part_file_stems]

    X_part_paths = [p for p in X_part_paths if p.exists()]
    y_part_paths = [p for p in y_part_paths if p.exists()]

    if not X_part_paths:
        print(f"  nothing to stitch for {output_X_name}")
        return

    total_rows = 0
    n_features = None
    for path in X_part_paths:
        part_array = np.load(path, mmap_mode="r")
        total_rows += part_array.shape[0]
        n_features = part_array.shape[1]
        del part_array

    combined_X = np.lib.format.open_memmap(
        out_path / output_X_name, mode="w+", dtype=np.float32,
        shape=(total_rows, n_features)
    )

    row_cursor = 0
    for path in X_part_paths:
        part_array = np.load(path, mmap_mode="r")
        combined_X[row_cursor:row_cursor + part_array.shape[0]] = part_array
        row_cursor += part_array.shape[0]
        del part_array

    combined_X.flush()
    del combined_X

    combined_labels = np.concatenate([np.load(path) for path in y_part_paths])
    np.save(out_path / output_y_name, combined_labels)

    n_malware = int(combined_labels.sum())
    n_benign = len(combined_labels) - n_malware
    print(f"  {output_X_name}: {total_rows} x {n_features} (malware {n_malware}, benign {n_benign})")


def main():
    """
    Reimplements EMBER's official feature vector (2381 dims) using batched
    FeatureHasher calls, without ever loading a full dataset into memory.

    Phase 1: each JSONL file is vectorised and saved as its own .npy part.
    Phase 2: parts are stitched into X_train.npy / y_train.npy via a
    memory-mapped array, so the full matrix is never fully in RAM.

    Safe to re-run: phase 1 skips files whose output already exists.
    """
    root_dir = Path(__file__).parent.parent
    data_path = root_dir / "data" / "raw" / "ember_2017_v2"
    out_path = root_dir / "data" / "processed_2017"
    parts_path = out_path / "parts"

    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    print("Phase 1: vectorising each file")
    run_phase1(data_path, parts_path)

    print("\nPhase 2: stitching")
    train_file_stems = [p.stem for p in sorted(data_path.glob("train_features_*.jsonl"))]
    stitch_parts_into_single_array(parts_path, out_path, train_file_stems, "X_train.npy", "y_train.npy")
    stitch_parts_into_single_array(parts_path, out_path, ["test_features"], "X_test.npy", "y_test.npy")

    print(f"\nDone. Output in {out_path}")
    print("Part files kept in data/processed/parts/ - delete once verified.")


if __name__ == "__main__":
    main()