import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction import FeatureHasher

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

DEFAULT_SAMPLE = 5000
RANDOM_SEED = 42

# feature layout, mirrors extract_and_save.py exactly
BLOCKS = [
    (0, 256, "byte_histogram"),
    (256, 512, "byte_entropy"),
    (512, 616, "strings"),
    (616, 626, "general"),
    (626, 688, "header"),
    (688, 943, "sections"),
    (943, 2223, "imports"),
    (2223, 2351, "exports"),
    (2351, 2381, "datadirectories"),
]

GENERAL_FIELDS = ["size", "vsize", "has_debug", "exports_count", "imports_count",
                  "has_relocations", "has_resources", "has_signature",
                  "has_tls", "symbols"]

HEADER_NUMERIC = ["major_image_version", "minor_image_version",
                  "major_linker_version", "minor_linker_version",
                  "major_os_version", "minor_os_version",
                  "major_subsystem_version", "minor_subsystem_version",
                  "sizeof_code", "sizeof_headers", "sizeof_heap_commit"]

SECTION_COUNTS = ["num_sections", "num_zero_size_sections", "num_unnamed_sections",
                  "num_read_execute_sections", "num_writable_sections"]

DATADIR_NAMES = ["EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE",
                 "EXCEPTION_TABLE", "CERTIFICATE_TABLE", "BASE_RELOCATION_TABLE",
                 "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "TLS_TABLE",
                 "LOAD_CONFIG_TABLE", "BOUND_IMPORT", "IAT",
                 "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER"]


def feature_name(i):
    """
    Human-readable name for feature index i.
    """
    if i < 256:
        return f"byte_histogram[0x{i:02X}]"
    if i < 512:
        return f"byte_entropy[bin {i - 256}]"
    if i < 616:
        j = i - 512
        if j == 0: return "strings.numstrings"
        if j == 1: return "strings.avg_length"
        if j == 2: return "strings.printable_count"
        if j < 99: return f"strings.printable_dist[{j - 3}]"
        return ["strings.entropy", "strings.paths", "strings.urls",
                "strings.registry", "strings.MZ_count"][j - 99]
    if i < 626:
        return f"general.{GENERAL_FIELDS[i - 616]}"
    if i < 688:
        j = i - 626
        if j == 0: return "header.timestamp"
        if j < 11: return f"header.machine_hash[{j - 1}]"
        if j < 21: return f"header.coff_characteristics_hash[{j - 11}]"
        if j < 31: return f"header.subsystem_hash[{j - 21}]"
        if j < 41: return f"header.dll_characteristics_hash[{j - 31}]"
        if j < 51: return f"header.magic_hash[{j - 41}]"
        return f"header.{HEADER_NUMERIC[j - 51]}"
    if i < 943:
        j = i - 688
        if j < 5: return f"sections.{SECTION_COUNTS[j]}"
        if j < 55: return f"sections.size_hash[{j - 5}]"
        if j < 105: return f"sections.entropy_hash[{j - 55}]"
        if j < 155: return f"sections.vsize_hash[{j - 105}]"
        if j < 205: return f"sections.entry_name_hash[{j - 155}]"
        return f"sections.entry_props_hash[{j - 205}]"
    if i < 2223:
        j = i - 943
        if j < 256: return f"imports.dll_hash[{j}]"
        return f"imports.function_hash[{j - 256}]"
    if i < 2351:
        return f"exports.hash[{i - 2223}]"
    j = i - 2351
    return f"datadir.{DATADIR_NAMES[j // 2]}.{'size' if j % 2 == 0 else 'vaddr'}"


def block_of(i):
    """
    Returns which semantic block a feature index belongs to.
    """
    for start, end, name in BLOCKS:
        if start <= i < end:
            return name
    return "unknown"


COMMON_DLLS = [
    "kernel32.dll", "user32.dll", "advapi32.dll", "ws2_32.dll", "wininet.dll",
    "shell32.dll", "ole32.dll", "oleaut32.dll", "gdi32.dll", "msvcrt.dll",
    "ntdll.dll", "crypt32.dll", "wsock32.dll", "urlmon.dll", "shlwapi.dll",
    "psapi.dll", "netapi32.dll", "comctl32.dll", "version.dll", "winmm.dll",
    "mpr.dll", "userenv.dll", "secur32.dll", "iphlpapi.dll", "dnsapi.dll",
]

SUSPICIOUS_APIS = [
    "kernel32.dll:VirtualAlloc", "kernel32.dll:VirtualProtect",
    "kernel32.dll:WriteProcessMemory", "kernel32.dll:CreateRemoteThread",
    "kernel32.dll:LoadLibraryA", "kernel32.dll:GetProcAddress",
    "kernel32.dll:CreateProcessA", "kernel32.dll:WinExec",
    "advapi32.dll:RegSetValueExA", "advapi32.dll:RegCreateKeyExA",
    "advapi32.dll:CryptEncrypt", "advapi32.dll:AdjustTokenPrivileges",
    "ws2_32.dll:socket", "ws2_32.dll:connect", "ws2_32.dll:send",
    "wininet.dll:InternetOpenA", "wininet.dll:InternetReadFile",
    "urlmon.dll:URLDownloadToFileA",
    "user32.dll:SetWindowsHookExA", "user32.dll:GetAsyncKeyState",
    "ntdll.dll:NtUnmapViewOfSection", "ntdll.dll:ZwQueryInformationProcess",
]

COMMON_SECTIONS = [".text", ".data", ".rdata", ".rsrc", ".reloc", ".bss",
                   ".idata", ".edata", ".tls", "UPX0", "UPX1", ".aspack",
                   ".themida", ".vmp0", ".petite", ".nsp0"]


def bucket_map(strings, n_buckets):
    """
    Maps each candidate string to the bucket it hashes into.
    """
    hasher = FeatureHasher(n_buckets, input_type="string")
    out = {}
    for s in strings:
        row = hasher.transform([[s]]).toarray()[0]
        nz = np.nonzero(row)[0]
        if len(nz):
            out.setdefault(int(nz[0]), []).append((s, float(row[nz[0]])))
    return out


def explain_bucket(name):
    """
    Suggests what strings might live in a hash bucket, if the feature is one.
    """
    if name.startswith("imports.dll_hash["):
        idx = int(name.split("[")[1].rstrip("]"))
        hits = bucket_map(COMMON_DLLS, 256).get(idx, [])
        return ", ".join(s for s, _ in hits) if hits else ""
    if name.startswith("imports.function_hash["):
        idx = int(name.split("[")[1].rstrip("]"))
        hits = bucket_map(SUSPICIOUS_APIS, 1024).get(idx, [])
        return ", ".join(s for s, _ in hits) if hits else ""
    if "entry_name_hash[" in name or "size_hash[" in name or \
            "entropy_hash[" in name or "vsize_hash[" in name:
        idx = int(name.split("[")[1].rstrip("]"))
        hits = bucket_map(COMMON_SECTIONS, 50).get(idx, [])
        return ", ".join(s for s, _ in hits) if hits else ""
    return ""


def main():
    """
    SHAP interpretation on LightGBM. Maps the 2381 anonymous feature indices
    back to their semantic names, aggregates importance by block, and reverse-
    looks-up hashed buckets against candidate strings (suggestive, not exact,
    since hash collisions are possible).

    TreeExplainer is used instead of KernelExplainer since it's exact for tree
    models and runs in seconds instead of hours at this width.
    """
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLE
    RESULTS.mkdir(parents=True, exist_ok=True)

    try:
        import shap
    except ImportError:
        print("ERROR: shap not installed. pip install shap")
        return

    model_path = MODELS / "lgb_model.pkl"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run model_training.py lgb first.")
        return

    model = pickle.load(open(model_path, "rb"))

    X = np.load(PROCESSED / "X_test.npy", mmap_mode="r")
    y = np.load(PROCESSED / "y_test.npy")

    # stratified sample - importance ranking stabilises well before 200k rows
    rng = np.random.default_rng(RANDOM_SEED)
    half = n_sample // 2
    idx = np.concatenate([
        rng.choice(np.flatnonzero(y == 1), half, replace=False),
        rng.choice(np.flatnonzero(y == 0), half, replace=False)])
    idx.sort()

    X_s = np.asarray(X[idx])
    y_s = y[idx]
    print(f"SHAP sample: {len(idx)} rows ({int(y_s.sum())} malware, "
          f"{len(y_s) - int(y_s.sum())} benign)")

    print("Computing SHAP values (TreeExplainer, exact for tree models)")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_s)

    # older shap returns a list per class, newer returns a single array
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    print(f"  SHAP matrix {sv.shape}")

    mean_abs = np.abs(sv).mean(axis=0)
    signed = sv.mean(axis=0)

    # --- importance by block ---
    print(f"\n{'Block':<22}{'total |SHAP|':>14}{'share':>9}{'dims':>7}"
          f"{'per-dim':>11}")
    print("-" * 63)

    block_rows = []
    total = mean_abs.sum()
    for start, end, name in BLOCKS:
        s = float(mean_abs[start:end].sum())
        dims = end - start
        block_rows.append({"block": name, "total": s, "share": s / total,
                           "dims": dims, "per_dim": s / dims})
    for r in sorted(block_rows, key=lambda r: -r["total"]):
        print(f"{r['block']:<22}{r['total']:>14.5f}{r['share']:>8.1%}"
              f"{r['dims']:>7}{r['per_dim']:>11.6f}")

    print("\nPer-dim divides by block width, since imports has 1280 columns")
    print("and a large total there may just reflect size, not importance.")

    # --- top individual features ---
    top = np.argsort(mean_abs)[::-1][:30]
    print(f"\n{'#':<4}{'Feature':<42}{'|SHAP|':>10}{'dir':>6}   possible contents")
    print("-" * 100)
    top_records = []
    for rank, i in enumerate(top, 1):
        name = feature_name(int(i))
        direction = "mal" if signed[i] > 0 else "ben"
        hint = explain_bucket(name)
        print(f"{rank:<4}{name:<42}{mean_abs[i]:>10.5f}{direction:>6}   {hint}")
        top_records.append({
            "rank": rank, "index": int(i), "name": name,
            "block": block_of(int(i)),
            "mean_abs_shap": float(mean_abs[i]),
            "mean_signed_shap": float(signed[i]),
            "pushes_toward": "malware" if signed[i] > 0 else "benign",
            "possible_contents": hint,
        })

    print("\n'dir' is whether higher values push toward malware or benign.")
    print("'possible contents' is a reverse lookup - hash collisions mean")
    print("other strings could share the bucket, so treat it as a suggestion.")

    # --- cross-check against LightGBM's native importance ---
    try:
        gain = model.booster_.feature_importance(importance_type="gain")
        top_gain = set(np.argsort(gain)[::-1][:30])
        overlap = len(top_gain & set(top.tolist()))
        print(f"\nAgreement with LightGBM's native gain importance: "
              f"{overlap}/30 features in common.")
    except Exception:
        pass

    # --- figures ---
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = sorted(block_rows, key=lambda r: r["total"])
    ax.barh([r["block"] for r in rows], [r["total"] for r in rows])
    ax.set_xlabel("Total mean |SHAP| across block")
    ax.set_title("Feature importance by semantic block")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "shap_by_block.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    t20 = top[:20][::-1]
    colours = ["tab:red" if signed[i] > 0 else "tab:blue" for i in t20]
    ax.barh([feature_name(int(i)) for i in t20], mean_abs[t20], color=colours)
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title("Top 20 features (red pushes toward malware, blue toward benign)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "shap_top_features.png", dpi=150)
    plt.close(fig)

    try:
        names = [feature_name(i) for i in range(sv.shape[1])]
        shap.summary_plot(sv, X_s, feature_names=names, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(RESULTS / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"\n(beeswarm plot skipped: {e})")

    with open(RESULTS / "shap_analysis.json", "w") as f:
        json.dump({"n_samples": int(len(idx)),
                   "blocks": block_rows,
                   "top_features": top_records}, f, indent=2)

    print("\nWritten to results/:")
    for filename in ["shap_analysis.json", "shap_by_block.png",
                     "shap_top_features.png", "shap_beeswarm.png"]:
        print(f"  {filename}")


if __name__ == "__main__":
    main()