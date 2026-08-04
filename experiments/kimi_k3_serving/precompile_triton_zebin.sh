#!/bin/bash
# Pre-compile all Triton kernel SPIR-V files to native zebin for Aurora PVC.
#
# Run this after the serving venv has generated its Triton cache. It is safe on
# a login node; ocloc uses the IGC compiler directly and does not need an XPU.
#
# Why: Triton caches old SPV entries with generate_native_code=false.
#   On first PIECEWISE vLLM generate call, Triton calls zeModuleCreate(SPIRV)
#   → L0 driver JIT via IGC → 50+ min per kernel (flash_attention_kernel is biggest).
#   Pre-compiling all SPVs to zebin + updating metadata eliminates this hang.
#
# Re-run when: torch211 venv is rebuilt or vLLM version changes (new Triton key =
#   new cache dirs with fresh SPV files that need pre-compilation).
#
# Usage: VENV=/path/to/venv bash experiments/kimi_k3_serving/precompile_triton_zebin.sh

set -e

VENV=${VENV:-}
CACHE_DIR=${TRITON_CACHE_DIR:-${VENV:+$VENV/triton-cache}}
CACHE_DIR=${CACHE_DIR:-${HOME}/.triton/cache}
TMPDIR="/tmp/triton_zebin_$$"
mkdir -p "$TMPDIR"

command -v ocloc >/dev/null || { echo "ERROR: ocloc is required" >&2; exit 1; }
[[ -d "$CACHE_DIR" ]] || { echo "ERROR: Triton cache does not exist: $CACHE_DIR" >&2; exit 1; }

echo "=== Triton zebin pre-compilation for Aurora PVC ==="
echo "Cache dir: $CACHE_DIR"
echo "Tmp dir:   $TMPDIR"
echo ""

# Step 1: Compile all SPV files that don't have a companion zebin
compiled=0
skipped=0
failed=0

while IFS= read -r spv; do
    dir=$(dirname "$spv")
    base=$(basename "$spv" .spv)
    zebin="${dir}/${base}.zebin"

    if [ -f "$zebin" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    outname="${base}_COMPILED"
    if ocloc compile -spirv_input -file "$spv" -device pvc -output "$outname" -out_dir "$TMPDIR" -q 2>/dev/null; then
        bin="${TMPDIR}/${outname}_pvc.bin"
        if [ -f "$bin" ]; then
            cp "$bin" "$zebin"
            compiled=$((compiled + 1))
        else
            echo "WARNING: ocloc succeeded but no output for $base"
            failed=$((failed + 1))
        fi
    else
        echo "WARNING: ocloc failed for $base ($(wc -c < "$spv")B SPV)"
        failed=$((failed + 1))
    fi
done < <(find "$CACHE_DIR" -name "*.spv" | sort)

echo "Step 1 done: compiled=$compiled skipped=$skipped failed=$failed"
echo ""

# Step 2: Update metadata JSONs to enable zebin loading path
TRITON_ZEBIN_CACHE_DIR="$CACHE_DIR" python3 - <<'PYEOF'
import json, os, glob

CACHE_DIR = os.environ["TRITON_ZEBIN_CACHE_DIR"]
fixed = 0
skipped = 0

for dirpath in sorted(glob.glob(f"{CACHE_DIR}/*")):
    if not os.path.isdir(dirpath):
        continue
    for json_file in glob.glob(f"{dirpath}/*.json"):
        if os.path.basename(json_file).startswith("__grp__"):
            continue
        try:
            with open(json_file) as f:
                metadata = json.load(f)
        except Exception:
            continue

        if metadata.get("generate_native_code", False):
            skipped += 1
            continue

        kernel_name = os.path.basename(json_file).replace(".json", "")
        zebin_file = os.path.join(dirpath, f"{kernel_name}.zebin")
        if not os.path.exists(zebin_file):
            continue  # no zebin compiled (ocloc failed or kernel not SPV-based)

        # Update metadata to use zebin
        metadata["generate_native_code"] = True
        metadata["binary_ext"] = "zebin"
        tmp = json_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(metadata, f)
        os.replace(tmp, json_file)

        # Update __grp__ manifest to include zebin path
        grp_file = os.path.join(dirpath, f"__grp__{kernel_name}.json")
        if os.path.exists(grp_file):
            try:
                with open(grp_file) as f:
                    grp_data = json.load(f)
                zebin_key = f"{kernel_name}.zebin"
                child_paths = grp_data.get("child_paths", {})
                if zebin_key not in child_paths:
                    child_paths[zebin_key] = zebin_file
                    grp_data["child_paths"] = child_paths
                    tmp = grp_file + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(grp_data, f, indent=4)
                    os.replace(tmp, grp_file)
            except Exception as e:
                print(f"WARNING: could not update {grp_file}: {e}")

        fixed += 1

print(f"Step 2 done: fixed={fixed} skipped_already_native={skipped}")
PYEOF

echo ""
echo "=== Done. All Triton kernels pre-compiled to zebin. ==="
echo "Next run will load zebin via zeModuleCreateWithNativeCode (instant)."

# Cleanup
rm -rf "$TMPDIR"
