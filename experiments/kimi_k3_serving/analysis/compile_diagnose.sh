#!/usr/bin/env bash
# Diagnose a compile leg: did Inductor actually run, and did it fuse?
#
# Exists because "compile made no difference" has two very different causes
# that look identical in a tok/s number:
#   (a) compilation never engaged (mode fell back to NONE, or every region
#       hit a graph break) -- the run measured eager and the lever is untested
#   (b) it engaged and genuinely did not help
#
# Only (b) is a result. Run this before believing either.
#
# Usage: bash compile_diagnose.sh <leg_dir>
set -uo pipefail
D=${1:?usage: compile_diagnose.sh <leg_dir>}
S="$D/server.log"
[[ -f "$S" ]] || { echo "no server.log in $D"; exit 2; }

echo "=== 1. What mode did the engine actually resolve? ==="
grep -oE "'mode': <CompilationMode\.[A-Z_]+" "$S" | head -1 || echo "  (no mode line)"
grep -oE "cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$S" | head -1 || true
grep -oE "'splitting_ops': \[[^]]{0,60}" "$S" | head -1 || true
echo "  NOTE: mode=NONE here means compile did NOT engage -- the leg is void"
echo "        as a compile measurement regardless of its tok/s."

echo
echo "=== 2. Did Inductor compile anything? ==="
for pat in "Compiling a graph" "torch.compile takes" "Dynamo bytecode transform" \
           "Inductor compilation" "Compilation of .* took"; do
    # grep -c prints 0 AND exits 1 on no-match, so `|| echo 0` appends a
    # second line -> "0\n0" and the [[ != 0 ]] test then passes. Use a
    # single-value form.
    n=$(grep -c "$pat" "$S" 2>/dev/null); n=${n:-0}
    [[ "$n" -gt 0 ]] && echo "  ${n} x  $pat"
done

echo
echo "=== 3. Graph breaks (the usual reason fusion underdelivers) ==="
gb=$(grep -c "graph break\|Graph break\|gb[0-9]\{4\}" "$S" 2>/dev/null); gb=${gb:-0}
echo "  graph-break mentions: $gb"
if [[ "$gb" -gt 0 ]]; then
    grep -oE "gb[0-9]{4}|Graph break[^.]{0,70}" "$S" 2>/dev/null \
        | sort | uniq -c | sort -rn | head -5
    echo "  -> many breaks = the fx graph is shredded; Inductor cannot fuse"
    echo "     across them. Re-run with TORCH_LOGS=graph_breaks for detail."
fi

echo
echo "=== 4. Compile-time cost (shows up as a slow first request) ==="
grep -oE "took [0-9.]+ s" "$S" 2>/dev/null | sort -rn -k2 | head -3 || true

echo
echo "=== 5. Errors that would force a silent eager fallback ==="
grep -iE "falling back|fallback to eager|compilation failed|BackendCompilerFailed" "$S" \
    2>/dev/null | head -4 || echo "  none"
