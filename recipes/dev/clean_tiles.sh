#!/bin/bash
# Diagnose and clean GPU tile memory on Aurora compute nodes.
#
# Usage:
#   bash recipes/dev/clean_tiles.sh          # Show tile memory + GPU processes
#   bash recipes/dev/clean_tiles.sh --kill   # Kill YOUR orphaned GPU processes
#   bash recipes/dev/clean_tiles.sh --check  # Quick tile memory check only
#
# Card-to-tile mapping (ZE_FLAT_DEVICE_HIERARCHY=FLAT):
#   renderD128 = card0 = tiles 0,1    renderD131 = card3 = tiles 6,7
#   renderD129 = card1 = tiles 2,3    renderD132 = card4 = tiles 8,9
#   renderD130 = card2 = tiles 4,5    renderD133 = card5 = tiles 10,11
#
# Common cause of "leaked" tile memory: orphaned vLLM subprocesses or other
# users' jobs on shared nodes. Use --status to diagnose before assuming a leak.

set -e

XPU_SMI=$(find /opt/aurora/26.26.0 -name 'xpu-smi' -type f 2>/dev/null | sort -V | tail -1)
ACTION=${1:-"--status"}
ME=$(whoami)

check_tiles() {
    module load frameworks 2>/dev/null || true
    python3 -c "
import torch
for i in range(12):
    free, total = torch.xpu.mem_get_info(i)
    f = free/1024**3
    card = i // 2
    status = 'CLEAN' if f > 55 else ('USABLE' if f > 20 else 'FULL')
    print(f'Tile {i:2d} (card {card}): {f:5.1f} / {total/1024**3:.0f} GiB [{status}]')
"
}

if [ "$ACTION" = "--check" ]; then
    check_tiles
    exit 0
fi

echo "=== Tile Memory ==="
check_tiles
echo ""

echo "=== GPU Processes ==="
if [ -n "$XPU_SMI" ]; then
    "$XPU_SMI" ps 2>/dev/null | grep -v "^$" || echo "(xpu-smi ps failed, falling back to fuser)"
    echo ""
fi

for card in 0 1 2 3 4 5; do
    dev="/dev/dri/renderD$((128 + card))"
    tile_a=$((card * 2))
    tile_b=$((card * 2 + 1))
    pids=$(fuser "$dev" 2>/dev/null | tr -s ' ') || true
    if [ -n "$pids" ]; then
        echo "Card $card (tiles $tile_a,$tile_b):"
        for pid in $pids; do
            info=$(ps -p "$pid" -o pid=,user=,etime=,comm= 2>/dev/null | head -1) || true
            owner=$(ps -p "$pid" -o user= 2>/dev/null) || true
            marker=""
            [ "$owner" = "$ME" ] && marker=" ← yours"
            echo "  PID $info$marker"
        done
    fi
done

if [ "$ACTION" = "--kill" ]; then
    echo ""
    echo "=== Killing YOUR orphaned GPU processes ==="
    killed=0
    for card in 0 1 2 3 4 5; do
        dev="/dev/dri/renderD$((128 + card))"
        pids=$(fuser "$dev" 2>/dev/null | tr -s ' ') || true
        for pid in $pids; do
            owner=$(ps -p "$pid" -o user= 2>/dev/null) || true
            if [ "$owner" = "$ME" ]; then
                cmd=$(ps -p "$pid" -o comm= 2>/dev/null) || true
                echo "  Killing PID $pid ($cmd) on card $card"
                kill -9 "$pid" 2>/dev/null || true
                killed=$((killed + 1))
            fi
        done
    done
    if [ $killed -eq 0 ]; then
        echo "  No orphaned processes found belonging to $ME"
    else
        echo "  Killed $killed processes. Waiting for cleanup..."
        sleep 2
    fi
    echo ""
    echo "=== Post-cleanup ==="
    check_tiles
fi
