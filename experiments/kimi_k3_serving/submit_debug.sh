#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NODES=${1:-1}

case "$NODES" in
    1) SCRIPT="$SCRIPT_DIR/hold_Nnode.sh" ;;
    2) SCRIPT="$SCRIPT_DIR/hold_2node.sh" ;;
    3) SCRIPT="$SCRIPT_DIR/hold_3node.sh" ;;
    *)
        echo "usage: $0 [1|2|3]" >&2
        exit 2
        ;;
esac

echo "Submitting Kimi serving hold to queue=debug nodes=$NODES"
qsub "$SCRIPT"
