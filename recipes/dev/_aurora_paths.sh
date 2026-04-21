#!/bin/bash
# Shared Aurora launcher path bootstrap.
#
# Resolves this checkout as the torchtune root unless TORCHTUNE_DIR is already
# provided, and keeps the external TRL checkout configurable via TRL_DIR.

if [[ -z "${TORCHTUNE_DIR:-}" ]]; then
    _aurora_paths_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    TORCHTUNE_DIR="$(cd -- "${_aurora_paths_dir}/../.." && pwd)"
fi
export TORCHTUNE_DIR

export LOG_DIR="${LOG_DIR:-${TORCHTUNE_DIR}/logs}"

if [[ -z "${TRL_DIR:-}" ]] && [[ -d "/flare/ModCon/ngetty/trl" ]]; then
    TRL_DIR="/flare/ModCon/ngetty/trl"
fi
export TRL_DIR

aurora_pythonpath() {
    local combined=""
    local path=""

    for path in "$@"; do
        [[ -n "${path}" ]] || continue
        [[ -d "${path}" ]] || continue
        if [[ -n "${combined}" ]]; then
            combined="${combined}:${path}"
        else
            combined="${path}"
        fi
    done

    if [[ -n "${PYTHONPATH:-}" ]]; then
        if [[ -n "${combined}" ]]; then
            combined="${combined}:${PYTHONPATH}"
        else
            combined="${PYTHONPATH}"
        fi
    fi

    printf '%s' "${combined}"
}

aurora_export_pythonpath() {
    export PYTHONPATH
    PYTHONPATH="$(aurora_pythonpath "$@")"
}
