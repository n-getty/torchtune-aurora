#!/usr/bin/env python
"""Explain a K3 KV-cache pool size from the per-rank block-profile records.

vLLM prints one "GPU KV cache size: N tokens" line and one "Available KV cache
memory: X GiB" line per *unique message* (info_once), so on a 32-rank job you
see ~3 memory numbers and no way to tell which rank set the pool size. That
matters because `get_kv_cache_configs` clamps every rank to `min_num_blocks`
BEFORE `initialize_from_config` writes these records -- so all ranks store the
same post-clamp `num_blocks`, and a single starved rank is invisible.

This reads K3_BLOCK_PROFILE_DIR/rank_*.json and reports, per rank, the
num_blocks each rank's own available_memory WOULD have supported
(available // page_size // max_layers_per_group), so the rank that actually
set the floor is named.

Usage:  analyze_block_profile.py <block_profile_dir>
"""

import json
import os
import sys

GIB = 1024**3


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    directory = sys.argv[1]
    names = sorted(
        (name for name in os.listdir(directory) if name.startswith("rank_")),
        key=lambda name: int(name.split("_")[1].split(".")[0]),
    )
    if not names:
        print(f"no rank_*.json in {directory}")
        return 1

    records = []
    for name in names:
        with open(os.path.join(directory, name)) as handle:
            records.append(json.load(handle))

    print(f"{len(records)} ranks in {directory}\n")
    header = (
        f"{'rank':>4} {'host':<16} {'blocks':>7} {'avail GiB':>10} "
        f"{'page B':>10} {'grp':>4} {'lyr/grp':>8} {'own blocks':>11} {'alloc GiB':>10}"
    )
    print(header)
    print("-" * len(header))

    own_blocks_by_rank = {}
    for record in records:
        rank = record["rank"]
        avail = record.get("available_kv_cache_memory_bytes")
        pages = record.get("page_size_bytes") or []
        page = pages[0] if pages else None
        per_group = record.get("max_layers_per_group") or 0
        # What this rank alone could have afforded, i.e. its pre-clamp value.
        own = (
            avail // page // per_group
            if (avail and page and per_group)
            else None
        )
        own_blocks_by_rank[rank] = own
        print(
            f"{rank:>4} {record.get('hostname', '?')[:16]:<16} "
            f"{record['profiled_num_gpu_blocks']:>7} "
            f"{(avail / GIB if avail else float('nan')):>10.2f} "
            f"{(page if page else 0):>10} "
            f"{record.get('num_groups', 0):>4} {per_group:>8} "
            f"{(own if own is not None else -1):>11} "
            f"{(record.get('kv_cache_tensor_total_bytes', 0) / GIB):>10.2f}"
        )

    effective = {record["profiled_num_gpu_blocks"] for record in records}
    print(f"\npost-clamp num_blocks across ranks: {sorted(effective)}")

    known = {r: b for r, b in own_blocks_by_rank.items() if b is not None}
    if not known:
        print(
            "\nNo available_kv_cache_memory_bytes in these records -- they predate\n"
            "the gpu_worker.py instrumentation. Re-run the server to populate it."
        )
        return 0

    floor_rank = min(known, key=lambda r: known[r])
    floor, ceiling = known[floor_rank], max(known.values())
    print(f"pre-clamp own-blocks: min={floor} (rank {floor_rank}), max={ceiling}")
    if floor == ceiling:
        print(
            "\nVERDICT: every rank computed the SAME num_blocks. The pool is not\n"
            "         limited by one starved rank -- the divisor or the profiled\n"
            "         available_memory is uniformly wrong. Compare 'avail GiB' to\n"
            "         'alloc GiB': if alloc << avail, the memory was never used."
        )
    else:
        print(
            f"\nVERDICT: rank {floor_rank} set the floor at {floor} blocks while the\n"
            f"         best rank could afford {ceiling} ({ceiling / max(floor, 1):.1f}x more).\n"
            f"         The pool is capped by ONE rank -- investigate its\n"
            f"         non_kv_cache_memory (weights + peak activation) on\n"
            f"         host {next(r['hostname'] for r in records if r['rank'] == floor_rank)}."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
