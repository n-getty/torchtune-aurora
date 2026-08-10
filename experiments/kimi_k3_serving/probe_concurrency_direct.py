#!/usr/bin/env python
"""Concurrency ladder over raw HTTP, bypassing `vllm bench serve`.

Why not just use the bench client: on the 48B surrogate it returns
"Service Unavailable" for every request while plain `/v1/completions` POSTs to
the same server succeed, so its failures say nothing about the server. This
probe issues the identical requests directly and reports the same quantities.

Reports both numbers that matter and are easy to conflate:
  * aggregate tok/s  -- total completion tokens / wall clock, the throughput
    figure comparable across configurations.
  * per-user tok/s   -- median of (tokens / that request's own latency), what
    one interactive user actually experiences. These diverge sharply under
    load: K3 at c=128 managed 65.2 aggregate but ~0.5 per user.

Also prints how many requests completed before any failure, because the
open `banned:1` fault on K3 kills the engine after a small, roughly fixed
number of completions -- so "how far did it get" is itself the measurement.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def one_request(base_url, model, prompt_tokens, max_tokens, timeout):
    # A synthetic prompt of roughly the requested token count. " word" is
    # ~1 token for this tokenizer, which is close enough for a load shape.
    prompt = "The capital of France is" + " word" * max(0, prompt_tokens - 5)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    url = base_url.rstrip("/")
    if not url.endswith("/v1/completions"):
        url = f"{url}/v1/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with OPENER.open(request, timeout=timeout) as response:
            body = json.loads(response.read())
        elapsed = time.perf_counter() - started
        used = body.get("usage", {}).get("completion_tokens", 0)
        return {"ok": True, "elapsed": elapsed, "tokens": used}
    except Exception as error:  # noqa: BLE001 - probe reports, never raises
        return {
            "ok": False,
            "elapsed": time.perf_counter() - started,
            "tokens": 0,
            "error": repr(error)[:200],
        }


def cell(url, model, concurrency, rounds, prompt_tokens, max_tokens, timeout):
    total = concurrency * rounds
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(
            pool.map(
                lambda _: one_request(url, model, prompt_tokens, max_tokens, timeout),
                range(total),
            )
        )
    wall = time.perf_counter() - started

    ok = [r for r in results if r["ok"]]
    tokens = sum(r["tokens"] for r in ok)
    per_user = [
        r["tokens"] / r["elapsed"] for r in ok if r["elapsed"] > 0 and r["tokens"]
    ]
    row = {
        "concurrency": concurrency,
        "requests": total,
        "successful": len(ok),
        "failed": total - len(ok),
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "wall_s": round(wall, 2),
        "aggregate_tok_s": round(tokens / wall, 3) if wall else 0.0,
        "per_user_tok_s": round(statistics.median(per_user), 3) if per_user else 0.0,
        "mean_latency_s": round(statistics.mean(r["elapsed"] for r in ok), 2)
        if ok
        else None,
    }
    first_error = next((r["error"] for r in results if not r["ok"]), None)
    if first_error:
        row["first_error"] = first_error
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--ladder", default="1,4,16,32")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--out", default="concurrency_ladder.json")
    args = parser.parse_args()

    rows = []
    print(f"{'conc':>5} {'ok':>5} {'fail':>5} {'wall_s':>8} "
          f"{'aggregate':>10} {'per_user':>9} {'lat_s':>7}")
    for concurrency in [int(c) for c in args.ladder.split(",")]:
        row = cell(args.url, args.model, concurrency, args.rounds,
                   args.prompt_tokens, args.max_tokens, args.timeout)
        rows.append(row)
        print(f"{row['concurrency']:>5} {row['successful']:>5} {row['failed']:>5} "
              f"{row['wall_s']:>8} {row['aggregate_tok_s']:>10} "
              f"{row['per_user_tok_s']:>9} {str(row['mean_latency_s']):>7}",
              flush=True)
        if row["failed"]:
            print(f"      first error: {row.get('first_error')}", flush=True)
            # Keep climbing anyway: on K3 the interesting datum is how many
            # requests completed before the engine died, per rung.
    with open(args.out, "w") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    good = [r for r in rows if r["successful"] and not r["failed"]]
    if good:
        best = max(good, key=lambda r: r["aggregate_tok_s"])
        print(f"\npeak aggregate {best['aggregate_tok_s']} tok/s at c="
              f"{best['concurrency']}; per-user there {best['per_user_tok_s']} tok/s")
    return 0 if all(not r["failed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
