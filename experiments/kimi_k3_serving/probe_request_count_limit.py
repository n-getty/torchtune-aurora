#!/usr/bin/env python3
"""Is there a hard limit on how many requests this server survives?

WHY
---
Across THREE separate topologies (tp1, tp2, tp8ep) the engine died after
EXACTLY 29 successful completions, on the 30th request. Doubling the KV cache
(8 -> 16 blocks, 1024 -> 2048 tokens) did NOT move the number, which refutes
the cache-starvation explanation I initially committed.

29 is also exactly where the single-step probe finishes and the depth ladder
makes its first call, so two explanations remain and they are easy to confuse:

  (a) CUMULATIVE  -- the engine degrades with request count and dies at ~30
                     regardless of what the 30th request is.
  (b) TRIGGERED   -- something specific about the ladder's first request kills
                     it, and the count is a coincidence.

This distinguishes them with the dumbest possible experiment: fire N identical
trivial requests at a fresh server and see whether it dies near 30. Same
max_tokens, same logprobs, same everything -- no ladder involved.

  dies near 30      -> (a) cumulative. A resource leak per request. The probes
                       are innocent; the engine cannot sustain a long session.
  survives 60+      -> (b) the ladder's request is the trigger, and the next
                       step is to bisect what is different about it.

Either answer is worth having, and neither costs an allocation beyond a server
that is already up.

Usage:
  probe_request_count_limit.py <base_url> <model> [n_requests]
"""

import json
import sys
import time
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, payload, timeout=120):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read())


def main():
    base = sys.argv[1]
    model = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    print(f"firing {n} identical trivial requests at {base}")
    print(f"{'#':>4} {'elapsed':>8}  status")
    for i in range(1, n + 1):
        started = time.time()
        try:
            result = post(
                f"{base}/v1/completions",
                {
                    "model": model,
                    "prompt": "The capital of France is",
                    "max_tokens": 2,
                    "temperature": 0,
                    "logprobs": 20,
                    "return_tokens_as_token_ids": True,
                },
            )
            took = time.time() - started
            ok = "choices" in result
            # Print every request near the suspected boundary, and a sample
            # elsewhere, so a slow drift is visible without 60 lines of noise.
            if i <= 3 or 25 <= i <= 35 or i % 10 == 0:
                print(f"{i:>4} {took:>7.2f}s  {'ok' if ok else 'NO CHOICES: ' + str(result)[:80]}")
            if not ok:
                print(f"\nFAILED at request {i} (server returned an error body)")
                return 1
        except Exception as error:
            took = time.time() - started
            print(f"{i:>4} {took:>7.2f}s  EXCEPTION {type(error).__name__}: {str(error)[:90]}")
            print(f"\nDIED at request {i}")
            if 25 <= i <= 35:
                print("=> consistent with the ~30-request limit seen at tp1, tp2 and")
                print("   tp8ep. CUMULATIVE: the engine degrades with request count,")
                print("   independent of what the request is. The probes are innocent.")
            else:
                print("=> does NOT match the ~30 boundary; different failure.")
            return 1

    print(f"\nsurvived all {n} requests.")
    print("=> the ~30-request deaths were NOT a simple per-request limit;")
    print("   something specific to the depth ladder's request is the trigger.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
