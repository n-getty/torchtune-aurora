#!/usr/bin/env python
"""Isolate the NaN-logprob failure that appears at concurrency >= 2.

Job 8745846 died with, from the SERVER side:

    ValueError: Out of range float values are not JSON compliant: nan
      vllm/entrypoints/openai/completion/api_router.py:61
    RuntimeError: concurrency 2 failed   (k3_generation_gate.py:290)

and the same signature killed job 8745694's gate, where it was filed as "an
external Ray-node SIGTERM". Single-request decode is clean. That is a
correctness signal in the batched path, not infrastructure flakiness -- but
before acting on it we need to know WHICH of three very different things it is:

  (A) The logits themselves go NaN when the batch has >1 sequence. Sampled
      TEXT would then also be corrupt, and every batched throughput number
      is meaningless.
  (B) The logits are fine and only the returned logprob VALUES are NaN
      (e.g. a masked/padded slot in the batched top-k gather being read back).
      Text stays correct; the bug is confined to the logprobs response path,
      which the throughput probe never requests.
  (C) Not concurrency at all -- just the second request in a process, or the
      `logprobs` parameter on its own.

The three cells below separate those. They are deliberately ordered cheapest
first and each is independent, so a crash in one still leaves the earlier
verdicts on disk.

  cell 1  c=1, logprobs on   -- baseline; must pass or nothing else means anything
  cell 2  c=2, logprobs OFF  -- does batching alone corrupt TEXT? (A vs B)
  cell 3  c=2, logprobs on   -- reproduce the reported failure
  cell 4  c=1, logprobs on, run twice sequentially -- rules out (C)

Usage:
    python probe_batched_nan_logprobs.py --url http://127.0.0.1:8000 \
        --model Kimi-K3 --out results.json
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

PROMPTS = [
    "The capital of France is",
    "One two three four five six seven",
]


def post(url, payload, timeout=600):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with OPENER.open(request, timeout=timeout) as response:
            raw = response.read() or b"{}"
            try:
                return response.status, json.loads(raw), None
            except json.JSONDecodeError:
                # A NaN in the payload makes the SERVER fail to serialize, so
                # the interesting case can arrive as unparseable text.
                return response.status, None, raw.decode(errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, None, error.read().decode(errors="replace")
    except Exception as error:  # noqa: BLE001 - probe reports, never raises
        return None, None, repr(error)


def make_payload(model, prompt, max_tokens, logprobs):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 123,
        "stream": False,
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
    return payload


def scan_nonfinite(value, path="response"):
    """Every non-finite leaf, with its JSON path. -inf is NOT the same bug."""
    hits = []
    if isinstance(value, float) and not math.isfinite(value):
        hits.append((path, value))
    elif isinstance(value, dict):
        for key, item in value.items():
            hits.extend(scan_nonfinite(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(scan_nonfinite(item, f"{path}[{index}]"))
    return hits


def looks_degenerate(text):
    """K3's known failure mode is a correct first token then repeated garbage."""
    if not text:
        return True, "empty"
    words = text.split()
    if len(words) >= 6:
        tail = words[-6:]
        if len(set(tail)) == 1:
            return True, f"last 6 tokens all {tail[0]!r}"
    non_ascii = sum(1 for character in text if ord(character) > 0x2E80)
    if non_ascii > max(4, 0.3 * len(text)):
        return True, f"{non_ascii} CJK-range chars in {len(text)} chars"
    return False, ""


def run_cell(name, url, model, prompts, max_tokens, logprobs):
    print(f"\n=== {name}: concurrency={len(prompts)} logprobs={logprobs} ===",
          flush=True)
    endpoint = f"{url}/v1/completions"
    started = time.time()
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        outcomes = list(
            pool.map(
                lambda prompt: post(
                    endpoint, make_payload(model, prompt, max_tokens, logprobs)
                ),
                prompts,
            )
        )
    elapsed = time.time() - started

    cell = {
        "name": name,
        "concurrency": len(prompts),
        "logprobs": logprobs,
        "max_tokens": max_tokens,
        "elapsed_seconds": elapsed,
        "requests": [],
        "verdict": "PASS",
    }
    for prompt, (status, body, raw) in zip(prompts, outcomes):
        record = {"prompt": prompt, "status": status}
        if body is None:
            record["unparseable_response"] = (raw or "")[:2000]
            # The server itself refused to serialize -> NaN reached the wire.
            record["nan_in_response"] = "nan" in (raw or "").lower()
            cell["verdict"] = "FAIL"
        else:
            text = (body.get("choices") or [{}])[0].get("text", "")
            record["text"] = text
            degenerate, why = looks_degenerate(text)
            record["degenerate"] = degenerate
            if degenerate:
                record["degenerate_reason"] = why
                cell["verdict"] = "FAIL"
            nonfinite = scan_nonfinite(body)
            record["nonfinite"] = [
                {"path": path, "value": repr(value)} for path, value in nonfinite
            ]
            # -inf on a zero-probability token is legitimate; NaN never is.
            if any(math.isnan(value) for _, value in nonfinite):
                record["nan_logprobs"] = True
                cell["verdict"] = "FAIL"
        cell["requests"].append(record)
        print(f"  status={record['status']} verdict_so_far={cell['verdict']} "
              f"text={record.get('text', record.get('unparseable_response', ''))[:90]!r}",
              flush=True)
    print(f"  -> {cell['verdict']} in {elapsed:.1f}s", flush=True)
    return cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Kimi-K3")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--out", default="batched_nan_probe.json")
    args = parser.parse_args()

    cells = []
    cells.append(run_cell("cell1_c1_logprobs_on", args.url, args.model,
                          PROMPTS[:1], args.max_tokens, 5))
    cells.append(run_cell("cell2_c2_logprobs_off", args.url, args.model,
                          PROMPTS, args.max_tokens, None))
    cells.append(run_cell("cell3_c2_logprobs_on", args.url, args.model,
                          PROMPTS, args.max_tokens, 5))
    cells.append(run_cell("cell4_c1_logprobs_on_repeat", args.url, args.model,
                          PROMPTS[1:2], args.max_tokens, 5))

    by_name = {cell["name"]: cell["verdict"] for cell in cells}
    if by_name["cell1_c1_logprobs_on"] != "PASS":
        conclusion = "INDETERMINATE: the c=1 baseline itself failed"
    elif by_name["cell3_c2_logprobs_on"] == "PASS":
        conclusion = "NOT REPRODUCED at c=2 with logprobs"
    elif by_name["cell2_c2_logprobs_off"] == "PASS":
        conclusion = (
            "(B) logprobs-only: batched TEXT is clean, NaN is confined to the "
            "logprob response path -- throughput numbers (no logprobs) stand"
        )
    else:
        conclusion = (
            "(A) batched logits are corrupt: TEXT degrades at c=2 even without "
            "logprobs -- every batched throughput number is suspect"
        )
    if by_name["cell4_c1_logprobs_on_repeat"] != "PASS":
        conclusion += " | NOTE: a second sequential c=1 request also failed, so "
        conclusion += "this may be request-ordinal, not concurrency"

    print(f"\nCONCLUSION: {conclusion}", flush=True)
    with open(args.out, "w") as handle:
        json.dump({"cells": cells, "conclusion": conclusion}, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.out}", flush=True)
    return 0 if all(cell["verdict"] == "PASS" for cell in cells) else 1


if __name__ == "__main__":
    sys.exit(main())
