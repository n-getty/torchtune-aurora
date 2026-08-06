#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


PROMPTS = (
    "The capital of France is",
    "Complete this sentence with punctuation: Aurora is a supercomputer",
    "Count upward using words: one, two, three,",
    "Explain in one sentence why water freezes when its temperature falls below zero degrees Celsius.",
)

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def require_finite_json(value, path="response"):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite numeric value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json(item, f"{path}[{index}]")


def request_json(url, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with OPENER.open(request, timeout=timeout) as response:
            body = response.read() or b"{}"
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, {"error": "invalid_json", "raw": body.decode(errors="replace")}
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body.decode(errors="replace")}
        return error.code, parsed
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return None, {"error": repr(error)}


def completion_payload(model, prompt, max_tokens, logprobs=5):
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 123,
        "logprobs": logprobs,
        "return_tokens_as_token_ids": True,
        "stream": False,
    }


def completion(url, model, prompt, max_tokens, logprobs=5):
    started = time.time()
    status, body = request_json(
        f"{url}/v1/completions", completion_payload(model, prompt, max_tokens, logprobs)
    )
    if status == 200:
        require_finite_json(body)
    return {
        "status": status,
        "elapsed_seconds": time.time() - started,
        "request": completion_payload(model, prompt, max_tokens, logprobs),
        "response": body,
    }


def signature(result):
    choice = result["response"]["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return choice.get("text"), logprobs.get("tokens")


def first_token_signature(result):
    """Repeat-stable signature for the first sampled token.

    Deliberately excludes raw logprob VALUES and top-k set membership.
    vLLM implements supports_batch_invariance() only for flash_attn /
    flex_attention / triton_attn -- not the XPU MLA/KDA path K3 uses -- so
    repeated identical requests legitimately differ by FP-nonassociative
    reduction order under EP=32.

    Measured across production repeats of an identical prompt
    (experiments/kimi_k3_serving/logs/k3_tp32_prod_*/gate/first_token.json):

      argmax token      stable 5/5   <- gated here
      output text       stable 5/5   <- gated via require_identical_text_bytes
      top-5 set         stable 4/5   (job 8737024 swapped members 459 / 2791)
      logprob values    spread up to 0.37 nats

    A correct model fails a value-equality or set-equality bar on this
    hardware, so neither may gate. Finiteness IS still enforced: NaN/inf
    logits are a real defect, distinct from reduction-order jitter.
    """
    choice = result["response"]["choices"][0]
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens") or []
    token_logprobs = logprobs.get("token_logprobs") or []
    top_logprobs = logprobs.get("top_logprobs") or []
    top = top_logprobs[0] if top_logprobs else {}
    if (
        not tokens
        or not token_logprobs
        or not math.isfinite(token_logprobs[0])
        or len(top) < 5
        or any(not math.isfinite(value) for value in top.values())
    ):
        raise RuntimeError("first-token logprob is missing or non-finite")
    return (tokens[0],)


def require_identical_signatures(results, label):
    signatures = [signature(result) for result in results]
    if len({json.dumps(value, sort_keys=True) for value in signatures}) != 1:
        raise RuntimeError(f"{label} results differ")


def require_identical_text_bytes(results, label):
    outputs = [result["response"]["choices"][0]["text"].encode("utf-8") for result in results]
    if len(set(outputs)) != 1:
        raise RuntimeError(f"{label} output bytes differ")


def require_not_degenerate(result, label, min_tokens=8):
    """Reject the observed K3 decode failure: collapse to a repeated token.

    Reproducibility checks alone CANNOT catch this. The broken TP=32 run
    degenerated to runs of newline (198) and to ' of of of' *identically*
    across repeats -- a deterministically broken model passes every
    self-consistency bar. This asserts the output carries signal.

    Thresholds are calibrated against recorded artifacts, not guessed:

      known-good (48B, 36 completions, logs/kimi_decode_repro_8740405/)
          top_share 0.12-0.50, distinct_ratio 0.25-1.00
      known-broken (K3 TP=32, 6 completions, logs/k3_daos_87352*/87346*)
          top_share 0.44-0.94, distinct_ratio 0.12-0.50

    Neither metric separates alone -- both ranges overlap at top_share 0.50.
    Jointly they do: every good sample at top_share>=0.5 has distinct_ratio
    >=0.56, while broken ones there sit at <=0.31. Hence the AND.

    Caveat: calibrated on a small sample (36 good / 6 broken) from one 48B
    model, and it catches 5 of the 6 known-broken cases -- the miss is an
    alternating two-token pattern at top_share 0.44. This is a floor that
    catches gross collapse, NOT a proof of output quality. Legitimate text
    can repeat tokens (enumerations, indentation), so it is deliberately
    tuned to avoid false positives at the cost of missing marginal cases.
    """
    choice = result["response"]["choices"][0]
    tokens = (choice.get("logprobs") or {}).get("tokens") or []
    if len(tokens) < min_tokens:
        return
    counts = {token: tokens.count(token) for token in set(tokens)}
    top_share = max(counts.values()) / len(tokens)
    distinct_ratio = len(counts) / len(tokens)
    if len(counts) == 1:
        raise RuntimeError(
            f"{label} degenerate: all {len(tokens)} sampled tokens identical "
            f"({tokens[0]})"
        )
    if top_share >= 0.5 and distinct_ratio <= 0.5:
        raise RuntimeError(
            f"{label} degenerate: most common token covers {top_share:.0%} of "
            f"{len(tokens)} tokens with only {len(counts)} distinct "
            f"(distinct_ratio {distinct_ratio:.2f})"
        )


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Kimi-K3")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-metadata", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = None
    if args.run_metadata is not None:
        run_metadata = json.loads(args.run_metadata.read_text())

    health_status, health_body = request_json(f"{args.base_url}/health", timeout=30)
    save(args.output_dir / "health.json", {"status": health_status, "response": health_body})
    if health_status != 200:
        raise RuntimeError(f"health failed: HTTP {health_status}")

    first_token = [
        completion(args.base_url, args.model, PROMPTS[0], 1) for _ in range(3)
    ]
    save(args.output_dir / "first_token.json", first_token)
    if any(result["status"] != 200 for result in first_token):
        raise RuntimeError("first-token request failed")
    first_signatures = [first_token_signature(result) for result in first_token]
    if len({json.dumps(value, sort_keys=True) for value in first_signatures}) != 1:
        raise RuntimeError("first-token results differ")

    deterministic = [
        completion(args.base_url, args.model, PROMPTS[0], 16) for _ in range(3)
    ]
    save(args.output_dir / "deterministic.json", deterministic)
    if any(result["status"] != 200 for result in deterministic):
        raise RuntimeError("deterministic request failed")
    require_identical_signatures(deterministic, "deterministic")

    short_decode = [
        completion(args.base_url, args.model, PROMPTS[3], 4) for _ in range(3)
    ]
    save(args.output_dir / "decode_4.json", short_decode)
    if any(result["status"] != 200 for result in short_decode):
        raise RuntimeError("4-token decode failed")
    require_identical_signatures(short_decode, "4-token decode")

    long_decode = [
        completion(args.base_url, args.model, PROMPTS[3], 128) for _ in range(3)
    ]
    save(args.output_dir / "decode_128.json", long_decode)
    if any(result["status"] != 200 for result in long_decode):
        raise RuntimeError("128-token decode failed")
    require_identical_signatures(long_decode, "128-token")
    require_identical_text_bytes(long_decode, "128-token")
    for index, result in enumerate(long_decode):
        require_not_degenerate(result, f"128-token repeat {index}")

    fixed_prompts = [
        completion(args.base_url, args.model, prompt, 32) for prompt in PROMPTS
    ]
    save(args.output_dir / "fixed_prompts.json", fixed_prompts)
    if any(result["status"] != 200 for result in fixed_prompts):
        raise RuntimeError("fixed-prompt correctness request failed")
    for prompt, result in zip(PROMPTS, fixed_prompts):
        require_not_degenerate(result, f"fixed prompt {prompt[:32]!r}")

    concurrency_results = {}
    for concurrency in (1, 2, 4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    completion, args.base_url, args.model, PROMPTS[0], 32
                )
                for index in range(concurrency)
            ]
        results = [future.result() for future in futures]
        concurrency_results[str(concurrency)] = results
        if any(result["status"] != 200 for result in results):
            raise RuntimeError(f"concurrency {concurrency} failed")
        require_identical_signatures(results, f"concurrency {concurrency}")
    save(args.output_dir / "concurrency.json", concurrency_results)
    save(
        args.output_dir / "gate.json",
        {
            "status": "pass",
            "model": args.model,
            "run_metadata": run_metadata,
            "deterministic_requests": 3,
            "short_decode_tokens": 4,
            "short_decode_requests": 3,
            "first_token_requests": 3,
            "decode_tokens": 128,
            "decode_requests": 3,
            "concurrency": [1, 2, 4],
            "fixed_prompts": len(PROMPTS),
        },
    )
    print(json.dumps({"phase": "generation_gate", "status": "pass"}))


if __name__ == "__main__":
    main()
