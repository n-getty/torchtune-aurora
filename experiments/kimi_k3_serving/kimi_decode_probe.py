#!/usr/bin/env python3
"""Decode-path probe for Kimi-Linear / Kimi-K3 vLLM servers.

Motivation: on Kimi-K3 (TP=32/EP-on) the *first* token is correct and
plausible (` Paris`, logprob -0.33, coherent top-5) while tokens 2+
degenerate into repetition and language-mixing. Prefill is therefore
substantially right and the defect lives in the decode-only XPU
fallbacks (`_forward_mqa_xpu` in triton_mla.py, and the KDA
`causal_conv1d_update` / recurrent-state branch in kda.py).

This probe is deliberately NOT a pass/fail acceptance gate. It records
what the model actually emits at increasing decode lengths so the first
divergent step can be located. Bit-exact repeatability is not required
and not checked as a failure: vLLM only implements batch invariance for
flash_attn/flex_attention/triton_attn, none of which is the XPU MLA/KDA
path, so small run-to-run logprob jitter is expected and benign.

What matters, and what this reports:
  * does greedy decode stay coherent past token 1?
  * do repeated identical requests produce identical *text*?
  * at which token index do repeated runs first differ?
"""

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

# Prompts with an unambiguous greedy continuation, so degeneration is
# obvious by inspection rather than by reference comparison.
PROMPTS = {
    "capital": "The capital of France is",
    "count": "Count upward using words: one, two, three,",
    "alphabet": "The English alphabet begins: a, b, c, d,",
}

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


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
                return response.status, {
                    "error": "invalid_json",
                    "raw": body.decode(errors="replace"),
                }
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body.decode(errors="replace")}
        return error.code, parsed
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return None, {"error": repr(error)}


def completion(url, model, prompt, max_tokens, logprobs=5):
    payload = {
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
    started = time.time()
    status, body = request_json(f"{url}/v1/completions", payload)
    return {
        "status": status,
        "elapsed_seconds": time.time() - started,
        "request": payload,
        "response": body,
    }


def extract(result):
    """Return (text, token_ids, token_logprobs) or None on a failed request."""
    if result["status"] != 200:
        return None
    try:
        choice = result["response"]["choices"][0]
    except (KeyError, IndexError):
        return None
    logprobs = choice.get("logprobs") or {}
    return (
        choice.get("text"),
        list(logprobs.get("tokens") or []),
        list(logprobs.get("token_logprobs") or []),
    )


def first_divergence(sequences):
    """Index of the first position where token sequences disagree, else None."""
    sequences = [s for s in sequences if s is not None]
    if len(sequences) < 2:
        return None
    for index in range(min(len(s) for s in sequences)):
        if len({s[index] for s in sequences}) != 1:
            return index
    if len({len(s) for s in sequences}) != 1:
        return min(len(s) for s in sequences)
    return None


def degenerate(token_ids):
    """Heuristic: flag obvious repetition collapse in a decoded sequence.

    Deliberately conservative. A legitimate enumeration ("one, two, three,
    ... fifty-one,") reuses separator and prefix tokens heavily and tripped an
    earlier unique-ratio test, so ratio alone is not evidence of collapse.
    What distinguishes real degeneration (K3's
    " of of of of of of of of") is a long run of the SAME token, or a short
    cycle repeating many times over.
    """
    if len(token_ids) < 8:
        return False
    tail = token_ids[1:]  # token 0 is known-good; judge the decode tail

    # Immediate repetition of a single token.
    longest_run = best_run = 1
    for previous, current in zip(tail, tail[1:]):
        longest_run = longest_run + 1 if current == previous else 1
        best_run = max(best_run, longest_run)
    if best_run >= 5:
        return True

    # Short cycle dominating the output (e.g. "of the of the of the ...").
    # Long-period looping is NOT included: a small model re-reciting a correct
    # multi-token pattern ("...x, y, z.\n\nThe English alphabet begins: a, b,
    # c...") on an open-ended prompt is ordinary behavior, not the token-level
    # collapse this probe exists to detect.
    for period in (1, 2, 3):
        if len(tail) < period * 6:
            continue
        matches = sum(
            1
            for index in range(len(tail) - period)
            if tail[index] == tail[index + period]
        )
        if matches / (len(tail) - period) > 0.9:
            return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--lengths",
        default="1,2,4,8,16,32,128",
        help="comma-separated max_tokens ladder",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]

    health_status, health_body = request_json(f"{args.base_url}/health", timeout=30)
    (args.output_dir / "health.json").write_text(
        json.dumps({"status": health_status, "response": health_body}, indent=2) + "\n"
    )
    if health_status != 200:
        raise SystemExit(f"health failed: HTTP {health_status}")

    records = []
    summary = []
    for prompt_name, prompt in PROMPTS.items():
        for max_tokens in lengths:
            results = [
                completion(args.base_url, args.model, prompt, max_tokens)
                for _ in range(args.repeats)
            ]
            records.append(
                {
                    "prompt_name": prompt_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "results": results,
                }
            )
            extracted = [extract(result) for result in results]
            failures = sum(1 for value in extracted if value is None)
            texts = [value[0] for value in extracted if value is not None]
            token_sequences = [value[1] for value in extracted if value is not None]
            row = {
                "prompt_name": prompt_name,
                "max_tokens": max_tokens,
                "http_failures": failures,
                "texts_identical": len(set(texts)) == 1 if texts else False,
                "first_divergent_token": first_divergence(token_sequences),
                "degenerate": (
                    degenerate(token_sequences[0]) if token_sequences else None
                ),
                "sample_text": texts[0] if texts else None,
                "distinct_texts": sorted(set(texts)),
            }
            summary.append(row)
            print(json.dumps({"phase": "decode_probe", **row}), flush=True)

    (args.output_dir / "decode_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "decode_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    # Headline verdict, for the log tail. A model whose decode path is
    # sound should stay non-degenerate and text-identical at 128 tokens.
    long_rows = [row for row in summary if row["max_tokens"] == max(lengths)]
    reproduces = any(
        row["degenerate"] or not row["texts_identical"] for row in long_rows
    )
    verdict = {
        "phase": "decode_probe_verdict",
        "model": args.model,
        "longest_decode": max(lengths),
        "decode_defect_reproduces": reproduces,
        "prompts_degenerate": [
            row["prompt_name"] for row in long_rows if row["degenerate"]
        ],
        "prompts_nondeterministic_text": [
            row["prompt_name"] for row in long_rows if not row["texts_identical"]
        ],
    }
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
