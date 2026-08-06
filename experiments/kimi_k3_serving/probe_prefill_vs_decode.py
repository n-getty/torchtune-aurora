#!/usr/bin/env python3
"""The decisive K3 experiment: does a token's logit depend on HOW it was reached?

THE DISCRIMINATOR
-----------------
K3 emits a correct FIRST token (produced by a prefill over the whole prompt)
and then degenerates from token 2 onward (each produced by a decode step with
num_tokens=1). Every component-level suspect has been eliminated -- router,
attention residual, SiTU, latent-MoE reductions, and the fused MoE at M=1 (all
verified against the HF reference or on hardware). What has NOT been tested is
the thing the symptom actually points at: whether the prefill path and the
decode path agree.

This probe tests exactly that, and nothing else:

    (a) prefill  : send prompt P (N tokens), read the logits for position N-1
    (b) decode   : send prompt P[:-1] (N-1 tokens) with max_tokens=2, so the
                   model prefills N-1 tokens and then DECODES one step whose
                   input is P[-1]

Both compute the distribution for the same next-token position, conditioned on
the same context. A correct implementation must agree to reduction-order noise.
If they disagree, the prefill/decode divergence is real and localized to a
step, which is a far stronger handle than "the output is degenerate".

HOW TO READ THE RESULT
----------------------
  DIVERGE -> the bug is reproduced OUTSIDE a 3-node capacity load. Bisect by
             layer next (hidden-state sums), then by component within a layer.
  AGREE   -> prefill and decode agree at this scale/topology. This does NOT
             clear K3: the reduced harness runs single-node, so it cannot see
             a TP=32/EP-on defect. The follow-up is to scale the REDUCED
             model's TP/EP degree, not to spend another 1.5 TiB load.

An AGREE is a null result. Report it as one.

WHY LOGPROBS AND NOT TEXT
-------------------------
Text comparison only sees the argmax and hides sub-threshold drift. Comparing
the top-k logprob vectors at the same position detects divergence long before
it flips a token, which matters because K3's first WRONG token may be only
slightly wrong.

Bit-exactness is NOT the bar: vLLM implements supports_batch_invariance() only
for flash_attn / flex_attention / triton_attn, not the XPU MLA/KDA path, so
repeated identical requests legitimately differ by FP reduction order. The
probe therefore measures the prefill-vs-decode gap against the SAME-PATH
repeat noise (a built-in control) and only calls DIVERGE when the gap is
clearly larger.

USAGE
-----
    python3 probe_prefill_vs_decode.py --base-url http://127.0.0.1:8077 \
        --model reduced-k3 --output-dir logs/<run>/prefill_vs_decode

On Aurora, note no_proxy lists localhost but NOT 127.0.0.1; this script builds
an opener that bypasses the proxy explicitly (see OPENER).
"""

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

# Aurora's no_proxy covers "localhost" but not "127.0.0.1", so a plain urlopen
# to a local server gets intercepted by the ALCF proxy and returns an HTML
# error page. Force a direct connection.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

PROMPTS = [
    "The capital of France is Paris and the capital of Germany is",
    "One two three four five six seven",
    "def add(a, b):\n    return a +",
]


def post(url, payload, timeout=300):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read())


def completion(base_url, model, prompt, max_tokens, logprobs=20):
    return post(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "logprobs": logprobs,
            "return_tokens_as_token_ids": True,
            "stream": False,
        },
    )


def top_at(result, index):
    """top-k logprob dict for the token sampled at position `index`."""
    logprobs = result["choices"][0].get("logprobs") or {}
    tops = logprobs.get("top_logprobs") or []
    if len(tops) <= index:
        raise RuntimeError(f"no logprobs at index {index} (got {len(tops)})")
    return tops[index]


def compare(a, b):
    """Compare two top-k logprob dicts over their shared support.

    Returns (argmax_match, jaccard, max_abs_diff, mean_abs_diff).
    """
    argmax_a = max(a, key=a.get)
    argmax_b = max(b, key=b.get)
    shared = set(a) & set(b)
    union = set(a) | set(b)
    diffs = [abs(a[k] - b[k]) for k in shared if math.isfinite(a[k]) and math.isfinite(b[k])]
    return (
        argmax_a == argmax_b,
        len(shared) / len(union) if union else 0.0,
        max(diffs) if diffs else float("inf"),
        # Plain arithmetic rather than statistics.fmean: that needs py3.8+ and
        # Aurora's default login-node python is 3.6, so importing it works but
        # calling it does not.
        (sum(diffs) / len(diffs)) if diffs else float("inf"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8077")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--tolerance-factor",
        type=float,
        default=4.0,
        help="DIVERGE if the prefill-vs-decode gap exceeds this multiple of "
        "the measured same-path repeat noise",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    verdicts = []

    for prompt in PROMPTS:
        # Split the prompt so the last whitespace-delimited word becomes the
        # decode step's input token. Using a word boundary keeps both requests
        # tokenizing the shared prefix identically, which a mid-token split
        # would not.
        head, _, tail = prompt.rpartition(" ")
        if not head:
            print(f"skip (no split point): {prompt!r}")
            continue

        # (a) FULL prompt, 1 token. Position 0 is produced by a PREFILL over
        #     all N tokens.
        prefill = [
            completion(args.base_url, args.model, prompt, 1)
            for _ in range(args.repeats)
        ]

        # (b) SHORT prompt (N-1 tokens), 2 tokens. Position 0 comes from a
        #     prefill over N-1 tokens; position 1 comes from a DECODE step
        #     whose single input token is whatever was sampled at position 0.
        #
        #     The comparison is only valid if that sampled token equals the
        #     word we removed -- otherwise the two runs condition on different
        #     contexts and any difference is expected, not a bug. We cannot
        #     force the sample, so we CHECK it and skip the prompt when it
        #     does not match. Skipping is the honest move: a mismatched pair
        #     would silently manufacture a DIVERGE.
        decode = [
            completion(args.base_url, args.model, head, 2)
            for _ in range(args.repeats)
        ]

        prefill_tokens = prefill[0]["choices"][0]["logprobs"]["tokens"]
        decode_tokens = decode[0]["choices"][0]["logprobs"]["tokens"]
        bridge = decode_tokens[0]
        # The decode step's input must be the token the full prompt ends with.
        # Ask the server to tokenize `prompt` so we can name that token id.
        # add_special_tokens defaults to True and would append/prepend BOS,
        # which can make tokens[-1] not the real last content token.
        tokenized = post(
            f"{args.base_url}/tokenize",
            {"model": args.model, "prompt": prompt, "add_special_tokens": False},
        )
        expected_last = f"token_id:{tokenized['tokens'][-1]}"
        if bridge != expected_last:
            print(
                f"\nprompt: {prompt!r}\n"
                f"  SKIP: contexts would differ. The short prompt sampled "
                f"{bridge} at position 0, but the full prompt ends with "
                f"{expected_last}. Comparing them would test two different "
                f"conditionings, not prefill-vs-decode."
            )
            continue

        # Same-path repeat noise: the control. Any prefill-vs-decode gap must
        # be judged against this, not against zero.
        noise = []
        for i in range(1, len(prefill)):
            _, _, mx, _ = compare(top_at(prefill[0], 0), top_at(prefill[i], 0))
            noise.append(mx)
        noise_max = max(noise) if noise else 0.0

        # The cross-path comparison: position 0 of the full-prompt prefill vs
        # position 1 of the short-prompt run (which is a decode step).
        argmax_match, jaccard, max_diff, mean_diff = compare(
            top_at(prefill[0], 0), top_at(decode[0], 1)
        )

        threshold = max(args.tolerance_factor * noise_max, 1e-3)
        diverged = (not argmax_match) or (max_diff > threshold)
        verdicts.append(diverged)

        record = {
            "prompt": prompt,
            "bridge_token": bridge,
            "prefill_token": prefill_tokens[0],
            "same_path_noise_max": noise_max,
            "threshold": threshold,
            "cross_path_argmax_match": argmax_match,
            "cross_path_topk_jaccard": jaccard,
            "cross_path_max_abs_diff": max_diff,
            "cross_path_mean_abs_diff": mean_diff,
            "diverged": diverged,
        }
        records.append(record)

        print(f"\nprompt: {prompt!r}")
        print(f"  same-path repeat noise (max) : {noise_max:.6f}")
        print(f"  threshold ({args.tolerance_factor}x noise)      : {threshold:.6f}")
        print(f"  cross-path argmax match      : {argmax_match}")
        print(f"  cross-path top-k jaccard     : {jaccard:.3f}")
        print(f"  cross-path max abs diff      : {max_diff:.6f}")
        print(f"  -> {'DIVERGE' if diverged else 'agree'}")

    # json.dumps emits bare `Infinity` for float('inf'), which is invalid
    # strict JSON and fails to load back. Disjoint top-k support legitimately
    # produces inf here, so serialize it as a string instead of writing an
    # artifact that cannot be re-read.
    def jsonable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {k: jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [jsonable(v) for v in value]
        return value

    (args.output_dir / "prefill_vs_decode.json").write_text(
        json.dumps(jsonable(records), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    if not records:
        print("\nno comparable prompts", file=sys.stderr)
        return 2

    if any(verdicts):
        print(
            f"\nVERDICT: DIVERGE ({sum(verdicts)}/{len(verdicts)} prompts). "
            "Prefill and decode disagree beyond same-path noise -- the bug is "
            "reproduced outside a capacity load. Bisect by layer next."
        )
        return 1

    print(
        f"\nVERDICT: agree ({len(verdicts)}/{len(verdicts)} prompts). "
        "NULL RESULT, not an all-clear: this harness is single-node and cannot "
        "see a TP=32/EP-on defect. Next step is scaling the REDUCED model's "
        "TP/EP degree, not another 1.5 TiB load."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
