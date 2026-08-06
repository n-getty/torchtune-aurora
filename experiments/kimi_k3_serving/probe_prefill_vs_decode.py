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
    parser.add_argument(
        "--flat-band-nats",
        type=float,
        default=1.0,
        help="treat a prompt as INCONCLUSIVE when its whole top-k lies within "
        "this many nats of the top-1 (near-tied => top-k membership is "
        "arbitrary). 1.0 nat still admits any genuinely peaked distribution.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    verdicts = []

    for prompt in PROMPTS:
        # Construct the pair from the model's OWN sample rather than hoping it
        # reproduces a token we removed.
        #
        # An earlier design split the prompt at the last word and hoped the
        # short prompt would sample the removed word back. With random weights
        # (or any imperfect model) it essentially never does, so every prompt
        # got skipped. Worse, on a real model it would silently bias the probe
        # toward prompts the model already predicts well.
        #
        # This construction is always valid:
        #   run A: prompt = T,       max_tokens=2
        #          position 0 <- prefill over T; call the sampled token X
        #          position 1 <- DECODE step, context T+[X]
        #   run B: prompt = T+[X],   max_tokens=1
        #          position 0 <- PREFILL over T+[X]
        # A[1] and B[0] condition on exactly the same token sequence T+[X],
        # one reached by decoding and one by prefilling. That is precisely the
        # prefill-vs-decode question, with no assumption about what the model
        # predicts. Token ids are passed straight through, so there is no
        # detokenize/retokenize round trip to corrupt the bridge.
        tokenized = post(
            f"{args.base_url}/tokenize",
            {"model": args.model, "prompt": prompt, "add_special_tokens": False},
        )
        base_ids = tokenized["tokens"]

        # run A: prefill over T, then one decode step.
        decode = [
            completion(args.base_url, args.model, base_ids, 2)
            for _ in range(args.repeats)
        ]
        bridge_str = decode[0]["choices"][0]["logprobs"]["tokens"][0]
        bridge_id = int(bridge_str.split(":")[-1])

        # run B: pure prefill over T+[X].
        extended_ids = base_ids + [bridge_id]
        prefill = [
            completion(args.base_url, args.model, extended_ids, 1)
            for _ in range(args.repeats)
        ]

        # Guard against silent divergence in what the two runs actually saw.
        if decode[0]["usage"]["prompt_tokens"] + 1 != prefill[0]["usage"][
            "prompt_tokens"
        ]:
            print(
                f"\nprompt: {prompt!r}\n"
                f"  SKIP: prompt lengths inconsistent "
                f"(decode {decode[0]['usage']['prompt_tokens']}, "
                f"prefill {prefill[0]['usage']['prompt_tokens']})."
            )
            continue

        prefill_tokens = prefill[0]["choices"][0]["logprobs"]["tokens"]
        bridge = bridge_str

        # Same-path repeat noise: the control. Any cross-path gap must be
        # judged against this, not against zero, because the XPU MLA/KDA path
        # has no batch invariance. Measure it on BOTH paths and take the
        # larger: the decode path plausibly carries more reduction-order noise
        # than the prefill path, and using only the prefill figure would set
        # the bar too low and manufacture a DIVERGE.
        noise = []
        for i in range(1, len(prefill)):
            _, _, mx, _ = compare(top_at(prefill[0], 0), top_at(prefill[i], 0))
            noise.append(mx)
        for i in range(1, len(decode)):
            _, _, mx, _ = compare(top_at(decode[0], 1), top_at(decode[i], 1))
            noise.append(mx)
        noise_max = max(noise) if noise else 0.0

        # The cross-path comparison: position 0 of run B (a pure PREFILL over
        # T+[X]) vs position 1 of run A (a DECODE step whose context is T+[X]).
        argmax_match, jaccard, max_diff, mean_diff = compare(
            top_at(prefill[0], 0), top_at(decode[0], 1)
        )

        # Flatness guard. On a near-uniform distribution the top-k ordering is
        # decided by tie-breaking, not by the model, so argmax_match and the
        # top-k Jaccard both become meaningless and the probe cannot return a
        # verdict.
        #
        # The right diagnostic is DISTANCE FROM UNIFORM, not the adjacent-token
        # gap. Measured on the random-weight reduced K3: every top-20 logprob
        # sat between -9.04 and -9.60 against a uniform value of -11.93 over a
        # 151936-token vocab. That is thousands of near-tied tokens, so two
        # top-20 lists barely intersect (Jaccard 0.05) purely by sampling --
        # yet max |diff| on the SHARED support was only 0.037, below same-path
        # noise. An earlier version of this guard tested the top1-top2 gap
        # (0.375) and wrongly judged that distribution "peaked", producing a
        # false DIVERGE.
        top_prefill = top_at(prefill[0], 0)
        ranked = sorted(top_prefill.values(), reverse=True)
        top1_gap = (ranked[0] - ranked[1]) if len(ranked) > 1 else float("inf")
        # A distribution whose entire top-k lies within `flat_band_nats` of its
        # own maximum has no usable peak: the members are effectively tied and
        # which ones surface in the top-k is arbitrary.
        flat_band = ranked[0] - ranked[-1] if len(ranked) > 1 else float("inf")
        flat = (top1_gap <= noise_max) or (flat_band <= args.flat_band_nats)

        # Threshold floor. If a single set of repeats happens to measure zero
        # noise, 4*0 collapses the bar to 1e-3 and any genuine FP difference
        # trips it. Floor the noise estimate at the observed top-1 gap scale
        # so a lucky zero cannot manufacture a DIVERGE.
        effective_noise = max(noise_max, abs(top1_gap) if math.isfinite(top1_gap) else 0.0)
        threshold = max(args.tolerance_factor * effective_noise, 1e-3)

        if flat:
            diverged = False
            verdict = "INCONCLUSIVE (flat distribution)"
        else:
            diverged = (not argmax_match) or (max_diff > threshold)
            verdict = "DIVERGE" if diverged else "agree"
        verdicts.append(diverged)

        record = {
            "prompt": prompt,
            "bridge_token": bridge,
            "prefill_token": prefill_tokens[0],
            "same_path_noise_max": noise_max,
            "top1_gap": top1_gap,
            "flat_band": flat_band,
            "flat_distribution": flat,
            "effective_noise": effective_noise,
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
        print(f"  top1-top2 gap                : {top1_gap:.6f}")
        print(f"  threshold ({args.tolerance_factor}x eff. noise)  : {threshold:.6f}")
        print(f"  cross-path argmax match      : {argmax_match}")
        print(f"  cross-path top-k jaccard     : {jaccard:.3f}")
        print(f"  cross-path max abs diff      : {max_diff:.6f}")
        print(f"  -> {verdict}")
        if flat:
            print(
                "     (top-1 gap is within same-path noise: the ordering here "
                "is decided by FP noise, not the model, so no verdict is "
                "possible on this prompt)"
            )

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

    conclusive = [r for r in records if not r["flat_distribution"]]
    if not conclusive:
        print(
            f"\nVERDICT: INCONCLUSIVE (0/{len(records)} prompts had a usable "
            "signal). Every distribution was flat enough that FP noise decides "
            "the ordering. This is the expected outcome on a RANDOM-WEIGHT "
            "model: it says nothing about prefill-vs-decode. Run against real "
            "weights (the 48B, or a trained checkpoint) for a real verdict."
        )
        return 2

    if any(verdicts):
        print(
            f"\nVERDICT: DIVERGE ({sum(verdicts)}/{len(conclusive)} conclusive "
            f"prompts, {len(records) - len(conclusive)} inconclusive). "
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
