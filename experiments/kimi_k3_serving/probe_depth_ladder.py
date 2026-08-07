"""Prefill-vs-decode agreement as a function of DECODE DEPTH.

WHY THIS AND NOT JUST THE SINGLE-STEP PROBE
-------------------------------------------
probe_prefill_vs_decode.py compares ONE decode step against a prefill, and on
real K3 weights it agreed at TP=1 and TP=8+EP. But K3's actual symptom is
PROGRESSIVE: token 1 is correct, then output degenerates. A defect that drifts
cumulatively -- e.g. a KDA recurrent state that is slightly wrong per update --
would look clean at depth 1 and only diverge after many steps. Depth is the one
dimension the single-step probe structurally cannot see.

For each depth d: generate d+1 tokens from T, so the token at position d comes
from the (d+1)-th step -- a DECODE whose context is T + the d tokens already
produced. Then re-PREFILL that exact realized sequence and compare the
distribution for the same position. Divergence that grows with d is the
signature of an accumulating state bug; a flat profile rules it out.

Both branches condition on identical token sequences by construction (the
prefill is fed the decode run's own output), so no assumption is made about
what the model predicts.
"""
import json, sys, urllib.request

OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(u, p):
    r = urllib.request.Request(u, data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"},
                               method="POST")
    return json.loads(OP.open(r, timeout=300).read())


BASE, MODEL = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None
PROMPT = "The capital of France is Paris and the capital of Germany is"
ids = post(f"{BASE}/tokenize",
           {"model": MODEL, "prompt": PROMPT, "add_special_tokens": False})["tokens"]


def comp(prompt_ids, n):
    return post(f"{BASE}/v1/completions",
                {"model": MODEL, "prompt": prompt_ids, "max_tokens": n,
                 "temperature": 0, "logprobs": 20,
                 "return_tokens_as_token_ids": True})


records = []
print(f"{'depth':>6} {'argmax':>7} {'jaccard':>8} {'maxdiff':>9} {'top1gap':>8} {'band':>7}  verdict")
for depth in (1, 2, 4, 8, 16, 32):
    run = comp(ids, depth + 1)
    toks = [int(t.split(":")[-1]) for t in run["choices"][0]["logprobs"]["tokens"]]
    if len(toks) <= depth:
        print(f"{depth:>6}  (model stopped early at {len(toks)} tokens)")
        break
    dec = run["choices"][0]["logprobs"]["top_logprobs"][depth]
    pre = comp(ids + toks[:depth], 1)["choices"][0]["logprobs"]["top_logprobs"][0]

    shared, union = set(dec) & set(pre), set(dec) | set(pre)
    md = max((abs(dec[k] - pre[k]) for k in shared), default=float("inf"))
    ranked = sorted(pre.values(), reverse=True)
    gap = ranked[0] - ranked[1]
    band = ranked[0] - ranked[-1]
    am = max(dec, key=dec.get) == max(pre, key=pre.get)
    jac = len(shared) / len(union)
    # Same flatness discipline as the single-step probe: a near-tied
    # distribution cannot support a verdict, so say so instead of scoring a
    # coin-flip argmax.
    flat = band <= 1.0
    verdict = "INCONCL" if flat else ("DIVERGE" if (not am or md > 1.0) else "agree")
    print(f"{depth:>6} {str(am):>7} {jac:>8.3f} {md:>9.4f} {gap:>8.3f} {band:>7.3f}  {verdict}")
    records.append({"depth": depth, "argmax_match": am, "jaccard": jac,
                    "max_abs_diff": md if md != float("inf") else "inf",
                    "top1_gap": gap, "flat_band": band, "verdict": verdict})

if OUT:
    import os
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "depth_ladder.json"), "w") as f:
        json.dump(records, f, indent=2)

concl = [r for r in records if r["verdict"] != "INCONCL"]
div = [r for r in concl if r["verdict"] == "DIVERGE"]
print()
if div:
    print(f"VERDICT: DIVERGE at depths {[r['depth'] for r in div]} -- drift GROWS "
          f"with decode depth. This is the accumulating-state signature.")
elif concl:
    print(f"VERDICT: agree at all {len(concl)} conclusive depths "
          f"({len(records) - len(concl)} inconclusive). No cumulative drift.")
else:
    print("VERDICT: INCONCLUSIVE at every depth (distributions too flat).")
