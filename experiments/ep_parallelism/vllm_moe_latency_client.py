"""
GRPO-mode latency client: send a fixed batch of G requests concurrently to a
vLLM server and measure wall-clock time until ALL responses complete.
This mirrors how the GRPO recipe uses vLLM — a synchronous burst of G sequences,
not a sustained throughput stream.

Usage:
  python3 vllm_moe_latency_client.py \
    --url http://localhost:8001 \
    --model Qwen3-30B-A3B \
    --batch 4 \
    --max-tokens 256 \
    [--runs 3]

Output (TSV to stdout):
  label  tp  ep  batch  max_tokens  best_s  avg_s  tok_per_s
"""
import argparse
import sys
import time
import urllib.request
import urllib.error
import json
import threading

parser = argparse.ArgumentParser()
parser.add_argument("--url",        required=True)
parser.add_argument("--model",      required=True)
parser.add_argument("--batch",      type=int, default=4)
parser.add_argument("--max-tokens", type=int, default=256)
parser.add_argument("--input-len",  type=int, default=128,
                    help="Approximate prompt token count (repeated phrase)")
parser.add_argument("--runs",       type=int, default=3)
parser.add_argument("--label",      default="")
parser.add_argument("--tp",         default="?")
parser.add_argument("--ep",         default="no")
args = parser.parse_args()

# Build a prompt roughly --input-len tokens long.
PHRASE = "Solve step by step showing all work: "
prompt = (PHRASE * ((args.input_len // len(PHRASE.split()) + 1)))[:args.input_len * 4]

def one_request(url, model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
    return sum(len(c["text"].split()) for c in result.get("choices", []))

def bench_once(batch):
    results = [None] * batch
    errors  = []

    def worker(i):
        try:
            results[i] = one_request(args.url, args.model, prompt, args.max_tokens)
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(batch)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.perf_counter() - t0

    if errors:
        print(f"ERRORS: {errors}", file=sys.stderr)
        return None, None
    return elapsed, args.max_tokens * batch

# Warmup
print(f"# Warming up (1 req, 16 tokens)...", file=sys.stderr, flush=True)
try:
    one_request(args.url, args.model, prompt[:50], 16)
except Exception as e:
    print(f"Warmup failed: {e}", file=sys.stderr)
    sys.exit(1)

times, toks = [], []
for r in range(args.runs):
    elapsed, total_toks = bench_once(args.batch)
    if elapsed is None:
        sys.exit(1)
    times.append(elapsed)
    toks.append(total_toks)
    print(f"# run {r+1}/{args.runs}: {elapsed:.1f}s, {total_toks/elapsed:.0f} tok/s",
          file=sys.stderr, flush=True)

best_s = min(times)
avg_s  = sum(times) / len(times)
avg_tok_s = sum(t for t in toks) / sum(times)

# TSV header on first call (caller should print header separately)
print(f"{args.label}\t{args.tp}\t{args.ep}\t{args.batch}\t{args.max_tokens}"
      f"\t{best_s:.1f}\t{avg_s:.1f}\t{avg_tok_s:.0f}")
