#!/usr/bin/env python3
"""Phase 2 prototype: validate vLLM HTTP /v1/completions with prompt_embeds.

Builds a dummy random prompt_embeds tensor (matching Qwen3-4B hidden_size=2560),
serializes via torch.save → bytes → base64, POSTs to vLLM HTTP server,
prints the generated text (or error).

Usage: python3 test_vllm_prompt_embeds.py [URL]
"""
import sys
import io
import base64
import json
import time
import urllib.request
import urllib.error

import torch

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
HIDDEN = 2560  # Qwen3-4B hidden_size
PROMPT_LEN = 16  # short dummy prompt

def encode_embed(t: torch.Tensor) -> str:
    """torch.save a tensor and base64-encode its bytes (matches vLLM API expected format)."""
    buf = io.BytesIO()
    torch.save(t, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def main():
    print(f"=== Probing {URL} ===")
    # Health
    try:
        with urllib.request.urlopen(f"{URL}/health", timeout=5) as r:
            print(f"GET /health -> {r.status}")
    except Exception as e:
        print(f"GET /health failed: {e}")
        return 1

    # Models
    try:
        with urllib.request.urlopen(f"{URL}/v1/models", timeout=5) as r:
            data = json.loads(r.read())
            print(f"models: {[m['id'] for m in data.get('data', [])]}")
    except Exception as e:
        print(f"GET /v1/models failed: {e}")
        return 1

    # Build dummy prompt_embeds
    embed = torch.randn(PROMPT_LEN, HIDDEN, dtype=torch.bfloat16)
    print(f"\nBuilt dummy prompt_embeds: shape={tuple(embed.shape)} dtype={embed.dtype}")
    encoded = encode_embed(embed)
    print(f"Encoded size: {len(encoded)} bytes")

    body = {
        "model": "/tmp/torchtune/bioreason-pro-sft",
        "prompt_embeds": encoded,
        "max_tokens": 32,
        "temperature": 0.8,
        "top_p": 1.0,
    }
    req = urllib.request.Request(
        f"{URL}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print("\nPOST /v1/completions ...")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read())
            dt = time.perf_counter() - t0
            print(f"OK in {dt:.1f}s")
            print(json.dumps(payload, indent=2)[:1500])
    except urllib.error.HTTPError as e:
        body_err = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {body_err[:1500]}")
        return 2
    except Exception as e:
        print(f"Request failed: {e}")
        return 3
    return 0

if __name__ == "__main__":
    sys.exit(main())
