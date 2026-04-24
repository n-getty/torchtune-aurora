#!/usr/bin/env python3
"""Fix Gemma4 tokenizer_config.json for older transformers (4.57.x).

Gemma4's tokenizer_config.json has extra_special_tokens as a list,
but transformers 4.57.6 expects a dict and calls .keys() on it.
This script converts the list to a dict format.
"""
import json
import sys

path = sys.argv[1]
with open(path) as f:
    d = json.load(f)

est = d.get("extra_special_tokens")
if isinstance(est, list):
    d["extra_special_tokens"] = {t: t for t in est}
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"Fixed extra_special_tokens: list -> dict ({est})")
else:
    print(f"No fix needed, type={type(est)}")
