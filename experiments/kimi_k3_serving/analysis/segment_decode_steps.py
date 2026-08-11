import gzip, json, sys, collections, re, numpy as np
p = sys.argv[1]
with gzip.open(p) as f: tr = json.load(f)
ev=[e for e in tr["traceEvents"] if e.get("ph")=="X" and e.get("cat") in ("kernel","gpu_memcpy")]
ev.sort(key=lambda e: e["ts"])
def norm(n):
    n=re.sub(r'<[^>]*>','<>',n); n=re.sub(r'\d+','N',n); return n[:70]
MARK = "at::native::xpu::ReduceKernel<>, unsigned int, long, N> >"
marks=[i for i,e in enumerate(ev) if e.get("cat")=="kernel" and norm(e["name"])==MARK]
print(f"marker occurrences: {len(marks)}")
ts=np.array([e["ts"] for e in ev])
prev=0
rows=[]
for j,i in enumerate(marks):
    seg=ev[prev:i+1]
    span=(ev[i]["ts"]-ev[prev]["ts"])/1000.0
    k=sum(1 for e in seg if e["cat"]=="kernel"); m=len(seg)-k
    rows.append((j,len(seg),k,m,span))
    prev=i+1
print(f"{'step':>4} {'events':>7} {'kernels':>8} {'memcpy':>7} {'span_ms':>9}")
for r in rows: print(f"{r[0]:4} {r[1]:7} {r[2]:8} {r[3]:7} {r[4]:9.1f}")
tail=len(ev)-prev
print(f"tail (after last marker): {tail} events")
# decode steps = all but the first (prefill)
dec=rows[1:]
if dec:
    import statistics
    print(f"\nDECODE steps (excluding step 0 = prefill): n={len(dec)}")
    print(f"  events/step median={statistics.median(r[1] for r in dec):.0f}")
    print(f"  kernels/step median={statistics.median(r[2] for r in dec):.0f}")
    print(f"  span_ms median={statistics.median(r[4] for r in dec):.1f}")
