import gzip, json, sys, collections, re, numpy as np
p=sys.argv[1]
with gzip.open(p) as f: tr=json.load(f)
ev=[e for e in tr["traceEvents"] if e.get("ph")=="X" and e.get("cat")=="kernel"]
ev.sort(key=lambda e:e["ts"])
def norm(n):
    n=re.sub(r'<[^>]*>','<>',n); n=re.sub(r'\d+','N',n); return n[:60]
MARK="at::native::xpu::ReduceKernel<>, unsigned int, long, N> >"
marks=[i for i,e in enumerate(ev) if norm(e["name"])==MARK]
seg=ev[marks[5]+1:marks[6]+1]
d=np.array([e.get("dur",0.0) for e in seg])
print(f"ONE DECODE STEP: {len(seg)} kernels, {d.sum()/1000:.1f} ms device-busy")
print(f"  mean kernel   {d.mean():8.2f} us")
print(f"  median kernel {np.median(d):8.2f} us")
for q in (25,75,90,99):
    print(f"  p{q:<3}          {np.percentile(d,q):8.2f} us")
print(f"  max           {d.max():8.2f} us")
print()
# how much of the busy time is in kernels shorter than N us?
for thr in (5,10,20,50):
    m=d<thr
    print(f"  kernels <{thr:3}us: {m.sum():6} ({100*m.sum()/len(d):4.1f}%) contributing {d[m].sum()/1000:6.1f} ms ({100*d[m].sum()/d.sum():4.1f}% of busy)")
