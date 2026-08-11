EVENTS=19642; BUSY=130.3
REAL=961.0; FUSED=879.0
COLL_N=463; COLL_MS=0.557       # measured 14 KiB 32-rank AR, isolated
coll = COLL_N*COLL_MS
print("=== c=1 decode step budget, REAL step, measured terms only ===\n")
for lbl,step in (("baseline",REAL),("AR-fused (default)",FUSED)):
    n = COLL_N-92 if "fused" in lbl else COLL_N
    c = n*COLL_MS
    other = step-BUSY-c
    print(f"{lbl}: {step:.0f} ms")
    print(f"   device-busy (compute)      {BUSY:6.0f} ms  {100*BUSY/step:5.1f}%")
    print(f"   collectives ({n} x {COLL_MS} ms) {c:6.0f} ms  {100*c/step:5.1f}%   [isolated LOWER bound]")
    print(f"   host/dispatch residual     {other:6.0f} ms  {100*other/step:5.1f}%   <- the target")
    print(f"   residual per event         {other*1000/EVENTS:6.1f} us\n")
step=FUSED; n=COLL_N-92; c=n*COLL_MS; resid=step-BUSY-c
per=resid*1000/EVENTS
KDA_L=85; KDA_N=69; kda=KDA_L*KDA_N
print("=== fused KDA decode kernel, charging ONLY the residual ===")
print(f"residual {resid:.0f} ms / {EVENTS} events = {per:.1f} us/event")
for keep in (4,8,27):
    removed=(KDA_L-keep)*KDA_N
    new=step-removed*per/1000
    print(f"  keep {keep:2}/layer: -{removed} launches -> {new:.0f} ms, "
          f"{1000/step:.3f}->{1000/new:.3f} tok/s ({100*(step-new)/step:+.1f}%)")
print(f"\nKDA is {100*kda/EVENTS:.0f}% of launches; the other {EVENTS-kda} are MoE/MLA/norm/sampler.")
print("Full-model fusion (not just KDA) is where the remaining 70% sits.")
