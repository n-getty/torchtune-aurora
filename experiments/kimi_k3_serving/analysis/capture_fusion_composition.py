EVENTS=19642; BUSY=130.3; STEP=879.0
COLL=(463-92)*0.557
RESID=STEP-BUSY-COLL
PER=RESID*1000/EVENTS
KDA=85*69
print(f"step {STEP:.0f} ms = busy {BUSY:.0f} + collectives {COLL:.0f} + residual {RESID:.0f}")
print(f"residual per launch = {PER:.1f} us\n")
print("KEY STRUCTURAL FACT: vllm::kda_attention is in CompilationConfig._attention_ops,")
print("which is the DEFAULT splitting_ops list. So under PIECEWISE:")
print("  - the graph is CUT at every kda_attention call (69 of them)")
print("  - kda_attention's body runs EAGERLY (no_compile_layers), never captured")
print(f"  - => all {KDA} KDA launches ({100*KDA/EVENTS:.0f}%) are OUTSIDE the graph\n")
noncapt = KDA
capt = EVENTS-noncapt
print(f"{'':22} {'launches':>9} {'resid ms':>9}")
print(f"{'capturable (in graph)':22} {capt:9} {capt*PER/1000:9.0f}")
print(f"{'NOT capturable (KDA)':22} {noncapt:9} {noncapt*PER/1000:9.0f}")
print()
print("=== so PIECEWISE capture's upper bound is NOT 62% ===")
best = BUSY+COLL+noncapt*PER/1000
print(f"perfect capture of everything capturable -> {best:.0f} ms "
      f"= {1000/best:.2f} tok/s ({100*(STEP-best)/STEP:.0f}% faster)")
print("(still large, but KDA fusion is the part capture CANNOT reach)")
print()
print("=== and that makes the two levers COMPLEMENTARY, not substitutes ===")
after_fuse = noncapt-4*69
best2 = BUSY+COLL+4*69*PER/1000
print(f"capture + KDA fused to 4/layer -> {best2:.0f} ms = {1000/best2:.2f} tok/s")
print(f"\nNOTE the fused-RMSNorm anti-stacking lesson does NOT apply here:")
print("that was a Triton kernel placed INSIDE a compiled region, blocking")
print("Inductor fusion. kda_attention is already a graph-splitting custom op,")
print("so a Triton kernel inside it displaces EAGER ops, blocking nothing.")
