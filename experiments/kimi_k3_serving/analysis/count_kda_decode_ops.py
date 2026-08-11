"""Count aten dispatches in one KDA layer's decode path (CPU, no XPU needed).

Mirrors op-for-op the code actually taken at c=1 decode in kda.py:
  _causal_conv1d_update_xpu (vectorized) x3  ->  fused_kda_gate + clamp
  -> _kda_recurrent_xpu (vectorized decode) -> FusedRMSNormGated.
Shapes from Kimi-K3 config at TP=32.  Projections (GEMMs) excluded: they are
not fusable into the KDA kernel and are counted separately.
"""
import torch, torch.nn.functional as F, collections
from torch.utils._python_dispatch import TorchDispatchMode

NUM_HEADS = 96 // 32     # local heads at TP=32
HEAD_DIM = 128
CONV_W = 4
N = 1                    # c=1 decode: one token
DIM = NUM_HEADS * HEAD_DIM
NUM_BLOCKS = 8
KDA_LAYERS = 69
GATE_LOWER_BOUND = -5.0
EPS = 1e-5

class Counter(TorchDispatchMode):
    def __init__(self, tag): self.tag, self.c = tag, collections.Counter()
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.c[str(func)] += 1
        return func(*args, **(kwargs or {}))

STAGES = collections.OrderedDict()
def stage(name):
    class _S:
        def __enter__(s):
            s.m = Counter(name); s.m.__enter__(); return s
        def __exit__(s, *a):
            s.m.__exit__(*a)
            STAGES.setdefault(name, collections.Counter()).update(s.m.c)
            return False
    return _S()

def conv1d_update(x, conv_state, weight, bias, idx):
    original_dtype = x.dtype
    x = x.unsqueeze(-1)
    state_len = conv_state.shape[-1]
    state = conv_state[idx]
    history = torch.cat((state, x[:, :, 0, None]), dim=-1)
    value = (history.float() * weight.float()).sum(-1)
    value = value + bias.float()
    value = F.silu(value)
    conv_state[idx] = history[:, :, -state_len:]
    return value.to(original_dtype).unsqueeze(-1).squeeze(-1)

def fused_kda_gate(g, A, head_k_dim, g_bias):
    original_shape = g.shape[:-1]
    num_heads = A.numel()
    g = g.reshape(-1, num_heads, head_k_dim).float()
    g = g + g_bias.reshape(1, num_heads, head_k_dim).float()
    softplus = F.softplus(g, beta=4.0, threshold=20.0)
    out = -torch.exp(A.float()).reshape(1, num_heads, 1) * softplus
    return out.reshape(*original_shape, num_heads, head_k_dim)

def kda_recurrent(q, k, v, g, beta, state, idx):
    output_dtype = v.dtype
    q_f, k_f, v_f = q.float(), k.float(), v.float()
    g_f, b_f = g.float(), beta.float()
    q_f = q_f / torch.sqrt(torch.sum(q_f * q_f, dim=-1, keepdim=True) + 1e-6)
    k_f = k_f / torch.sqrt(torch.sum(k_f * k_f, dim=-1, keepdim=True) + 1e-6)
    rs = state[idx].float()
    rs = rs * torch.exp(g_f[0])[:, :, None, :]
    delta = v_f[0] - torch.einsum("nhvk,nhk->nhv", rs, k_f[0])
    rs = rs + b_f[0, :, :, None, None] * (delta[:, :, :, None] * k_f[0, :, :, None, :])
    out = torch.einsum("nhvk,nhk->nhv", rs, q_f[0] * (q.shape[-1] ** -0.5))
    state[idx] = rs.to(state.dtype)
    return out.unsqueeze(0).to(output_dtype)

def o_norm(x, g, weight, eps=EPS):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = out * weight.float()
    out = out * g.float() * torch.sigmoid(g.float())
    return out.to(x.dtype)

def build():
    d = torch.bfloat16
    return dict(
        x=torch.randn(N, DIM, dtype=d),
        conv_state=torch.randn(NUM_BLOCKS, DIM, CONV_W - 1, dtype=d),
        weight=torch.randn(DIM, CONV_W, dtype=d),
        bias=torch.randn(DIM, dtype=d),
        idx=torch.arange(N, dtype=torch.long),
        g_raw=torch.randn(N, NUM_HEADS * HEAD_DIM, dtype=d),
        A=torch.randn(NUM_HEADS, dtype=d),
        g_bias=torch.randn(NUM_HEADS * HEAD_DIM, dtype=d),
        beta_raw=torch.randn(N, NUM_HEADS, dtype=d),
        rstate=torch.randn(NUM_BLOCKS, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=d),
        g2=torch.randn(1, N, NUM_HEADS, HEAD_DIM, dtype=d),
        onorm_w=torch.randn(HEAD_DIM, dtype=d),
    )

def run_decode(t, counted=False):
    S = stage if counted else (lambda name: __import__("contextlib").nullcontext())
    with S("conv1d_update x3 (q,k,v)"):
        q = conv1d_update(t["x"], t["conv_state"], t["weight"], t["bias"], t["idx"])
        k = conv1d_update(t["x"], t["conv_state"], t["weight"], t["bias"], t["idx"])
        v = conv1d_update(t["x"], t["conv_state"], t["weight"], t["bias"], t["idx"])
    with S("gate (fused_kda_gate + clamp + sigmoid)"):
        beta = t["beta_raw"].float().sigmoid().unsqueeze(0)
        g1 = fused_kda_gate(t["g_raw"], t["A"], HEAD_DIM, t["g_bias"])
        g1 = g1.clamp(min=GATE_LOWER_BOUND).unsqueeze(0)
    with S("recurrence (_kda_recurrent_xpu)"):
        qq, kk, vv = (z.view(1, N, NUM_HEADS, HEAD_DIM) for z in (q, k, v))
        out = kda_recurrent(qq, kk, vv, g1, beta, t["rstate"], t["idx"])
    with S("o_norm (FusedRMSNormGated)"):
        out = o_norm(out, t["g2"], t["onorm_w"])
    return out

if __name__ == "__main__":
    t = build()
    run_decode(t)
    run_decode(t, counted=True)
    grand = sum(sum(c.values()) for c in STAGES.values())
    print(f"KDA decode, c=1, TP=32 ({NUM_HEADS} local heads, head_dim={HEAD_DIM})")
    print(f"{'ops':>5}  stage")
    for name, c in STAGES.items():
        print(f"{sum(c.values()):5}  {name}")
    print(f"{grand:5}  TOTAL per KDA layer")
    print(f"\n{grand*KDA_LAYERS:5}  ops/token across {KDA_LAYERS} KDA layers")
    print("\nper-op detail:")
    allc = collections.Counter()
    for c in STAGES.values(): allc.update(c)
    for name, n in allc.most_common():
        print(f"{n:5}  {name}")

# --- kernel-launching subset -------------------------------------------------
VIEW_OPS = {
    "aten.view.default", "aten.unsqueeze.default", "aten.permute.default",
    "aten.select.int", "aten.slice.Tensor", "aten.squeeze.dim",
    "aten.expand.default", "aten.reshape.default", "aten.t.default",
    "aten.transpose.int", "aten.detach.default", "aten._unsafe_view.default",
}
def kernel_launching(counter):
    return sum(n for op, n in counter.items() if op not in VIEW_OPS)
