"""Validate the chunked delta-rule algebra against the per-token loop, on CPU."""
import torch

def per_token(q,k,v,g,beta,S0,scale):
    """Reference: the loop currently in kda.py (single sequence)."""
    T,H,Dk = k.shape; Dv = v.shape[-1]
    S = S0.clone()
    out = torch.empty(T,H,Dv, dtype=torch.float32)
    for t in range(T):
        S = S * torch.exp(g[t])[:,None,:]              # [H,Dv,Dk]
        u = v[t] - torch.einsum("hvk,hk->hv", S, k[t])
        S = S + beta[t][:,None,None] * (u[:,:,None] * k[t][:,None,:])
        out[t] = torch.einsum("hvk,hk->hv", S, q[t]*scale)
    return out, S

def chunked(q,k,v,g,beta,S0,scale,C=16):
    """Chunked form: O(T/C) iterations, each solving a CxC triangular system."""
    T,H,Dk = k.shape; Dv = v.shape[-1]
    S = S0.clone()
    out = torch.empty(T,H,Dv, dtype=torch.float32)
    for s in range(0, T, C):
        e = min(s+C, T); L = e-s
        gc = g[s:e]                                     # [L,H,Dk]
        A = torch.exp(torch.cumsum(gc, dim=0))          # [L,H,Dk]
        kA  = k[s:e] * A                                # k_t * A_t
        kdA = k[s:e] / A                                # k_i / A_i
        qA  = q[s:e] * A
        # w_t = v_t - S @ (A_t k_t)
        w = v[s:e] - torch.einsum("hvk,lhk->lhv", S, kA)
        # M[t,i] = beta_i <k_i/A_i, A_t k_t>, strictly lower triangular
        M = torch.einsum("lhk,mhk->hlm", kA, kdA) * beta[s:e].permute(1,0)[:,None,:]
        M = M * torch.tril(torch.ones(L,L), -1)
        u = torch.linalg.solve_triangular(
            torch.eye(L) + M, w.permute(1,0,2), upper=False)   # [H,L,Dv]
        bu = u * beta[s:e].permute(1,0)[:,:,None]              # beta_i u_i
        # o_t = S @ (A_t q_t) + sum_{i<=t} bu_i <k_i/A_i, A_t q_t>
        Qk = torch.einsum("lhk,mhk->hlm", qA, kdA) * torch.tril(torch.ones(L,L))
        out[s:e] = (torch.einsum("hvk,lhk->lhv", S, qA)
                    + torch.einsum("hlm,hmv->lhv", Qk, bu)) * scale
        # S_end = A_L * (S + sum_i bu_i (x) k_i/A_i)
        S = (S + torch.einsum("hlv,lhk->hvk", bu.permute(0,1,2), kdA)) * A[-1][:,None,:]
    return out, S

torch.manual_seed(0)
fails=0
for T,H,Dk,Dv,C in [(16,3,8,8,16),(32,2,16,16,16),(37,3,8,8,16),(64,2,8,8,8)]:
    q=torch.randn(T,H,Dk); k=torch.randn(T,H,Dk); v=torch.randn(T,H,Dv)
    q=torch.nn.functional.normalize(q,dim=-1); k=torch.nn.functional.normalize(k,dim=-1)
    g=-torch.rand(T,H,Dk)*5.0            # gate_lower_bound = -5
    beta=torch.rand(T,H)
    S0=torch.randn(H,Dv,Dk)*0.1
    scale=Dk**-0.5
    o1,S1=per_token(q,k,v,g,beta,S0,scale)
    o2,S2=chunked(q,k,v,g,beta,S0,scale,C)
    do=(o1-o2).abs().max().item(); ds=(S1-S2).abs().max().item()
    ok = do<2e-3 and ds<2e-3
    fails += 0 if ok else 1
    print(f"T={T:3} H={H} C={C}: max|dout|={do:.2e} max|dstate|={ds:.2e}  {'OK' if ok else 'FAIL'}")
print("FAILURES:", fails)
