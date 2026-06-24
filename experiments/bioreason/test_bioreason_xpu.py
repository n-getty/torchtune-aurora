"""
BioReason-Pro XPU smoke tests.

T1: ESM3 import (SKIP — gated HF model, blocked offline)
T2: vLLM enable_prompt_embeds confirmed present in EngineArgs
T3: Scatter-replace pipeline on XPU
T4: BioReason imports work (sys.modules stub trick)
T5: ESM3 load + per-residue embeddings on XPU (7.4s load, 1.2s encode)
T6: BioReasonModel full integration (init + build_prompt_embeds + forward)
T7: Dataset + reward function
T8: End-to-end GRPO pipeline
"""
import os, sys, types, torch, time

PROJDIR = "/lus/flare/projects/ModCon/ngetty/torchtune"
BIOREASON_SRC = "/flare/ModCon/ngetty/BioReason-Pro"
BIOREASON_DEPS = "/lus/flare/projects/ModCon/ngetty/bioreason_deps"
MODEL_DIR = "/tmp/torchtune/bioreason-pro-sft"

# CRITICAL: INFRA_PROVIDER must be set before any esm.* import
os.environ["INFRA_PROVIDER"] = "local"
os.chdir(PROJDIR)
sys.path.insert(0, BIOREASON_DEPS)
sys.path.insert(0, BIOREASON_SRC)
sys.path.insert(0, PROJDIR)

for pkg_name, pkg_path in [
    ("bioreason2", f"{BIOREASON_SRC}/bioreason2"),
    ("bioreason2.models", f"{BIOREASON_SRC}/bioreason2/models"),
]:
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [pkg_path]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

device = torch.device("xpu:0")
PASS, FAIL, SKIP = [], [], []


def run(name, fn):
    try:
        fn()
        print(f"  PASS {name}")
        PASS.append(name)
    except Exception as e:
        import traceback
        print(f"  FAIL {name}: {e}")
        traceback.print_exc()
        FAIL.append(name)


print("=== BioReason XPU tests ===")

# T5: ESM3 load + encode
def t5():
    from esm.pretrained import LOCAL_MODEL_REGISTRY
    from esm.sdk.api import ESMProtein, SamplingConfig
    model = LOCAL_MODEL_REGISTRY["esm3_sm_open_v1"](device).eval()
    protein = ESMProtein(sequence="MKTAYIAKQRQISFVKSHFS")
    with torch.no_grad():
        pt = model.encode(protein)
        out = model.forward_and_sample(pt, SamplingConfig(return_per_residue_embeddings=True))
    emb = out.per_residue_embedding
    assert emb.shape == (22, 1536), f"Expected [22,1536] got {emb.shape}"
run("T5: ESM3 encode on XPU", t5)

# T6: BioReasonModel init + forward
def t6():
    from torchtune.dev.bioreason.model import BioReasonModel
    model = BioReasonModel(ckpt_dir=MODEL_DIR, device=device, dtype=torch.bfloat16)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    tokens = torch.zeros(2, 128, dtype=torch.long)
    tokens[:, 5:10] = model.protein_token_id   # 5 protein tokens
    tokens[:, 20:220] = model.go_token_id       # 200 GO tokens
    protein_seqs = ["MKTAY"] * 2
    pe = model.build_prompt_embeds(tokens, protein_seqs)
    assert pe.shape == (2, 128, 2560), f"Bad prompt_embeds shape: {pe.shape}"
    fe = model.build_full_embeds(pe, torch.randint(100, 1000, (2, 10)))
    with torch.no_grad():
        logits = model(inputs_embeds=fe)
    assert logits.shape == (2, 138, 151671), f"Bad logits shape: {logits.shape}"
run("T6: BioReasonModel full pipeline", t6)

# T7: Dataset + reward
def t7():
    from torchtune.dev.bioreason.dataset import bioreason_tokenizer, BioReasonRLDataset, bioreason_collate_fn, PROTEIN_PAD, GO_PAD
    tok = bioreason_tokenizer(MODEL_DIR)
    pid = tok.encode(PROTEIN_PAD)
    gid = tok.encode(GO_PAD)
    assert len(pid) == 1 and len(gid) == 1, "Special tokens must be single tokens"
    DATA = "/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl"
    ds = BioReasonRLDataset(DATA, tok, max_seq_len=256, max_protein_len=32, num_go_tokens=30)
    assert len(ds) == 9197, f"Expected 9197 examples, got {len(ds)}"
    batch = bioreason_collate_fn([ds[0], ds[1]], padding_idx=tok.pad_id, max_seq_len=256)
    from torchtune.dev.bioreason.reward import bioreason_reward_fn
    rewards, _ = bioreason_reward_fn([batch['answers'][0]] * 2, batch['answers'] * 1)
    assert rewards[0].item() == 1.0, "Self-reward should be 1.0"
run("T7: Dataset + reward function", t7)

# T8: End-to-end GRPO pipeline (prompt_embeds → full_embeds → forward → loss)
def t8():
    from torchtune.dev.bioreason.model import BioReasonModel
    model = BioReasonModel(ckpt_dir=MODEL_DIR, device=device, dtype=torch.bfloat16)
    tok = model.tokenizer
    pad_id = tok.pad_id if hasattr(tok, 'pad_id') else tok.eos_token_id
    B, ctx_len, comp_len = 2, 64, 16
    # Build tokens with protein+GO placeholders
    tokens = torch.full((B, ctx_len), pad_id, dtype=torch.long)
    tokens[:, 5:10] = model.protein_token_id
    tokens[:, 20:220] = model.go_token_id
    protein_seqs = ["MKTAY"] * B
    # Build prompt embeds (no grad — encoder is frozen)
    with torch.no_grad():
        pe = model.build_prompt_embeds(tokens.to(device), protein_seqs)  # [B, ctx_len, H]
    # Build full embeds (prompt + completion) and run training forward
    comp_ids = torch.randint(100, 1000, (B, comp_len))
    fe = model.build_full_embeds(pe, comp_ids)  # [B, ctx_len+comp_len, H]
    assert fe.shape == (B, ctx_len + comp_len, 2560), f"full_embeds shape: {fe.shape}"
    logits = model(inputs_embeds=fe)
    assert logits.shape == (B, ctx_len + comp_len, model.vocab_size), f"logits shape: {logits.shape}"
    loss = logits[:, ctx_len:].mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "No gradients flowed"
run("T8: Full pipeline (prompt_embeds→full_embeds→fwd→bwd)", t8)

# T9: Recipe integration hooks (vllm_param_iter, GRPOTrajectory.prompt_embeds)
def t9():
    from torchtune.dev.bioreason.model import BioReasonModel
    from torchtune.dev.rl.types import GRPOTrajectory
    model = BioReasonModel(ckpt_dir=MODEL_DIR, device=device, dtype=torch.bfloat16)

    # Test vllm_param_iter: yields backbone params with HF names (no 'backbone.' prefix)
    params = list(model.vllm_param_iter())
    names = [n for n, _ in params]
    assert len(params) > 0, "vllm_param_iter() yielded nothing"
    # Backbone HF names start with 'model.' (Qwen3 uses 'model.layers.N...')
    assert all(not n.startswith("backbone.") for n in names), \
        f"vllm_param_iter() should strip 'backbone.' prefix, got: {names[:3]}"
    # Projectors must NOT appear (they are not synced to vLLM)
    assert not any("protein_projection" in n or "go_projection" in n for n in names), \
        "Projectors must not be in vllm_param_iter()"

    # Test GRPOTrajectory.prompt_embeds field
    B, G, P, H = 2, 4, 64, 2560
    pe = torch.randn(B * G, P, H)
    traj = GRPOTrajectory(
        query_responses=torch.zeros(B * G, P + 16, dtype=torch.long),
        prompt_embeds=pe,
    )
    assert traj.prompt_embeds is pe, "prompt_embeds not stored in trajectory"

    # Test trajectory slicing (replicates _slice_trajectory logic — recipes/ can't be imported)
    sliced_fields = {}
    for field_name in traj._fields:
        val = getattr(traj, field_name)
        if isinstance(val, torch.Tensor):
            sliced_fields[field_name] = val[0:4]
        elif isinstance(val, list):
            sliced_fields[field_name] = val[0:4]
        else:
            sliced_fields[field_name] = val
    sliced = GRPOTrajectory(**sliced_fields)
    assert sliced.prompt_embeds.shape == (4, P, H), f"sliced shape: {sliced.prompt_embeds.shape}"
run("T9: Recipe hooks (vllm_param_iter + GRPOTrajectory.prompt_embeds)", t9)

print(f"\n=== Results: {len(PASS)} PASS, {len(FAIL)} FAIL ===")
if FAIL:
    sys.exit(1)
