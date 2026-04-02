"""Test if 32B model can handle forward_batch_size=4 without OOM.

Run: torchrun --standalone --nproc_per_node=10 recipes/dev/test_32b_fbs.py
"""
import os, sys, time, types, datetime
import torch
import torch.distributed as dist

# Pre-register torchtune
if "torchtune" not in sys.modules:
    sys.modules["torchtune"] = types.ModuleType("torchtune")
    sys.modules["torchtune"].__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "torchtune")]

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.xpu.set_device(local_rank)
device = torch.device(f"xpu:{local_rank}")
dist.init_process_group("xccl", timeout=datetime.timedelta(minutes=5))
rank = dist.get_rank()
world_size = dist.get_world_size()

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# Monkey-patch FSDP2 for XPU (avoid ReduceOp.AVG which XPU doesn't support)
try:
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as _fsdp_coll
    _orig_get_gradient_divide_factors = _fsdp_coll._get_gradient_divide_factors
    def _patched_get_gradient_divide_factors(reduce_scatter_group, all_reduce_group, reduce_dtype,
                                              device_type='', factor=None, force_sum_reduction_for_comms=False):
        return _orig_get_gradient_divide_factors(
            reduce_scatter_group, all_reduce_group, reduce_dtype,
            device_type=device_type, factor=factor, force_sum_reduction_for_comms=True)
    _fsdp_coll._get_gradient_divide_factors = _patched_get_gradient_divide_factors
except Exception:
    pass

from torchtune.models.qwen3 import qwen3_32b
from torchtune.training._distributed import shard_model
from torchtune import training
from functools import partial

if rank == 0:
    print(f"World size: {world_size}")
    print(f"Testing 32B forward with batch sizes 1, 2, 4")

# Build and shard model
with training.set_default_dtype(torch.bfloat16), torch.device("meta"):
    model = qwen3_32b()
model.train()
# Keep requires_grad=True for training-phase test

from torchtune.modules import TransformerSelfAttentionLayer
training.set_activation_checkpointing(model, auto_wrap_policy={TransformerSelfAttentionLayer})

shard_conditions = [partial(training.get_shard_conditions, names_to_match=None)]
shard_model(model=model, shard_conditions=shard_conditions, cpu_offload=False,
            reshard_after_forward=True, dp_mesh=None)

with training.set_default_dtype(torch.bfloat16), device:
    for m in model.modules():
        if hasattr(m, "rope_init"):
            m.rope_init()

# Load checkpoint
ckpt_path = "/tmp/torchtune/Qwen3-32B"
from torchtune.training import FullModelHFCheckpointer
checkpointer = FullModelHFCheckpointer(
    checkpoint_dir=ckpt_path,
    checkpoint_files=[f"model-{i:05d}-of-00017.safetensors" for i in range(1, 18)],
    output_dir="/tmp/torchtune/test_fbs",
    model_type="QWEN3",
)
ckpt = checkpointer.load_checkpoint()
training.load_from_full_model_state_dict(model, ckpt["model"], device, strict=True, cpu_offload=False)
training.validate_no_params_on_meta_device(model)
del ckpt

if rank == 0:
    mem = training.get_memory_stats(device=device)
    print(f"After model load: {mem.get('peak_memory_active', 0)/1e9:.1f} GiB active, {mem.get('peak_memory_reserved', 0)/1e9:.1f} GiB reserved")

dist.barrier()

seq_len = 640  # 512 prompt + 128 gen tokens
for bs in [1, 2, 4]:
    if rank == 0:
        print(f"\n=== Batch size {bs} ===")

    torch.xpu.synchronize()
    dist.barrier()

    input_ids = torch.randint(0, 151936, (bs, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)

    try:
        # Test 1: no_grad forward (gen phase)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(input_ids, input_pos=position_ids)
        torch.xpu.synchronize()
        nograd_time = time.perf_counter() - t0

        if rank == 0:
            mem_alloc = torch.xpu.memory_allocated(device) / 1e9
            mem_reserved = torch.xpu.memory_reserved(device) / 1e9
            print(f"  no_grad forward: {nograd_time:.2f}s  mem: {mem_alloc:.1f}/{mem_reserved:.1f} GiB alloc/reserved")
            print(f"  Output shape: {logits.shape}")
        del logits

        # Test 2: grad-enabled forward (training phase)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        logits = model(input_ids, input_pos=position_ids)
        torch.xpu.synchronize()
        grad_time = time.perf_counter() - t0

        if rank == 0:
            mem_alloc = torch.xpu.memory_allocated(device) / 1e9
            mem_reserved = torch.xpu.memory_reserved(device) / 1e9
            print(f"  grad forward:    {grad_time:.2f}s  mem: {mem_alloc:.1f}/{mem_reserved:.1f} GiB alloc/reserved")

        # Test 3: backward
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        loss = logits.sum()
        loss.backward()
        torch.xpu.synchronize()
        bwd_time = time.perf_counter() - t0

        if rank == 0:
            mem_alloc = torch.xpu.memory_allocated(device) / 1e9
            mem_reserved = torch.xpu.memory_reserved(device) / 1e9
            print(f"  backward:        {bwd_time:.2f}s  mem: {mem_alloc:.1f}/{mem_reserved:.1f} GiB alloc/reserved")

        del logits, loss, input_ids, position_ids
        model.zero_grad(set_to_none=True)
        torch.xpu.synchronize()
    except RuntimeError as e:
        if rank == 0:
            print(f"  OOM or error: {e}")
        break

    dist.barrier()

dist.destroy_process_group()
