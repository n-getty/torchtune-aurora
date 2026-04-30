# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# vLLM initialization and setup methods for GRPO training on Aurora/XPU.
#
# Extracted from grpo_full_finetune_distributed_xpu.py to reduce recipe file size.
# All functions take `self` as first argument and are bound to the recipe class via
# method binding in GRPOFullFinetuneDistributedXPU.
#
# Covers three initialization paths:
#   _init_vllm_early / _init_vllm_early_dedicated — early XPU context setup
#   _init_vllm_tp1 / _init_vllm_tp — tensor-parallel vLLM process group init
#   _setup_vllm_server_mode / _setup_vllm_colocate_mode — mode-specific wiring

import os
import threading
import time

import torch
from torchtune import utils

log = utils.get_logger("DEBUG")

def _init_vllm_early(self, cfg):
    """Initialize colocated vLLM engine(s) before the training PG.

    Two modes based on ``vllm_tensor_parallel_size``:

    **TP=1** (default): Each rank creates its own isolated vLLM engine
    using a gloo PG (world_size=1). Sequential init via file barriers.

    **TP>1**: Ranks are grouped into DP groups of size tp_size. Each group
    creates one shared vLLM engine via ``external_launcher`` with an XCCL
    PG of size tp_size. All ranks in a group call generate() together.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # Match the recipe's _xpu_device_index logic: with single-tile affinity
    # the rank sees its tile as xpu:0; otherwise select by LOCAL_RANK.
    _affinity = os.environ.get("ZE_AFFINITY_MASK", "")
    _aff_tiles = _affinity.split(",") if _affinity else []
    local_rank = 0 if len(_aff_tiles) == 1 else int(os.environ.get("LOCAL_RANK", "0"))

    vllm_mode = cfg.get("vllm_mode", "colocate")
    tp_size = cfg.get("vllm_tensor_parallel_size", 1)

    # For colocate_sleep mode, apply XPU sleep patches BEFORE importing vLLM LLM
    if vllm_mode == "colocate_sleep":
        from torchtune.dev.xpu_sleep import patch_vllm_for_xpu_sleep
        patch_vllm_for_xpu_sleep()

    from vllm import LLM

    model_path = cfg.base_model_path
    gpu_mem = cfg.get("vllm_gpu_memory_utilization", 0.3)
    max_model_len = cfg.get("vllm_max_model_len", 2048)

    log.info(
        "Rank %d: initializing colocated vLLM engine on xpu:%d (tp=%d, gpu_mem=%.2f, max_model_len=%d)",
        rank, local_rank, tp_size, gpu_mem, max_model_len,
    )

    # vLLM V1: disable multiprocessing to avoid EngineCore subprocess hangs.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Disable torch.compile for vLLM init (not viable on XPU for vLLM).
    _prev_compile_disable = os.environ.get("TORCH_COMPILE_DISABLE")
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

    # Each rank generates all grpo_samples for its own prompts (no split)
    max_num_seqs = max(cfg.batch_size * cfg.grpo_samples, 1)

    if tp_size > 1:
        self._init_vllm_tp(
            cfg, rank, world_size, local_rank, tp_size,
            model_path, gpu_mem, max_model_len, max_num_seqs, vllm_mode, LLM,
        )
    else:
        self._init_vllm_tp1(
            cfg, rank, world_size, local_rank,
            model_path, gpu_mem, max_model_len, max_num_seqs, vllm_mode, LLM,
        )

    # Restore TORCH_COMPILE_DISABLE so training model can use torch.compile
    if _prev_compile_disable is not None:
        os.environ["TORCH_COMPILE_DISABLE"] = _prev_compile_disable
    elif "TORCH_COMPILE_DISABLE" in os.environ:
        del os.environ["TORCH_COMPILE_DISABLE"]

    # Re-set the XPU device to our tile (vLLM may have called set_device)
    torch.xpu.set_device(local_rank)

    log.info("Rank %d: colocated vLLM engine initialized on xpu:%d, PG reset for training", rank, local_rank)



def _init_vllm_early_dedicated(self, cfg):
    """Initialize vLLM on the dedicated vLLM rank BEFORE the CCL process group.

    Must be called before init_xpu_process_group() because vLLM creates
    gloo sub-groups that crash if CCL is already the default backend.
    The CCL world PG is created afterward in __init__ via init_xpu_process_group.
    """
    from vllm import LLM

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    log.info("Rank %d (vLLM server): early vLLM init on tile %d", rank, local_rank)

    # Sequential init: wait for rank-1 before starting to avoid resource conflicts.
    import tempfile
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
    barrier_dir = f"/tmp/torchtune/vllm_init_barriers_{run_id}"
    os.makedirs(barrier_dir, exist_ok=True)

    # Override env vars so vLLM sees world_size=1 for its internal PG.
    saved_env = {}
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK",
                "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "TORCHELASTIC_RUN_ID", "ZE_AFFINITY_MASK"):
        saved_env[key] = os.environ.pop(key, None)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29599 + rank)
    os.environ["ZE_AFFINITY_MASK"] = str(local_rank)

    _store_file = tempfile.mktemp(prefix=f"vllm_gloo_store_ded_r{rank}_")
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{_store_file}",
        world_size=1,
        rank=0,
    )

    _orig_new_group = torch.distributed.new_group
    def _gloo_new_group(*args, **kwargs):
        kwargs["backend"] = "gloo"
        return _orig_new_group(*args, **kwargs)
    torch.distributed.new_group = _gloo_new_group

    _orig_all_reduce = torch.distributed.all_reduce
    def _safe_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM,
                         group=None, async_op=False):
        if group is not None and group.size() == 1:
            return None
        if tensor.is_xpu:
            return None
        return _orig_all_reduce(tensor, op=op, group=group, async_op=async_op)
    torch.distributed.all_reduce = _safe_all_reduce

    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    _orig_distributed_args = UniProcExecutor._distributed_args
    _correct_local_rank = local_rank
    def _patched_distributed_args(self_exec):
        method, _rank, _lr = _orig_distributed_args(self_exec)
        return method, _rank, _correct_local_rank
    UniProcExecutor._distributed_args = _patched_distributed_args

    # Disable vLLM V1 multiprocessing: XPU is already initialized (ZE_AFFINITY_MASK set),
    # so vLLM would force 'spawn' mode and its EngineCore subprocess would hang.
    # Same fix as colocate mode (_init_vllm_tp1).
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    _prev_compile_disable = os.environ.get("TORCH_COMPILE_DISABLE")
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

    gpu_mem = cfg.get("vllm_gpu_memory_utilization", 0.7)
    max_model_len = cfg.get("vllm_max_model_len", 2048)
    max_num_seqs = cfg.get("vllm_max_num_seqs", 64)

    self._vllm_llm = LLM(
        model=cfg.base_model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enforce_eager=True,
        dtype="bfloat16",
        disable_custom_all_reduce=True,
        enable_sleep_mode=False,
        enable_prompt_embeds=True,
    )

    # Restore TORCH_COMPILE_DISABLE so training ranks can use torch.compile.
    if _prev_compile_disable is not None:
        os.environ["TORCH_COMPILE_DISABLE"] = _prev_compile_disable
    elif "TORCH_COMPILE_DISABLE" in os.environ:
        del os.environ["TORCH_COMPILE_DISABLE"]

    # Restore monkey-patches before destroying PG.
    torch.distributed.new_group = _orig_new_group
    torch.distributed.all_reduce = _orig_all_reduce

    # Destroy vLLM's gloo PG so rank 11 can join the CCL world PG.
    # vLLM TP=1 doesn't need the gloo PG at inference time (single process).
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    try:
        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._WORLD = None
    except Exception:
        pass

    try:
        os.unlink(_store_file)
    except OSError:
        pass

    # Restore env for the CCL process group init that follows.
    for key, val in saved_env.items():
        if val is not None:
            os.environ[key] = val
        else:
            os.environ.pop(key, None)

    torch.xpu.set_device(local_rank)
    log.info("Rank %d (vLLM server): early vLLM init complete on tile %d", rank, local_rank)



def _init_vllm_tp1(self, cfg, rank, world_size, local_rank,
                    model_path, gpu_mem, max_model_len, max_num_seqs,
                    vllm_mode, LLM):
    """Initialize TP=1 colocated vLLM: one isolated engine per rank.

    Uses gloo PG (world_size=1) with sequential init via file barriers.
    """
    # Sequential init to avoid port/resource conflicts.
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
    barrier_dir = f"/tmp/torchtune/vllm_init_barriers_{run_id}"
    os.makedirs(barrier_dir, exist_ok=True)
    if rank > 0:
        prev_barrier = os.path.join(barrier_dir, f"rank_{rank - 1}_done")
        log.info("Rank %d: waiting for rank %d to finish vLLM init...", rank, rank - 1)
        while not os.path.exists(prev_barrier):
            time.sleep(0.5)

    # Override torchrun env vars so vLLM sees world_size=1
    saved_env = {}
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK",
                "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "TORCHELASTIC_RUN_ID", "ZE_AFFINITY_MASK"):
        saved_env[key] = os.environ.pop(key, None)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29599 + rank)
    # torch.xpu.mem_get_info() ignores its device argument and always returns
    # the first visible tile's stats. Without an affinity mask, rank 1's vLLM
    # mem_get_info queries tile 0's memory (which rank 0's vLLM has allocated),
    # making rank 1 think tile 1 is out of memory. Restricting each process to
    # only its own tile gives vLLM accurate per-tile memory stats.
    os.environ["ZE_AFFINITY_MASK"] = str(local_rank)

    # Pre-init a gloo PG (world_size=1) so vLLM skips its own init.
    import tempfile
    _store_file = tempfile.mktemp(prefix=f"vllm_gloo_store_r{rank}_")
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{_store_file}",
        world_size=1,
        rank=0,
    )

    # Monkey-patch new_group to force gloo backend.
    _orig_new_group = torch.distributed.new_group
    def _gloo_new_group(*args, **kwargs):
        kwargs["backend"] = "gloo"
        return _orig_new_group(*args, **kwargs)
    torch.distributed.new_group = _gloo_new_group

    # Monkey-patch all_reduce to skip XPU tensor ops on gloo.
    _orig_all_reduce = torch.distributed.all_reduce
    def _safe_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM,
                         group=None, async_op=False):
        if group is not None and group.size() == 1:
            return None
        if tensor.is_xpu:
            return None
        return _orig_all_reduce(tensor, op=op, group=group,
                                async_op=async_op)
    torch.distributed.all_reduce = _safe_all_reduce

    # Monkey-patch _distributed_args for correct local_rank.
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    _orig_distributed_args = UniProcExecutor._distributed_args
    _correct_local_rank = local_rank
    def _patched_distributed_args(self_exec):
        method, _rank, _lr = _orig_distributed_args(self_exec)
        return method, _rank, _correct_local_rank
    UniProcExecutor._distributed_args = _patched_distributed_args

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enforce_eager=True,
        dtype="bfloat16",
        disable_custom_all_reduce=True,
    )
    if vllm_mode == "colocate_sleep":
        llm_kwargs["enable_sleep_mode"] = True
    if cfg.get("vllm_enable_prompt_embeds", False):
        llm_kwargs["enable_prompt_embeds"] = True

    self._vllm_llm = LLM(**llm_kwargs)

    # colocate_sleep: sleep immediately after init if not the last rank.
    # torch.xpu.mem_get_info always returns tile 0's memory stats regardless
    # of which device is queried (L0 bug). Sequential init means rank 0's vLLM
    # weights + KV appear in tile 0's mem_get_info when rank 1 profiles, making
    # rank 1's memory budget check report 0 KV blocks. Sleeping rank 0's vLLM
    # moves tensors to CPU; empty_cache() then releases PyTorch's reserved GPU
    # pool back to the L0 driver so mem_get_info shows ~64 GiB free on tile 0.
    # Safe to call here: FSDP is not yet initialized (no IPC handles open), so
    # the XPU empty_cache UR-handle-leak bug (FSDP+empty_cache) does not apply.
    if vllm_mode == "colocate_sleep" and rank < world_size - 1:
        log.info(
            "Rank %d: sleeping vLLM immediately after init (next rank needs clean mem_get_info)",
            rank,
        )
        self._vllm_llm.sleep(level=1)
        self._vllm_is_sleeping = True
        if self._device.type == "xpu":
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
            log.info("Rank %d: empty_cache() after sleep — GPU memory released to driver", rank)

    # Restore monkey-patches
    UniProcExecutor._distributed_args = _orig_distributed_args
    torch.distributed.new_group = _orig_new_group
    torch.distributed.all_reduce = _orig_all_reduce

    # Destroy vLLM's gloo PG so we can init the training XCCL PG.
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    import vllm.distributed.parallel_state as vllm_ps
    vllm_ps._WORLD = None

    try:
        os.unlink(_store_file)
    except OSError:
        pass

    # Restore torchrun env vars for the training PG
    for key, val in saved_env.items():
        if val is not None:
            os.environ[key] = val
        elif key in os.environ:
            del os.environ[key]

    # Signal next rank that we're done
    my_barrier = os.path.join(barrier_dir, f"rank_{rank}_done")
    with open(my_barrier, "w") as f:
        f.write("done")

    # Wait for ALL ranks to finish vLLM init before returning.
    # CCL init (called immediately after _init_vllm_early) maps IPC handles
    # for every peer tile. If rank 0 enters CCL while rank 1 is still in
    # its vLLM memory-profiling run, CCL maps rank 0's vLLM weights+KV
    # (~24 GiB) as non-torch IPC memory on tile 1, causing rank 1's vLLM
    # KV-cache budget check to fail ("No available memory for cache blocks").
    for wait_rank in range(world_size):
        wait_barrier = os.path.join(barrier_dir, f"rank_{wait_rank}_done")
        while not os.path.exists(wait_barrier):
            time.sleep(0.2)
    log.info(
        "Rank %d: all %d ranks completed vLLM init, proceeding to CCL init",
        rank, world_size,
    )

    # Last rank cleans up barrier files
    if rank == world_size - 1:
        time.sleep(0.5)
        for i in range(world_size):
            try:
                os.unlink(os.path.join(barrier_dir, f"rank_{i}_done"))
            except OSError:
                pass



def _init_vllm_tp(self, cfg, rank, world_size, local_rank, tp_size,
                   model_path, gpu_mem, max_model_len, max_num_seqs,
                   vllm_mode, LLM):
    """Initialize TP>1 colocated vLLM: one engine per DP group via external_launcher.

    Groups ranks into DP groups of tp_size. Each group creates an XCCL PG
    of size tp_size and a shared vLLM engine with external_launcher.
    All ranks in a group must call generate() with identical prompts.

    Pattern:
    1. File-based barrier (all ranks in DP group signal ready)
    2. Override env: WORLD_SIZE, RANK, LOCAL_RANK, MASTER_PORT; disable elastic agent store
    3. Create TP-sized XCCL PG per DP group (TCP rendezvous, unique port)
    4. Create vLLM LLM with external_launcher
    5. Destroy TP PG, restore env (training PG created later via elastic agent store)
    """
    assert world_size % tp_size == 0, (
        f"world_size={world_size} must be divisible by vllm_tensor_parallel_size={tp_size}"
    )
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size
    tp_rank = rank % tp_size

    # Store TP/DP info for later use
    self._vllm_dp_rank = dp_rank
    self._vllm_tp_rank = tp_rank
    self._vllm_dp_size = dp_size

    log.info(
        "Rank %d: TP init — dp_rank=%d, tp_rank=%d, dp_size=%d, tp_size=%d",
        rank, dp_rank, tp_rank, dp_size, tp_size,
    )

    # Step 1: File-based barrier — all ranks signal ready before any creates PG.
    # torchrun sets TORCHELASTIC_USE_AGENT_STORE=True; consuming it for a
    # throwaway global PG would prevent reuse for the training PG later.
    _run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
    _barrier_dir = f"/tmp/torchtune/vllm_tp_barrier_{_run_id}"
    os.makedirs(_barrier_dir, exist_ok=True)
    # Signal this rank is ready
    with open(os.path.join(_barrier_dir, f"rank_{rank}"), "w") as f:
        f.write("ready")
    # Wait for all ranks in our DP group to be ready
    dp_group_ranks = list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
    for r in dp_group_ranks:
        while not os.path.exists(os.path.join(_barrier_dir, f"rank_{r}")):
            time.sleep(0.1)
    log.info("Rank %d: all %d ranks in DP group %d ready", rank, tp_size, dp_rank)

    # Step 2: Override env for TP subgroup
    saved_env = {}
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_PORT",
                "TORCHELASTIC_USE_AGENT_STORE"):
        saved_env[key] = os.environ.get(key)

    original_port = os.environ.get("MASTER_PORT", "29500")
    os.environ["WORLD_SIZE"] = str(tp_size)
    os.environ["RANK"] = str(tp_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    # Unique port per DP group to avoid collisions
    os.environ["MASTER_PORT"] = str(int(original_port) + dp_rank + 1)
    # Disable elastic agent store — use TCP rendezvous for TP subgroup
    os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)

    # Step 3: Create XCCL PG for this TP subgroup
    torch.distributed.init_process_group(
        backend="xccl",
        world_size=tp_size,
        rank=tp_rank,
    )
    log.info("Rank %d: TP subgroup PG initialized (dp_group=%d, size=%d)",
             rank, dp_rank, torch.distributed.get_world_size())

    # Step 5: Create vLLM LLM engine
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp_size,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enforce_eager=True,
        dtype="bfloat16",
    )
    if vllm_mode == "colocate_sleep":
        llm_kwargs["enable_sleep_mode"] = True

    self._vllm_llm = LLM(**llm_kwargs)

    # Step 6: Destroy TP PG — training will create its own global XCCL PG
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    import vllm.distributed.parallel_state as vllm_ps
    vllm_ps._WORLD = None

    # Restore env vars
    for key, val in saved_env.items():
        if val is not None:
            os.environ[key] = val
        elif key in os.environ:
            del os.environ[key]

    # Clean up barrier files (best-effort)
    if rank == 0:
        import shutil
        try:
            shutil.rmtree(_barrier_dir, ignore_errors=True)
        except OSError:
            pass



def _setup_vllm_server_mode(self):
    """Initialize vLLM in server mode.

    Only global rank 0 creates HTTP clients and calls vLLM for generation.
    Results are broadcast to all ranks via the world process group.

    NOTE: XCCL on Aurora deadlocks when using shard-level sub-communicators,
    so per-node vLLM generation with shard-level broadcast is not possible.
    Only rank 0's vLLM server is used; other nodes' vLLM servers are idle.
    """
    should_init_client = self._is_rank_zero

    if should_init_client:
        from torchtune.dev.rl.vllm_client import VLLMClient

        # Create clients for all vLLM URLs (supports DP vLLM replicas)
        self._vllm_clients = []
        for url in self._vllm_urls:
            client = VLLMClient(
                base_url=url,
                group_port=self._vllm_group_port,
                connection_timeout=900.0,
            )
            self._vllm_clients.append(client)
        self._vllm_client = self._vllm_clients[0]  # backward compat

        if self._vllm_weight_sync:
            # On XPU, creating a second ProcessGroupXCCL (for weight sync)
            # alongside the training XCCL PG causes SIGABRT. Use file-based
            # weight sync instead: save to /tmp, POST to vLLM to reload.
            self._build_tune_to_hf_map()
            # Use /dev/shm (RAM-backed tmpfs, 504 GB on Aurora) for fast I/O.
            # Raw bytes format (.raw) replaces safetensors — uses memcpy at DDR5
            # bandwidth instead of CPU serialization. For 62 GB BF16: ~2s vs ~47s.
            # Async: gather is synchronous (FSDP collective), but save+HTTP runs
            # in a background thread overlapping with the next generation step.
            # Default to node-local /dev/shm (RAM-backed tmpfs, fastest).
            # In multi-node server mode (train and vLLM on different nodes),
            # /dev/shm is NOT shared — override with TORCHTUNE_WEIGHT_SYNC_PATH
            # to point at a shared FS path (e.g. /lus/flare/.../weight_update.raw).
            self._weight_sync_path = os.environ.get(
                "TORCHTUNE_WEIGHT_SYNC_PATH",
                "/dev/shm/torchtune/weight_update.raw",
            )
            # Async sync state: event is set when no sync is in progress (safe to generate).
            self._sync_done_event = threading.Event()
            self._sync_done_event.set()  # initially clear (no sync running)
            self._sync_error = None      # captured exception from background thread
            self._xccl_bcast_buf = None  # static GPU buffer for XCCL broadcast (VA pinning)
            # Sync dispatch counter — used by raw_bytes _sync_weights_to_vllm to
            # tag each in-flight sync. xccl path initializes these in
            # _init_sender_pool but the raw_bytes path (BioReason) calls
            # _sync_weights_to_vllm directly and would AttributeError without
            # these defaults.
            self._sync_id_counter = 0
            self._pending_sync_id = None
            log.info(
                "Rank %d: vLLM %d client(s) initialized: %s (%d params mapped, async raw-bytes sync via %s, interval=%d)",
                self.rank,
                len(self._vllm_clients),
                ",".join(self._vllm_urls),
                len(self._tune_to_hf_map),
                self._weight_sync_path,
                self._vllm_weight_sync_interval,
            )
        else:
            log.info(
                "Rank %d: vLLM %d client(s) initialized (generation only, no weight sync): %s",
                self.rank, len(self._vllm_clients), ",".join(self._vllm_urls),
            )

    # NOTE: Do NOT use shard_pg for any explicit collectives on Aurora.
    # XCCL deadlocks when creating sub-communicators from device_mesh groups.
    # FSDP2 internally manages its own communicators and works fine, but
    # explicit operations on dp_mesh.get_group("dp_shard") deadlock.
    if not self._production_mode:
        torch.distributed.barrier()

    # Phase 1: rollout producer. Driven by _async_lookahead_iter() inside
    # train(). The async path stashes prefetched query_responses here;
    # generate_trajectory's server-mode branch consumes it. Always None
    # in sync mode.
    self._pending_async_query_responses = None
    # Step 2 (Phase 2 plan): producer-class handles. The producer itself
    # is constructed lazily inside _async_lookahead_iter when it sees
    # the first epoch's dataloader; the version tracker is global so we
    # can bump it from the wsync site even before the first batch.
    from torchtune.dev.rl.async_rollout import WeightVersionTracker
    self._weight_versions = WeightVersionTracker()
    self._rollout_producer = None
    self._last_rollout_item = None
    if self._async_generation_enabled:
        log.info(
            "Rank %d: async generation enabled (max_staleness=%d).",
            self.rank, self._async_generation_max_staleness,
        )



def _setup_vllm_colocate_mode(self, cfg):
    """Initialize vLLM in colocate mode: every rank has its own vLLM engine.

    All ranks have vLLM engines (created in _init_vllm_early). Each rank
    generates all completions for its own prompts locally (no cross-rank
    communication during generation). Weight sync loads gathered FSDP2
    params into each rank's local engine.

    Called from setup() AFTER model init.
    """
    # Build param name mapping for weight sync
    self._build_tune_to_hf_map()

    log.info(
        "Rank %d: colocated vLLM engine ready (%d params mapped, local generation)",
        self.rank, len(self._tune_to_hf_map),
    )
    torch.distributed.barrier()


def _create_dedicated_pgs(self) -> None:
    """Create _training_pg (xccl), _wsync_pg (gloo/xccl), and _gen_pg (gloo) for dedicated_rank mode.

    Must be called in the same order on ALL ranks — training ranks and the dedicated
    vLLM rank both call this so that new_group() ordering is consistent (any mismatch
    deadlocks). PG backend is controlled by TORCHTUNE_WSYNC_BACKEND (default: gloo).

    `_gen_pg` is the Phase-2 generation ring buffer: a 2-rank gloo group [0, vllm_rank]
    that carries generation requests/responses, so the dedicated vLLM rank no longer
    has to participate in world-PG broadcasts each step. Always gloo (XCCL would
    deadlock with the producer-thread/consumer-thread overlap on the same XPU).
    """
    _training_ranks = list(range(self.world_size - 1))
    self._training_pg = torch.distributed.new_group(_training_ranks, backend="xccl")

    import torch.distributed.distributed_c10d as _dc10d

    _wsync_backend = os.environ.get("TORCHTUNE_WSYNC_BACKEND", "gloo")
    if _wsync_backend == "gloo":
        _default_pg = _dc10d._get_default_group()
        _orig_bound = _default_pg.bound_device_id
        _default_pg.bound_device_id = None
        try:
            self._wsync_pg = torch.distributed.new_group(
                [0, self._vllm_dedicated_rank], backend=_wsync_backend
            )
        finally:
            _default_pg.bound_device_id = _orig_bound
    else:
        self._wsync_pg = torch.distributed.new_group(
            [0, self._vllm_dedicated_rank], backend=_wsync_backend
        )

    _default_pg = _dc10d._get_default_group()
    _orig_bound = _default_pg.bound_device_id
    _default_pg.bound_device_id = None
    try:
        self._gen_pg = torch.distributed.new_group(
            [0, self._vllm_dedicated_rank], backend="gloo"
        )
    finally:
        _default_pg.bound_device_id = _orig_bound

    # Phase 2: gloo fan-out PG for query_responses broadcast among training
    # ranks ([0..N-2]). The xccl _training_pg is brittle for the *first* ever
    # collective fired on it (FSDP2 reduce_scatter is patched to gloo, so the
    # xccl sub-PG is otherwise unexercised → fi_cq_readerr EPERM on first use).
    # Gloo is fast enough for the small qr payload (~49 KB at G=8 max_gen=512).
    _default_pg = _dc10d._get_default_group()
    _orig_bound = _default_pg.bound_device_id
    _default_pg.bound_device_id = None
    try:
        self._training_fanout_pg = torch.distributed.new_group(
            _training_ranks, backend="gloo"
        )
    finally:
        _default_pg.bound_device_id = _orig_bound


def _setup_dedicated_vllm_rank(self, cfg) -> None:
    """Generic setup for the dedicated vLLM generation server rank.

    Creates _training_pg (xccl) and _wsync_pg (gloo) then seeds generation
    hyperparams (_total_steps, _temperature, etc.) from cfg. Called after
    init_process_group() completes on the vLLM rank.

    Subclasses that use dedicated_rank mode (e.g. BioReason) call this first,
    then layer model-specific setup (embed model, _compute_wsync_layout) on top.
    """
    _create_dedicated_pgs(self)

    self._total_steps = cfg.get("num_steps", 10000)
    self._max_generated_tokens = cfg.get("max_generated_tokens", 512)
    self._temperature = cfg.get("temperature", 0.8)
    self._top_k = cfg.get("top_k", None)
    self.grpo_samples = cfg.get("grpo_samples", 8)

    log.info(
        "Rank %d (dedicated vLLM): training_pg=[0..%d] wsync_pg=[0,%d] backend=%s total_steps=%d",
        self.rank, self.world_size - 2, self._vllm_dedicated_rank,
        os.environ.get("TORCHTUNE_WSYNC_BACKEND", "gloo"), self._total_steps,
    )


def _setup_dedicated_training_pgs(self, cfg) -> None:
    """Create process groups on training ranks for dedicated_rank weight sync.

    Mirrors _setup_dedicated_vllm_rank: both must call new_group in identical
    order. Call this from setup() before any FSDP wrapping that uses _training_pg.
    """
    _create_dedicated_pgs(self)

    log.info(
        "Rank %d (dedicated training): training_pg=[0..%d] wsync_pg=[0,%d] backend=%s",
        self.rank, self.world_size - 2, self._vllm_dedicated_rank,
        os.environ.get("TORCHTUNE_WSYNC_BACKEND", "gloo"),
    )

