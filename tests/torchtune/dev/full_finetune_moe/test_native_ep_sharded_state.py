# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from torchtune import training


WORLD_SIZE = 2
RECIPE_PATH = (
    Path(__file__).parents[4] / "recipes/dev/full_finetune_moe_distributed_xpu.py"
)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(8, 8, bias=False)
        self.experts = torch.nn.Linear(8, 8, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dense(inputs) + self.experts(inputs)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers[0](inputs)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _run_native_ep_state(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        torch.manual_seed(0)
        model = _Model()
        expert_param = model.layers[0].experts.weight
        expert_id = id(expert_param)
        mesh = init_device_mesh("cpu", (WORLD_SIZE,))

        training.shard_model(
            model,
            [training.get_shard_conditions],
            cpu_offload=False,
            dp_mesh=mesh,
            reshard_after_forward=False,
            ignored_params={expert_param},
        )

        assert id(model.layers[0].experts.weight) == expert_id
        assert not isinstance(model.layers[0].experts.weight, DTensor)
        assert isinstance(model.layers[0].dense.weight, DTensor)

        model(torch.randn(4, 8)).sum().backward()

        assert isinstance(model.layers[0].experts.weight.grad, torch.Tensor)
        assert not isinstance(model.layers[0].experts.weight.grad, DTensor)
        assert isinstance(model.layers[0].dense.weight.grad, DTensor)
    finally:
        dist.destroy_process_group()


def test_native_ep_parameters_remain_outside_nested_fsdp() -> None:
    mp.spawn(
        _run_native_ep_state,
        args=(_free_port(),),
        nprocs=WORLD_SIZE,
        join=True,
    )


def test_native_fsdp_reduce_requires_native_expert_state() -> None:
    source = RECIPE_PATH.read_text()
    assert "native_fsdp_grad_reduce requires native_ep_sharded_experts" in source
    assert "if self._ep_active and not self._native_fsdp_grad_reduce:" in source
    assert 'with _moe_timed("manual_grad_release_total"):' in source
