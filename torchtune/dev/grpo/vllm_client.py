# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Lightweight vLLM client for torchtune GRPO on XPU.

Communicates with TRL's ``vllm_serve.py`` (``WeightSyncWorkerExtension``) via HTTP
for generation and XCCL for weight synchronization. No dependency on TRL at runtime.
"""

import atexit
import logging
import socket
import time
from typing import Optional
from urllib.parse import urlparse

import torch
import torch.distributed.distributed_c10d as c10d

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for a vLLM generation server with weight-sync support over XCCL.

    The server is expected to expose:
      - ``GET  /health/``              – readiness probe
      - ``POST /generate/``            – token-id-in / token-id-out generation
      - ``GET  /get_world_size/``      – TP size of the server
      - ``POST /init_communicator/``   – bootstrap weight-update XCCL group
      - ``POST /update_named_param/``  – per-parameter weight push
      - ``POST /close_communicator/``  – tear down XCCL group
      - ``POST /reset_prefix_cache/``  – invalidate KV prefix cache

    These endpoints are provided by TRL's ``WeightSyncWorkerExtension``.

    Args:
        base_url: e.g. ``"http://localhost:8001"``
        group_port: TCP port for the weight-update ``TCPStore``.
        connection_timeout: seconds to wait for the server to become healthy.
    """

    def __init__(
        self,
        base_url: str,
        group_port: int = 51216,
        connection_timeout: float = 120.0,
    ):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.session = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            status=3,
            status_forcelist=[500, 502, 503],
            backoff_factor=2,
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        parsed = urlparse(base_url)
        self.host = socket.gethostbyname(parsed.hostname)
        scheme = parsed.scheme or "http"
        self.base_url = f"{scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
        self.group_port = group_port

        self.communicator: Optional[c10d.ProcessGroupXCCL] = None
        self.rank: Optional[int] = None

        self.check_server(connection_timeout)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def check_server(
        self, total_timeout: float = 120.0, retry_interval: float = 5.0
    ) -> None:
        """Block until the vLLM server responds to ``/health/``."""
        import requests as _requests

        url = f"{self.base_url}/health/"
        t0 = time.time()
        while True:
            try:
                r = _requests.get(url, timeout=10)
                if r.status_code == 200:
                    logger.info("vLLM server is up at %s", self.base_url)
                    break
            except _requests.exceptions.RequestException:
                pass
            if time.time() - t0 >= total_timeout:
                raise ConnectionError(
                    f"vLLM server not reachable at {self.base_url} after {total_timeout}s"
                )
            logger.info("Waiting for vLLM server… retrying in %.0fs", retry_interval)
            time.sleep(retry_interval)

        # Detect API type: TRL vllm_serve (/generate/) vs OpenAI API (/v1/completions)
        try:
            r = _requests.get(f"{self.base_url}/v1/models", timeout=10)
            if r.status_code == 200:
                self._api_type = "openai"
                data = r.json()
                self._model_name = data["data"][0]["id"] if data.get("data") else "default"
                logger.info("Detected OpenAI API server (model=%s)", self._model_name)
                return
        except Exception:
            pass
        self._api_type = "trl"
        self._model_name = None
        logger.info("Detected TRL vllm_serve API")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: list[list[int]],
        n: int = 1,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> list[list[int]]:
        """Send prompt token-IDs to vLLM, return completion token-IDs.

        Args:
            prompts: list of token-ID lists (one per sequence).
            n: number of completions per prompt.
            max_tokens: maximum generated tokens per completion.
            temperature: sampling temperature.
            top_k: top-k sampling (0 = disabled).

        Returns:
            ``completion_ids`` — list of token-ID lists, length ``len(prompts) * n``.
        """
        if self._api_type == "openai":
            return self._generate_openai(prompts, n, max_tokens, temperature, top_k)
        return self._generate_trl(prompts, n, max_tokens, temperature, top_k)

    def _generate_trl(
        self,
        prompts: list[list[int]],
        n: int,
        max_tokens: int,
        temperature: float,
        top_k: int,
    ) -> list[list[int]]:
        """Generate via TRL's /generate/ endpoint (token-ids in/out)."""
        url = f"{self.base_url}/generate/"
        payload = {
            "prompts": prompts,
            "n": n,
            "temperature": temperature,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "logprobs": None,
        }
        r = self.session.post(url, json=payload, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"vLLM /generate/ failed: {r.status_code} {r.text}")
        return r.json()["completion_ids"]

    def _generate_openai(
        self,
        prompts: list[list[int]],
        n: int,
        max_tokens: int,
        temperature: float,
        top_k: int,
    ) -> list[list[int]]:
        """Generate via OpenAI-compatible /v1/completions endpoint.

        The OpenAI API accepts prompt as token IDs via the ``prompt`` field.
        We batch all prompts in a single request so vLLM can schedule them
        concurrently (continuous batching).
        """
        comp_url = f"{self.base_url}/v1/completions"
        tok_url = f"{self.base_url}/tokenize"

        # Batch all prompts in one request — vLLM /v1/completions accepts
        # a list of prompts and returns choices ordered by prompt index.
        payload = {
            "model": self._model_name,
            "prompt": prompts,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "echo": False,
        }
        r = self.session.post(comp_url, json=payload, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(
                f"vLLM /v1/completions failed: {r.status_code} {r.text}"
            )
        data = r.json()
        if "choices" not in data:
            error_msg = data.get("error", data.get("message", str(data)))
            raise RuntimeError(f"vLLM returned error response: {error_msg}")

        all_completion_ids = []
        texts_to_tokenize = []
        text_indices = []

        for i, choice in enumerate(data["choices"]):
            token_ids = choice.get("token_ids")
            if token_ids:
                all_completion_ids.append(list(token_ids))
            else:
                # Collect texts for batch re-tokenization
                all_completion_ids.append(None)  # placeholder
                texts_to_tokenize.append(choice["text"])
                text_indices.append(i)

        # Re-tokenize any text outputs via /tokenize endpoint
        for idx, text in zip(text_indices, texts_to_tokenize):
            tok_r = self.session.post(
                tok_url,
                json={"model": self._model_name, "prompt": text},
                timeout=30,
            )
            if tok_r.status_code != 200:
                raise RuntimeError(
                    f"vLLM /tokenize failed: {tok_r.status_code} {tok_r.text}"
                )
            all_completion_ids[idx] = tok_r.json()["tokens"]

        return all_completion_ids

    # ------------------------------------------------------------------
    # Weight sync – XCCL communicator
    # ------------------------------------------------------------------
    def init_communicator(self, device: torch.device) -> None:
        """Bootstrap a weight-update XCCL process group with the vLLM server.

        The client (this process) joins as the *last* rank in the group.
        """
        import requests as _requests

        # 1. Learn the server's TP world size
        r = _requests.get(f"{self.base_url}/get_world_size/", timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"/get_world_size/ failed: {r.status_code} {r.text}")
        vllm_world_size = r.json()["world_size"]

        world_size = vllm_world_size + 1  # +1 for this client
        self.rank = vllm_world_size       # client is the last rank

        # 2. Get device UUID (best-effort; Aurora may not expose it yet)
        if hasattr(torch.xpu, "get_device_properties"):
            props = torch.xpu.get_device_properties(device)
            client_uuid = str(getattr(props, "uuid", "42"))
        else:
            client_uuid = "42"

        # 3. Tell the server to initialize its side of the communicator
        r = self.session.post(
            f"{self.base_url}/init_communicator/",
            json={
                "host": "0.0.0.0",
                "port": self.group_port,
                "world_size": world_size,
                "client_device_uuid": client_uuid,
            },
            timeout=120,
        )
        if r.status_code != 200:
            raise RuntimeError(f"/init_communicator/ failed: {r.status_code} {r.text}")

        time.sleep(0.5)  # let server socket bind

        # 4. Create our side of the XCCL process group
        store = torch.distributed.TCPStore(
            host_name=self.host,
            port=self.group_port,
            world_size=world_size,
            is_master=(self.rank == 0),
        )
        prefixed_store = c10d.PrefixStore("client2server", store)
        xccl_options = c10d.ProcessGroupXCCL.Options()
        self.communicator = c10d.ProcessGroupXCCL(
            store=prefixed_store,
            rank=self.rank,
            size=world_size,
            options=xccl_options,
        )
        logger.info(
            "XCCL weight-sync communicator ready (rank=%d, world=%d)",
            self.rank,
            world_size,
        )
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        """Push a single named parameter to the vLLM server via XCCL broadcast."""
        if self.communicator is None:
            raise RuntimeError("Communicator not initialized — call init_communicator first")

        dtype_str = str(weights.dtype)
        shape = tuple(weights.shape)

        r = self.session.post(
            f"{self.base_url}/update_named_param/",
            json={"name": name, "dtype": dtype_str, "shape": shape},
            timeout=120,
        )
        if r.status_code != 200:
            raise RuntimeError(f"/update_named_param/ failed: {r.status_code} {r.text}")

        # Broadcast from client (root=self.rank) to all server workers
        self.communicator.broadcast(weights, root=self.rank)
        self.communicator.barrier()

    def reset_prefix_cache(self) -> None:
        """Invalidate vLLM's prefix cache after weight update."""
        try:
            r = self.session.post(f"{self.base_url}/reset_prefix_cache/", timeout=30)
            if r.status_code != 200:
                logger.warning("reset_prefix_cache failed: %s", r.text)
        except Exception:
            logger.warning("reset_prefix_cache request failed", exc_info=True)

    def close_communicator(self) -> None:
        """Tear down the weight-update XCCL group."""
        try:
            self.session.post(f"{self.base_url}/close_communicator/", timeout=10)
        except Exception:
            pass  # server may already be down
        if self.communicator is not None:
            del self.communicator
            self.communicator = None
            self.rank = None
