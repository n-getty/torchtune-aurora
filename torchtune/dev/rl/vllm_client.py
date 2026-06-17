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
        stop_token_ids: Optional[list[int]] = None,
        stop: Optional[list[str]] = None,
    ) -> list[list[int]]:
        """Send prompt token-IDs to vLLM, return completion token-IDs.

        Args:
            prompts: list of token-ID lists (one per sequence).
            n: number of completions per prompt.
            max_tokens: maximum generated tokens per completion.
            temperature: sampling temperature.
            top_k: top-k sampling (0 = disabled).
            stop_token_ids: token IDs that terminate generation. Forwarded to
                vLLM's SamplingParams.stop_token_ids. ``None`` preserves the
                server-side default (which on most stacks is just the model EOS
                returned via tokenizer config — NOT the tokenizer's full
                ``stop_tokens`` list).
            stop: list of strings that terminate generation. Forwarded to
                vLLM's SamplingParams.stop. Required when the model never
                naturally emits EOS (raw pretraining checkpoints) and the
                only learnable stop signal is a format marker like
                ``</answer>`` or a conversational turn ``User:``.

        Returns:
            ``completion_ids`` — list of token-ID lists, length ``len(prompts) * n``.
        """
        if self._api_type == "openai":
            return self._generate_openai(
                prompts, n, max_tokens, temperature, top_k, stop_token_ids, stop
            )
        return self._generate_trl(
            prompts, n, max_tokens, temperature, top_k, stop_token_ids, stop
        )

    def _generate_trl(
        self,
        prompts: list[list[int]],
        n: int,
        max_tokens: int,
        temperature: float,
        top_k: int,
        stop_token_ids: Optional[list[int]] = None,
        stop: Optional[list[str]] = None,
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
        if stop_token_ids:
            payload["stop_token_ids"] = list(stop_token_ids)
        if stop:
            payload["stop"] = list(stop)
            # Keep the stop string in the output (see _generate_openai for why).
            payload["include_stop_str_in_output"] = True
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
        stop_token_ids: Optional[list[int]] = None,
        stop: Optional[list[str]] = None,
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
        if stop_token_ids:
            # vLLM's OpenAI server accepts SamplingParams extras at top level.
            payload["stop_token_ids"] = list(stop_token_ids)
        if stop:
            payload["stop"] = list(stop)
            # Keep the stop string in the output so downstream regex-based
            # reward extractors (e.g. ThinkingAnswerFormattingReward, which
            # requires ``</answer>`` to match) still see the format markers.
            payload["include_stop_str_in_output"] = True
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
    # Multimodal generation (prompt_embeds, OpenAI API only)
    # ------------------------------------------------------------------
    def generate_from_embeds(
        self,
        prompt_embeds: list[torch.Tensor],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> list[list[int]]:
        """Send pre-computed prompt embeddings to vLLM and return completion token IDs.

        Used by BioReason where the embeddings come from a multimodal pipeline
        (ESM3 + GO encoder + projectors) computed on the training side; vLLM only
        runs the LM backbone.

        The wire format matches vLLM's CompletionRequest spec:
          prompt_embeds: bytes (base64 of torch.save'd tensor) for one prompt,
                         OR list[bytes] for a batch.

        Args:
            prompt_embeds: list of length B, each element a [P_i, H] bf16/fp16 tensor.
            max_tokens, temperature, top_k, top_p: sampling params.

        Returns:
            list[list[int]] of length B — token IDs for each completion.
        """
        if self._api_type != "openai":
            raise RuntimeError(
                "generate_from_embeds requires the OpenAI API server "
                "(launch vLLM with vllm.entrypoints.openai.api_server "
                "--enable-prompt-embeds)."
            )

        import base64
        import io

        comp_url = f"{self.base_url}/v1/completions"
        encoded = []
        for t in prompt_embeds:
            buf = io.BytesIO()
            # NOTE: use torch.save (NOT tensor.numpy().tobytes()) — vLLM's
            # OpenAIServingCompletion calls torch.load(...) on the bytes.
            torch.save(t.detach().cpu().contiguous(), buf)
            encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))

        payload = {
            "model": self._model_name,
            "prompt_embeds": encoded if len(encoded) > 1 else encoded[0],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_k and top_k > 0:
            payload["top_k"] = top_k

        r = self.session.post(comp_url, json=payload, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(
                f"vLLM /v1/completions (prompt_embeds) failed: "
                f"{r.status_code} {r.text[:500]}"
            )
        data = r.json()
        if "choices" not in data:
            error_msg = data.get("error", data.get("message", str(data)))
            raise RuntimeError(f"vLLM error response: {error_msg}")

        # Token IDs may not be present in every choice — fall back to /tokenize.
        tok_url = f"{self.base_url}/tokenize"
        out = []
        for choice in data["choices"]:
            ids = choice.get("token_ids")
            if ids:
                out.append(list(ids))
            else:
                tok_r = self.session.post(
                    tok_url,
                    json={"model": self._model_name, "prompt": choice["text"]},
                    timeout=30,
                )
                if tok_r.status_code != 200:
                    raise RuntimeError(
                        f"/tokenize failed: {tok_r.status_code} {tok_r.text[:500]}"
                    )
                out.append(tok_r.json()["tokens"])
        return out

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


def vllm_http_generate(
    batch_input_ids: torch.Tensor,
    context_length: int,
    *,
    vllm_clients: list,
    pad_id: int,
    eos_id: Optional[int],
    max_generated_tokens: int,
    vllm_max_model_len: int,
    temperature: float,
    top_k: Optional[int],
    stop_token_ids: Optional[torch.Tensor] = None,
    stop_strings: Optional[list] = None,
    device=None,
) -> torch.Tensor:
    """Single-generator vLLM HTTP round-trip — no collectives, producer-safe.

    Shared by the dense GRPO recipe and the LoRA-GRPO recipe so both get the
    same prompt-truncation / stop-token / EOS-injection behavior. The caller is
    responsible for broadcasting the result to other ranks. All generators share
    one ``vllm_clients`` pool and fan their prompts across it round-robin.

    Args:
        batch_input_ids: [bsz, context_length] prompt token ids (padded).
        context_length: prompt length; output is [bsz, context_length + max_gen].
        vllm_clients: list of VLLMClient; >1 fans out round-robin via threads.
        pad_id: tokenizer pad id (stripped from prompts, fills the output).
        eos_id: tokenizer eos id; injected at the first pad position after a
            short completion so downstream stop-token truncation finds a real
            boundary. ``None`` disables injection.
        max_generated_tokens: max new tokens per prompt.
        vllm_max_model_len: prompts are left-truncated to
            ``vllm_max_model_len - max_generated_tokens`` to avoid vLLM overflow.
        temperature, top_k: sampling params (top_k None/0 → 0).
        stop_token_ids: optional tensor of stop token ids forwarded to vLLM.
        stop_strings: optional list of stop strings forwarded to vLLM.
        device: device for the output tensor (defaults to input's device).

    Returns:
        [bsz, context_length + max_generated_tokens] query+response tensor.
    """
    bsz = batch_input_ids.shape[0]
    total_len = context_length + max_generated_tokens
    if device is None:
        device = batch_input_ids.device

    # Strip padding, left-truncate to the model-len budget, convert to lists.
    max_prompt_len = vllm_max_model_len - max_generated_tokens
    prompts = []
    for i in range(bsz):
        ids = batch_input_ids[i].cpu().tolist()
        ids = [t for t in ids if t != pad_id]
        prompts.append(ids[-max_prompt_len:] if len(ids) > max_prompt_len else ids)

    gen_kwargs = dict(
        n=1,  # prompts already expanded by grpo_samples
        max_tokens=max_generated_tokens,
        temperature=temperature,
        top_k=top_k or 0,
    )
    # Forward stop tokens so generation halts at EOS instead of running to
    # max_tokens. Forward stop strings for raw checkpoints that never emit EOS
    # (e.g. ``</answer>``) — only when configured.
    if stop_token_ids is not None and stop_token_ids.numel() > 0:
        gen_kwargs["stop_token_ids"] = stop_token_ids.cpu().tolist()
    if stop_strings:
        gen_kwargs["stop"] = list(stop_strings)

    t0 = time.perf_counter()
    num_clients = len(vllm_clients)
    if num_clients > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunks = [prompts[i::num_clients] for i in range(num_clients)]

        def _call_vllm(client, chunk):
            return client.generate(prompts=chunk, **gen_kwargs) if chunk else []

        with ThreadPoolExecutor(max_workers=num_clients) as pool:
            futures = {
                pool.submit(_call_vllm, client, chunk): idx
                for idx, (client, chunk) in enumerate(zip(vllm_clients, chunks))
            }
            chunk_results = [None] * num_clients
            for future in as_completed(futures):
                idx = futures[future]
                chunk_results[idx] = future.result()

        completions = [None] * bsz
        for i in range(bsz):
            completions[i] = chunk_results[i % num_clients][i // num_clients]
    else:
        completions = vllm_clients[0].generate(prompts=prompts, **gen_kwargs)
    gen_time = time.perf_counter() - t0

    query_responses = batch_input_ids.new_full((bsz, total_len), pad_id)
    query_responses[:, :context_length] = batch_input_ids
    # Inject one EOS at the first pad after the generated tokens so the
    # stop-token truncation logic finds the real boundary (no-op when the
    # completion already filled max_generated_tokens or contains EOS).
    for i, comp in enumerate(completions):
        length = min(len(comp), max_generated_tokens)
        query_responses[i, context_length : context_length + length] = torch.tensor(
            comp[:length], dtype=batch_input_ids.dtype, device=device
        )
        if eos_id is not None and length < max_generated_tokens:
            query_responses[i, context_length + length] = eos_id

    total_tokens = sum(len(c) for c in completions)
    logger.info(
        "vLLM generation: %d sequences (%d clients), %d tokens in %.1fs (%.1f tok/s)",
        bsz, num_clients, total_tokens, gen_time, total_tokens / max(gen_time, 0.01),
    )
    return query_responses
