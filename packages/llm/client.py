"""Provider-agnostic chat-LLM client.

The pipeline currently calls 12 sites with the Bedrock Converse API:

    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": "..."}, {"image": {...}}]}],
        inferenceConfig={"temperature": 0.0, "maxTokens": 4096},
    )
    text = response["output"]["message"]["content"][0]["text"]

Rather than refactor every callsite, this module ships an `OpenAIChatShim`
and a `LocalVLLMChatShim` that expose the SAME
`.converse(modelId=, messages=, inferenceConfig=)` API on top of, respectively,
the OpenAI Python SDK and an in-process vLLM bundle.

Provider selection is driven by the `LLM_PROVIDER` environment variable so
no code changes are needed to swap backends:

    LLM_PROVIDER=bedrock                     (default)
    LLM_PROVIDER=openai      OPENAI_API_KEY=sk-...
    LLM_PROVIDER=azure_openai
        AZURE_OPENAI_API_KEY=...
        AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/v1/
    LLM_PROVIDER=local       LOCAL_LLM_MODEL_PATH=/path/to/hf-model-dir
        Optional: LOCAL_LLM_MAX_MODEL_LEN=8192
                  LOCAL_LLM_GPU_MEMORY_UTILIZATION=0.9
                  LOCAL_LLM_DTYPE=auto
                  LOCAL_LLM_TENSOR_PARALLEL_SIZE=1

The OpenAI shim translates only the message envelope:

  - Bedrock content blocks `{"text": "..."}` and
    `{"image": {"format": "png", "source": {"bytes": b"..."}}}` map to
    OpenAI's `{"type": "text", ...}` and
    `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`.
  - Response `choices[0].message.content` is wrapped back into
    `{"output": {"message": {"content": [{"text": "..."}]}}}` so existing
    callers' parsing code (`response["output"]["message"]["content"][0]["text"]`)
    keeps working unchanged.

The local vLLM shim is text-only: image content blocks are dropped silently.
The OCR reparse path (apps/pdf_checker/ingest/reference_segmenter.py) loads
its own multimodal vLLM bundle and is NOT replaced by this shim.
"""
from __future__ import annotations

import base64
import os
import queue as _queue
import threading
import time as _time
from concurrent.futures import Future as _Future
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# OpenAI-compatible shim
# ---------------------------------------------------------------------------

class OpenAIChatShim:
    """Wraps an `openai.OpenAI` client to expose Bedrock's `.converse(...)` API.

    Only translates the message envelope and the response shape. Inference
    parameters (temperature, max_tokens) pass through unchanged.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def converse(
        self,
        modelId: str,
        messages: list[dict[str, Any]],
        inferenceConfig: dict[str, Any] | None = None,
        **_unused_kwargs: Any,
    ) -> dict[str, Any]:
        cfg = inferenceConfig or {}
        max_tokens = int(cfg.get("maxTokens", 4096))
        temperature = float(cfg.get("temperature", 0.0))

        openai_messages = [self._translate_message(m) for m in messages]

        kwargs: dict[str, Any] = {
            "model": modelId,
            "messages": openai_messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # GPT-5 / o-series only accept the default temperature. Strip it
            # and retry once when the API rejects the parameter.
            msg = str(exc).lower()
            if "temperature" in msg and ("unsupported" in msg or "does not support" in msg):
                kwargs.pop("temperature", None)
                response = self._client.chat.completions.create(**kwargs)
            else:
                raise
        text = ""
        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            text = ""
        return {"output": {"message": {"content": [{"text": text}]}}}

    @staticmethod
    def _translate_message(msg: dict[str, Any]) -> dict[str, Any]:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            return {"role": role, "content": content}
        if not isinstance(content, list):
            return {"role": role, "content": str(content) if content is not None else ""}
        out_blocks: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if "text" in block:
                out_blocks.append({"type": "text", "text": str(block["text"])})
                continue
            if "image" in block:
                img = block["image"] or {}
                fmt = (img.get("format") or "png").lower()
                src = img.get("source") or {}
                raw = src.get("bytes")
                if isinstance(raw, (bytes, bytearray)):
                    b64 = base64.b64encode(bytes(raw)).decode("ascii")
                    out_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{fmt};base64,{b64}"},
                    })
                continue
            # Unrecognised block type: drop silently rather than failing.
        if not out_blocks:
            out_blocks.append({"type": "text", "text": ""})
        return {"role": role, "content": out_blocks}


# ---------------------------------------------------------------------------
# Local vLLM shim (text-only)
# ---------------------------------------------------------------------------

class _VLLMBatcher:
    """Coalesce concurrent .converse() calls into batched vLLM inference.

    vLLM's offline ``LLM.generate`` is most efficient when handed many prompts
    at once (continuous batching saturates the GPU). Caller threads call
    ``submit(prompt, sampling_params)`` and block on a Future; a background
    dispatcher thread pops up to ``BATCH_MAX`` requests within a short
    coalescing window and dispatches them as a single ``LLM.generate`` call,
    so a Python ThreadPoolExecutor with N workers naturally yields N-way
    batched generation rather than N serial single-prompt generations.
    """
    _BATCH_WINDOW_S = 0.05
    _BATCH_MAX = 64
    _SHUTDOWN_TIMEOUT = 5.0

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._q: _queue.Queue = _queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._loop, name="vllm-batcher", daemon=True,
        )
        self._worker.start()

    def submit(self, prompt: str, sampling_params: Any) -> str:
        fut: _Future = _Future()
        self._q.put((prompt, sampling_params, fut))
        return fut.result()

    def shutdown(self) -> None:
        self._stop.set()
        self._worker.join(timeout=self._SHUTDOWN_TIMEOUT)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._q.get(timeout=0.5)
            except _queue.Empty:
                continue
            items = [first]
            deadline = _time.perf_counter() + self._BATCH_WINDOW_S
            while len(items) < self._BATCH_MAX and _time.perf_counter() < deadline:
                wait = max(0.0, deadline - _time.perf_counter())
                try:
                    items.append(self._q.get(timeout=wait))
                except _queue.Empty:
                    break

            prompts = [it[0] for it in items]
            sp_list = [it[1] for it in items]
            futures = [it[2] for it in items]
            try:
                outputs = self._llm.generate(prompts, sp_list, use_tqdm=False)
                for fut, out in zip(futures, outputs):
                    text = ""
                    try:
                        text = out.outputs[0].text
                    except (IndexError, AttributeError):
                        text = ""
                    if not fut.done():
                        fut.set_result(text)
            except Exception as exc:  # noqa: BLE001 — propagate to all queued futures
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)


class LocalVLLMChatShim:
    """Wrap a ``vllm.LLM`` instance in the Bedrock-style ``.converse()`` API.

    Loaded once per (model_path, dtype, tp-size, max_model_len, gmu) tuple per
    process — see :data:`_LOCAL_LLM_SINGLETON`. Concurrent ``.converse()``
    calls are coalesced by :class:`_VLLMBatcher` so the GPU sees real batched
    inference even from a Python ThreadPoolExecutor.

    Image content blocks in the messages are dropped silently. The OCR reparse
    path uses its own multimodal vLLM bundle in
    ``apps/pdf_checker/ingest/reference_segmenter.py`` and is NOT served by
    this shim.
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 8192,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
    ) -> None:
        try:
            from vllm import LLM  # noqa: F401  (import-time dep check)
        except ImportError as exc:
            raise RuntimeError(
                "vllm is not installed; required for LLM_PROVIDER=local. "
                "`pip install vllm` (this environment already has vllm 0.8.5)."
            ) from exc
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed; required for LLM_PROVIDER=local."
            ) from exc

        from vllm import LLM
        print(
            f"[local-llm] loading vLLM bundle from {model_path} "
            f"(max_model_len={max_model_len} dtype={dtype} "
            f"gpu_memory_utilization={gpu_memory_utilization} "
            f"tp={tensor_parallel_size})",
            flush=True,
        )
        self._model_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code,
        )
        self._llm = LLM(
            model=model_path,
            max_model_len=max_model_len,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            disable_log_stats=True,
        )
        self._batcher = _VLLMBatcher(self._llm)
        print(f"[local-llm] ready ({model_path})", flush=True)

    def converse(
        self,
        modelId: str,  # noqa: N803 — Bedrock-API-compatible signature
        messages: list[dict[str, Any]],
        inferenceConfig: dict[str, Any] | None = None,  # noqa: N803
        **_unused_kwargs: Any,
    ) -> dict[str, Any]:
        from vllm import SamplingParams
        cfg = inferenceConfig or {}
        max_tokens = int(cfg.get("maxTokens", 4096))
        temperature = float(cfg.get("temperature", 0.0))
        chat_msgs = self._messages_to_chat(messages)
        prompt_text = self._tokenizer.apply_chat_template(
            chat_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0 if temperature == 0.0 else 0.95,
        )
        text = self._batcher.submit(prompt_text, sp)
        return {"output": {"message": {"content": [{"text": text}]}}}

    @staticmethod
    def _messages_to_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content")
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                out.append({
                    "role": role,
                    "content": str(content) if content is not None else "",
                })
                continue
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if "text" in block:
                    parts.append(str(block["text"]))
                # image blocks dropped (text-only shim).
            out.append({"role": role, "content": "\n".join(parts)})
        return out


_LOCAL_LLM_SINGLETON: dict[str, LocalVLLMChatShim] = {}
_LOCAL_LLM_LOCK = threading.Lock()


def _build_local_vllm_client(model_path: str | None = None) -> LocalVLLMChatShim:
    path = model_path or os.getenv("LOCAL_LLM_MODEL_PATH")
    if not path:
        raise RuntimeError(
            "local provider needs LOCAL_LLM_MODEL_PATH env var (or model_path arg) "
            "pointing at a HuggingFace-format model directory."
        )
    if not Path(path).is_dir():
        raise RuntimeError(f"LOCAL_LLM_MODEL_PATH={path!r} is not a directory.")

    with _LOCAL_LLM_LOCK:
        if path in _LOCAL_LLM_SINGLETON:
            return _LOCAL_LLM_SINGLETON[path]

        kwargs: dict[str, Any] = {}
        for env_key, arg_key, cast in (
            ("LOCAL_LLM_MAX_MODEL_LEN",          "max_model_len",           int),
            ("LOCAL_LLM_DTYPE",                  "dtype",                   str),
            ("LOCAL_LLM_GPU_MEMORY_UTILIZATION", "gpu_memory_utilization",  float),
            ("LOCAL_LLM_TENSOR_PARALLEL_SIZE",   "tensor_parallel_size",    int),
        ):
            ev = os.getenv(env_key)
            if ev:
                try:
                    kwargs[arg_key] = cast(ev)
                except ValueError:
                    pass
        client = LocalVLLMChatShim(path, **kwargs)
        _LOCAL_LLM_SINGLETON[path] = client
        return client


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _normalize_provider(provider: str | None) -> str:
    if not provider:
        provider = os.getenv("LLM_PROVIDER", "bedrock")
    return (provider or "bedrock").strip().lower()


def build_chat_client(
    provider: str | None = None,
    *,
    region: str = "us-east-1",
    bearer_token: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model_path: str | None = None,
) -> Any:
    """Return a chat client that exposes a Bedrock-compatible `.converse(...)`.

    `provider` defaults to the `LLM_PROVIDER` env var (or "bedrock" if unset).

    Bedrock kwargs:
        region        — AWS region (default us-east-1)
        bearer_token  — sets AWS_BEARER_TOKEN_BEDROCK in the process env

    OpenAI / Azure OpenAI kwargs:
        api_key       — overrides OPENAI_API_KEY / AZURE_OPENAI_API_KEY
        base_url      — overrides OPENAI_BASE_URL / AZURE_OPENAI_ENDPOINT

    Local-vLLM kwargs:
        model_path    — overrides LOCAL_LLM_MODEL_PATH; HF-format dir
                        (additional knobs via LOCAL_LLM_MAX_MODEL_LEN,
                        LOCAL_LLM_DTYPE, LOCAL_LLM_GPU_MEMORY_UTILIZATION,
                        LOCAL_LLM_TENSOR_PARALLEL_SIZE env vars)
    """
    p = _normalize_provider(provider)

    if p == "local":
        return _build_local_vllm_client(model_path=model_path)

    if p == "bedrock":
        os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is not installed. Install boto3 for the Bedrock provider, "
                "or set LLM_PROVIDER=openai / azure_openai."
            ) from exc
        if bearer_token:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = str(bearer_token)
        return boto3.client(service_name="bedrock-runtime", region_name=region)

    if p in ("openai", "azure_openai"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed. `pip install openai>=1.0` for the "
                "openai/azure_openai provider."
            ) from exc
        if p == "azure_openai":
            key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            url = base_url or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL")
            if not key:
                raise RuntimeError(
                    "azure_openai provider needs AZURE_OPENAI_API_KEY (or OPENAI_API_KEY)."
                )
            if not url:
                raise RuntimeError(
                    "azure_openai provider needs AZURE_OPENAI_ENDPOINT, e.g. "
                    "https://<resource>.openai.azure.com/openai/v1/"
                )
        else:  # plain openai
            key = api_key or os.getenv("OPENAI_API_KEY")
            url = base_url or os.getenv("OPENAI_BASE_URL")
            if not key:
                raise RuntimeError("openai provider needs OPENAI_API_KEY.")
        client_kwargs: dict[str, Any] = {"api_key": key}
        if url:
            client_kwargs["base_url"] = url
        return OpenAIChatShim(OpenAI(**client_kwargs))

    raise ValueError(
        f"Unknown LLM provider {p!r}. Use 'bedrock' (default), 'openai', "
        "'local' (in-process vLLM), "
        "or 'azure_openai'."
    )
