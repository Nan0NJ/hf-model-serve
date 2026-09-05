"""Configuration loading, validation, normalization, and hashing."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised for an invalid user configuration."""


DEFAULTS: dict[str, Any] = {
    "backend": "transformers",
    "model": None,
    "served_model_name": None,
    "quantization": "none",
    "dtype": "auto",
    "attention": "auto",
    "gpus": "0",
    "host": "127.0.0.1",
    "port": 8000,
    "max_context": 4096,
    "max_output_tokens": 256,
    "max_concurrency": 1,
    "tensor_parallel": 1,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 4,
    "minimum_vram_headroom": 0.08,
    "trust_remote_code": False,
    "revision": None,
    "download_dir": None,
    "offline": False,
    "api_key_env": None,
    "lora_adapter": None,
    "lora_name": None,
    "bnb_4bit_type": "nf4",
    "bnb_4bit_double_quant": True,
    "bnb_compute_dtype": "float32",
    "detach": False,
}


@dataclasses.dataclass(slots=True)
class ServerConfig:
    backend: str = "transformers"
    model: str | None = None
    served_model_name: str | None = None
    quantization: str = "none"
    dtype: str = "auto"
    attention: str = "auto"
    gpus: str = "0"
    host: str = "127.0.0.1"
    port: int = 8000
    max_context: int = 4096
    max_output_tokens: int = 256
    max_concurrency: int = 1
    tensor_parallel: int = 1
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 4
    minimum_vram_headroom: float = 0.08
    trust_remote_code: bool = False
    revision: str | None = None
    download_dir: str | None = None
    offline: bool = False
    api_key_env: str | None = None
    lora_adapter: str | None = None
    lora_name: str | None = None
    bnb_4bit_type: str = "nf4"
    bnb_4bit_double_quant: bool = True
    bnb_compute_dtype: str = "float32"
    detach: bool = False

    def normalized(self) -> "ServerConfig":
        data = dataclasses.asdict(self)
        data["gpus"] = ",".join(str(x) for x in parse_gpu_list(self.gpus))
        if not data["served_model_name"] and data["model"]:
            data["served_model_name"] = Path(str(data["model"])).name
        return ServerConfig(**data)

    def validate(self, require_model: bool = True) -> None:
        if self.backend not in {"transformers", "vllm"}:
            raise ConfigError("backend must be transformers or vllm")
        if require_model and not self.model:
            raise ConfigError("--model is required (or set model in --config)")
        if self.quantization not in {"none", "int8", "int4"}:
            raise ConfigError("quantization must be none, int8, or int4")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ConfigError("dtype must be auto, float32, float16, or bfloat16")
        if self.attention not in {"auto", "sdpa", "eager", "flash_attention_2"}:
            raise ConfigError("attention must be auto, sdpa, eager, or flash_attention_2")
        parse_gpu_list(self.gpus)
        if not 1 <= self.port <= 65535:
            raise ConfigError("port must be between 1 and 65535")
        for name in ("max_context", "max_output_tokens", "max_concurrency", "tensor_parallel", "max_num_seqs"):
            if int(getattr(self, name)) < 1:
                raise ConfigError(f"{name} must be at least 1")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ConfigError("gpu_memory_utilization must be in (0, 1]")
        if not 0.05 <= self.minimum_vram_headroom < 1.0:
            raise ConfigError("minimum_vram_headroom must be at least 0.05 and below 1.0")
        if self.tensor_parallel > len(parse_gpu_list(self.gpus)):
            raise ConfigError("tensor_parallel cannot exceed the number of selected GPUs")
        if self.backend == "transformers" and self.tensor_parallel != 1:
            raise ConfigError("Transformers uses single-GPU or device_map=auto; tensor_parallel is vLLM-only")
        if self.backend == "vllm" and self.quantization != "none":
            raise ConfigError(
                "generic int8/int4 flags are Transformers BitsAndBytes modes and are not mapped to a vLLM method; "
                "use a vLLM-supported pre-quantized checkpoint with --quantization none (auto-detected)"
            )
        if self.quantization == "int4" and self.bnb_4bit_type not in {"nf4", "fp4"}:
            raise ConfigError("bnb_4bit_type must be nf4 or fp4")
        if self.bnb_compute_dtype not in {"float32", "float16", "bfloat16"}:
            raise ConfigError("invalid bnb_compute_dtype")
        if bool(self.lora_adapter) != bool(self.lora_name):
            raise ConfigError("--lora-adapter and --lora-name must be supplied together")
        if self.backend == "vllm" and self.api_key_env:
            raise ConfigError(
                "--api-key-env is currently supported by the Transformers API only; for vLLM keep localhost binding "
                "or use a reverse proxy rather than exposing a key in process arguments"
            )
        if self.api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ConfigError("api_key_env must be a valid environment variable name")

    def as_dict(self, redact: bool = True) -> dict[str, Any]:
        # Only the environment variable name is stored, never its value.
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        raw = json.dumps(self.normalized().as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def parse_gpu_list(value: str | int | list[int]) -> list[int]:
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        text = str(value).strip()
        if not text:
            raise ConfigError("GPU list cannot be empty")
        if not re.fullmatch(r"\d+(,\d+)*", text):
            raise ConfigError("GPU list must look like 0 or 0,1,2")
        values = [int(item) for item in text.split(",")]
    if any(v < 0 for v in values) or len(set(values)) != len(values):
        raise ConfigError("GPU indices must be unique non-negative integers")
    return values


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if (value[0:1], value[-1:]) in {("'", "'"), ('"', '"')}:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the intentionally flat profile format without requiring PyYAML."""
    result: dict[str, Any] = {}
    allowed = set(DEFAULTS)
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if raw[:1].isspace() or ":" not in line:
            raise ConfigError(f"{path}:{number}: profiles must contain flat key: value pairs")
        key, value = line.split(":", 1)
        key = key.strip().replace("-", "_")
        if key not in allowed:
            raise ConfigError(f"{path}:{number}: unknown configuration key {key!r}")
        result[key] = _scalar(value.split(" #", 1)[0])
    return result


def build_config(profile: str | None, overrides: Mapping[str, Any], require_model: bool = True) -> ServerConfig:
    values = dict(DEFAULTS)
    if profile:
        values.update(load_yaml(profile))
    values.update({k: v for k, v in overrides.items() if k in DEFAULTS and v is not None})
    try:
        cfg = ServerConfig(**values).normalized()
        cfg.validate(require_model=require_model)
    except TypeError as exc:
        raise ConfigError(str(exc)) from exc
    return cfg


def dump_yaml(config: ServerConfig) -> str:
    lines = []
    for key, value in config.normalized().as_dict().items():
        if value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, str) and (not value or any(c in value for c in ":#{}[],'\"")):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines) + "\n"


def child_environment(config: ServerConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = config.gpus
    for key in ("NO_PROXY", "no_proxy"):
        current = [x.strip() for x in env.get(key, "").split(",") if x.strip()]
        for host in ("127.0.0.1", "localhost"):
            if host not in current:
                current.append(host)
        env[key] = ",".join(current)
    if config.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env
