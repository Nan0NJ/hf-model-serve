"""Official vLLM CLI argument construction (inference remains in vLLM)."""

from __future__ import annotations

from ..config import ServerConfig


def build_vllm_args(config: ServerConfig, executable: str = "vllm") -> list[str]:
    args = [
        executable, "serve", str(config.model),
        "--served-model-name", str(config.served_model_name),
        "--host", config.host,
        "--port", str(config.port),
        "--tensor-parallel-size", str(config.tensor_parallel),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--max-model-len", str(config.max_context),
        "--max-num-seqs", str(config.max_num_seqs),
        "--dtype", config.dtype,
    ]
    if config.revision:
        args += ["--revision", config.revision]
    if config.download_dir:
        args += ["--download-dir", config.download_dir]
    if config.trust_remote_code:
        args.append("--trust-remote-code")
    if config.quantization != "none":
        args += ["--quantization", config.quantization]
    if config.lora_adapter:
        args += ["--enable-lora", "--lora-modules", f"{config.lora_name}={config.lora_adapter}"]
    return args

