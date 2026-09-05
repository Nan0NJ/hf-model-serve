"""Preflight diagnostics with machine-readable PASS/WARN/FAIL results."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import ServerConfig, parse_gpu_list
from .process import Paths, port_is_free, read_state


@dataclass(slots=True)
class Check:
    status: str
    name: str
    message: str
    remedy: str | None = None


@dataclass(slots=True)
class DoctorReport:
    checks: list[Check]
    metadata: dict[str, Any]

    @property
    def blocking(self) -> bool:
        return any(c.status == "FAIL" for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        counts = {s: sum(c.status == s for c in self.checks) for s in ("PASS", "WARN", "FAIL")}
        return {"ok": not self.blocking, "summary": counts, "checks": [asdict(c) for c in self.checks], "metadata": self.metadata}


class Doctor:
    def __init__(self, config: ServerConfig, root: Path):
        self.config = config
        self.root = root
        self.checks: list[Check] = []
        self.metadata: dict[str, Any] = {}

    def add(self, status: str, name: str, message: str, remedy: str | None = None) -> None:
        if status == "FAIL" and not remedy:
            raise AssertionError(f"blocking check {name} needs an exact remedy")
        self.checks.append(Check(status, name, message, remedy))

    def run(self) -> DoctorReport:
        self._platform()
        self._resources()
        gpu = self._nvidia()
        runtime = self._python_stack()
        model_meta = self._model_access()
        self._configuration(gpu, runtime, model_meta)
        self._port_and_process()
        self._proxy()
        return DoctorReport(self.checks, self.metadata)

    def _platform(self) -> None:
        system, machine = platform.system(), platform.machine()
        self.metadata.update(os=system, architecture=machine, python=platform.python_version())
        if system == "Linux" and machine in {"x86_64", "amd64"}:
            self.add("PASS", "platform", f"{system} {machine}")
        else:
            self.add("WARN", "platform", f"{system} {machine}; Linux x86_64 is the primary supported GPU platform")
        if sys.version_info >= (3, 10):
            self.add("PASS", "control_python", f"Python {platform.python_version()}")
        else:
            self.add("FAIL", "control_python", f"Python {platform.python_version()} is too old", "Install user-accessible Python 3.10–3.12 and rerun setup with that python3.")

    def _resources(self) -> None:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram = pages * page_size
            self.metadata["free_ram_bytes"] = ram
            status = "PASS" if ram >= 8 * 2**30 else "WARN"
            self.add(status, "free_ram", f"{ram / 2**30:.1f} GiB available")
        except (ValueError, OSError):
            self.add("WARN", "free_ram", "Could not determine available RAM")
        disk = shutil.disk_usage(self.root).free
        self.metadata["free_disk_bytes"] = disk
        if disk >= 20 * 2**30:
            self.add("PASS", "free_disk", f"{disk / 2**30:.1f} GiB free")
        else:
            self.add("FAIL", "free_disk", f"Only {disk / 2**30:.1f} GiB free", "Free at least 20 GiB in the repository filesystem or set --download-dir to a larger filesystem.")

    def _nvidia(self) -> dict[str, Any]:
        executable = shutil.which("nvidia-smi")
        if not executable:
            self.add("FAIL", "nvidia_driver", "nvidia-smi is not visible", "Install/enable a compatible NVIDIA driver or run on a machine with an NVIDIA GPU.")
            return {"gpus": []}
        query = "index,name,memory.total,memory.free,driver_version,compute_cap"
        try:
            result = subprocess.run(
                [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=10,
            )
            gpus = []
            for line in result.stdout.splitlines():
                fields = [x.strip() for x in line.split(",")]
                if len(fields) >= 6:
                    gpus.append({"index": int(fields[0]), "name": fields[1], "total_mib": int(fields[2]), "free_mib": int(fields[3]), "driver": fields[4], "compute_capability": fields[5]})
        except (subprocess.SubprocessError, ValueError) as exc:
            self.add("FAIL", "nvidia_driver", f"nvidia-smi query failed: {exc}", "Repair NVIDIA driver visibility, then verify with nvidia-smi.")
            return {"gpus": []}
        self.metadata["gpus"] = gpus
        self.add("PASS", "nvidia_driver", f"Driver {gpus[0]['driver']}; {len(gpus)} GPU(s) visible" if gpus else "NVIDIA driver responded")
        requested = parse_gpu_list(self.config.gpus)
        missing = [i for i in requested if i not in {g["index"] for g in gpus}]
        if missing:
            self.add("FAIL", "gpu_selection", f"Requested GPU indices do not exist: {missing}", "Choose indices printed by ./model-server list-gpus.")
        else:
            selected = [g for g in gpus if g["index"] in requested]
            details = "; ".join(f"{g['index']} {g['name']} {g['free_mib']}/{g['total_mib']} MiB free CC {g['compute_capability']}" for g in selected)
            self.add("PASS", "gpu_selection", details)
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is None:
            self.add("PASS", "cuda_visible_devices", f"Unset in parent; child will set it to {self.config.gpus} before importing Torch")
        elif cuda_visible == self.config.gpus:
            self.add("PASS", "cuda_visible_devices", f"Parent already selects {cuda_visible}")
        else:
            self.add("WARN", "cuda_visible_devices", f"Parent is {cuda_visible!r}; managed child will override it with {self.config.gpus!r}")
        return {"gpus": gpus}

    def _python_stack(self) -> dict[str, Any]:
        python = self.root / ".runtime" / self.config.backend / "bin" / "python"
        if not python.exists():
            self.add("FAIL", "backend_environment", f"{python} does not exist", f"Run ./model-server setup --backend {self.config.backend}.")
            return {}
        self.add("PASS", "backend_environment", str(python))
        script = r'''
import importlib, json, os
out = {"python": __import__("platform").python_version(), "packages": {}}
for name in ("torch", "transformers", "accelerate", "bitsandbytes", "peft", "vllm"):
    try:
        mod = importlib.import_module(name)
        out["packages"][name] = getattr(mod, "__version__", "unknown")
    except Exception as e:
        out["packages"][name] = None
        out.setdefault("import_errors", {})[name] = type(e).__name__ + ": " + str(e)[:300]
try:
    import torch
    out["torch_cuda_available"] = torch.cuda.is_available()
    out["torch_cuda_runtime"] = torch.version.cuda
    out["torch_devices"] = [{"name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]
except Exception:
    pass
print(json.dumps(out))
'''
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.config.gpus
        try:
            proc = subprocess.run([str(python), "-c", script], env=env, capture_output=True, text=True, timeout=40)
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except (subprocess.SubprocessError, json.JSONDecodeError, IndexError) as exc:
            self.add("FAIL", "backend_imports", f"Could not inspect backend: {exc}", f"Rerun ./model-server setup --backend {self.config.backend}; inspect pip output.")
            return {}
        self.metadata["runtime"] = data
        required = ["torch", "transformers", "accelerate"] if self.config.backend == "transformers" else ["torch", "vllm"]
        if self.config.quantization in {"int8", "int4"} and self.config.backend == "transformers":
            required.append("bitsandbytes")
        if self.config.lora_adapter and self.config.backend == "transformers":
            required.append("peft")
        for name in required:
            version = data.get("packages", {}).get(name)
            if version:
                self.add("PASS", f"package_{name}", f"{name} {version}")
            else:
                detail = data.get("import_errors", {}).get(name, "not installed")
                self.add("FAIL", f"package_{name}", detail, f"Rerun ./model-server setup --backend {self.config.backend} to install the pinned {name} dependency.")
        if data.get("torch_cuda_available"):
            self.add("PASS", "torch_cuda", f"CUDA available; PyTorch wheel runtime {data.get('torch_cuda_runtime')}")
        else:
            self.add("FAIL", "torch_cuda", "torch.cuda.is_available() is false", "Install the pinned CUDA-enabled Torch wheel, verify the NVIDIA driver, and ensure selected GPUs are not masked.")
        nvcc = shutil.which("nvcc")
        if nvcc:
            self.add("PASS", "nvcc", f"Optional compiler found at {nvcc}")
        else:
            self.add("PASS", "nvcc", "Not found; nvcc is not required by compatible prebuilt PyTorch/BitsAndBytes/vLLM wheels")
        return data

    def _token(self) -> str | None:
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if os.environ.get(key):
                return os.environ[key]
        token_path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
        try:
            return token_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def _model_access(self) -> dict[str, Any]:
        token = self._token()
        self.add("PASS" if token else "WARN", "huggingface_auth", "Credentials detected (token hidden)" if token else "No Hugging Face token detected; public/local models can still work")
        model = str(self.config.model)
        local = Path(model).expanduser()
        if local.exists():
            config_path = local / "config.json" if local.is_dir() else local.parent / "config.json"
            meta: dict[str, Any] = {}
            if config_path.exists():
                try: meta = json.loads(config_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError): pass
            self.add("PASS", "model_access", f"Local model path exists: {local}")
            return meta
        if self.config.offline:
            python = self.root / ".runtime" / self.config.backend / "bin" / "python"
            if python.exists() and self.config.backend == "transformers":
                script = """import json, sys
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(sys.argv[1], revision=None if sys.argv[2] == '-' else sys.argv[2], cache_dir=None if sys.argv[3] == '-' else sys.argv[3], local_files_only=True, trust_remote_code=False)
print(json.dumps(cfg.to_dict()))
"""
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = self.config.gpus
                result = subprocess.run(
                    [str(python), "-c", script, model, self.config.revision or "-", self.config.download_dir or "-"],
                    env=env, capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    try:
                        meta = json.loads(result.stdout.strip().splitlines()[-1])
                        self.add("PASS", "model_access", f"Offline cache contains model config for {model}")
                        return meta
                    except (json.JSONDecodeError, IndexError):
                        pass
            self.add("FAIL", "model_access", f"Offline mode cannot find cached model metadata for {model}", "Pre-download the model into --download-dir, pass its local path, or remove --offline.")
            return {}
        url = "https://huggingface.co/api/models/" + urllib.parse.quote(model, safe="/")
        if self.config.revision:
            url += "/revision/" + urllib.parse.quote(self.config.revision, safe="")
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                meta = json.load(response)
            self.add("PASS", "model_access", f"Hugging Face model metadata is accessible for {model}")
            return meta.get("config", {}) | {"safetensors": meta.get("safetensors", {})}
        except urllib.error.HTTPError as exc:
            remedy = "Run huggingface-cli login (or set HF_TOKEN) and accept any model license, then retry." if exc.code in {401, 403} else "Verify the model ID/revision and network access to huggingface.co."
            self.add("FAIL", "model_access", f"Hugging Face returned HTTP {exc.code} for {model}", remedy)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            self.add("FAIL", "model_access", f"Could not reach Hugging Face: {exc}", "Fix proxy/network settings, or pre-download the model and use --offline with a local path.")
        return {}

    def _configuration(self, gpu_info: dict[str, Any], runtime: dict[str, Any], model: dict[str, Any]) -> None:
        model_id = str(self.config.model).lower()
        selected = [g for g in gpu_info.get("gpus", []) if g["index"] in parse_gpu_list(self.config.gpus)]
        capabilities = []
        for g in selected:
            try: capabilities.append(float(g["compute_capability"]))
            except ValueError: pass
        if not capabilities:
            for dev in runtime.get("torch_devices", []):
                cap = dev.get("capability", [0, 0]); capabilities.append(float(f"{cap[0]}.{cap[1]}"))
        minimum_cc = min(capabilities) if capabilities else None

        if self.config.backend == "vllm" and ("gams3" in model_id or "gams-3" in model_id):
            self.add("FAIL", "backend_model_compatibility", "GaMS3 has known Gemma 3 rope-scaling and BitsAndBytes weight-shape failures under vLLM", "Use --backend transformers and configs/gams3-v100-int8.yaml.")
        else:
            self.add("PASS", "backend_model_compatibility", f"No repository deny-list rule blocks {self.config.backend} for this model; runtime loading remains the definitive check")
        if self.config.trust_remote_code:
            self.add("WARN", "remote_code", "trust_remote_code is enabled; model repository Python code will execute locally—review the pinned revision first")
        else:
            self.add("PASS", "remote_code", "Remote model code execution is disabled")

        if self.config.dtype == "bfloat16" and minimum_cc is not None and minimum_cc < 8.0:
            self.add("FAIL", "dtype_compatibility", f"BF16 requires Ampere-class CC 8.0+ for this server; selected minimum is {minimum_cc:.1f}", "Choose --dtype float32/float16 as model-compatible, or use an Ampere-or-newer GPU. For GaMS3 on V100 use INT8 + float32.")
        elif "gams3" in model_id and self.config.quantization == "none" and self.config.dtype == "float16" and minimum_cc == 7.0:
            self.add("FAIL", "dtype_compatibility", "GaMS3 unquantized FP16 is known to load but decode empty output on V100 CC 7.0", "Use the proven --quantization int8 --dtype float32 --attention sdpa profile.")
        elif self.config.quantization == "int4" and self.config.bnb_compute_dtype == "bfloat16" and minimum_cc is not None and minimum_cc < 8.0:
            self.add("FAIL", "dtype_compatibility", f"INT4 BF16 compute is unsupported on selected CC {minimum_cc:.1f}", "Set --bnb-compute-dtype float32 (safe) or float16, or use Ampere-or-newer hardware.")
        else:
            self.add("PASS", "dtype_compatibility", f"Requested dtype/compute dtype is numerically permitted by known rules{f' on minimum CC {minimum_cc:.1f}' if minimum_cc else ''}")

        if self.config.attention == "flash_attention_2" and minimum_cc is not None and minimum_cc < 8.0:
            self.add("FAIL", "attention_compatibility", "FlashAttention 2 is not a safe default on pre-Ampere GPUs", "Use --attention sdpa or eager on V100-class GPUs.")
        else:
            self.add("PASS", "attention_compatibility", f"Attention implementation: {self.config.attention}")

        if self.config.quantization in {"int8", "int4"} and minimum_cc is not None and minimum_cc < 6.0:
            self.add("FAIL", "quantization_compatibility", f"BitsAndBytes quantization requires CC 6.0+; selected minimum is {minimum_cc:.1f}", "Use a Pascal-or-newer NVIDIA GPU or unquantized CPU/GPU loading.")
        elif self.config.quantization == "int8" and minimum_cc is not None and minimum_cc < 7.5:
            if "gams3" in model_id and minimum_cc == 7.0 and self.config.dtype == "float32":
                self.add("WARN", "quantization_compatibility", "Current generic BitsAndBytes docs list LLM.int8() at CC 7.5+, but this exact GaMS3 INT8/float32 profile is user-verified on V100 CC 7.0; runtime import/load remains required")
            else:
                self.add("WARN", "quantization_compatibility", f"Generic LLM.int8 documentation targets CC 7.5+; selected minimum is {minimum_cc:.1f}. 8-bit quantization kernels may vary by model/path")
        else:
            self.add("PASS", "quantization_compatibility", f"Quantization mode {self.config.quantization} passes known hardware rules")

        requested_gpus = len(parse_gpu_list(self.config.gpus))
        if self.config.backend == "vllm" and self.config.tensor_parallel != requested_gpus:
            self.add("WARN", "tensor_parallel", f"TP={self.config.tensor_parallel} while {requested_gpus} GPUs are visible; unused GPUs are intentional only if planned")
        else:
            self.add("PASS", "tensor_parallel", f"TP={self.config.tensor_parallel}, visible GPUs={requested_gpus}")
        heads = model.get("num_attention_heads")
        if not heads and isinstance(model.get("text_config"), dict):
            heads = model["text_config"].get("num_attention_heads")
        if self.config.backend == "vllm" and heads and int(heads) % self.config.tensor_parallel:
            self.add("FAIL", "tensor_parallel_architecture", f"{heads} attention heads are not divisible by TP={self.config.tensor_parallel}", "Choose a tensor-parallel value that divides the model's attention-head count.")
        elif self.config.backend == "vllm":
            self.add("PASS", "tensor_parallel_architecture", "Attention-head divisibility is compatible or metadata is unavailable")

        max_position = model.get("max_position_embeddings") or (model.get("text_config", {}).get("max_position_embeddings") if isinstance(model.get("text_config"), dict) else None)
        if max_position and self.config.max_context > int(max_position):
            self.add("FAIL", "context_length", f"Requested {self.config.max_context} exceeds model metadata limit {max_position}", f"Set --max-context {max_position} or lower.")
        else:
            self.add("PASS", "context_length", f"Requested context {self.config.max_context}; model limit {max_position or 'not declared'}")
        if self.config.max_output_tokens >= self.config.max_context:
            self.add("FAIL", "output_length", "max_output_tokens must be smaller than max_context", "Lower --max-output-tokens or raise --max-context within the model limit.")
        else:
            self.add("PASS", "output_length", f"Maximum output {self.config.max_output_tokens} tokens")
        if self.config.backend == "transformers" and self.config.max_concurrency > 1:
            self.add("WARN", "concurrency", f"Concurrency {self.config.max_concurrency} is semaphore-limited generation, not vLLM continuous batching; memory use can multiply")
        else:
            self.add("PASS", "concurrency", f"Concurrency {self.config.max_concurrency}")
        if self.config.backend == "vllm" and self.config.gpu_memory_utilization > 1 - self.config.minimum_vram_headroom:
            self.add("FAIL", "vllm_memory_headroom", f"gpu_memory_utilization {self.config.gpu_memory_utilization:.2f} violates requested minimum headroom {self.config.minimum_vram_headroom:.2f}", f"Set --gpu-memory-utilization to at most {1-self.config.minimum_vram_headroom:.2f}.")
        elif self.config.backend == "vllm":
            self.add("PASS", "vllm_memory_headroom", f"vLLM utilization leaves at least {1-self.config.gpu_memory_utilization:.0%} nominal headroom")

        params = model.get("safetensors", {}).get("total") if isinstance(model.get("safetensors"), dict) else None
        if not params and "12b" in model_id: params = 12_000_000_000
        if params:
            bytes_per = {"int4": 0.65, "int8": 1.15, "none": {"float16": 2.0, "bfloat16": 2.0, "float32": 4.0, "auto": 2.0}[self.config.dtype]}[self.config.quantization]
            estimate = int(int(params) * bytes_per)
            self.metadata["estimated_weight_bytes"] = estimate
            if selected:
                available = sum(g["free_mib"] for g in selected) * 2**20
                safe = int(available * (1 - self.config.minimum_vram_headroom))
                if estimate > safe:
                    self.add("FAIL", "vram_headroom", f"Estimated weights {estimate/2**30:.1f} GiB exceed safe free VRAM {safe/2**30:.1f} GiB after {self.config.minimum_vram_headroom:.0%} headroom", "Use more/larger GPUs, stop other GPU jobs, or choose a supported lower-memory quantization.")
                else:
                    self.add("PASS", "vram_headroom", f"Estimated weights {estimate/2**30:.1f} GiB; safe free VRAM {safe/2**30:.1f} GiB, excluding KV cache/runtime overhead")
            else:
                self.add("WARN", "vram_headroom", f"Estimated weights {estimate/2**30:.1f} GiB; GPU availability could not be measured")
        else:
            self.add("WARN", "vram_headroom", "Parameter count unavailable; cannot estimate weight memory. Context/KV cache and allocator overhead are additional")

        if self.config.lora_adapter:
            adapter_config = Path(self.config.lora_adapter) / "adapter_config.json"
            if Path(self.config.lora_adapter).exists() and not adapter_config.exists():
                self.add("FAIL", "lora_compatibility", "Local LoRA path lacks adapter_config.json", "Point --lora-adapter at a PEFT adapter directory, or pass a merged model as --model without LoRA flags.")
            elif adapter_config.exists():
                try:
                    adapter_meta = json.loads(adapter_config.read_text(encoding="utf-8"))
                    base = adapter_meta.get("base_model_name_or_path")
                except (OSError, json.JSONDecodeError):
                    base = None
                if base and str(base).rstrip("/") != str(self.config.model).rstrip("/"):
                    self.add("FAIL", "lora_compatibility", f"Adapter declares base {base!r}, not requested model {self.config.model!r}", "Set --model to the adapter's declared base model, or use the correctly matched adapter.")
                else:
                    self.add("PASS", "lora_compatibility", f"Local adapter metadata matches the requested base{f' {base}' if base else ''}")
            else:
                self.add("WARN", "lora_compatibility", "Adapter/base compatibility will be verified from adapter_config.json by PEFT/vLLM at load time")
        else:
            self.add("PASS", "lora_compatibility", "No separate LoRA adapter requested")

    def _port_and_process(self) -> None:
        state = read_state(Paths(self.root))
        if state and not state.get("stale"):
            same = state.get("host") == self.config.host and int(state.get("port", -1)) == self.config.port
            self.add("FAIL", "managed_instance", f"Managed PID {state['pid']} is already running", "Run ./model-server stop before starting another model." if same else "Stop the managed instance or use its existing endpoint.")
        elif state and state.get("stale"):
            self.add("WARN", "managed_instance", "Stale PID state found; start will replace it safely")
        else:
            self.add("PASS", "managed_instance", "No active managed instance")
        if port_is_free(self.config.host, self.config.port):
            self.add("PASS", "port", f"{self.config.host}:{self.config.port} is free")
        else:
            self.add("FAIL", "port", f"{self.config.host}:{self.config.port} is occupied", "Stop the process using this port or select another --port.")
        if self.config.host in {"0.0.0.0", "::"}:
            self.add("WARN", "network_binding", "Explicit all-interface binding exposes the API to the network; use an API key and firewall")
        else:
            self.add("PASS", "network_binding", f"Local-only binding {self.config.host}")
        if self.config.api_key_env:
            if os.environ.get(self.config.api_key_env):
                self.add("PASS", "api_key", f"API key variable {self.config.api_key_env} is set (value hidden)")
            else:
                self.add("FAIL", "api_key", f"API key variable {self.config.api_key_env} is not set", f"Export {self.config.api_key_env} before doctor/start, without placing its value in a profile.")
        else:
            self.add("PASS", "api_key", "No API key required for the localhost-only endpoint")

    def _proxy(self) -> None:
        proxy_names = [k for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy") if os.environ.get(k)]
        self.add("WARN" if proxy_names else "PASS", "proxy_variables", f"Proxy variables set: {', '.join(proxy_names)}" if proxy_names else "No proxy variables detected")
        for key in ("NO_PROXY", "no_proxy"):
            values = {v.strip().lower() for v in os.environ.get(key, "").split(",") if v.strip()}
            missing = {"127.0.0.1", "localhost"} - values
            if missing:
                self.add("WARN", key, f"Parent {key} lacks {', '.join(sorted(missing))}; managed children and built-in tests add both automatically")
            else:
                self.add("PASS", key, "Includes 127.0.0.1 and localhost")


def render_human(report: DoctorReport) -> str:
    lines = []
    for check in report.checks:
        lines.append(f"{check.status:<4} {check.name}: {check.message}")
        if check.remedy:
            lines.append(f"     Remedy: {check.remedy}")
    counts = report.as_dict()["summary"]
    lines.append(f"Summary: {counts['PASS']} PASS, {counts['WARN']} WARN, {counts['FAIL']} FAIL")
    return "\n".join(lines)
