"""Dependency-light command line interface."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConfigError, ServerConfig, build_config, dump_yaml
from .doctor import Doctor, render_human
from .process import Paths, ProcessError, read_state, saved_config, start_server, stop_server

ROOT = Path(__file__).resolve().parents[1]


def _model_options(parser: argparse.ArgumentParser, include_detach: bool = False) -> None:
    parser.add_argument("--config", dest="profile", help="flat YAML profile")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--served-model-name", dest="served_model_name", default=None)
    parser.add_argument("--quantization", choices=["none", "int8", "int4"], default=None)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--attention", choices=["auto", "sdpa", "eager", "flash_attention_2"], default=None)
    parser.add_argument("--max-context", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--gpus", default=None, help="physical GPU indices, e.g. 0 or 0,1")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--tensor-parallel", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--minimum-vram-headroom", type=float, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--lora-name", default=None)
    parser.add_argument("--bnb-4bit-type", choices=["nf4", "fp4"], default=None)
    parser.add_argument("--bnb-4bit-double-quant", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bnb-compute-dtype", choices=["float32", "float16", "bfloat16"], default=None)
    if include_detach:
        parser.add_argument("--detach", action=argparse.BooleanOptionalAction, default=None)


def _cfg(args: argparse.Namespace, require_model: bool = True) -> ServerConfig:
    values = vars(args).copy()
    profile = values.pop("profile", None)
    for key in ("command", "json", "skip_doctor", "func"):
        values.pop(key, None)
    return build_config(profile, values, require_model=require_model)


def cmd_setup(args: argparse.Namespace) -> int:
    targets = ["transformers", "vllm"] if args.all else [args.backend]
    if not targets or targets == [None]:
        raise ConfigError("choose --backend transformers|vllm or --all")
    for backend in targets:
        destination = ROOT / ".runtime" / backend
        python = destination / "bin" / "python"
        if not python.exists():
            print(f"Creating {destination}")
            result = subprocess.run([sys.executable, "-m", "venv", str(destination)])
            if result.returncode:
                raise ProcessError(
                    "Python venv creation failed. On Ubuntu, ask an administrator to run "
                    f"sudo apt install python{sys.version_info.major}.{sys.version_info.minor}-venv "
                    "(or python3-venv). This tool does not attempt root changes."
                )
        requirements = [ROOT / "requirements" / "control.lock", ROOT / "requirements" / f"{backend}.lock"]
        command = [str(python), "-m", "pip", "install"]
        for requirement in requirements:
            command += ["-r", str(requirement)]
        print(f"Installing pinned {backend} stack (no root access required)...")
        code = subprocess.run(command).returncode
        if code:
            raise ProcessError(f"pip failed for {backend}; review the output above. The partial venv is preserved at {destination}")
        print(f"PASS {backend}: {python}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    report = Doctor(cfg, ROOT).run()
    print(json.dumps(report.as_dict(), indent=2) if args.json else render_human(report))
    return 2 if report.blocking else 0


def cmd_start(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    if not args.skip_doctor:
        report = Doctor(cfg, ROOT).run()
        print(render_human(report))
        if report.blocking:
            print("Start refused because doctor found blocking failures.", file=sys.stderr)
            return 2
    state = start_server(cfg, ROOT)
    print(f"Started PID {state['pid']} at http://{state['host']}:{state['port']}/v1")
    print(f"Log: {state['log_path']}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state = read_state(Paths(ROOT))
    if not state:
        print("STOPPED: no managed server state")
        return 3
    print(json.dumps(state, indent=2) if args.json else "\n".join(f"{k}: {v}" for k, v in state.items() if k != "command"))
    return 0 if not state.get("stale") else 3


def cmd_stop(args: argparse.Namespace) -> int:
    result = stop_server(ROOT, timeout=args.timeout)
    if result.get("stale_removed"):
        print(f"Removed stale state for PID {result.get('pid')}; no process was signaled.")
    else:
        print(f"Stopped PID {result['pid']}" + (" with SIGKILL after timeout" if result.get("forced") else " gracefully"))
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    cfg = saved_config(ROOT)
    try:
        stop_server(ROOT, timeout=args.timeout)
    except ProcessError as exc:
        if "no managed" not in str(exc):
            raise
    state = start_server(cfg, ROOT, detach=True)
    print(f"Restarted PID {state['pid']} at http://{state['host']}:{state['port']}/v1")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    state = read_state(Paths(ROOT))
    if not state or not state.get("log_path"):
        raise ProcessError("no server log is recorded; start the server first")
    path = Path(state["log_path"])
    if not path.exists():
        raise ProcessError(f"recorded log does not exist: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if args.follow:
            handle.seek(0, os.SEEK_END)
        else:
            lines = handle.readlines()
            print("".join(lines[-args.lines:]), end="")
            return 0
        try:
            while True:
                line = handle.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    state = read_state(Paths(ROOT))
                    if not state or state.get("stale"):
                        return 0
                    time.sleep(0.25)
        except KeyboardInterrupt:
            return 130


def _endpoint() -> tuple[str, dict[str, Any]]:
    state = read_state(Paths(ROOT))
    if not state or state.get("stale"):
        raise ProcessError("no running managed server; run ./model-server start first")
    return f"http://{state['host']}:{state['port']}", state


def _request(url: str, method: str = "GET", payload: dict[str, Any] | None = None, api_key_env: str | None = None, timeout: float = 120.0) -> tuple[dict[str, Any], float]:
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise ProcessError(f"API key environment variable {api_key_env} is not set")
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1000]
        raise ProcessError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProcessError(f"request failed for {url}: {exc}") from exc
    return body, time.perf_counter() - started


def _one_completion(base: str, state: dict[str, Any], prompt: str, max_tokens: int) -> dict[str, Any]:
    cfg = saved_config(ROOT)
    payload = {
        "model": state["served_model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    body, latency = _request(base + "/v1/chat/completions", "POST", payload, cfg.api_key_env)
    try:
        text = body["choices"][0]["message"]["content"]
        usage = body["usage"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProcessError(f"invalid chat-completion response: {body}") from exc
    empty = not str(text).strip()
    generated = int(usage.get("completion_tokens", 0))
    return {"latency_seconds": latency, "text": text, "empty": empty, "generated_but_empty": empty and generated > 0, **usage, "tokens_per_second": generated / latency if latency else None}


def cmd_test(args: argparse.Namespace) -> int:
    base, state = _endpoint()
    cfg = saved_config(ROOT)
    health, health_latency = _request(base + "/health")
    if health.get("status") != "ok":
        raise ProcessError(f"health check is not ready: {health}")
    models, _ = _request(base + "/v1/models", api_key_env=cfg.api_key_env)
    ids = [m.get("id") for m in models.get("data", [])]
    if state["served_model_name"] not in ids:
        raise ProcessError(f"served model missing from /v1/models: {ids}")
    result = _one_completion(base, state, args.prompt, args.max_tokens)
    print(json.dumps({"health_latency_seconds": health_latency, "model": state["served_model_name"], **result}, indent=2, ensure_ascii=False))
    if result["generated_but_empty"]:
        print("FAIL: tokens were generated but decoded to an empty string; check model/dtype compatibility.", file=sys.stderr)
        return 4
    if result["empty"]:
        print("FAIL: model returned empty output.", file=sys.stderr)
        return 4
    return 0


def _gpu_memory() -> list[dict[str, int]] | None:
    command = shutil.which("nvidia-smi")
    if not command:
        return None
    try:
        out = subprocess.run([command, "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10).stdout
        return [{"index": int(a), "used_mib": int(b), "free_mib": int(c)} for a, b, c in ([x.strip() for x in line.split(",")] for line in out.splitlines())]
    except (subprocess.SubprocessError, ValueError):
        return None


def cmd_benchmark(args: argparse.Namespace) -> int:
    base, state = _endpoint()
    cfg = saved_config(ROOT)
    health, _ = _request(base + "/health")
    if health.get("status") != "ok":
        raise ProcessError("model is not loaded")
    before = _gpu_memory()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_one_completion, base, state, args.prompt, args.max_tokens) for _ in range(args.requests)]
        first_done = None
        for future in concurrent.futures.as_completed(futures):
            if first_done is None: first_done = time.perf_counter() - started
            try: results.append(future.result())
            except Exception as exc: results.append({"error": str(exc), "empty": False})
    total = time.perf_counter() - started
    after = _gpu_memory()
    successes = [r for r in results if "error" not in r and not r.get("empty")]
    output_tokens = sum(int(r.get("completion_tokens", 0)) for r in results)
    try:
        metrics, _ = _request(base + "/internal/metrics", api_key_env=cfg.api_key_env)
    except ProcessError:
        metrics = {}
    report = {
        "model_load_status": "loaded", "requests": args.requests, "concurrency": args.concurrency,
        "success_count": len(successes), "failure_count": args.requests - len(successes),
        "empty_output_count": sum(bool(r.get("empty")) for r in results),
        "time_to_first_completed_response_seconds": first_done,
        "total_generation_latency_seconds": total, "output_tokens": output_tokens,
        "aggregate_tokens_per_second": output_tokens / total if total else None,
        "gpu_memory_before": before, "gpu_memory_after": after,
        "peak_gpu_memory_bytes": metrics.get("peak_gpu_memory_bytes"),
        "minimum_vram_headroom": args.minimum_vram_headroom,
        "results": results,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if len(successes) == args.requests else 4


def cmd_list_gpus(args: argparse.Namespace) -> int:
    command = shutil.which("nvidia-smi")
    if not command:
        raise ProcessError("nvidia-smi not found")
    return subprocess.run([command, "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version", "--format=csv,noheader"]).returncode


def cmd_config(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    if args.json:
        print(json.dumps({"configuration": cfg.as_dict(), "configuration_hash": cfg.config_hash()}, indent=2))
    else:
        print(dump_yaml(cfg), end="")
        print(f"# configuration_hash: {cfg.config_hash()}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    print(f"model-server {__version__}")
    targets = [args.backend] if args.backend else ["transformers", "vllm"]
    script = """import importlib.metadata as m
names = ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'peft', 'fastapi', 'uvicorn', 'vllm']
for name in names:
    try:
        print(name, m.version(name))
    except m.PackageNotFoundError:
        pass
"""
    for backend in targets:
        python = ROOT / ".runtime" / backend / "bin" / "python"
        print(f"[{backend}]")
        if python.exists(): subprocess.run([str(python), "-c", script])
        else: print("not installed")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="model-server", description="Install, diagnose, serve, and test local Hugging Face LLMs")
    sub = result.add_subparsers(dest="command", required=True)
    p = sub.add_parser("setup", help="create isolated backend virtual environments")
    p.add_argument("--backend", choices=["transformers", "vllm"]); p.add_argument("--all", action="store_true"); p.set_defaults(func=cmd_setup)
    p = sub.add_parser("doctor", help="run blocking preflight checks"); _model_options(p); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_doctor)
    p = sub.add_parser("start", help="start one managed model process"); _model_options(p, True); p.add_argument("--skip-doctor", action="store_true", help="advanced: skip repeated preflight"); p.set_defaults(func=cmd_start)
    p = sub.add_parser("stop", help="gracefully stop the managed process"); p.add_argument("--timeout", type=float, default=20); p.set_defaults(func=cmd_stop)
    p = sub.add_parser("restart", help="restart using the last effective configuration"); p.add_argument("--timeout", type=float, default=20); p.set_defaults(func=cmd_restart)
    p = sub.add_parser("status", help="show managed process state"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_status)
    p = sub.add_parser("logs", help="show the current server log"); p.add_argument("--follow", "-f", action="store_true"); p.add_argument("--lines", type=int, default=100); p.set_defaults(func=cmd_logs)
    p = sub.add_parser("test", help="validate health, models, and nonempty generation"); p.add_argument("--prompt", default="Koliko je 2 + 2? Odgovori v slovenščini."); p.add_argument("--max-tokens", type=int, default=64); p.set_defaults(func=cmd_test)
    p = sub.add_parser("benchmark", help="run a small bounded benchmark"); p.add_argument("--requests", type=int, default=3); p.add_argument("--concurrency", type=int, default=1); p.add_argument("--prompt", default="Napiši en kratek pozdrav."); p.add_argument("--max-tokens", type=int, default=64); p.add_argument("--minimum-vram-headroom", type=float, default=0.08); p.set_defaults(func=cmd_benchmark)
    p = sub.add_parser("list-gpus", help="list NVIDIA GPU indices and memory"); p.set_defaults(func=cmd_list_gpus)
    p = sub.add_parser("config", help="print validated effective configuration"); _model_options(p); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_config)
    p = sub.add_parser("version", help="show CLI and installed backend versions"); p.add_argument("--backend", choices=["transformers", "vllm"]); p.set_defaults(func=cmd_version)
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        return int(args.func(args))
    except (ConfigError, ProcessError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
