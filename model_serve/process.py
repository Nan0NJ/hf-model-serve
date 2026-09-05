"""Safe single-instance process management."""

from __future__ import annotations

import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends.vllm_backend import build_vllm_args
from .config import ServerConfig, child_environment


class ProcessError(RuntimeError):
    pass


@dataclass(slots=True)
class Paths:
    root: Path

    @property
    def state_dir(self) -> Path: return self.root / ".runtime" / "state"
    @property
    def state(self) -> Path: return self.state_dir / "server.json"
    @property
    def config(self) -> Path: return self.state_dir / "effective-config.json"
    @property
    def lock(self) -> Path: return self.state_dir / "start.lock"
    @property
    def logs_dir(self) -> Path: return self.root / ".runtime" / "logs"


def pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def pid_matches_state(pid: int, state: dict[str, Any]) -> bool:
    """Refuse to own a reused PID whose command no longer matches our record."""
    command = state.get("command")
    if not command or not isinstance(command, list):
        return False
    try:
        actual = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        actual_text = [part.decode(errors="replace") for part in actual if part]
    except OSError:
        # Non-Linux platforms lack /proc; kill(0) is the best available check.
        return platform.system() != "Linux"
    if not actual_text:
        return False
    expected_executable = str(Path(str(command[0])).resolve())
    actual_paths = {str(Path(part).resolve()) for part in actual_text if "/" in part}
    executable_matches = expected_executable in actual_paths or str(command[0]) in actual_text
    return executable_matches and all(str(part) in actual_text for part in command[1:3])


def read_state(paths: Paths) -> dict[str, Any] | None:
    if not paths.state.exists():
        return None
    try:
        data = json.loads(paths.state.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"status": "invalid", "stale": True}
    pid = int(data.get("pid", -1))
    data["stale"] = not pid_alive(pid) or not pid_matches_state(pid, data)
    data["status"] = "stale" if data["stale"] else "running"
    return data


def port_is_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _runtime_python(root: Path, backend: str) -> Path:
    return root / ".runtime" / backend / "bin" / "python"


def _write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def command_for(config: ServerConfig, paths: Paths) -> list[str]:
    python = _runtime_python(paths.root, config.backend)
    if not python.exists():
        raise ProcessError(f"backend environment is missing; run ./model-server setup --backend {config.backend}")
    if config.backend == "transformers":
        return [str(python), "-m", "uvicorn", "model_server.api:app", "--host", config.host, "--port", str(config.port), "--workers", "1"]
    vllm = paths.root / ".runtime" / "vllm" / "bin" / "vllm"
    if not vllm.exists():
        raise ProcessError("vLLM executable is missing; rerun ./model-server setup --backend vllm")
    return build_vllm_args(config, str(vllm))


def start_server(config: ServerConfig, root: Path, detach: bool | None = None) -> dict[str, Any]:
    paths = Paths(root)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    old = read_state(paths)
    if old and not old.get("stale"):
        raise ProcessError(f"a managed server is already running as PID {old['pid']} on {old['host']}:{old['port']}")
    if not port_is_free(config.host, config.port):
        raise ProcessError(f"{config.host}:{config.port} is occupied; stop its owner or choose --port")
    try:
        lock_fd = os.open(paths.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        if old and old.get("stale"):
            paths.lock.unlink(missing_ok=True)
            return start_server(config, root, detach)
        raise ProcessError("another start operation is in progress") from exc
    try:
        normalized = config.normalized()
        _write_json(paths.config, normalized.as_dict())
        command = command_for(normalized, paths)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = paths.logs_dir / f"server-{timestamp}.log"
        env = child_environment(normalized)
        env["MODEL_SERVER_CONFIG"] = str(paths.config)
        env["PYTHONUNBUFFERED"] = "1"
        use_detach = normalized.detach if detach is None else detach
        log_handle = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=use_detach,
        )
        state = {
            "backend": normalized.backend,
            "model": normalized.model,
            "served_model_name": normalized.served_model_name,
            "pid": process.pid,
            "host": normalized.host,
            "port": normalized.port,
            "visible_gpus": normalized.gpus,
            "dtype": normalized.dtype,
            "quantization": normalized.quantization,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "log_path": str(log_path),
            "configuration_hash": normalized.config_hash(),
            "command": command,
        }
        _write_json(paths.state, state)
        if use_detach:
            time.sleep(0.4)
            if process.poll() is not None:
                state["status"] = "failed"
                state["exit_code"] = process.returncode
                _write_json(paths.state, state)
                suffix = "; if this is model compatibility, try an explicit Transformers profile" if normalized.backend == "vllm" else ""
                raise ProcessError(f"server exited immediately; inspect {log_path}{suffix}")
            return state
        try:
            code = process.wait()
            state["exit_code"] = code
            return state
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=15)
            return state
    finally:
        os.close(lock_fd)
        paths.lock.unlink(missing_ok=True)


def stop_server(root: Path, timeout: float = 20.0) -> dict[str, Any]:
    paths = Paths(root)
    state = read_state(paths)
    if not state:
        raise ProcessError("no managed server state exists")
    if state.get("stale"):
        paths.state.unlink(missing_ok=True)
        return {"stopped": False, "stale_removed": True, "pid": state.get("pid")}
    pid = int(state["pid"])
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            paths.state.unlink(missing_ok=True)
            return {"stopped": True, "forced": False, "pid": pid}
        time.sleep(0.2)
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and pid_alive(pid):
        time.sleep(0.1)
    paths.state.unlink(missing_ok=True)
    return {"stopped": True, "forced": True, "pid": pid}


def saved_config(root: Path) -> ServerConfig:
    path = Paths(root).config
    if not path.exists():
        raise ProcessError("no saved effective configuration exists; run start first")
    return ServerConfig(**json.loads(path.read_text(encoding="utf-8"))).normalized()
