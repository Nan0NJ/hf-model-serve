import json
import os
import sys
from pathlib import Path

from model_server.config import ServerConfig
from model_server.process import Paths, command_for, read_state


def test_stale_pid_handling(tmp_path: Path):
    paths = Paths(tmp_path)
    paths.state_dir.mkdir(parents=True)
    paths.state.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")
    state = read_state(paths)
    assert state["stale"] is True
    assert state["status"] == "stale"


def test_live_pid_state(tmp_path: Path):
    paths = Paths(tmp_path)
    paths.state_dir.mkdir(parents=True)
    paths.state.write_text(json.dumps({"pid": os.getpid(), "command": [sys.executable]}), encoding="utf-8")
    assert read_state(paths)["status"] == "running"


def test_transformers_command_has_exactly_one_worker(tmp_path: Path):
    python = tmp_path / ".runtime" / "transformers" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    command = command_for(ServerConfig(model="x"), Paths(tmp_path))
    assert command.count("--workers") == 1
    assert command[command.index("--workers") + 1] == "1"
