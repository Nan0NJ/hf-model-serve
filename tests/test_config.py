import dataclasses
from pathlib import Path

import pytest

from model_server.cli import parser
from model_server.config import ConfigError, ServerConfig, build_config, child_environment, parse_gpu_list


def test_cli_parsing_all_required_commands():
    commands = {a.dest for a in parser()._subparsers._group_actions[0]._name_parser_map.values() for a in a._actions}
    expected = {"setup", "doctor", "start", "stop", "restart", "status", "logs", "test", "benchmark", "list-gpus", "config", "version"}
    assert expected == set(parser()._subparsers._group_actions[0]._name_parser_map)


def test_yaml_and_cli_precedence(tmp_path: Path):
    profile = tmp_path / "p.yaml"
    profile.write_text("model: org/base\nport: 8000\ngpus: \"0\"\n", encoding="utf-8")
    cfg = build_config(str(profile), {"port": 9000})
    assert cfg.model == "org/base"
    assert cfg.port == 9000
    assert cfg.served_model_name == "base"


@pytest.mark.parametrize(("raw", "expected"), [("0", [0]), ("1", [1]), ("0,1", [0, 1]), ("0,1,2,3", [0, 1, 2, 3])])
def test_gpu_list(raw, expected):
    assert parse_gpu_list(raw) == expected


@pytest.mark.parametrize("raw", ["", "0,", "-1", "0,0", "gpu0", "0 1"])
def test_invalid_gpu_list(raw):
    with pytest.raises(ConfigError):
        parse_gpu_list(raw)


def test_lora_configuration_validation():
    cfg = ServerConfig(model="base", lora_adapter="adapter", lora_name=None)
    with pytest.raises(ConfigError, match="supplied together"):
        cfg.validate()
    ServerConfig(model="base", lora_adapter="adapter", lora_name="named").validate()


def test_configuration_hash_stable_and_sensitive():
    first = ServerConfig(model="org/model", gpus="0").config_hash()
    same = ServerConfig(model="org/model", gpus="0").config_hash()
    changed = ServerConfig(model="org/model", gpus="1").config_hash()
    assert first == same
    assert first != changed
    assert len(first) == 16


def test_proxy_bypass_added_without_overwriting(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "example.org")
    monkeypatch.delenv("no_proxy", raising=False)
    env = child_environment(ServerConfig(model="x", gpus="0"))
    assert set(env["NO_PROXY"].split(",")) >= {"example.org", "127.0.0.1", "localhost"}
    assert set(env["no_proxy"].split(",")) >= {"127.0.0.1", "localhost"}
    assert env["CUDA_VISIBLE_DEVICES"] == "0"

