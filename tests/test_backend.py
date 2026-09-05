from model_server.backends.base import fold_system_messages
from model_server.backends.vllm_backend import build_vllm_args
from model_server.config import ServerConfig


def test_system_role_fallback_is_deterministic():
    source = [
        {"role": "system", "content": "First"},
        {"role": "system", "content": "Second"},
        {"role": "user", "content": "Question"},
    ]
    assert fold_system_messages(source) == [{"role": "user", "content": "First\n\nSecond\n\nQuestion"}]


def test_vllm_subprocess_argument_construction_no_shell():
    cfg = ServerConfig(
        backend="vllm", model="org/model", served_model_name="served", gpus="0,1",
        dtype="float16", tensor_parallel=2, lora_adapter="org/adapter", lora_name="a",
    )
    args = build_vllm_args(cfg, "/venv/bin/vllm")
    assert args[:3] == ["/venv/bin/vllm", "serve", "org/model"]
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--lora-modules") + 1] == "a=org/adapter"
    assert all(";" not in item for item in args)

