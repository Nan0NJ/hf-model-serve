# HF Model Serve

A standalone, localhost-first CLI for installing, diagnosing, running, monitoring, testing, and stopping local Hugging Face causal language models. It supports a generic Transformers engine and delegates vLLM inference to the official `vllm serve` command.

No existing project is imported or modified. Runtime environments, state, and logs stay under this repository's `.runtime/` directory. RAG is intentionally outside this project: a caller should retrieve evidence and place it in the chat messages.

## Quick start: GaMS3 on a V100

```bash
unzip hf-model-serve.zip
cd hf-model-serve

./model-server setup --backend transformers
./model-server doctor --config configs/gams3-v100-int8.yaml
./model-server start --config configs/gams3-v100-int8.yaml
./model-server status
./model-server logs --follow
```

In another terminal:

```bash
./model-server test \
  --prompt "Koliko je 2 + 2? Odgovori z enim kratkim stavkom."

./model-server benchmark --requests 3 --concurrency 1
./model-server stop
```

The endpoint root for chat-completion clients is `http://127.0.0.1:8000/v1`. Available routes are:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Only non-streaming chat completion is supported. `temperature: 0` is deterministic. Example:

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gams3-12b","messages":[{"role":"user","content":"Pozdrav!"}],"temperature":0}'
```

## Why the GaMS3 profile is specific

The supplied profile preserves the known working load path:

```text
Transformers + BitsAndBytes INT8 + float32 + SDPA + device_map={"": 0}
```

On a Tesla V100 (compute capability 7.0), native BF16 is not supported. For this checkpoint, unquantized FP16 is also deny-listed on V100 because it can load yet decode to empty output. Prior GaMS3 vLLM attempts are deny-listed because of known Gemma 3 rope-scaling/config and BitsAndBytes weight-shape failures. The doctor refuses those configurations instead of silently changing precision or backend.

Current generic BitsAndBytes documentation lists LLM.int8() at compute capability 7.5+, while broader 8-bit quantization kernels support older NVIDIA generations. Because this exact GaMS3 configuration has been independently demonstrated on V100 CC 7.0, the doctor emits a warning—not a false blanket pass—and requires the actual import/load test on the target machine.

## Commands

```text
model-server setup       create/update isolated backend venvs
model-server doctor      preflight hardware, software, model, network, and config
model-server start       start exactly one model process
model-server stop        terminate the recorded PID gracefully, then force if needed
model-server restart     restart the last effective configuration
model-server status      print structured process state
model-server logs        tail the recorded log; add --follow
model-server test        validate health/models/chat and reject empty output
model-server benchmark   run a bounded, non-OOM-seeking benchmark
model-server list-gpus   list physical GPU indices, VRAM, CC, and driver
model-server config      print merged/validated configuration and hash
model-server version     print CLI and installed backend versions
```

Run `./model-server COMMAND --help` for all options.

## Isolated installation

```bash
./model-server setup --backend transformers
./model-server setup --backend vllm
./model-server setup --all
```

The environments are deliberately separate:

```text
.runtime/transformers/
.runtime/vllm/
```

The launcher always calls the selected environment's Python or vLLM executable; manual activation is unnecessary. Setup uses `python -m venv` and `pip` without root. If Ubuntu lacks venv support, setup prints the exact `python3-venv` package remedy but never runs `sudo`.

Top-level versions are pinned in `requirements/*.lock`. The Transformers profile uses the official PyTorch CUDA 12.6 wheel index. vLLM owns its tightly coupled Torch/CUDA dependency set, so it lives in another environment.

### Driver, toolkit, and wheel CUDA are different

The NVIDIA driver lets processes use the GPU and must be new enough for the CUDA runtime packaged with a wheel. The system CUDA toolkit provides developer tools such as `nvcc`; it does not replace the runtime libraries bundled with ordinary PyTorch/vLLM wheels. Therefore a local CUDA 13.0 toolkit does not force PyTorch to use CUDA 13.0, and missing `nvcc` is not a failure when compatible prebuilt wheels are used.

The pins are a conservative compatibility snapshot for Python 3.12 and the GaMS3-era Transformers API, not arbitrary newest packages. Relevant primary documentation:

- [PyTorch local installation and CUDA wheel selector](https://pytorch.org/get-started/locally/)
- [PyTorch version-specific wheel commands](https://pytorch.org/get-started/previous-versions/)
- [Transformers BitsAndBytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [BitsAndBytes installation and hardware matrix](https://huggingface.co/docs/bitsandbytes/installation)
- [Accelerate installation](https://huggingface.co/docs/accelerate/basic_tutorials/install)
- [PEFT installation](https://huggingface.co/docs/peft/install)
- [vLLM installation](https://docs.vllm.ai/en/stable/getting_started/installation/)
- [FastAPI deployment concepts](https://fastapi.tiangolo.com/deployment/)
- [Uvicorn settings](https://www.uvicorn.org/settings/)

## Doctor

The doctor reports every check as `PASS`, `WARN`, or `FAIL`; every `FAIL` has a concrete remedy and causes exit status 2. JSON output is stable enough for automation:

```bash
./model-server doctor --config configs/gams3-v100-int8.yaml --json
```

Checks cover OS/architecture, Python, free RAM/disk, NVIDIA driver, GPU identity/index/VRAM/compute capability, parent and child CUDA visibility, backend imports and versions, `torch.cuda`, wheel CUDA runtime, optional `nvcc`, Hugging Face credentials (never their value), model metadata access, port ownership, model/backend rules, dtype and attention compatibility, weight-memory/headroom estimates, context/output lengths, concurrency, tensor-parallel/head divisibility, LoRA metadata, proxies, and localhost bypass.

Memory estimates are explicitly estimates. They include a quantization loading factor but cannot exactly predict KV cache, temporary buffers, allocator fragmentation, architecture-specific tensors, or other processes. Fitting weights is never treated as proof of numerical compatibility.

## Configuration and precedence

Profiles are flat YAML `key: value` documents. The loader intentionally rejects nested/unknown keys, and command-line options override the profile:

```bash
./model-server config \
  --config configs/gams3-v100-int8.yaml \
  --port 8100 --max-context 2048
```

The effective configuration is saved as `.runtime/state/effective-config.json`, and process state records a deterministic configuration hash. Included examples cover V100 INT8, FP16, Ampere+ BF16, NF4 INT4, LoRA, and two-GPU vLLM.

GPU lists are physical indices such as `--gpus 0`, `--gpus 1`, or `--gpus 0,1`. The child gets `CUDA_VISIBLE_DEVICES` before importing Torch. One selected Transformers GPU uses explicit placement; multiple selected GPUs use `device_map=auto`, which performs layer placement and is not vLLM tensor parallelism or continuous batching. vLLM validates `--tensor-parallel`, then receives `--tensor-parallel-size` unchanged.

## Transformers behavior

The backend uses `AutoTokenizer` and `AutoModelForCausalLM`, not `AutoProcessor`, so text-only Gemma 3 derivatives do not need image processor assets. It uses the tokenizer chat template. If the template rejects the `system` role, system instructions are deterministically joined and prepended to the first user message.

Generation is guarded by a semaphore. This protects memory and provides controlled admission, but it is not vLLM-style continuous batching. Uvicorn is hard-coded to one worker so the model is loaded once.

Quantization modes:

- `none`: selected model dtype, unchanged.
- `int8`: `BitsAndBytesConfig(load_in_8bit=True)`.
- `int4`: NF4 by default, double quantization by default, explicit compute dtype.

An adapter requires both `--lora-adapter` and `--lora-name`. Transformers uses PEFT; vLLM receives `--enable-lora --lora-modules NAME=PATH`. A previously merged adapter is simply an ordinary `--model`.

## vLLM behavior

The project does not reimplement vLLM inference. It validates configuration and launches the installed official CLI. Example:

```bash
./model-server setup --backend vllm
./model-server doctor --config configs/vllm-multigpu-example.yaml
./model-server start --config configs/vllm-multigpu-example.yaml
```

Failed starts retain logs. When architecture or quantized-weight compatibility is uncertain, inspect the log and try an explicit Transformers profile; no backend, dtype, or quantization fallback occurs silently.

The generic `int8` and `int4` CLI modes mean Transformers BitsAndBytes and are therefore rejected for vLLM. For vLLM, choose a checkpoint in a quantization format supported by that vLLM release and leave this setting as `none` so vLLM can use the checkpoint metadata; the server never guesses or remaps a quantization method.

## Processes, logs, and security

Detached mode uses a new session, a precise PID file, and append-only timestamped logs. Stop signals only that PID: broad `pkill` is never used. A lock prevents concurrent starts; active state and occupied ports prevent duplicate model loads; stale state is recognized and removed safely.

The default host is `127.0.0.1`. Binding `0.0.0.0` or `::` requires an explicit flag and produces a warning. For Transformers authentication, pass `--api-key-env MY_SERVER_KEY`; the variable name is recorded, but its value is never printed or written. Clients then send `Authorization: Bearer ...`. This control plane deliberately rejects that option for vLLM instead of placing a secret in process arguments; keep vLLM on localhost or put an authenticated reverse proxy in front of it.

## Proxy handling

Managed children always receive both:

```text
NO_PROXY=127.0.0.1,localhost
no_proxy=127.0.0.1,localhost
```

The built-in test and benchmark use an explicit no-proxy opener, so an institutional Squid proxy cannot intercept localhost. For manual tests use `curl --noproxy '*'`.

## Interactive and managed alternatives

Detached mode is simplest. For an interactive terminal, override a profile's detach setting:

```bash
./model-server start --config configs/gams3-v100-int8.yaml --no-detach
```

To keep the CLI in tmux:

```bash
tmux new -s model-server
./model-server start --config configs/gams3-v100-int8.yaml --no-detach
```

For `systemd --user`, copy `deploy/systemd/hf-model-serve.service.example` to `~/.config/systemd/user/hf-model-serve.service`, replace the absolute path placeholder, then run `systemctl --user daemon-reload` and `systemctl --user enable --now hf-model-serve`. Do not combine systemd management with the CLI's detached instance.

Docker and Kubernetes files are future deployment templates only. Neither is needed—or assumed installed—for one local process. Review paths, secrets, storage, and network policy before production use.

## Offline and model access

Use a local model path or a populated cache with `--offline`; the launcher sets both Hugging Face offline variables. Use `--revision` for reproducibility and `--download-dir` for a large filesystem. Authenticate with `huggingface-cli login` or `HF_TOKEN`; doctor detects credentials and tests model metadata without displaying the token.

## Testing the repository

Tests do not download models or initialize CUDA:

```bash
python -m compileall model_server tests
.runtime/transformers/bin/python -m pytest
```
