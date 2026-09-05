"""Generic text-only Transformers causal-language-model backend."""

from __future__ import annotations

import os
import threading
from typing import Any

from .base import Backend, GenerationResult, fold_system_messages
from ..config import ServerConfig, parse_gpu_list


class TransformersBackend(Backend):
    def __init__(self, config: ServerConfig):
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._gate = threading.BoundedSemaphore(config.max_concurrency)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        # api.py sets this before importing this module; keep the invariant explicit.
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpus
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        common: dict[str, Any] = {
            "revision": self.config.revision,
            "trust_remote_code": self.config.trust_remote_code,
            "cache_dir": self.config.download_dir,
            "local_files_only": self.config.offline,
        }
        common = {k: v for k, v in common.items() if v is not None}
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model, **common)

        dtypes = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        model_args = dict(common)
        model_args.update({
            "torch_dtype": dtypes[self.config.dtype],
            "low_cpu_mem_usage": True,
        })
        if self.config.attention != "auto":
            model_args["attn_implementation"] = self.config.attention
        if self.config.quantization == "int8":
            model_args["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.config.quantization == "int4":
            compute = dtypes[self.config.bnb_compute_dtype]
            model_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_double_quant,
                bnb_4bit_compute_dtype=compute,
            )
        model_args["device_map"] = {"": 0} if len(parse_gpu_list(self.config.gpus)) == 1 else "auto"

        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model, **model_args)
            if self.config.lora_adapter:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(
                    self._model,
                    self.config.lora_adapter,
                    adapter_name=self.config.lora_name,
                    revision=self.config.revision,
                )
            self._model.eval()
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA out of memory while loading the model. Stop other GPU jobs, lower context/concurrency, "
                "or select INT8/INT4. The server never changes precision automatically."
            ) from exc

    def _template(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        kwargs = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        normalized = messages
        try:
            if not getattr(self._tokenizer, "chat_template", None):
                raise LookupError("tokenizer has no chat template")
            return self._tokenizer.apply_chat_template(normalized, **kwargs)
        except (ValueError, TypeError, jinja2_error()):
            normalized = fold_system_messages(messages)
            return self._tokenizer.apply_chat_template(normalized, **kwargs)
        except LookupError:
            normalized = fold_system_messages(messages)
            text = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in normalized)
            text += "\n\nAssistant:"
            return self._tokenizer(text, return_tensors="pt")

    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, top_p: float) -> GenerationResult:
        if not self.loaded:
            raise RuntimeError("model is not loaded")
        inputs = self._template(messages)
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])
        allowed_output = min(max_tokens, self.config.max_output_tokens)
        if prompt_tokens + allowed_output > self.config.max_context:
            raise ValueError(
                f"prompt ({prompt_tokens}) + requested output ({allowed_output}) exceeds max context "
                f"({self.config.max_context}); shorten the prompt or output"
            )
        args: dict[str, Any] = {
            "max_new_tokens": allowed_output,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            args.update(temperature=temperature, top_p=top_p)
        with self._gate, self._torch.inference_mode():
            try:
                output = self._model.generate(**inputs, **args)
            except self._torch.cuda.OutOfMemoryError as exc:
                self._torch.cuda.empty_cache()
                raise RuntimeError("CUDA out of memory during generation; reduce max tokens, context, or concurrency") from exc
        generated = output[0, prompt_tokens:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        completion_tokens = int(generated.shape[0])
        # Preserve empty text so clients/tests can identify this numerical failure mode.
        finish_reason = "length" if completion_tokens >= allowed_output else "stop"
        return GenerationResult(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, finish_reason=finish_reason)

    def metrics(self) -> dict[str, Any]:
        if self._torch and self._torch.cuda.is_available():
            return {"peak_gpu_memory_bytes": int(self._torch.cuda.max_memory_allocated())}
        return {}


def jinja2_error():
    """Return Jinja's template error class without making Jinja a control-plane dependency."""
    try:
        from jinja2 import TemplateError
        return TemplateError
    except ImportError:
        return RuntimeError
