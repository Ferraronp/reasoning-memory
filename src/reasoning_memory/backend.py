from dataclasses import dataclass, field
import gc
import platform
import time

from .protocol import STOP_STRINGS


class ContextLimit(RuntimeError):
    pass


class InferenceOutOfMemory(RuntimeError):
    """The current generation could not fit in GPU memory."""


@dataclass
class Generation:
    text: str
    input_tokens: int
    generated_tokens: int
    seconds: float = 0.0
    finish_reason: str = "stop"
    token_ids: list[int] = field(default_factory=list)


class MockBackend:
    """Deterministic plumbing check, not an LLM and not quality evidence."""
    def prompt(self, system, task):
        return f"SYSTEM: {system}\nUSER: {task}\nASSISTANT: <think>\n"

    def next_user_turn(self, task):
        return "\nUSER: " + task + "\nASSISTANT: <think>\n"

    def count(self, text):
        return len(text)  # Explicitly character units for mock only.

    def generate(self, text, limit, seed, stop_strings=None):
        if stop_strings is not None:
            if '</experiment>' in stop_strings:
                out = '2 plus 3 equals 5.</think>'
            elif text.endswith('Conclusion: '):
                out = 'The sum is 5.</think>'
            else:
                out = '5</answer>'
        elif '<summary id="e1">' not in text.split("ASSISTANT:")[-1]:
            out = '<experiment id="e1">For the smoke task, 2 plus 3 equals 5.</experiment>\n<summary id="e1">The sum is 5.</summary>'
        else:
            out = '</think><answer>5</answer>'
        out = out[:limit]
        stops = STOP_STRINGS if stop_strings is None else stop_strings
        return Generation(out, self.count(text), len(out), finish_reason="stop" if any(s in out for s in stops) else "length")

    def metadata(self):
        return {"backend": "mock", "units": "characters", "quality_evidence": False}


class HFBackend:
    """Batch size 1. Every call prefills the entire canonical text from scratch."""
    def __init__(self, config):
        import os
        os.environ["USE_TF"] = "0"  # This backend uses PyTorch only.
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch, self.transformers, self.config = torch, transformers, config
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Select a GPU runtime or use a CPU/float32 config.")
        if config.device == "cpu" and config.dtype != "float32":
            raise ValueError("Use float32 for the initial CPU check")
        if config.quantization == "int8" and config.device != "cuda":
            raise ValueError("This starter supports INT8 only on CUDA")
        if config.dtype == "bfloat16" and config.device == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 unsupported on this GPU; use float16")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision, trust_remote_code=False)
        kwargs = dict(revision=config.revision, dtype=getattr(torch, config.dtype),
                      device_map={"": config.device}, trust_remote_code=False)
        if config.quantization == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs).eval()
        self.offloaded_cache_retries = 0
        maximum = getattr(self.model.config, "max_position_embeddings", config.max_context_tokens)
        self.context_limit = min(config.max_context_tokens, maximum)
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def prompt(self, system, task):
        # Apply template once. Subsequent calls continue this exact assistant turn.
        prefix = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": task}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        # v0 adapter is explicitly for Qwen3 dense thinking checkpoints.
        if not prefix.rstrip().endswith("<think>"):
            if prefix.rstrip().endswith("<|im_start|>assistant"):
                prefix += "<think>\n"
            else:
                raise ValueError("Unsupported chat-template suffix; add a model-specific thinking adapter")
        return prefix

    def next_user_turn(self, task):
        # Qwen3 adapter: append only the new turn. Re-rendering old assistant
        # messages through a template can remove their thinking content.
        if not {"<|im_start|>", "<|im_end|>"}.issubset(set(self.tokenizer.all_special_tokens)):
            raise ValueError("Unsupported chat turn delimiters; expected Qwen3")
        return "<|im_end|>\n<|im_start|>user\n" + task + "<|im_end|>\n<|im_start|>assistant\n<think>\n"

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, text, limit, seed, stop_strings=None):
        torch, config = self.torch, self.config
        self.transformers.set_seed(seed)
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").to(config.device)
        n = inputs.input_ids.shape[-1]
        if n >= self.context_limit:
            raise ContextLimit(f"Context {n} >= limit {self.context_limit}; never silently truncated")
        limit = min(limit, self.context_limit - n)
        # Fresh GenerationConfig prevents inherited forced tokens/penalties/cache settings.
        eos = self.model.generation_config.eos_token_id
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = eos[0] if isinstance(eos, list) else eos
        stops = STOP_STRINGS if stop_strings is None else stop_strings
        params = dict(max_new_tokens=limit, do_sample=config.temperature > 0,
                      use_cache=True, num_beams=1, eos_token_id=eos, pad_token_id=pad,
                      bos_token_id=self.model.generation_config.bos_token_id,
                      stop_strings=stops)
        if config.temperature > 0:
            params.update(temperature=config.temperature, top_p=config.top_p, top_k=config.top_k)
        if config.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        # No past_key_values accepted or returned. Cache lives only within this call.
        first_oom = False
        with torch.inference_mode():
            try:
                output = self.model.generate(**inputs,
                    generation_config=self.transformers.GenerationConfig(**params), tokenizer=self.tokenizer)
            except torch.OutOfMemoryError:
                first_oom = True
        if first_oom:
            gc.collect()
            if config.device == "cuda":
                torch.cuda.empty_cache()
            if not config.retry_offloaded_on_oom:
                raise InferenceOutOfMemory("CUDA out of memory during generation")
            self.offloaded_cache_retries += 1
            # A fresh attempt with the same seed moves most KV cache to CPU.
            # The prompt is still prefetched from scratch for this attempt.
            self.transformers.set_seed(seed)
            second_oom = False
            with torch.inference_mode():
                try:
                    output = self.model.generate(**inputs,
                        generation_config=self.transformers.GenerationConfig(
                            **{**params, "cache_implementation": "offloaded"}), tokenizer=self.tokenizer)
                except torch.OutOfMemoryError:
                    second_oom = True
            if second_oom:
                gc.collect()
                if config.device == "cuda":
                    torch.cuda.empty_cache()
                raise InferenceOutOfMemory("CUDA out of memory even with offloaded KV cache")
        if config.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        ids = output[0, n:].tolist()
        del output, inputs
        decoded = self.tokenizer.decode(ids, skip_special_tokens=False)
        ends = [decoded.find(s) + len(s) for s in stops if s in decoded]
        # Stop strings can end mid-token. Record original IDs but use canonical text up to stop.
        if ends:
            decoded, reason = decoded[:min(ends)], "stop"
        else:
            eos_ids = eos if isinstance(eos, list) else [eos]
            reason = "eos" if ids and ids[-1] in eos_ids else "length"
            if reason == "eos":
                decoded = self.tokenizer.decode(ids[:-1], skip_special_tokens=False)
        return Generation(decoded, n, len(ids), elapsed, reason, ids)

    def metadata(self):
        torch = self.torch
        return {"backend": "hf", "model_id": self.config.model_id,
                "resolved_model_revision": getattr(self.model.config, "_commit_hash", None),
                "tokenizer_revision": self.tokenizer.init_kwargs.get("_commit_hash"),
                "torch": torch.__version__, "transformers": self.transformers.__version__,
                "python": platform.python_version(), "context_limit": self.context_limit,
                "offloaded_cache_retries": self.offloaded_cache_retries,
                "gpu": torch.cuda.get_device_name() if self.config.device == "cuda" else None,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if self.config.device == "cuda" else None}


def make_backend(config):
    return MockBackend() if config.backend == "mock" else HFBackend(config)
