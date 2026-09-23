from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass
class Config:
    backend: str = "mock"
    protocol: str = "autonomous"
    model_id: str = "Qwen/Qwen3-0.6B"
    revision: str = "main"
    dtype: str = "float16"
    quantization: str = "none"
    device: str = "cuda"
    max_context_tokens: int = 8192
    max_new_tokens: int = 1024
    max_total_new_tokens: int = 4096
    max_events: int = 12
    max_restores: int = 2
    allow_restore: bool = False
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 42
    summary_max_new_tokens: int = 256
    answer_max_new_tokens: int = 128
    stage2_hide_source: bool = False

    def __post_init__(self):
        for name, choices in {
            "backend": {"hf", "mock"}, "protocol": {"autonomous", "guided_single", "guided_two_stage", "guided_chat_two_stage"},
            "dtype": {"float16", "bfloat16", "float32"},
            "quantization": {"none", "int8"}, "device": {"cuda", "cpu"}
        }.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"Invalid {name}: {getattr(self, name)}")
        for name in ("max_context_tokens", "max_new_tokens", "max_total_new_tokens", "max_events", "summary_max_new_tokens", "answer_max_new_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_restores < 0 or self.temperature < 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("Invalid sampling/restore limits")
        if self.protocol.startswith("guided_") and self.allow_restore:
            raise ValueError("Guided diagnostics do not support restore")
        if self.stage2_hide_source and self.protocol != "guided_chat_two_stage":
            raise ValueError("stage2_hide_source requires guided_chat_two_stage")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)
