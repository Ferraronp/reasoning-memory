from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Callable

from .backend import ContextLimit
from .protocol import ProtocolError, guided_prompt, parse, system_prompt


@dataclass
class State:
    prefix: str
    chunks: list[dict] = field(default_factory=list)
    archive: dict = field(default_factory=dict)
    restored: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    generated_tokens: int = 0
    prefill_tokens: int = 0
    generation_seconds: float = 0.0
    status: str = "running"
    answer: str | None = None
    phase: str = "autonomous"
    pending_body: str = ""
    followup: str | None = None
    stage: int = 1

    @property
    def text(self):
        return self.prefix + "".join(c["text"] for c in self.chunks)


class Engine:
    def __init__(self, backend, config):
        self.backend, self.config = backend, config

    def start(self, prompt, followup=None):
        if self.config.protocol in {"guided_two_stage", "guided_chat_two_stage"}:
            if not isinstance(followup, str) or not followup.strip():
                raise ValueError("Two-stage guided protocol requires a nonempty task.followup")
        elif followup is not None:
            raise ValueError("task.followup requires a two-stage guided protocol; it must not be silently ignored")
        if self.config.protocol.startswith("guided_"):
            state = State(self.backend.prompt(guided_prompt(), prompt), phase="experiment", followup=followup)
            # Let the model start its ordinary reasoning; bookkeeping tags stay external.
            state.chunks.append({"kind": "experiment", "id": "e1", "text": ""})
            return state
        return State(self.backend.prompt(system_prompt(self.config.allow_restore), prompt))

    @staticmethod
    def compact(state):
        # Original body remains only in the archive and event journal, never model input.
        state.chunks = [c for c in state.chunks if c["kind"] not in {"experiment", "restored"}]

    def apply(self, state, text, mode):
        event = parse(text)
        if event.kind == "experiment":
            if event.id in state.archive:
                raise ProtocolError(f"Duplicate experiment ID: {event.id}")
            state.archive[event.id] = {"id": event.id, "body": event.body, "summary": event.summary}
            state.chunks.extend([
                {"kind": "prose", "text": event.lead},
                {"kind": "experiment", "id": event.id,
                 "text": f'<experiment id="{event.id}">{event.body}</experiment>\n'},
                {"kind": "summary", "id": event.id,
                 "text": f'<summary id="{event.id}">{event.summary}</summary>\n'},
            ])
            if mode == "compact":
                self.compact(state)
        elif event.kind == "restore":
            if not self.config.allow_restore:
                raise ProtocolError("Restore is disabled")
            if event.id not in state.archive:
                raise ProtocolError(f"Unknown archive ID: {event.id}")
            if event.id in state.restored or len(state.restored) >= self.config.max_restores:
                raise ProtocolError("Restore limit reached or repeated ID")
            state.restored.append(event.id)
            state.chunks.append({"kind": "prose", "text": event.lead + f'<restore id="{event.id}"/>\n'})
            state.chunks.append({"kind": "restored", "id": event.id,
                "text": f'<restored id="{event.id}">{state.archive[event.id]["body"]}</restored>\n'})
        else:
            if not state.archive:
                raise ProtocolError("Final answer before any completed experiment")
            state.chunks.append({"kind": "answer", "text": text})
            state.answer, state.status = event.answer.strip(), "completed"
        return event

    def step(self, state, mode, emit: Callable | None = None):
        if self.config.protocol.startswith("guided_"):
            from .guided import step
            return step(self, state, mode, emit)
        cfg = self.config
        if mode not in {"full", "compact"}:
            raise ValueError("mode must be full or compact")
        remaining = cfg.max_total_new_tokens - state.generated_tokens
        if remaining <= 0 or len(state.events) >= cfg.max_events:
            state.status = "budget_exhausted"
            return
        before = state.text
        record = {"index": len(state.events), "mode": mode,
                  "active_before": before,
                  "input_sha256": hashlib.sha256(before.encode()).hexdigest(),
                  "seed": cfg.seed + len(state.events)}
        try:
            if self.backend.count(before) >= cfg.max_context_tokens:
                raise ContextLimit("Configured context cap reached")
            result = self.backend.generate(before, min(remaining, cfg.max_new_tokens), record["seed"])
            record["generation"] = asdict(result)
            state.generated_tokens += result.generated_tokens
            state.prefill_tokens += result.input_tokens
            state.generation_seconds += result.seconds
            if result.finish_reason != "stop":
                state.status = "incomplete_event_" + result.finish_reason
                record["error"] = (
                    f"Generation ended with {result.finish_reason} before a protocol event completed. "
                    "No compaction was performed; inspect generation.text."
                )
            else:
                event = self.apply(state, result.text, mode)
                record["event"] = asdict(event)
        except ProtocolError as exc:
            state.status, record["error"] = "protocol_error", str(exc)
        except ContextLimit as exc:
            state.status, record["error"] = "context_limit", str(exc)
        record["status"] = state.status
        record["active_after"] = state.text
        record["active_after_tokens"] = self.backend.count(state.text)
        state.events.append(record)
        if emit:
            emit(record)

    def run(self, state, mode, emit=None, stop_after_first=False):
        while state.status == "running":
            self.step(state, mode, emit)
            if stop_after_first and state.archive:
                break
        return state

    def fork(self, state, mode):
        fork = deepcopy(state)
        if mode == "compact":
            self.compact(fork)
        return fork

    def result(self, state, expected=None):
        # An unfinished run has no answer to score. Protocol completion is
        # reported separately from the accuracy of completed answers.
        if expected is None or state.status != "completed":
            correct = None
        elif isinstance(expected, list):
            # A grid is compared structurally: JSON whitespace does not
            # change correctness, but missing/extra cells do.
            try:
                decoded = json.loads(state.answer)
                correct = (isinstance(decoded, list) and decoded == expected and
                           all(isinstance(row, list) and all(type(cell) is int for cell in row)
                               for row in decoded))
            except (TypeError, ValueError):
                correct = False
        else:
            correct = state.answer == str(expected).strip()
        return {"status": state.status, "answer": state.answer, "expected": expected,
                "correct": correct, "generated_tokens": state.generated_tokens,
                "prefill_tokens": state.prefill_tokens, "generation_seconds": state.generation_seconds,
                "active_tokens": self.backend.count(state.text), "experiments": len(state.archive),
                "restores": len(state.restored), "active_text": state.text, "archive": state.archive}
