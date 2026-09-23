"""Controller-scaffolded diagnostics with one or two stages; not autonomous protocol use."""
from dataclasses import asdict
import hashlib

from .backend import ContextLimit
from .protocol import ProtocolError, RESERVED, guided_prompt


def step(engine, state, mode, emit=None):
    cfg = engine.config
    if mode not in {"full", "compact"}:
        raise ValueError("mode must be full or compact")
    remaining = cfg.max_total_new_tokens - state.generated_tokens
    if remaining <= 0 or len(state.events) >= cfg.max_events:
        state.status = "budget_exhausted"
        return
    # This transition happens only after the shared summary has been forked.
    # A normal compact run also reaches it only after compaction of stage 1.
    if state.phase == "next_stage":
        state.stage = 2
        if cfg.stage2_hide_source:
            # Both arms lose the source examples at the same boundary. The
            # detailed reasoning survives in full; compact retains its summary.
            state.prefix = engine.backend.prompt(
                guided_prompt(),
                "The Stage 1 examples are no longer available. Continue using "
                "only your earlier reasoning or its reusable conclusion."
            )
        if cfg.protocol == "guided_chat_two_stage":
            # Present the existing summary as the first assistant turn's answer.
            # Keep full's reasoning verbatim; compact has already removed it.
            summary_index = next(i for i, c in enumerate(state.chunks)
                                 if c["kind"] == "summary" and c["id"] == "e1")
            state.chunks.insert(summary_index, {"kind": "turn_boundary", "text": "</think>\n"})
            transition = engine.backend.next_user_turn(state.followup)
        else:
            transition = ("\nNext stage (provided by the controller):\n" + state.followup +
                          "\nI will use the saved conclusion to work through this stage.\n")
        state.chunks.append({"kind": "stage_instruction", "text": transition})
        state.chunks.append({"kind": "experiment", "id": "e2", "text": ""})
        state.phase = "experiment"
    phase = state.phase
    experiment_id = f"e{state.stage}"
    limits = {"experiment": cfg.max_new_tokens, "summary": cfg.summary_max_new_tokens,
              "answer": cfg.answer_max_new_tokens}
    stops = {"experiment": ["</experiment>", "</think>"],
             "summary": ["</summary>", "</think>"], "answer": ["</answer>"]}[phase]
    if phase == "answer":
        state.chunks.append({"kind": "answer", "text": '</think>\n\n'})
    before = state.text
    record = {"index": len(state.events), "mode": mode, "phase": phase,
              "protocol": cfg.protocol, "stage": state.stage, "controller_scaffolded": True,
              "active_before": before, "input_sha256": hashlib.sha256(before.encode()).hexdigest(),
              "seed": cfg.seed + len(state.events)}
    try:
        if engine.backend.count(before) >= cfg.max_context_tokens:
            raise ContextLimit("Configured context cap reached")
        result = engine.backend.generate(before, min(remaining, limits[phase]), record["seed"], stop_strings=stops)
        record["generation"] = asdict(result)
        state.generated_tokens += result.generated_tokens
        state.prefill_tokens += result.input_tokens
        state.generation_seconds += result.seconds
        boundary = next((s for s in stops if result.text.endswith(s)), None)
        raw_content = result.text[:-len(boundary)] if boundary else result.text
        # A length cap is a chunk boundary, not an experiment boundary. Keep
        # the unfinished text and let the model continue it in the same turn.
        # The experiment is archived only after a genuine closing tag and summary.
        if result.finish_reason == "length" and phase == "experiment" and cfg.continue_experiment_on_length:
            if not raw_content or RESERVED.search(state.chunks[-1]["text"] + raw_content):
                raise ProtocolError("Empty or nested protocol content in guided experiment continuation")
            state.chunks[-1]["text"] += raw_content
            record["continued"] = True
        elif result.finish_reason == "length" or (boundary is None and not (phase == "answer" and result.finish_reason == "eos")):
            state.status = "incomplete_phase_" + result.finish_reason
            record["error"] = f"{phase} did not reach its end boundary; no automatic repair"
        else:
            content = ((state.chunks[-1]["text"] + raw_content).strip() if phase == "experiment"
                       else raw_content.strip())
            if not content or RESERVED.search(content):
                raise ProtocolError(f"Empty or nested protocol content in guided {phase}")
            record["boundary"] = boundary or "eos"
            if phase == "experiment":
                state.pending_body = content
                state.chunks[-1]["text"] = content + "\n\n"
                if cfg.protocol == "guided_chat_two_stage" and state.stage == 2:
                    # This is a new assistant turn. After thinking closes, let the
                    # model answer normally instead of reopening a summary prompt.
                    state.phase = "answer"
                else:
                    state.chunks.append({"kind": "summary", "id": experiment_id, "text":
                        'Now I will state a short reusable conclusion without repeating the derivation.\nConclusion: '})
                    state.phase = "summary"
            elif phase == "summary":
                summary = "Conclusion: " + content
                state.chunks[-1]["text"] = summary + "\n"
                state.archive[experiment_id] = {"id": experiment_id, "body": state.pending_body, "summary": summary}
                state.pending_body = ""
                if mode == "compact":
                    engine.compact(state)
                state.phase = "next_stage" if state.followup is not None and state.stage == 1 else "answer"
            else:
                state.chunks[-1]["text"] += content
                if cfg.protocol == "guided_chat_two_stage" and state.stage == 2:
                    # The model's final answer is the last stage's conclusion.
                    # It is archived only after a complete, nonempty answer.
                    state.archive[experiment_id] = {"id": experiment_id,
                        "body": state.pending_body, "summary": content}
                    state.pending_body = ""
                    if mode == "compact":
                        engine.compact(state)
                state.answer, state.status, state.phase = content, "completed", "done"
    except ProtocolError as exc:
        state.status, record["error"] = "protocol_error", str(exc)
    except ContextLimit as exc:
        state.status, record["error"] = "context_limit", str(exc)
    record.update(status=state.status, active_after=state.text, active_after_tokens=engine.backend.count(state.text))
    state.events.append(record)
    if emit:
        emit(record)
