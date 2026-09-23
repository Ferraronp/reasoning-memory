import unittest
from dataclasses import replace

from reasoning_memory.backend import Generation, MockBackend
from reasoning_memory.config import Config
from reasoning_memory.engine import Engine
from reasoning_memory.protocol import ProtocolError, parse

E1 = '<experiment id="e1">UNIQUE_RAW_BODY_1</experiment><summary id="e1">Result A.</summary>'
E2 = '<experiment id="e2">UNIQUE_RAW_BODY_2</experiment><summary id="e2">Result B.</summary>'
FINAL = '</think><answer>5</answer>'


class ScriptedBackend(MockBackend):
    def __init__(self, texts):
        self.texts, self.inputs = iter(texts), []

    def generate(self, text, limit, seed):
        self.inputs.append(text)
        out = next(self.texts)
        return Generation(out, len(text), len(out))


class EngineTests(unittest.TestCase):
    def engine(self, texts=(), **kwargs):
        cfg = replace(Config(), max_context_tokens=16000, **kwargs)
        backend = ScriptedBackend(texts)
        return Engine(backend, cfg), backend

    def test_compaction_removes_body_from_next_input_not_archive(self):
        engine, backend = self.engine([E1, FINAL])
        state = engine.run(engine.start("task"), "compact")
        self.assertEqual(state.status, "completed")
        self.assertNotIn("UNIQUE_RAW_BODY_1", backend.inputs[1])
        self.assertIn("Result A.", backend.inputs[1])
        self.assertEqual(state.archive["e1"]["body"], "UNIQUE_RAW_BODY_1")

    def test_full_preserves_body(self):
        engine, backend = self.engine([E1, FINAL])
        engine.run(engine.start("task"), "full")
        self.assertIn("UNIQUE_RAW_BODY_1", backend.inputs[1])

    def test_forks_share_summary_and_do_not_mutate_parent(self):
        engine, _ = self.engine([E1])
        original = engine.run(engine.start("task"), "full", stop_after_first=True)
        compact = engine.fork(original, "compact")
        full = engine.fork(original, "full")
        self.assertIn("UNIQUE_RAW_BODY_1", original.text)
        self.assertNotIn("UNIQUE_RAW_BODY_1", compact.text)
        self.assertEqual(full.archive, compact.archive)
        compact.archive["e1"]["summary"] = "changed"
        self.assertEqual(original.archive["e1"]["summary"], "Result A.")

    def test_restore_then_rearchive_at_next_summary(self):
        engine, backend = self.engine([E1, '<restore id="e1"/>', E2, FINAL], allow_restore=True)
        state = engine.run(engine.start("task"), "compact")
        self.assertEqual(state.status, "completed")
        self.assertIn("UNIQUE_RAW_BODY_1", backend.inputs[2])
        self.assertNotIn("UNIQUE_RAW_BODY_1", backend.inputs[3])
        self.assertNotIn("UNIQUE_RAW_BODY_2", backend.inputs[3])
        self.assertIn("Result A.", backend.inputs[3])
        self.assertIn("Result B.", backend.inputs[3])

    def test_restore_disabled(self):
        engine, _ = self.engine([E1, '<restore id="e1"/>'])
        state = engine.run(engine.start("task"), "compact")
        self.assertEqual(state.status, "protocol_error")

    def test_unknown_and_repeated_restore(self):
        for requests in ([E1, '<restore id="missing"/>'], [E1, '<restore id="e1"/>', '<restore id="e1"/>']):
            engine, _ = self.engine(requests, allow_restore=True)
            self.assertEqual(engine.run(engine.start("task"), "compact").status, "protocol_error")

    def test_duplicate_id(self):
        engine, _ = self.engine([E1, E1])
        state = engine.run(engine.start("task"), "compact")
        self.assertEqual(state.status, "protocol_error")
        self.assertEqual(len(state.archive), 1)

    def test_final_requires_experiment(self):
        engine, _ = self.engine([FINAL])
        self.assertEqual(engine.run(engine.start("task"), "full").status, "protocol_error")

    def test_bad_grammar_never_silently_repairs(self):
        for text in (E1.replace('summary id="e1"', 'summary id="e2"'),
                     E1.replace("Result A.", ""), E1 + "extra",
                     E1.replace("UNIQUE_RAW_BODY_1", '<experiment id="nested">x</experiment>')):
            with self.assertRaises(ProtocolError):
                parse(text)

    def test_context_limit_before_generation(self):
        engine, backend = self.engine([E1])
        engine.config.max_context_tokens = 2
        state = engine.run(engine.start("task"), "compact")
        self.assertEqual(state.status, "context_limit")
        self.assertEqual(backend.inputs, [])

    def test_event_budget_stops_loop(self):
        engine, _ = self.engine([E1], max_events=1)
        state = engine.run(engine.start("task"), "compact")
        self.assertEqual(state.status, "budget_exhausted")
        self.assertFalse(engine.result(state, "5")["correct"])

    def test_incomplete_output_is_not_compacted(self):
        class Truncated(MockBackend):
            def generate(self, *args):
                return Generation(E1[:-4], 10, 20, finish_reason="length")
        engine = Engine(Truncated(), Config(max_context_tokens=16000))
        state = engine.run(engine.start("task"), "compact")
        self.assertEqual(state.status, "incomplete_event_length")
        self.assertEqual(state.archive, {})

    def test_no_expected_answer_in_model_input(self):
        engine, backend = self.engine([E1, FINAL])
        state = engine.run(engine.start("task"), "compact")
        engine.result(state, "SECRET_GOLD")
        self.assertTrue(all("SECRET_GOLD" not in text for text in backend.inputs))


if __name__ == "__main__":
    unittest.main()
