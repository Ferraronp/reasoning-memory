import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reasoning_memory import cli
from reasoning_memory.backend import Generation, MockBackend
from reasoning_memory.config import Config
from reasoning_memory.engine import Engine


class GuidedBackend(MockBackend):
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.inputs = []

    def generate(self, text, limit, seed, stop_strings=None):
        self.inputs.append(text)
        content, reason = next(self.outputs)
        return Generation(content, len(text), len(content), finish_reason=reason)


class RegressionTests(unittest.TestCase):
    def test_guided_same_summary_clean_compaction_and_eos_final(self):
        backend = GuidedBackend([
            ('UNIQUE_CALCULATION_BODY 2+3=5</think>', 'stop'),
            ('The result is 5.</think>', 'stop'),
            ('5', 'eos'), ('5</answer>', 'stop'),
        ])
        engine = Engine(backend, Config(protocol='guided_single', max_context_tokens=16000))
        shared = engine.run(engine.start('Compute the sum'), 'full', stop_after_first=True)
        self.assertEqual(len(shared.archive), 1)
        self.assertEqual(shared.phase, 'answer')
        full = engine.run(engine.fork(shared, 'full'), 'full')
        compact = engine.run(engine.fork(shared, 'compact'), 'compact')
        self.assertEqual(full.answer, '5')
        self.assertEqual(compact.answer, '5')
        self.assertIn('UNIQUE_CALCULATION_BODY', backend.inputs[2])
        self.assertNotIn('UNIQUE_CALCULATION_BODY', backend.inputs[3])
        self.assertIn('Conclusion: The result is 5.', backend.inputs[3])
        self.assertEqual(full.archive, compact.archive)
        self.assertNotIn('UNIQUE_CALCULATION_BODY', compact.text)

    def test_guided_does_not_compact_truncated_summary(self):
        backend = GuidedBackend([('some calculation</experiment>', 'stop'), ('unfinished', 'length')])
        engine = Engine(backend, Config(protocol='guided_single', max_context_tokens=16000))
        state = engine.run(engine.start('task'), 'compact')
        self.assertEqual(state.status, 'incomplete_phase_length')
        self.assertEqual(state.archive, {})
        self.assertIn('some calculation', state.text)

    def test_guided_rejects_body_eos_without_boundary(self):
        backend = GuidedBackend([('some calculation', 'eos')])
        engine = Engine(backend, Config(protocol='guided_single', max_context_tokens=16000))
        state = engine.run(engine.start('task'), 'full')
        self.assertEqual(state.status, 'incomplete_phase_eos')
        self.assertEqual(state.archive, {})

    def test_user_reported_eos_creates_one_shared_failure(self):
        backend = GuidedBackend([('Okay, the answer is 5.\n</think>\nfinal answer only: 5', 'eos')])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, tasks, out = root/'config.json', root/'tasks.jsonl', root/'run'
            config.write_text(json.dumps({'backend':'mock','max_context_tokens':16000}))
            tasks.write_text(json.dumps({'id':'smoke','prompt':'Compute 2+3','expected':'5'})+'\n')
            args = argparse.Namespace(config=config, tasks=tasks, output=out, seed=None, limit=None, command='pair')
            with patch.object(cli, 'make_backend', return_value=backend), contextlib.redirect_stdout(io.StringIO()):
                code = cli.execute(args)
            self.assertEqual(code, 2)
            results = json.loads((out/'results.json').read_text())
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['mode'], 'shared')
            self.assertIsNone(results[0]['correct'])
            self.assertFalse((out/'task_00000/full.json').exists())
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'finished_with_errors')
            self.assertEqual(manifest['completed_pairs'], 0)
            self.assertEqual(manifest['failed_shared_prefixes'], 1)
            self.assertNotIn('quality_evidence', manifest)
            summary = json.loads((out/'summary.json').read_text())
            self.assertIsNone(summary['shared']['accuracy'])

    def test_old_unpairable_rows_not_scored_as_arm_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'results.json').write_text(json.dumps([
                {'task_id':'smoke','mode':mode,'pairable':False,'status':'incomplete_event_eos',
                 'correct':False,'generated_tokens':241,'prefill_tokens':194}
                for mode in ['full','compact']]))
            with contextlib.redirect_stdout(io.StringIO()):
                cli.summarize(root)
            summary = json.loads((root/'summary.json').read_text())
            self.assertIsNone(summary['full']['accuracy'])
            self.assertEqual(summary['diagnostics']['unpairable_tasks'], 1)


if __name__ == '__main__':
    unittest.main()
