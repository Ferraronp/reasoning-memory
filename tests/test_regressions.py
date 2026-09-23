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
    def test_chat_stage_preserves_full_thinking_and_reveals_followup_as_user(self):
        from reasoning_memory.backend import HFBackend
        from types import SimpleNamespace
        class QwenScripted(GuidedBackend):
            next_user_turn = HFBackend.next_user_turn
        backend = QwenScripted([
            ('PRIVATE_BODY r=5</think>', 'stop'), ('r=5</think>', 'stop'),
            ('SECOND_BODY 17</think>', 'stop'), ('17</think>', 'stop'), ('17', 'eos'),
            ('SECOND_BODY 17</think>', 'stop'), ('17</think>', 'stop'), ('17', 'eos'),
        ])
        backend.tokenizer = SimpleNamespace(all_special_tokens=['<|im_start|>', '<|im_end|>'])
        engine = Engine(backend, Config(protocol='guided_chat_two_stage', max_context_tokens=16000))
        shared = engine.run(engine.start('Find r', 'NEW_TASK: compute 4*r-3'), 'full', stop_after_first=True)
        self.assertTrue(all('NEW_TASK' not in t for t in backend.inputs))
        full = engine.run(engine.fork(shared, 'full'), 'full')
        compact = engine.run(engine.fork(shared, 'compact'), 'compact')
        for i in (2, 5):
            self.assertIn('</think>\nConclusion: r=5\n<|im_end|>\n<|im_start|>user\nNEW_TASK', backend.inputs[i])
            self.assertTrue(backend.inputs[i].endswith('<|im_start|>assistant\n<think>\n'))
        self.assertIn('PRIVATE_BODY', backend.inputs[2])
        self.assertNotIn('PRIVATE_BODY', backend.inputs[5])
        self.assertNotIn('SECOND_BODY', backend.inputs[7])
        self.assertIn('PRIVATE_BODY', full.text)
        self.assertEqual(full.answer, '17')
        self.assertEqual(compact.answer, '17')
        self.assertEqual(len(compact.archive), 2)
        self.assertNotIn('NEW_TASK', shared.text)

    def test_two_stage_forks_before_followup_and_compacts_each_experiment(self):
        backend = GuidedBackend([
            ('FIRST_PRIVATE_BODY r=5</think>', 'stop'), ('r=5</think>', 'stop'),
            ('SECOND_FULL_BODY 4*5-3=17</think>', 'stop'), ('result=17</think>', 'stop'), ('17', 'eos'),
            ('SECOND_COMPACT_BODY 4*5-3=17</think>', 'stop'), ('result=17</think>', 'stop'), ('17', 'eos'),
        ])
        engine = Engine(backend, Config(protocol='guided_two_stage', max_context_tokens=16000))
        shared = engine.run(engine.start('Find r', 'FOLLOWUP_ONLY: compute 4*r-3'), 'full', stop_after_first=True)
        self.assertEqual(shared.phase, 'next_stage')
        self.assertEqual(len(shared.archive), 1)
        self.assertTrue(all('FOLLOWUP_ONLY' not in t for t in backend.inputs))
        full = engine.run(engine.fork(shared, 'full'), 'full')
        compact = engine.run(engine.fork(shared, 'compact'), 'compact')
        self.assertIn('FIRST_PRIVATE_BODY', backend.inputs[2])
        self.assertNotIn('FIRST_PRIVATE_BODY', backend.inputs[5])
        for i in (2, 5):
            self.assertIn('Conclusion: r=5', backend.inputs[i])
            self.assertIn('FOLLOWUP_ONLY', backend.inputs[i])
        self.assertEqual(full.events[2]['seed'], compact.events[2]['seed'])
        self.assertIn('SECOND_FULL_BODY', backend.inputs[4])
        self.assertNotIn('SECOND_COMPACT_BODY', backend.inputs[7])
        self.assertIn('Conclusion: result=17', backend.inputs[7])
        for state in (full, compact):
            self.assertEqual(state.status, 'completed')
            self.assertEqual(state.answer, '17')
            self.assertEqual(set(state.archive), {'e1', 'e2'})
            self.assertEqual([e['stage'] for e in state.events], [1, 1, 2, 2, 2])
        self.assertEqual(shared.phase, 'next_stage')
        self.assertEqual(len(shared.archive), 1)

    def test_two_stage_incomplete_second_summary_retains_second_body(self):
        backend = GuidedBackend([
            ('FIRST_BODY</think>', 'stop'), ('r=5</think>', 'stop'),
            ('SECOND_BODY</think>', 'stop'), ('unfinished', 'length'),
        ])
        engine = Engine(backend, Config(protocol='guided_two_stage', max_context_tokens=16000))
        state = engine.run(engine.start('Find r', 'Compute 4*r-3'), 'compact')
        self.assertEqual(state.status, 'incomplete_phase_length')
        self.assertEqual(set(state.archive), {'e1'})
        self.assertNotIn('FIRST_BODY', state.text)
        self.assertIn('SECOND_BODY', state.text)
        self.assertIsNone(state.answer)

    def test_followup_cannot_be_missing_or_silently_ignored(self):
        for protocol, followup in [('guided_two_stage', None), ('guided_two_stage', ''),
                                   ('guided_single', 'next'), ('autonomous', 'next')]:
            with self.subTest(protocol=protocol, followup=followup), self.assertRaises(ValueError):
                Engine(MockBackend(), Config(protocol=protocol)).start('task', followup)

    def test_guided_first_input_is_plain_reasoning_not_an_empty_xml_element(self):
        engine = Engine(MockBackend(), Config(protocol='guided_single', max_context_tokens=16000))
        state = engine.start('Compute 2 + 3')
        self.assertTrue(state.text.endswith('<think>\n'))
        self.assertNotIn('<experiment', state.text)
        self.assertNotIn('<summary', state.text)
        engine.step(state, 'full')
        self.assertEqual(state.phase, 'summary')
        self.assertIn('2 plus 3 equals 5.', state.text)
        self.assertTrue(state.text.endswith('Conclusion: '))
        self.assertNotIn('<experiment', state.text)

    def test_empty_experiment_still_fails_without_inventing_content(self):
        backend = GuidedBackend([('</experiment>', 'stop')])
        engine = Engine(backend, Config(protocol='guided_single', max_context_tokens=16000))
        state = engine.run(engine.start('Compute 2 + 3'), 'full')
        self.assertEqual(state.status, 'protocol_error')
        self.assertEqual(state.archive, {})

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
