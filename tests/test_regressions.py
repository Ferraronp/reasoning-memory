import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reasoning_memory import cli
from reasoning_memory.backend import Generation, InferenceOutOfMemory, MockBackend
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
    def test_full_oom_preserves_diagnostics_and_still_runs_compact(self):
        from reasoning_memory.backend import HFBackend
        from types import SimpleNamespace
        class OOMAfterShared(GuidedBackend):
            next_user_turn = HFBackend.next_user_turn

            def generate(self, text, limit, seed, stop_strings=None):
                self.inputs.append(text)
                item = next(self.outputs)
                if isinstance(item, Exception):
                    raise item
                content, reason = item
                return Generation(content, len(text), len(content), finish_reason=reason)

        backend = OOMAfterShared([
            ('Rotate the grid clockwise.</think>', 'stop'),
            ('Rotate 90 degrees clockwise.</think>', 'stop'),
            InferenceOutOfMemory('CUDA out of memory even with offloaded KV cache'),
            ('Apply clockwise rotation.</think>', 'stop'),
            ('[[0,2],[1,0]]', 'eos'),
        ])
        backend.tokenizer = SimpleNamespace(all_special_tokens=['<|im_start|>', '<|im_end|>'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, tasks, out = root/'config.json', root/'tasks.jsonl', root/'run'
            config.write_text(json.dumps({'backend':'mock', 'protocol':'guided_chat_two_stage',
                'stage2_hide_source': True, 'max_context_tokens':16000}))
            tasks.write_text(json.dumps({'id':'rotate','prompt':'Infer rotation',
                'followup':'Transform [[1,0],[2,0]]', 'expected':[[0,2],[1,0]]})+'\n')
            args = argparse.Namespace(config=config, tasks=tasks, output=out, seed=None,
                                      limit=None, task_id=None, command='pair')
            with patch.object(cli, 'make_backend', return_value=backend), contextlib.redirect_stdout(io.StringIO()):
                code = cli.execute(args)
            self.assertEqual(code, 2)
            rows = json.loads((out/'results.json').read_text())
            self.assertEqual([(r['mode'], r['status'], r['correct']) for r in rows],
                             [('full', 'resource_exhausted', None), ('compact', 'completed', True)])
            self.assertEqual(json.loads((out/'manifest.json').read_text())['status'], 'finished_with_errors')
            self.assertEqual(json.loads((out/'manifest.json').read_text())['completed_pairs'], 0)
            self.assertTrue((out/'task_00000/full.json').exists())
            self.assertTrue((out/'task_00000/compact.json').exists())
            self.assertTrue((out/'summary.json').exists())
            self.assertIn('resource_exhausted', (out/'task_00000/events.jsonl').read_text())

    def test_hf_oom_retries_with_offloaded_cache_and_original_seed(self):
        from reasoning_memory.backend import HFBackend
        from types import SimpleNamespace

        class FakeOOM(RuntimeError):
            pass

        class FakeTokens(dict):
            def __init__(self):
                super().__init__(input_ids=SimpleNamespace(shape=[1, 3]))
                self.input_ids = self['input_ids']

            def to(self, device):
                return self

        class FakeTokenizer:
            pad_token_id = 0

            def __call__(self, text, **kwargs):
                return FakeTokens()

            def decode(self, ids, **kwargs):
                return '5' if ids else ''

        class FakeModel:
            generation_config = SimpleNamespace(eos_token_id=2, bos_token_id=1)

            def __init__(self):
                self.cache_modes = []

            def generate(self, **kwargs):
                self.cache_modes.append(getattr(kwargs['generation_config'], 'cache_implementation', None))
                if len(self.cache_modes) == 1:
                    raise FakeOOM('GPU full')
                class FakeOutput:
                    def __getitem__(self, key):
                        return SimpleNamespace(tolist=lambda: [5, 2])
                return FakeOutput()

        model = FakeModel()
        seeds = []
        emptied = []
        backend = HFBackend.__new__(HFBackend)
        backend.model = model
        backend.tokenizer = FakeTokenizer()
        backend.context_limit = 8192
        backend.offloaded_cache_retries = 0
        backend.config = SimpleNamespace(device='cuda', temperature=0, top_p=1, top_k=0,
                                         retry_offloaded_on_oom=True)
        backend.transformers = SimpleNamespace(set_seed=seeds.append,
            GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs))
        backend.torch = SimpleNamespace(OutOfMemoryError=FakeOOM,
            inference_mode=contextlib.nullcontext,
            cuda=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: emptied.append(True)))
        result = backend.generate('text', 50, 42, stop_strings=['</answer>'])
        self.assertEqual(result.finish_reason, 'eos')
        self.assertEqual(model.cache_modes, [None, 'offloaded'])
        self.assertEqual(seeds, [42, 42])
        self.assertEqual(len(emptied), 1)
        self.assertEqual(backend.offloaded_cache_retries, 1)

    def test_task_id_runs_only_requested_task_and_records_source_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, tasks, out = root/'config.json', root/'tasks.jsonl', root/'run'
            config.write_text(json.dumps({'backend':'mock', 'protocol':'guided_single',
                                          'max_context_tokens':16000}))
            tasks.write_text(''.join(json.dumps({'id':name, 'prompt':'Compute 2+3',
                                                  'expected':'5'})+'\n'
                                     for name in ('skip_me', 'run_me')))
            args = argparse.Namespace(config=config, tasks=tasks, output=out, seed=None,
                                      limit=None, task_id='run_me', command='pair')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.execute(args), 0)
            self.assertEqual([task['id'] for task in json.loads((out/'tasks.json').read_text())],
                             ['run_me'])
            self.assertEqual({row['task_id'] for row in json.loads((out/'results.json').read_text())},
                             {'run_me'})
            self.assertEqual(json.loads((out/'manifest.json').read_text())['completed_pairs'], 1)

    def test_chat_stage_preserves_full_thinking_and_reveals_followup_as_user(self):
        from reasoning_memory.backend import HFBackend
        from types import SimpleNamespace
        class QwenScripted(GuidedBackend):
            next_user_turn = HFBackend.next_user_turn
        backend = QwenScripted([
            ('PRIVATE_BODY r=5</think>', 'stop'), ('r=5</think>', 'stop'),
            ('SECOND_BODY 17</think>', 'stop'), ('17', 'eos'),
            ('SECOND_BODY 17</think>', 'stop'), ('17', 'eos'),
        ])
        backend.tokenizer = SimpleNamespace(all_special_tokens=['<|im_start|>', '<|im_end|>'])
        engine = Engine(backend, Config(protocol='guided_chat_two_stage', max_context_tokens=16000))
        shared = engine.run(engine.start('Find r', 'NEW_TASK: compute 4*r-3'), 'full', stop_after_first=True)
        self.assertTrue(all('NEW_TASK' not in t for t in backend.inputs))
        full = engine.run(engine.fork(shared, 'full'), 'full')
        compact = engine.run(engine.fork(shared, 'compact'), 'compact')
        for i in (2, 4):
            self.assertIn('</think>\nConclusion: r=5\n<|im_end|>\n<|im_start|>user\nNEW_TASK', backend.inputs[i])
            self.assertTrue(backend.inputs[i].endswith('<|im_start|>assistant\n<think>\n'))
        self.assertIn('PRIVATE_BODY', backend.inputs[2])
        self.assertNotIn('PRIVATE_BODY', backend.inputs[4])
        self.assertIn('SECOND_BODY', backend.inputs[3])
        self.assertIn('SECOND_BODY', backend.inputs[5])
        self.assertNotIn('SECOND_BODY', compact.text)
        self.assertEqual([e['phase'] for e in compact.events], ['experiment', 'summary', 'experiment', 'answer'])
        self.assertIn('PRIVATE_BODY', full.text)
        self.assertEqual(full.answer, '17')
        self.assertEqual(compact.answer, '17')
        self.assertEqual(len(compact.archive), 2)
        self.assertEqual(compact.archive['e2']['summary'], '17')
        self.assertNotIn('NEW_TASK', shared.text)

    def test_length_chunk_continues_same_experiment_before_shared_fork(self):
        backend = GuidedBackend([
            ('The grid should ro', 'length'), ('tate clockwise.</think>', 'stop'),
            ('Rotate 90 degrees clockwise.</think>', 'stop'),
            ('Apply the rule.</think>', 'stop'), ('[[1,0]]', 'eos'),
            ('Apply the summary.</think>', 'stop'), ('[[1,0]]', 'eos'),
        ])
        cfg = Config(protocol='guided_chat_two_stage', max_context_tokens=16000,
                     max_total_new_tokens=5120, continue_experiment_on_length=True)
        engine = Engine(backend, cfg)
        shared = engine.start('Infer rotation.', 'Transform test grid.')
        engine.step(shared, 'full')
        self.assertEqual(shared.status, 'running')
        self.assertEqual(shared.archive, {})
        self.assertEqual(shared.phase, 'experiment')
        self.assertTrue(shared.events[0]['continued'])
        self.assertTrue(backend.inputs[1:] == [])
        engine.run(shared, 'full', stop_after_first=True)
        self.assertIn('The grid should ro', backend.inputs[1])
        self.assertEqual(shared.archive['e1']['body'], 'The grid should rotate clockwise.')
        self.assertEqual(shared.archive['e1']['summary'], 'Conclusion: Rotate 90 degrees clockwise.')
        full = engine.run(engine.fork(shared, 'full'), 'full')
        compact = engine.run(engine.fork(shared, 'compact'), 'compact')
        self.assertEqual(full.status, 'completed')
        self.assertEqual(compact.status, 'completed')
        self.assertIn('The grid should rotate clockwise.', backend.inputs[3])
        self.assertNotIn('The grid should rotate clockwise.', backend.inputs[5])
        self.assertEqual(full.answer, compact.answer)

    def test_continued_experiment_exhausts_total_budget_without_compacting(self):
        backend = GuidedBackend([('unfinished reasoning', 'length')])
        cfg = Config(protocol='guided_single', max_context_tokens=16000,
                     max_total_new_tokens=len('unfinished reasoning'),
                     continue_experiment_on_length=True)
        engine = Engine(backend, cfg)
        state = engine.run(engine.start('Infer rule'), 'compact')
        self.assertEqual(state.status, 'budget_exhausted')
        self.assertEqual(state.archive, {})
        self.assertIn('unfinished reasoning', state.text)
        self.assertIsNone(state.answer)

    def test_hidden_source_for_grid_transfer_removes_examples_from_both_arms(self):
        from reasoning_memory.backend import HFBackend
        from types import SimpleNamespace
        class QwenScripted(GuidedBackend):
            next_user_turn = HFBackend.next_user_turn
        backend = QwenScripted([
            ('The examples include SECRET_EXAMPLE; rotate clockwise.</think>', 'stop'),
            ('Rotate the grid 90 degrees clockwise.</think>', 'stop'),
            ('The new input becomes a two-row grid.</think>', 'stop'),
            ('[[0,2],[1,0]]', 'eos'),
            ('Use the clockwise rule saved in the conclusion.</think>', 'stop'),
            ('[[0, 2], [1, 0]]', 'eos'),
        ])
        backend.tokenizer = SimpleNamespace(all_special_tokens=['<|im_start|>', '<|im_end|>'])
        cfg = Config(protocol='guided_chat_two_stage', stage2_hide_source=True,
                     max_context_tokens=16000)
        engine = Engine(backend, cfg)
        shared = engine.run(engine.start('SECRET_EXAMPLE: infer the rule',
                                         'Apply to the new grid.'), 'full', stop_after_first=True)
        full = engine.run(engine.fork(shared, 'full'), 'full')
        compact = engine.run(engine.fork(shared, 'compact'), 'compact')
        for index in (2, 4):
            self.assertNotIn('SECRET_EXAMPLE: infer the rule', backend.inputs[index])
            self.assertIn('Apply to the new grid.', backend.inputs[index])
            self.assertIn('Conclusion: Rotate the grid 90 degrees clockwise.', backend.inputs[index])
        self.assertIn('The examples include SECRET_EXAMPLE', backend.inputs[2])
        self.assertNotIn('The examples include SECRET_EXAMPLE', backend.inputs[4])
        self.assertIn('SECRET_EXAMPLE: infer the rule', shared.text)
        self.assertTrue(engine.result(full, [[0, 2], [1, 0]])['correct'])
        self.assertTrue(engine.result(compact, [[0, 2], [1, 0]])['correct'])
        self.assertFalse(engine.result(compact, [[1, 0], [0, 2]])['correct'])
        self.assertRaises(ValueError, Config, protocol='guided_single', stage2_hide_source=True)

    def test_chat_incomplete_second_answer_does_not_archive_or_compact_body(self):
        from reasoning_memory.backend import HFBackend
        from types import SimpleNamespace
        class QwenScripted(GuidedBackend):
            next_user_turn = HFBackend.next_user_turn
        backend = QwenScripted([
            ('r=5</think>', 'stop'), ('5</think>', 'stop'),
            ('PRIVATE_STAGE_TWO</think>', 'stop'), ('17', 'length'),
        ])
        backend.tokenizer = SimpleNamespace(all_special_tokens=['<|im_start|>', '<|im_end|>'])
        engine = Engine(backend, Config(protocol='guided_chat_two_stage', max_context_tokens=16000))
        result = engine.run(engine.start('Compute r', 'Use r'), 'compact')
        self.assertEqual(result.status, 'incomplete_phase_length')
        self.assertEqual(set(result.archive), {'e1'})
        self.assertIn('PRIVATE_STAGE_TWO', result.text)
        self.assertIsNone(result.answer)

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

    def test_protocol_failure_is_not_scored_as_wrong_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'results.json').write_text(json.dumps([
                {'task_id':'a','mode':'full','pairable':True,'status':'protocol_error',
                 'correct':False,'generated_tokens':10,'prefill_tokens':20},
                {'task_id':'b','mode':'full','pairable':True,'status':'completed',
                 'correct':True,'generated_tokens':10,'prefill_tokens':20},
                {'task_id':'c','mode':'compact','pairable':True,'status':'completed',
                 'correct':False,'generated_tokens':10,'prefill_tokens':20},
            ]))
            with contextlib.redirect_stdout(io.StringIO()):
                cli.summarize(root)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['full']['failed'], 1)
            self.assertEqual(summary['full']['scored'], 1)
            self.assertEqual(summary['full']['accuracy'], 1.0)
            self.assertEqual(summary['compact']['accuracy'], 0.0)
            engine = Engine(MockBackend(), Config(max_context_tokens=16000))
            state = engine.start('task')
            state.status = 'protocol_error'
            self.assertIsNone(engine.result(state, '17')['correct'])


if __name__ == '__main__':
    unittest.main()
