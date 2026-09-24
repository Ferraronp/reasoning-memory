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
from reasoning_memory.controls import build_tasks, classify_answer
from reasoning_memory.engine import Engine
from reasoning_memory.report import report


class Scripted(MockBackend):
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []
        self.seeds = []

    def generate(self, text, limit, seed, stop_strings=None):
        self.prompts.append(text)
        self.seeds.append(seed)
        content, reason = next(self.outputs)
        return Generation(content, len(text), len(content), finish_reason=reason)


class ControlTests(unittest.TestCase):
    def test_frozen_dataset_answers_and_matched_conditions(self):
        tasks = build_tasks()
        frozen = [json.loads(line) for line in
                  (Path(__file__).parents[1]/'data/rotation_controls.jsonl').read_text().splitlines()]
        self.assertEqual(tasks, frozen)
        expected = [
            [[9, 0, 0], [0, 0, 7], [0, 0, 0], [0, 8, 0]],
            [[4, 1], [5, 2], [6, 3]],
            [[0, 3, 0, 1], [4, 0, 2, 0]],
        ]
        for i, correct in enumerate(expected):
            named, formula = tasks[2*i:2*i+2]
            self.assertEqual(named['input_grid'], formula['input_grid'])
            self.assertEqual(named['expected'], correct)
            self.assertEqual(formula['expected'], correct)
            self.assertEqual(named['condition'], 'named_rule')
            self.assertEqual(formula['condition'], 'coordinate_rule')
            self.assertNotIn('output[c]', named['prompt'])
            self.assertIn('output[c][m-1-r]', formula['prompt'])

    def test_direct_has_no_summary_archive_or_compaction(self):
        backend = Scripted([('Reasoning body.</think>', 'stop'), ('[[4,1],[5,2],[6,3]]', 'eos')])
        engine = Engine(backend, Config(protocol='guided_direct', max_context_tokens=16000))
        state = engine.run(engine.start('Apply the known rule'), 'full')
        self.assertEqual(state.status, 'completed')
        self.assertEqual([e['phase'] for e in state.events], ['experiment', 'answer'])
        self.assertEqual(state.archive, {})
        self.assertIn('Reasoning body.', backend.prompts[1])
        self.assertNotIn('Conclusion:', backend.prompts[1])
        self.assertTrue(engine.result(state, [[4,1],[5,2],[6,3]])['correct'])

    def test_controls_report_wrong_direction_and_keep_truncations_unscored(self):
        tasks = build_tasks()[:3]
        backend = Scripted([
            ('Reasoning.</think>', 'stop'), ('[[0,8,0],[0,0,0],[7,0,0],[0,0,9]]', 'eos'),
            ('Reasoning.</think>', 'stop'), (json.dumps(tasks[1]['expected']), 'eos'),
            ('Unfinished reasoning', 'length'),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, data, out = root/'config.json', root/'tasks.jsonl', root/'run'
            cfg.write_text(json.dumps({'backend':'mock','protocol':'guided_direct','max_context_tokens':16000}))
            data.write_text(''.join(json.dumps(t)+'\n' for t in tasks))
            args = argparse.Namespace(command='control', config=cfg, tasks=data,
                                      output=out, seed=None, limit=None, task_id=None)
            with patch.object(cli, 'make_backend', return_value=backend), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.execute(args), 2)
            rows = json.loads((out/'results.json').read_text())
            self.assertEqual([r['answer_kind'] for r in rows], ['counterclockwise', 'correct', 'unfinished'])
            self.assertEqual([r['correct'] for r in rows], [False, True, None])
            self.assertEqual({r['mode'] for r in rows}, {'control'})
            self.assertTrue(all(r['pairable'] is None for r in rows))
            self.assertEqual(backend.seeds, [42,43,42,43,42])
            summary = json.loads((out/'summary.json').read_text())['control_conditions']
            self.assertEqual(summary['named_rule']['scored'], 1)
            self.assertEqual(summary['named_rule']['tasks'], 2)
            self.assertEqual(summary['coordinate_rule']['correct'], 1)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['completed_pairs'], 0)
            self.assertEqual(manifest['evaluation_kind'], 'rule_execution_control')
            self.assertFalse((out/'task_00000/shared_prefix.json').exists())
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                report(out)
            self.assertIn('counterclockwise', stream.getvalue())
            self.assertIn('coordinate_rule: 1/1 correct', stream.getvalue())

    def test_control_protocol_cannot_silently_be_used_for_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            path.write_text(json.dumps({'protocol':'guided_direct'}))
            with patch.object(cli, 'make_backend') as factory:
                with self.assertRaisesRegex(ValueError, 'do not create'):
                    cli.execute(argparse.Namespace(config=path, command='pair'))
                factory.assert_not_called()

    def test_answer_classifier_rejects_boolean_and_ragged_grids(self):
        task = build_tasks()[0]
        self.assertEqual(classify_answer('[[true]]', task, 'completed'), 'invalid_grid')
        self.assertEqual(classify_answer('[[1],[2,3]]', task, 'completed'), 'invalid_grid')
        self.assertEqual(classify_answer('```json\n[]\n```', task, 'completed'), 'invalid_json')
        self.assertEqual(classify_answer(None, task, 'resource_exhausted'), 'unfinished')

    def test_report_handles_interrupted_run_without_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'manifest.json').write_text(json.dumps({'status':'failed','error':'CUDA OOM'}))
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                report(root)
            self.assertIn('CUDA OOM', stream.getvalue())
            self.assertIn('No completed result records', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
