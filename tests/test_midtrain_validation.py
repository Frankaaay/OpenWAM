"""CPU tests of the actual validation method, isolated from GPU model imports."""
import ast
import json
import logging
import math
from pathlib import Path
import random
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


def validation_method():
    source = Path(__file__).parents[1] / 'openwam/train/openwam_trainer.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OpenWAMTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'validate')
    namespace = dict(torch=torch, np=np, random=random, math=math, Path=Path,
                     json=json, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['validate']


class Leaf:
    action_dim = 80

    def __init__(self, root, episode, count, value):
        self._dataset_dir = root
        self._eps_df = {'episode_index': [episode]}
        self._cum_n_starts = np.array([0, count])
        self.value = value

    def __getitem__(self, index):
        return self.value


def mixture(*datasets):
    return SimpleNamespace(names=['droid', 'abc'], datasets=datasets, action_dim=80)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.validate = validation_method()
        self.train = mixture(Leaf('/droid', 0, 2, 1), Leaf('/abc', 0, 5, 3))
        self.val = mixture(Leaf('/droid', 1, 2, 1), Leaf('/abc', 1, 5, 3))
        self.model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout())
        self.model.train()
        self.model[0].eval()
        self.trainer = SimpleNamespace(
            cfg=SimpleNamespace(training=SimpleNamespace(validation=SimpleNamespace(seed=42, dataloader={}))),
            dataset=self.train, architecture=self.model,
            accelerator=SimpleNamespace(num_processes=1, process_index=0, device='cpu',
                                        is_main_process=True, reduce=lambda x, reduction: x),
            compute_loss=lambda samples: {k: torch.tensor(float(samples[0])) for k in ('total', 'action', 'video')},
        )
        self.rows = []
        self.run = SimpleNamespace(settings=SimpleNamespace(mode='offline'),
                                   log=lambda metrics, step: self.rows.append(dict(metrics)))
        self.registry = ModuleType('openwam.dataloader.registry')
        self.registry.build_dataset = lambda cfg, split: self.val

    def call(self, directory):
        with patch.dict('sys.modules', {'openwam.dataloader.registry': self.registry}):
            self.validate(self.trainer, 10, 10, self.run, directory)

    def test_offline_per_source_metrics_and_repeatability(self):
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            self.call(directory)
            self.call(directory)
            saved = [json.loads(line) for line in (Path(directory) / 'validation.jsonl').read_text().splitlines()]
        self.assertEqual(self.rows[0], self.rows[1])
        self.assertEqual(saved[0]['val/droid/windows'], 2)
        self.assertEqual(saved[0]['val/abc/windows'], 4)
        self.assertEqual(saved[0]['val/abc/action_loss'], 3)
        self.assertAlmostEqual(saved[0]['val/loss'], 14 / 6)
        self.assertEqual(random.getstate(), python_rng)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertTrue(self.model.training)
        self.assertFalse(self.model[0].training)

    def test_per_source_window_limit(self):
        self.trainer.cfg.training.validation.max_windows_per_source = 1
        with tempfile.TemporaryDirectory() as directory:
            self.call(directory)
        self.assertEqual(self.rows[0]['val/droid/windows'], 1)
        self.assertEqual(self.rows[0]['val/abc/windows'], 1)

    def test_overlap_in_second_source_is_rejected(self):
        self.val.datasets[1]._eps_df['episode_index'] = [0]
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'overlap'):
            self.call(directory)
        self.assertEqual(self.rows, [])
        self.assertTrue(self.model.training)

    def test_nonfinite_loss_is_rejected(self):
        self.trainer.compute_loss = lambda samples: {k: torch.tensor(float('nan')) for k in ('total', 'action', 'video')}
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            self.call(directory)
        self.assertEqual(self.rows, [])
        self.assertTrue(self.model.training)

    def test_legacy_episode_file_layout(self):
        train = Leaf('/robotwin', 0, 1, 1)
        val = Leaf('/robotwin', 1, 3, 2)
        train._episode_files = ['/robotwin/train.h5']
        val._episode_files = ['/robotwin/val.h5']
        val._window_index = [(0, 0), (0, 1), (0, 2)]
        self.trainer.dataset = SimpleNamespace(_sub_datasets=[train])
        self.val = SimpleNamespace(_sub_datasets=[val])
        with tempfile.TemporaryDirectory() as directory:
            self.call(directory)
        self.assertEqual(self.rows[0]['val/windows'], 3)
        self.assertEqual(self.rows[0]['val/loss'], 2)


if __name__ == '__main__':
    unittest.main()
