"""CPU checks of fixed-anchor state targets and explicit missing-state admission."""
import importlib.util
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

from openwam.dataloader.orion_midtrain import droid_states, droid_targets, scatter80


class MidtrainContractTests(unittest.TestCase):
    def test_window_exclusion_does_not_hide_other_errors(self):
        script = Path(__file__).parents[1] / 'scripts/prepare_orion_midtrain.py'
        spec = importlib.util.spec_from_file_location('prepare_midtrain', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._stats_dataset = SimpleNamespace(ids=[17], record=lambda _: dict(source_episode_id='episode', target_ns=[123]))
        with patch.object(module, 'droid_targets', side_effect=ValueError('DROID RGB anchor outside observed state timestamps: episode')):
            arrays, excluded = module._droid_target_batch([0])
        self.assertIsNone(arrays)
        self.assertEqual(excluded[0]['window_id'], 17)
        self.assertEqual(excluded[0]['anchor_ns'], 123)
        with patch.object(module, 'droid_targets', side_effect=ValueError('nonfinite state')):
            with self.assertRaisesRegex(ValueError, 'nonfinite state'):
                module._droid_target_batch([0])

    def test_observed_targets_fixed_anchor_and_missing_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'trajectory.h5'
            t = np.arange(41) * 50
            pose = np.zeros((41, 6))
            pose[:, 0] = t / 1000
            pose[:, 5] = np.pi / 2
            with h5py.File(source, 'w') as f:
                f['observation/robot_state/cartesian_position'] = pose
                f['observation/robot_state/gripper_position'] = t / 2000
                f['observation/timestamp/robot_state/read_start'] = t
                f['observation/timestamp/robot_state/read_end'] = t
            (root / 'result.json').write_text(json.dumps(dict(id='test', trajectory_h5=str(source))))
            record = dict(table_root=str(root), camera_specs=[dict(lance_path='wrist/frames.lance')], target_ns=[1_000_000_000])
            grid, relative, valid, _ = droid_targets(record)
            self.assertEqual(tuple(relative.shape), (34, 20))
            # Anchor at1s, first target0.95s: world-X displacement maps to local+Y.
            torch.testing.assert_close(relative[0, :3], torch.tensor([0., .05, 0.]), atol=1e-6, rtol=0)
            torch.testing.assert_close(relative[1, :3], torch.zeros(3), atol=1e-6, rtol=0)
            self.assertAlmostEqual(float(grid[0, 0, 9]), .5)
            self.assertFalse(valid[:, 10:].any())
            self.assertFalse(valid[-1].any())
            packed, mask = scatter80(relative, valid)
            self.assertFalse(mask[:, 34:44].any())
            self.assertEqual(float(packed[:, 34:44].abs().sum()), 0.)
            droid_states.cache_clear()

    def test_explicit_exclusion_and_recovered_source_are_indexed(self):
        script = Path(__file__).parents[1] / 'scripts/prepare_orion_midtrain.py'
        spec = importlib.util.spec_from_file_location('prepare_midtrain', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'droid-action-audit-summary.json').write_text(json.dumps(dict(full_action_schema_audited=True, windows={'train': 3})))
            audit = [dict(id='good', ok=True, trajectory='/good.h5'), dict(id='recover', ok=False), dict(id='missing', ok=False)]
            recovery = [dict(id='recover', status='recovered', trajectory_h5='/recovered.h5'), dict(id='missing', status='exclude_missing_state')]
            for name, rows in [('droid-action-audit', audit), ('droid-state-recovery', recovery)]:
                (root / (name + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows))
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(episodes=[dict(source='droid', source_episode_id=key, split='train') for key in ['good', 'recover', 'missing']])))
            module.index_droid(manifest, root, root)
            with closing(sqlite3.connect(root / 'droid.sqlite')) as con:
                rows = [json.loads(r[0]) for r in con.execute('select payload from windows order by id')]
            self.assertEqual([r['trajectory_h5'] for r in rows], ['/good.h5', '/recovered.h5'])
            admission = json.loads((root / 'droid-admission.json').read_text())
            self.assertEqual(admission['windows']['train'], 2)
            self.assertEqual(admission['excluded'][0]['windows']['train'], 1)


if __name__ == '__main__':
    unittest.main()
