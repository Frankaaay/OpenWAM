"""CPU checks of fixed-anchor state targets and explicit missing-state admission."""
import importlib.util
from contextlib import closing
import json
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

from openwam.dataloader.orion_midtrain import droid_states, droid_targets, scatter80, SparseABCOrionDataset


class MidtrainContractTests(unittest.TestCase):
    def test_sparse_abc_norm_matches_selected_anchor_phases(self):
        script = Path(__file__).parents[1] / 'scripts/prepare_orion_midtrain.py'
        spec = importlib.util.spec_from_file_location('prepare_midtrain', script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        states = torch.zeros(31, 2, 10)
        states[:, :, 3:9] = torch.tensor([1., 0., 0., 0., 1., 0.])
        states[:, 0, 0] = torch.arange(31).square() / 1000
        rows = {i: {'observation.state': states[i].flatten().tolist()} for i in range(31)}
        mask = torch.zeros(20, dtype=torch.bool); mask[:10] = True
        meta = dict(state_element_mask=mask, field_mask={}, field_mapping={},
                    camera_mask=torch.zeros(2, dtype=torch.bool), source_fps=30,
                    table_backend=SimpleNamespace(read_rows=lambda *args: rows))
        dataset = SimpleNamespace(_episodes=[dict(root_index=0, length=31, action_target_length=19, rgb_target_length=10)],
            _root_meta=[meta], camera_keys=[], _episode_window_ends=[2], source_phase_offsets=(0,1,2),
            timeline_layout=module.abc_orion.__globals__['LAYOUT'], action_steps_per_rgb_frame=2,
            action_rate_hz=20, rgb_rate_hz=10, _reshape_state_window=SparseABCOrionDataset._reshape_state_window,
            _state_interpolation_indices=SparseABCOrionDataset._state_interpolation_indices)
        result = module.compute_zscore_statistics(dataset, progress_every=0)
        absolute, relative = [], []
        for anchor, phase in [(0,0), (12,1)]:
            def x(t):
                position=t*1.5+phase
                lo=int(np.floor(position)); hi=int(np.ceil(position))
                return ((1-(position-lo))*lo**2+(position-lo)*hi**2)/1000
            for target in range(anchor-1,anchor+33):
                if 0 <= target < 19:
                    absolute.append(x(target)); relative.append(x(target)-x(anchor))
        self.assertEqual(result['statistics']['anchors'], 2)
        self.assertAlmostEqual(result['state_tcp_mean'][0][0], float(np.mean(absolute)), places=6)
        self.assertAlmostEqual(result['action_tcp_mean'][0][0], float(np.mean(relative)), places=6)

    def test_sparse_abc_anchors_keep_physical_episode_length(self):
        dataset = object.__new__(SparseABCOrionDataset)
        dataset._episodes = [dict(rgb_target_length=20), dict(rgb_target_length=9)]
        dataset._episode_window_ends = [4, 6]
        dataset._length = 6
        self.assertEqual([dataset._locate_window(i)[1] for i in range(6)], [0, 6, 12, 18, 0, 6])
        self.assertEqual(dataset._locate_window(3)[0]['rgb_target_length'], 20)

    def test_fixed_warmup_matches_optimizer_updates_at_world16(self):
        script = Path(__file__).parents[1] / 'openwam/train/utils/training_utils.py'
        spec = importlib.util.spec_from_file_location('training_utils', script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        build_cosine_scheduler = module.build_cosine_scheduler
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        cfg = SimpleNamespace(training=SimpleNamespace(learning_rate=1e-4, warmup_steps=500,
                              warmup_start_factor=.1, lr_min_ratio=.01))
        scheduler = build_cosine_scheduler(optimizer, total_opt_steps=1000, cfg=cfg, num_processes=16)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-5)
        for _ in range(500 * 16):
            optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-4)
        for _ in range(500 * 16):
            optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-6)

    def test_droid_reindex_uses_real_06s_windows_and_reports_gaps(self):
        script = Path(__file__).parents[1] / 'scripts/prepare_orion_midtrain.py'
        spec = importlib.util.spec_from_file_location('prepare_midtrain', script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(target_ns=i*100000000, capture_ns=i*100000000, source_frame=i, row=i) for i in range(30)]
            data = json.dumps(rows).encode(); (root/'sampling.json').write_bytes(data)
            camera = dict(sampling_path='sampling.json', sampling_sha256=hashlib.sha256(data).hexdigest())
            record = dict(table_root=str(root), camera_specs=[camera]*3, target_ns=[0],
                          time_tolerance_ns=1, cross_camera_tolerance_ns=1,
                          episode_id='test@0', source_episode_id='test', split='train')
            windows, rejected = module.droid_episode_windows(record)
            self.assertEqual([json.loads(x)['target_ns'][0] for x in windows], [0,600000000,1200000000])
            self.assertEqual(rejected, [])
            rows.pop(10); data=json.dumps(rows).encode(); (root/'sampling.json').write_bytes(data)
            camera['sampling_sha256']=hashlib.sha256(data).hexdigest()
            windows, rejected=module.droid_episode_windows(record)
            self.assertEqual(len(windows), 1)
            self.assertEqual([r['reason'] for r in rejected], ['missing_camera_target']*2)

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
            class SerialPool:
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def map(self, function, rows, **kwargs):
                    return (([json.dumps(row)], []) for row in rows)
            with patch.object(module, 'ProcessPoolExecutor', return_value=SerialPool()):
                module.index_droid(manifest, root, root)
            with closing(sqlite3.connect(root / 'droid.sqlite')) as con:
                rows = [json.loads(r[0]) for r in con.execute('select payload from windows order by id')]
            self.assertEqual([r['trajectory_h5'] for r in rows], ['/good.h5', '/recovered.h5'])
            admission = json.loads((root / 'droid-admission.json').read_text())
            self.assertEqual(admission['windows']['train'], 2)
            self.assertEqual(admission['excluded'][0]['windows']['train'], 1)


if __name__ == '__main__':
    unittest.main()
