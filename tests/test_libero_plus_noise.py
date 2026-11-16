"""Noise replay is paired and must not consume extra benchmark randomness."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


class NoiseReplayTest(unittest.TestCase):
    def test_pairs_replay_and_restore_rng(self):
        class FakeFresh:
            def __getattr__(self, name):
                return getattr(self._wrapped, name)

            def _images(self, obs):
                return {'agentview_image': np.full((2, 2, 3), 10, np.uint8),
                        'agentview_stereo_right_image': np.full((2, 2, 3), 30, np.uint8)}

        stub = types.ModuleType('XPolicyLab.utils.render_libero_stereo')
        stub.FreshStereoEvalEnv = FakeFresh
        path = Path(__file__).resolve().parents[1] / 'benchmarks/libero-plus/stereo_env.py'
        spec = importlib.util.spec_from_file_location('plus_noise_test', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'XPolicyLab.utils.render_libero_stereo': stub}):
            spec.loader.exec_module(module)
        def noise(image, severity):
            return np.asarray(image) + np.random.randint(1, 5, size=np.asarray(image).shape, dtype=np.uint8)
        raw = types.SimpleNamespace(step=lambda action: ({'agentview_image': np.zeros((2, 2, 3), np.uint8)}, 0, False, {}))
        wrapped = types.SimpleNamespace(env=raw, noise=3, get_sim_state=lambda: np.zeros(4))
        def official(action):
            obs, *rest = raw.step(action)
            obs['agentview_image'] = noise(obs['agentview_image'], 3)
            return obs, *rest
        wrapped.step = official
        env = module.PlusStereoEvalEnv()
        env._wrapped = wrapped
        env.stereo = True
        np.random.seed(42)
        official(None)
        expected_next = np.random.rand()
        np.random.seed(42)
        with patch.object(module, 'noise_image', noise):
            obs, *_ = env.step(None)
        np.testing.assert_array_equal(obs['agentview_stereo_right_image'] - obs['agentview_image'], np.full((2, 2, 3), 20))
        self.assertEqual(np.random.rand(), expected_next)


if __name__ == '__main__':
    unittest.main()
