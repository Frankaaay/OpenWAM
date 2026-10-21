"""OpenWAM's shared-plan bridge must not override the environment contract."""
import unittest
from unittest.mock import patch
import numpy as np
from benchmarks.robotwin import openwam2robotwin_interface as iface


class SharedPlanPolicyTests(unittest.TestCase):
    def test_bridge_retains_endpoint_and_eef_layout(self):
        with patch.object(iface, 'get_model') as factory:
            model = iface.SharedPlanPolicy(host='server', port=8848, stereo=False)
        factory.assert_called_once_with(dict(host='server', port=8848, action_type='ee', state_dim=20))
        model.reset()
        model.client.reset.assert_called_once_with(task_description='')
        self.assertTrue(model.should_request_observation())

    def test_bridge_requires_explicit_boolean_encoder_mode(self):
        with self.assertRaises(ValueError):
            iface.SharedPlanPolicy(host='server', port=8848, stereo='false')

    def test_bridge_uses_shared_environment_and_exact_observation(self):
        with patch.object(iface, 'get_model'):
            model = iface.SharedPlanPolicy(host='server', port=8848, stereo=True)
        env, observation = object(), {'raw': object()}
        with patch.object(iface, 'eval') as evaluate:
            model.step(env, observation)
        evaluate.assert_called_once_with(env, model.client, observation, shared_plan=True, stereo=True)

    def test_shared_eval_does_not_apply_model_step_limit(self):
        class Model:
            _action_type = 'ee'
            _send_state = False
            def step(self, example): return np.array([0, 0, 0, 1, 0, 0, 0, 0] * 2)
        class Env:
            step_lim = 400
            def get_instruction(self): return 'frozen'
            def take_action(self, action, action_type): self.action = action
        env = Env()
        observation = {'observation': {'head_camera': {'rgb': np.zeros((2, 2, 3), dtype=np.uint8)}}}
        with patch.object(iface, '_apply_step_lim_override', side_effect=AssertionError('override')):
            iface.eval(env, Model(), observation, shared_plan=True, stereo=False)
        self.assertEqual(env.step_lim, 400)
        self.assertEqual(len(env.action), 16)


if __name__ == '__main__': unittest.main()
