"""OpenWAM's admitted stereo cameras with LIBERO-Plus sensor-noise replay."""
import numpy as np
from PIL import Image
from XPolicyLab.utils.render_libero_stereo import FreshStereoEvalEnv


def noise_image(image, noise):
    from libero.libero.envs import env_wrapper as wrapper
    functions = [wrapper.motion_blur, wrapper.gaussian_blur, wrapper.zoom_blur,
                 wrapper.fog, wrapper.glass_blur]
    return np.asarray(functions[(noise - 1) // 10](
        Image.fromarray(image), severity=(noise - 1) % 10 + 1), dtype=np.uint8)


class PlusStereoEvalEnv(FreshStereoEvalEnv):
    def step(self, action):
        raw_step = self._wrapped.env.step
        captured = {}

        def capture(action):
            value = raw_step(action)
            captured['rng'] = np.random.get_state()
            captured['left'] = value[0]['agentview_image'].copy()
            return value

        self._wrapped.env.step = capture
        try:
            obs, reward, done, info = self._wrapped.step(action)
        finally:
            self._wrapped.env.step = raw_step
        official_left = obs['agentview_image'].copy()
        before = np.asarray(self.get_sim_state()).copy()
        obs = self._images(obs)
        np.testing.assert_array_equal(before, self.get_sim_state())
        noise = self._wrapped.noise
        if noise:
            current = np.random.get_state()
            try:
                np.random.set_state(captured['rng'])
                np.testing.assert_array_equal(noise_image(captured['left'], noise), official_left)
                # Apply the same official corruption to each admitted fresh eye.
                for key in ['agentview_image'] + (['agentview_stereo_right_image'] if self.stereo else []):
                    np.random.set_state(captured['rng'])
                    obs[key] = noise_image(obs[key], noise)
            finally:
                np.random.set_state(current)
        if self.stereo:
            assert not np.array_equal(obs['agentview_image'], obs['agentview_stereo_right_image'])
        if not getattr(self, '_plus_verified', False):
            print('PLUS_RENDER_VERIFIED', dict(noise=noise, stereo=self.stereo,
                  state_unchanged=True, official_noise_replay=True), flush=True)
            self._plus_verified = True
        return obs, reward, done, info
