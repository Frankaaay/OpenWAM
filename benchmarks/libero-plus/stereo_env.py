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
        # These cached camera images are replaced by _images below. Disable
        # their rendering while retaining all state/proprioception observables.
        cameras = [v for k, v in self._wrapped.env._observables.items()
                   if k.endswith('_image') and v.is_enabled()]

        def capture(action):
            obs, reward, done, info = raw_step(action)
            captured['rng'] = np.random.get_state()
            return self._images(obs), reward, done, info

        for camera in cameras:
            camera.set_enabled(False)
        self._wrapped.env.step = capture
        try:
            obs, reward, done, info = self._wrapped.step(action)
        finally:
            self._wrapped.env.step = raw_step
            for camera in cameras:
                camera.set_enabled(True)
        noise = self._wrapped.noise
        if noise and self.stereo:
            current = np.random.get_state()
            try:
                np.random.set_state(captured['rng'])
                obs['agentview_stereo_right_image'] = noise_image(
                    obs['agentview_stereo_right_image'], noise)
            finally:
                np.random.set_state(current)
        if self.stereo:
            assert not np.array_equal(obs['agentview_image'], obs['agentview_stereo_right_image'])
        if not getattr(self, '_plus_verified', False):
            print('PLUS_RENDER_VERIFIED', dict(noise=noise, stereo=self.stereo,
                  official_noise_replay=True, fresh_camera_only=True), flush=True)
            self._plus_verified = True
        return obs, reward, done, info
