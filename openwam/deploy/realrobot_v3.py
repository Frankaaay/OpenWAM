"""Serve the recorded-pose v3 checkpoints through ORION's AIRBOT protocol.

The normalizer belongs to this dataset contract, not ORION's TCP128 manifest.
Robot poses and grippers must already have the recorded-state units/frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


SLOTS = np.array([*range(10), *range(34, 44)])
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
PROMPTS = {"cup_stacking": "Stack the cups.", "object_handover": "Hand over the wooden stick."}
PREFIX = "A video recorded from a robot's point of view executing the following instruction: "


def finite(value, shape, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {value.shape}")
    return value


def cartesian_to_tcp(state):
    arms = finite(state, (16,), "Cartesian state").reshape(2, 8)
    if not np.allclose(np.linalg.norm(arms[:, 3:7], axis=-1), 1, atol=1e-3):
        raise ValueError("SDK xyzw quaternions must have unit norm")
    rows = Rotation.from_quat(arms[:, 3:7]).as_matrix()[:, :2, :].reshape(2, 6)
    return np.concatenate((arms[:, :3], rows, arms[:, 7:8]), axis=-1).reshape(20).astype(np.float32)


def rows_to_matrix(rows):
    rows = np.asarray(rows, dtype=np.float64)
    first, second = rows[..., :3], rows[..., 3:]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("Degenerate first Rot6D row")
    first = first / norm
    second = second - (first * second).sum(-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("Degenerate parallel Rot6D rows")
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=-2)


class V3Normalizer:
    """Raw TCP20 state -> normalized80; normalized80 action -> relative TCP20."""

    _dst_index = SLOTS
    _unify_dim = 80

    def __init__(self, stats):
        if (stats.get("supervision") != "recorded_pose_targets_not_joint_commands"
                or stats.get("gripper_transform") != "empirical_zscore_no_clipping"
                or stats.get("action_rate_hz") != 20 or stats.get("rgb_rate_hz") != 10):
            raise ValueError("Not a v3 recorded-pose normalization artifact")
        self.stats = {key: finite(stats[key], (20,), key) for key in
                      ("state_mean", "state_std", "action_mean", "action_std")}
        if any(np.any(self.stats[key] <= 0) for key in ("state_std", "action_std")):
            raise ValueError("Normalization std must be positive")

    def normalize(self, state):
        state = finite(state, (20,), "TCP state")
        output = np.zeros(80, dtype=np.float32)
        output[SLOTS] = (state - self.stats["state_mean"]) / self.stats["state_std"]
        return output

    def unnormalize(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 80 or not np.isfinite(action).all():
            raise ValueError("Expected finite [T,80] model actions")
        return action[:, SLOTS] * self.stats["action_std"] + self.stats["action_mean"]


def absolute_targets(anchor, relative, max_translation, max_rotation, count=32):
    """Every target uses the observation's fixed anchor, never the previous target."""
    anchor = finite(anchor, (20,), "anchor").reshape(2, 10)
    relative = finite(relative, (34, 20), "relative actions").reshape(34, 2, 10)
    # Indices 0/1 are t=-50ms/0. Only future targets may reach the robot.
    future = relative[2:2 + count]
    if np.any(np.linalg.norm(future[..., :3], axis=-1) > max_translation):
        raise ValueError("Predicted translation exceeds the anchor-relative limit")
    rotation = rows_to_matrix(future[..., 3:9])
    angle = Rotation.from_matrix(rotation.reshape(-1, 3, 3)).magnitude()
    if np.any(angle > max_rotation):
        raise ValueError("Predicted rotation exceeds the anchor-relative limit")
    base = rows_to_matrix(anchor[:, 3:9])
    positions = anchor[None, :, :3] + np.einsum("aij,taj->tai", base, future[..., :3])
    rotations = np.einsum("aij,tajk->taik", base, rotation)
    quaternions = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(count, 2, 4)
    return np.concatenate((positions, quaternions, future[..., 9:10]), axis=-1).reshape(count, 16).astype(np.float32)


def camera_mosaic(observation, *, right=False):
    # Match v3's published full-FOV 256x192 frames, then its PIL BILINEAR mosaic.
    canvas = Image.new("RGB", (320, 384))
    prefix = "observation/right/" if right else "observation/"
    for name, size, xy in zip(CAMERAS, ((320, 256), (160, 128), (160, 128)),
                              ((0, 0), (0, 256), (160, 256))):
        raw = np.asarray(observation[prefix + name])
        if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[-1] != 3 or min(raw.shape[:2]) < 1:
            raise ValueError(f"{prefix + name} must be uint8 RGB [H,W,3]")
        image = Image.fromarray(raw)
        if image.size != (256, 192):
            raise ValueError("Client must provide full-FOV 256x192 v3 frames; verify the capture resize first")
        canvas.paste(image.resize(size, Image.Resampling.BILINEAR), xy)
    return canvas


def validate_observation(observation, mode, max_skew_s):
    count = 6 if mode == "stereo" else 3
    # Float32 loses sub-frame precision for long-running monotonic clocks.
    times = np.asarray(observation["observation/camera_timestamps"], dtype=np.float64)
    if times.shape != (count,) or not np.isfinite(times).all():
        raise ValueError("Invalid camera timestamps")
    state_time = float(observation["observation/state_timestamp"])
    if not np.isfinite(state_time) or np.ptp(np.append(times, state_time)) > max_skew_s:
        raise ValueError("Camera/state acquisition timestamp skew is too large")
    # Timestamps are client-local; never subtract them from the server clock.
    if not isinstance(observation.get("observation_id"), str):
        raise ValueError("observation_id is required")


class V3Policy:
    def __init__(self, engine, task, mode, *, actions_per_chunk=4,
                 max_translation=0.08, max_rotation=0.5, max_skew_s=0.035):
        if task not in PROMPTS or mode not in ("mono-left", "stereo"):
            raise ValueError("Unknown v3 task/input mode")
        if not 1 <= actions_per_chunk <= 32:
            raise ValueError("actions_per_chunk must be in [1,32]")
        if not all(np.isfinite(x) and x > 0 for x in (max_translation, max_rotation, max_skew_s)):
            raise ValueError("Motion and skew limits must be finite and positive")
        self.engine, self.task, self.mode = engine, task, mode
        self.chunk, self.max_translation = actions_per_chunk, max_translation
        self.max_rotation, self.max_skew_s = max_rotation, max_skew_s

    def infer(self, observation):
        started = time.perf_counter()
        validate_observation(observation, self.mode, self.max_skew_s)
        if observation.get("prompt") != PROMPTS[self.task]:
            raise ValueError("Client task prompt does not match checkpoint task")
        anchor = cartesian_to_tcp(observation["observation/cartesian_state"])
        conditions = dict(first_frame_image=[camera_mosaic(observation)], proprio=anchor,
                          prompt=PREFIX + PROMPTS[self.task], num_frames=35, video_num_frames=17,
                          height=384, width=320)
        if self.mode == "stereo":
            conditions["right_first_frame_image"] = [camera_mosaic(observation, right=True)]
        result = self.engine.generate(conditions)
        # architecture.generate already applies V3Normalizer. Do not unnormalize twice.
        actions = absolute_targets(anchor, result["actions"], self.max_translation, self.max_rotation, self.chunk)
        return dict(actions=actions[:self.chunk], observation_id=observation["observation_id"],
                    task=self.task, input_mode=self.mode, action_rate_hz=20,
                    first_target_offset_s=0.05, inference_seconds=time.perf_counter() - started)


def load_policy(checkpoint, task, mode, **options):
    from omegaconf import OmegaConf
    from openwam.deploy.engine import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file() or checkpoint.suffix != ".safetensors":
        raise FileNotFoundError("Specify one exact final .safetensors file")
    root = checkpoint.parent
    contract = json.loads((root / "data-contract.json").read_text(encoding="utf-8"))
    expected_model = "stereo" if mode == "stereo" else "wan"
    if contract.get("task") != task or contract.get("model") != expected_model or contract.get("smoke") is not False:
        raise ValueError("Checkpoint task/model/formal contract mismatch")
    stats_bytes = (root / "realrobot_stats.json").read_bytes()
    if hashlib.sha256(stats_bytes).hexdigest() != contract.get("stats_sha256"):
        raise ValueError("Normalization artifact differs from the training contract")
    stats = json.loads(stats_bytes)
    normalizer = V3Normalizer(stats)
    # v3 normalized in its dataset. Reject an unrelated generic normalizer config.
    saved = OmegaConf.load(root / "config.yaml")
    if OmegaConf.select(saved, "dataloader.normalize_mode") not in (None, "none", "null", ""):
        raise ValueError("Unexpected generic normalizer in v3 checkpoint")
    cfg, architecture = load_from_checkpoint_dir(str(root), ckpt_name=checkpoint.name)
    if architecture.action_dim != 80:
        raise ValueError("v3 requires an 80-dimensional action head")
    encoder = OmegaConf.select(cfg, "model.video_backbone.encoder.name")
    if (encoder == "stereotok") != (mode == "stereo"):
        raise ValueError("Checkpoint codec does not match requested input mode")
    architecture.attach_normalizer(normalizer)
    cfg.inference = OmegaConf.create(dict(denoise_steps=options.pop("denoise_steps"), denoise_mode="sync"))
    cfg.optimization = OmegaConf.create(dict(decode_video=False, compile=dict(enabled=False),
                                            dit_cache=dict(enabled=False)))
    return V3Policy(JointInferenceEngine(cfg, architecture), task, mode, **options)


def main():
    import yaml
    from wam.deployment.airbot.policy_protocol import PolicyServer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    policy = load_policy(args.checkpoint, settings["task"], settings["input_mode"], **settings["inference"])
    with PolicyServer((args.host, args.port), policy) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
