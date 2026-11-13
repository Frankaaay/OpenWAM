"""P7 v3 client; inference-only unless --enable-motion is explicitly supplied.

Reuse ORION's SDK control/interpolation and transport; no model/GPU imports here.
PYTHONPATH must include the pinned orion-0/src directory.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path
import time
import uuid

import numpy as np
import yaml
from PIL import Image

from wam.deployment.airbot import CAMERA_NAMES
from wam.deployment.airbot.client import (
    LatestFrameCamera, policy_actions, send_target,
)
from wam.deployment.airbot.policy_protocol import PolicyClient
from wam.deployment.p7.client import (
    CoraImageStream, GRIPPER_RANGE_M, MM_PER_M, P7System, load_deployment_config,
)


LOGGER = logging.getLogger(__name__)
PROMPTS = {"cup_stacking": "Stack the cups.", "object_handover": "Hand over the wooden stick."}


class V3P7System(P7System):
    def __init__(self, config, right_topics=None):
        self._right_readers = []
        super().__init__(config)
        try:
            if right_topics is not None:
                import cv2
                for name, topic, rotate in zip(
                        CAMERA_NAMES, right_topics, config.camera_rotate_180, strict=True):
                    self._right_readers.append(LatestFrameCamera(
                        CoraImageStream(topic, rotate), cv2, name + "_right", config.camera_fps))
            for reader in self._right_readers:
                reader.latest(timeout=10.0)
        except Exception:
            self.close()
            raise

    def observation(self, prompt):
        observation = super().observation(prompt)
        times = list(observation["observation/camera_timestamps"])
        for name, reader in zip(CAMERA_NAMES, self._right_readers):
            frame, timestamp = reader.latest()
            observation["observation/right/" + name] = frame
            times.append(timestamp)
        observation["observation/camera_timestamps"] = np.asarray(times)
        # CORA publishes full 640x480 views. Preserve the full 4:3 field of view;
        # do not apply the canonical ORION server's crop/letterbox transforms.
        for key in ["observation/" + name for name in CAMERA_NAMES] + [
                "observation/right/" + name for name in CAMERA_NAMES if self._right_readers]:
            rgb = observation[key]
            if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                    or rgb.shape[1] * 3 != rgb.shape[0] * 4):
                raise ValueError("P7 cameras must provide full-view 4:3 RGB uint8 images")
            observation[key] = np.asarray(Image.fromarray(rgb).resize((256, 192), Image.Resampling.BILINEAR))
        return observation

    def close(self):
        for reader in self._right_readers:
            reader.close()
        super().close()


def validate_settings(settings, enable_motion):
    if settings["task"] not in PROMPTS or settings["input_mode"] not in ("mono-left", "stereo"):
        raise ValueError("Unknown task or camera mode")
    devices = settings["client"]["right_camera_topics"]
    if settings["input_mode"] == "stereo" and (
            not isinstance(devices, list) or len(devices) != 3 or len(set(devices)) != 3
            or any(not isinstance(x, str) or not x for x in devices)):
        raise ValueError("Stereo requires three explicit, distinct right-camera CORA topics")
    if enable_motion and settings["client"]["interfaces_verified"] is not True:
        raise ValueError("Verify recorded-pose/SDK frame, raw gripper and capture parity before enabling motion")
    limits = np.asarray(settings["client"]["raw_gripper_limits"], dtype=float)
    if limits.shape != (2, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("raw_gripper_limits must contain [min,max] for each SDK/raw gripper")
    if np.any(limits[:, 0] < GRIPPER_RANGE_M[0]) or np.any(limits[:, 1] > GRIPPER_RANGE_M[1]):
        raise ValueError("P7 gripper limits must remain within 0..0.095 metres")
    for key in ("max_round_trip_s", "max_observation_age_s", "max_anchor_translation", "max_anchor_rotation"):
        value = settings["client"][key]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if type(settings["client"]["max_steps"]) is not int or settings["client"]["max_steps"] < 1:
        raise ValueError("max_steps must be a positive integer")
    return limits


def validate_reply(reply, observation, settings, limits):
    if (reply.get("observation_id") != observation["observation_id"]
            or reply.get("task") != settings["task"]
            or reply.get("input_mode") != settings["input_mode"]
            or reply.get("action_rate_hz") != 20 or reply.get("first_target_offset_s") != 0.05):
        raise ValueError("Server reply does not match this v3 observation/contract")
    actions = policy_actions(reply, action_space="tcp")
    if len(actions) != settings["inference"]["actions_per_chunk"]:
        raise ValueError("Server/client execution chunk lengths differ")
    arms = actions.reshape(-1, 2, 8)
    if not np.allclose(np.linalg.norm(arms[..., 3:7], axis=-1), 1, atol=1e-3):
        raise ValueError("Server returned non-unit SDK quaternions")
    grippers = arms[..., 7]
    if np.any(grippers < limits[:, 0]) or np.any(grippers > limits[:, 1]):
        raise ValueError("Raw predicted gripper is outside verified SDK limits; no clipping applied")
    return actions


def check_anchor(robot, observation, settings):
    """Blocking inference is allowed only while the robot remains near its anchor."""
    measured = []
    for arm in robot._arms:
        pose, gripper = arm.get_end_pose(), arm.get_eef_joint_state()
        if (pose is None or gripper is None or len(pose.position) != 3
                or len(pose.orientation) != 4 or len(gripper.eef_pos) != 1):
            raise ValueError("Invalid P7 Cartesian/gripper feedback after inference")
        measured.extend([*pose.position, *pose.orientation, gripper.eef_pos[0] / MM_PER_M])
    current = np.asarray(measured, dtype=float).reshape(2, 8)
    anchor = np.asarray(observation["observation/cartesian_state"]).reshape(2, 8)
    if not np.isfinite(current).all() or not np.allclose(np.linalg.norm(current[:, 3:7], axis=-1), 1, atol=1e-3):
        raise ValueError("Invalid measured pose after inference")
    translation = np.linalg.norm(current[:, :3] - anchor[:, :3], axis=-1)
    angle = 2 * np.arccos(np.clip(np.abs(np.sum(current[:, 3:7] * anchor[:, 3:7], axis=-1)), 0, 1))
    if (np.any(translation > settings["client"]["max_anchor_translation"])
            or np.any(angle > settings["client"]["max_anchor_rotation"])):
        raise ValueError("Robot moved away from the inference anchor; refusing stale targets")
    return current.reshape(16).astype(np.float32)


def run(settings, config, *, enable_motion=False):
    limits = validate_settings(settings, enable_motion)
    if config.action_space != "tcp" or config.control_hz != 20 or config.async_inference:
        raise ValueError("v3 client requires TCP, 20 Hz and synchronous execution")
    if config.arms != "both" or config.ttrtc is not None:
        raise ValueError("v3 checkpoints require P7 arms: both and TTRTC disabled")
    if config.policy_host is None:
        raise ValueError("Set policy.host and policy.port in robot config")
    # No reset or stock gripper threshold/projection is used by this entrypoint.
    config = replace(config, check_only=not enable_motion, skip_reset=True,
                     gripper_binary_enabled=False, prompt=PROMPTS[settings["task"]])
    right = settings["client"]["right_camera_topics"] if settings["input_mode"] == "stereo" else None
    if len(set(config.camera_topics)) != 3 or (right is not None and set(right) & set(config.camera_topics)):
        raise ValueError("P7 camera topics must be distinct across mounts and eyes")
    policy = PolicyClient(config.policy_host, config.policy_port)
    robot = None
    try:
        robot = V3P7System(config, right)
        # Warm-up and check the full path before any mode switch or servo command.
        executed, enabled = 0, False
        while True:
            observation = robot.observation(config.prompt)
            observation["observation_id"] = uuid.uuid4().hex
            timestamps = np.append(observation["observation/camera_timestamps"], observation["observation/state_timestamp"])
            if time.monotonic() - timestamps.min() > settings["client"]["max_observation_age_s"]:
                raise ValueError("Observation too old before inference")
            started = time.monotonic()
            reply = policy.infer(observation)
            elapsed = time.monotonic() - started
            actions = validate_reply(reply, observation, settings, limits)
            LOGGER.info("Inference %.3fs; %d TCP16 targets; raw grippers=%s", elapsed, len(actions), actions[:, [7, 15]])
            if not enable_motion:
                return  # No reset, switch_mode, or servo calls in check-only mode.
            if not enabled:
                if input("Check passed. Press Enter to enable motion from CURRENT pose, or q to quit: ").strip():
                    return
                robot.enable_servo()
                robot.mark_moving()  # Partial enable failures are released by P7.close().
                enabled = True
                continue  # Discard the pre-confirmation observation and warm-up output.
            if elapsed > settings["client"]["max_round_trip_s"]:
                raise ValueError("Inference round trip exceeds configured motion budget")
            previous = check_anchor(robot, observation, settings)
            for target in actions[:settings["client"]["max_steps"] - executed]:
                send_target(robot, previous, target, config, time.monotonic())
                previous = target
                executed += 1
            if executed >= settings["client"]["max_steps"]:
                return
    finally:
        try:
            policy.close()
        finally:
            if robot is not None:
                try:
                    robot.stop_motion()
                finally:
                    robot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--robot-config", required=True, help="ORION P7 deployment YAML with arm_app ports and CORA topics")
    parser.add_argument("--enable-motion", action="store_true")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO)
    run(settings, load_deployment_config(args.robot_config), enable_motion=args.enable_motion)


if __name__ == "__main__":
    main()
