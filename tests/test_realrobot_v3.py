"""CPU contract tests; no model, GPU, SDK or robot connection is required."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Avoid deploy/__init__'s GPU/model imports for this CPU-only adapter test.
v3 = load_file("realrobot_v3_test_module", ROOT / "openwam/deploy/realrobot_v3.py")


def stats():
    return dict(state_mean=np.arange(20).tolist(), state_std=np.linspace(1, 3, 20).tolist(),
                action_mean=np.linspace(-2, 2, 20).tolist(), action_std=np.linspace(0.1, 2, 20).tolist(),
                supervision="recorded_pose_targets_not_joint_commands",
                gripper_transform="empirical_zscore_no_clipping", action_rate_hz=20, rgb_rate_hz=10)


def pose():
    return np.array([0.2, 0.1, 0.3, 0, 0, 0, 1, 0.037,
                     -0.1, 0.2, 0.4, 0, 0, 0, 1, 0.081], dtype=np.float32)


def relative():
    actions = np.zeros((34, 2, 10), dtype=np.float32)
    actions[..., 3:9] = [1, 0, 0, 0, 1, 0]
    actions[..., 9] = [0.041, 0.092]
    return actions.reshape(34, 20)


def observation(stereo=False):
    obs = {"observation/cartesian_state": pose(), "prompt": v3.PROMPTS["cup_stacking"],
           "observation/state_timestamp": 100.0, "observation_id": "test-request",
           "observation/camera_timestamps": np.full(6 if stereo else 3, 100.0)}
    for index, name in enumerate(v3.CAMERAS):
        obs["observation/" + name] = np.full((192, 256, 3), 20 + index, np.uint8)
        if stereo:
            obs["observation/right/" + name] = np.full((192, 256, 3), 120 + index, np.uint8)
    return obs


def test_stats_mapping_includes_raw_grippers_and_distinct_state_action_stats():
    norm = v3.V3Normalizer(stats())
    state = v3.cartesian_to_tcp(pose())
    packed = norm.normalize(state)
    np.testing.assert_allclose(packed[v3.SLOTS], (state - stats()["state_mean"]) / stats()["state_std"], atol=1e-6)
    assert np.count_nonzero(packed[np.setdiff1d(np.arange(80), v3.SLOTS)]) == 0
    raw = relative()
    normalized = np.zeros((34, 80), np.float32)
    normalized[:, v3.SLOTS] = (raw - stats()["action_mean"]) / stats()["action_std"]
    np.testing.assert_allclose(norm.unnormalize(normalized), raw, atol=1e-6)
    assert abs(norm.unnormalize(normalized)[0, 9] - 0.041) < 1e-6


@pytest.mark.parametrize("field,value", [("state_std", [0] * 20), ("action_mean", [float('nan')] * 20),
                                         ("gripper_transform", "canonical_openness")])
def test_bad_stats_rejected(field, value):
    invalid = stats()
    invalid[field] = value
    with pytest.raises(ValueError):
        v3.V3Normalizer(invalid)


def test_fixed_anchor_rotation_row_convention_and_timeline():
    state = pose().reshape(2, 8)
    state[0, 3:7] = Rotation.from_euler("z", 90, degrees=True).as_quat()
    anchor = v3.cartesian_to_tcp(state.reshape(16))
    raw = relative().reshape(34, 2, 10)
    raw[:2, :, :3] = 999  # Historical/current positions must never be executed.
    raw[2:, :, 0] = np.arange(1, 33)[:, None] * 0.001
    target_rot = Rotation.from_euler("x", 0.1).as_matrix()
    raw[2:, :, 3:9] = target_rot[:2].reshape(6)
    result = v3.absolute_targets(anchor, raw.reshape(34, 20), 0.1, 0.5).reshape(32, 2, 8)
    np.testing.assert_allclose(result[:, 0, 1], state[0, 1] + np.arange(1, 33) * 0.001, atol=1e-7)
    np.testing.assert_allclose(result[:, 1, 0], state[1, 0] + np.arange(1, 33) * 0.001, atol=1e-7)
    expected_rot = Rotation.from_quat(state[0, 3:7]).as_matrix() @ target_rot
    np.testing.assert_allclose(Rotation.from_quat(result[0, 0, 3:7]).as_matrix(), expected_rot, atol=1e-7)
    np.testing.assert_allclose(result[..., 7], np.tile([0.041, 0.092], (32, 1)), atol=1e-7)


def test_degenerate_rotation_and_large_motion_rejected():
    with pytest.raises(ValueError, match="Degenerate"):
        v3.rows_to_matrix(np.zeros(6))
    raw = relative()
    raw[2, 0] = 0.2
    with pytest.raises(ValueError, match="translation"):
        v3.absolute_targets(v3.cartesian_to_tcp(pose()), raw, 0.08, 0.5)


def test_mosaic_matches_training_pil_layout():
    obs = observation(True)
    rng = np.random.default_rng(42)
    for name in v3.CAMERAS:
        obs["observation/" + name] = rng.integers(0, 256, (192, 256, 3), dtype=np.uint8)
    transforms = load_file("v3_reference_multiview", ROOT / "openwam/dataloader/transforms/multiview.py")
    expected = transforms.assemble_multiview_layout(
        {name: Image.fromarray(obs["observation/" + name]) for name in v3.CAMERAS},
        list(v3.CAMERAS), out_h=384, out_w=320,
    )
    np.testing.assert_array_equal(v3.camera_mosaic(obs), expected)
    assert np.array(v3.camera_mosaic(obs, right=True))[0, 0, 0] == 120
    obs["observation/base_0_rgb"] = np.zeros((480, 640, 3), np.uint8)
    with pytest.raises(ValueError, match="256x192"):
        v3.camera_mosaic(obs)


@pytest.mark.parametrize("stereo", [False, True])
def test_policy_engine_boundary(stereo):
    captured = {}
    def generate(conditions):
        captured.update(conditions)
        return {"actions": relative()}
    policy = v3.V3Policy(SimpleNamespace(generate=generate), "cup_stacking", "stereo" if stereo else "mono-left")
    reply = policy.infer(observation(stereo))
    assert reply["actions"].shape == (4, 16)
    assert reply["observation_id"] == "test-request"
    assert captured["num_frames"] == 35 and captured["video_num_frames"] == 17
    assert ("right_first_frame_image" in captured) == stereo
    np.testing.assert_allclose(reply["actions"][0, [7, 15]], [0.041, 0.092], atol=1e-7)


def test_skew_missing_right_and_wrong_task_rejected():
    policy = v3.V3Policy(None, "cup_stacking", "stereo")
    obs = observation(True)
    obs["observation/camera_timestamps"][2] -= 0.1
    with pytest.raises(ValueError, match="skew"):
        policy.infer(obs)
    obs = observation(True)
    obs["prompt"] = "wrong task"
    with pytest.raises(ValueError, match="prompt"):
        policy.infer(obs)
    obs = observation(True)
    del obs["observation/right/base_0_rgb"]
    with pytest.raises(KeyError):
        policy.infer(obs)


def test_timestamp_precision_on_long_running_client():
    obs = observation()
    obs["observation/state_timestamp"] = 10_000_000.123
    obs["observation/camera_timestamps"][:] = 10_000_000.113
    v3.validate_observation(obs, "mono-left", 0.035)
    obs["observation/camera_timestamps"][0] -= 0.04
    with pytest.raises(ValueError, match="skew"):
        v3.validate_observation(obs, "mono-left", 0.035)


def test_client_check_only_never_enables_or_sends(monkeypatch):
    import yaml
    # ORION src is an explicit deployment dependency (same one used on the robot).
    client = load_file("realrobot_v3_client_test_module", ROOT / "scripts/realrobot_v3_client.py")
    settings = yaml.safe_load((ROOT / "configs/realrobot_v3_inference.yaml").read_text())
    calls = []

    class Robot:
        def __init__(self, config, right_topics):
            calls.append("connect")

        def observation(self, prompt):
            obs = observation()
            obs["observation/camera_timestamps"][:] = client.time.monotonic()
            obs["observation/state_timestamp"] = client.time.monotonic()
            return obs

        def close(self):
            calls.append("disconnect")

        def stop_motion(self):
            calls.append("stop")

        def reset(self, *args):
            pytest.fail("check-only must not reset")

        def enable_servo(self):
            pytest.fail("check-only must not enable motion")

        def send(self, *args):
            pytest.fail("check-only must not send commands")

    class Policy:
        def __init__(self, *args):
            pass

        def infer(self, obs):
            return dict(actions=np.tile(pose(), (4, 1)), observation_id=obs["observation_id"],
                        task="cup_stacking", input_mode="mono-left", action_rate_hz=20,
                        first_target_offset_s=0.05)

        def close(self):
            calls.append("policy_close")

    from dataclasses import make_dataclass
    fields = dict(action_space="tcp", arms="both", ttrtc=None, camera_topics=("head", "left", "right"),
                  control_hz=20, async_inference=False, policy_host="localhost", policy_port=8000,
                  check_only=False, skip_reset=False, gripper_binary_enabled=True, prompt="unused")
    config = make_dataclass("Config", [(key, type(value)) for key, value in fields.items()])(**fields)
    monkeypatch.setattr(client, "V3P7System", Robot)
    monkeypatch.setattr(client, "PolicyClient", Policy)
    client.run(settings, config)
    assert calls == ["connect", "policy_close", "stop", "disconnect"]
    with pytest.raises(ValueError, match="Verify"):
        client.run(settings, config, enable_motion=True)
    assert calls == ["connect", "policy_close", "stop", "disconnect"]  # gate before connecting


def test_client_rejects_wrong_reply_and_does_not_clip_gripper():
    import yaml
    client = load_file("realrobot_v3_client_validation", ROOT / "scripts/realrobot_v3_client.py")
    settings = yaml.safe_load((ROOT / "configs/realrobot_v3_inference.yaml").read_text())
    limits = client.validate_settings(settings, False)
    obs = observation()
    reply = dict(actions=np.tile(pose(), (4, 1)), observation_id=obs["observation_id"], task="cup_stacking",
                 input_mode="mono-left", action_rate_hz=20, first_target_offset_s=0.05)
    np.testing.assert_array_equal(client.validate_reply(reply, obs, settings, limits), reply["actions"])
    reply["observation_id"] = "old-request"
    with pytest.raises(ValueError, match="contract"):
        client.validate_reply(reply, obs, settings, limits)
    reply["observation_id"] = obs["observation_id"]
    reply["actions"][0, 7] = 2.0
    with pytest.raises(ValueError, match="gripper"):
        client.validate_reply(reply, obs, settings, limits)


def test_existing_orion_transport_round_trip():
    from threading import Thread
    from wam.deployment.airbot.policy_protocol import PolicyClient, PolicyServer

    engine = SimpleNamespace(generate=lambda conditions: {"actions": relative()})
    server = PolicyServer(("127.0.0.1", 0), v3.V3Policy(engine, "cup_stacking", "mono-left"))
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    client = PolicyClient(*server.server_address)
    try:
        obs = observation()
        result = client.infer(obs)
        assert result["observation_id"] == obs["observation_id"]
        np.testing.assert_allclose(result["actions"][0, [7, 15]], [0.041, 0.092], atol=1e-7)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_p7_feedback_and_send_preserve_tcp_and_convert_gripper_once():
    import yaml
    client = load_file("v3_p7_units", ROOT / "scripts/realrobot_v3_client.py")
    settings = yaml.safe_load((ROOT / "configs/realrobot_v3_inference.yaml").read_text())
    targets = pose().reshape(2, 8)
    sent = []

    class Arm:
        def __init__(self, index):
            self.index = index

        def get_end_pose(self):
            return SimpleNamespace(position=targets[self.index, :3], orientation=targets[self.index, 3:7])

        def get_eef_joint_state(self):
            return SimpleNamespace(eef_pos=[float(targets[self.index, 7]) * 1000])

        def move_end_pose(self, value, options, **kwargs):
            sent.append((self.index, "pose", value))
            return True

        def move_eef(self, value, options, **kwargs):
            sent.append((self.index, "gripper_mm", value))
            return True

    robot = client.P7System.__new__(client.P7System)
    robot._arms = [Arm(0), Arm(1)]
    robot._active = (True, True)
    robot._action_space = "tcp"
    robot._pose = SimpleNamespace
    robot._servo_pose = robot._eef_async = None
    robot._gripper_mm = [None, None]
    measured = client.check_anchor(robot, observation(), settings)
    np.testing.assert_allclose(measured, pose())
    robot.send(measured)
    for index in range(2):
        np.testing.assert_allclose(sent[2 * index][2].position, targets[index, :3])
        np.testing.assert_allclose(sent[2 * index][2].orientation, targets[index, 3:7])
        np.testing.assert_allclose(sent[2 * index + 1][2], [targets[index, 7] * 1000])
    shifted = observation()
    shifted["observation/cartesian_state"][0] += 0.02
    with pytest.raises(ValueError, match="anchor"):
        client.check_anchor(robot, shifted, settings)


@pytest.mark.parametrize("stereo", [False, True])
def test_p7_cora_camera_mapping_resize_and_timestamps(monkeypatch, stereo):
    client = load_file("v3_p7_cameras", ROOT / "scripts/realrobot_v3_client.py")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:240, :320, 0] = 255
    obs = observation()
    for index, name in enumerate(v3.CAMERAS):
        obs["observation/" + name] = frame + index
    monkeypatch.setattr(client.P7System, "observation", lambda self, prompt: obs.copy())
    robot = client.V3P7System.__new__(client.V3P7System)
    robot._right_readers = ([SimpleNamespace(latest=lambda i=i: (frame + i + 10, 100.01 + i * 0.001))
                             for i in range(3)] if stereo else [])
    actual = robot.observation("test")
    for i, name in enumerate(v3.CAMERAS):
        expected = Image.fromarray(frame + i).resize((256, 192), Image.Resampling.BILINEAR)
        np.testing.assert_array_equal(actual["observation/" + name], expected)
        if stereo:
            expected_right = Image.fromarray(frame + i + 10).resize((256, 192), Image.Resampling.BILINEAR)
            np.testing.assert_array_equal(actual["observation/right/" + name], expected_right)
    np.testing.assert_allclose(actual["observation/camera_timestamps"],
                               [100, 100, 100, 100.01, 100.011, 100.012] if stereo else [100, 100, 100])
    obs["observation/base_0_rgb"] = np.zeros((352, 640, 3), np.uint8)
    with pytest.raises(ValueError, match="4:3"):
        robot.observation("test")


def test_p7_config_rejects_single_arm_and_overwide_grippers():
    from dataclasses import replace
    import yaml
    from wam.deployment.p7.client import DEFAULT_DEPLOYMENT_CONFIG
    client = load_file("v3_p7_config", ROOT / "scripts/realrobot_v3_client.py")
    settings = yaml.safe_load((ROOT / "configs/realrobot_v3_inference.yaml").read_text())
    config = client.load_deployment_config(DEFAULT_DEPLOYMENT_CONFIG)
    with pytest.raises(ValueError, match="arms: both"):
        client.run(settings, replace(config, arms="right"))
    with pytest.raises(ValueError, match="distinct"):
        client.run(settings, replace(config, arms="both", camera_topics=("same",) * 3))
    settings["client"]["raw_gripper_limits"][0][1] = 1.0
    with pytest.raises(ValueError, match="0.095"):
        client.validate_settings(settings, False)


@pytest.mark.parametrize("failure", ["enable", "send"])
def test_p7_motion_failure_releases_and_never_executes_warmup(monkeypatch, failure):
    from dataclasses import replace
    import yaml
    from wam.deployment.p7.client import DEFAULT_DEPLOYMENT_CONFIG
    client = load_file("v3_p7_cleanup", ROOT / "scripts/realrobot_v3_client.py")
    settings = yaml.safe_load((ROOT / "configs/realrobot_v3_inference.yaml").read_text())
    settings["client"]["interfaces_verified"] = True
    config = replace(client.load_deployment_config(DEFAULT_DEPLOYMENT_CONFIG), arms="both")
    calls = []

    class Robot:
        def __init__(self, *args):
            pass

        def observation(self, prompt):
            obs = observation()
            obs["observation/camera_timestamps"][:] = client.time.monotonic()
            obs["observation/state_timestamp"] = client.time.monotonic()
            calls.append("observe")
            return obs

        def enable_servo(self):
            calls.append("enable")
            if failure == "enable":
                raise RuntimeError("enable failed")

        def mark_moving(self):
            calls.append("mark")

        def stop_motion(self):
            calls.append("stop")
            if failure == "send":
                raise RuntimeError("stop failed")

        def close(self):
            calls.append("release")

    class Policy:
        def __init__(self, *args):
            pass

        def infer(self, obs):
            calls.append("infer")
            return dict(actions=np.tile(pose(), (4, 1)), observation_id=obs["observation_id"],
                        task="cup_stacking", input_mode="mono-left", action_rate_hz=20,
                        first_target_offset_s=0.05)

        def close(self):
            calls.append("policy_close")

    def send(*args):
        calls.append("send")
        raise RuntimeError("send failed")

    monkeypatch.setattr(client, "V3P7System", Robot)
    monkeypatch.setattr(client, "PolicyClient", Policy)
    monkeypatch.setattr(client, "check_anchor", lambda *args: pose())
    monkeypatch.setattr(client, "send_target", send)
    monkeypatch.setattr("builtins.input", lambda *args: "")
    with pytest.raises(RuntimeError):
        client.run(settings, config, enable_motion=True)
    assert calls[:3] == ["observe", "infer", "enable"]
    if failure == "enable":
        assert "mark" not in calls and "send" not in calls
    else:
        assert calls[3:7] == ["mark", "observe", "infer", "send"]
    assert calls[-3:] == ["policy_close", "stop", "release"]
