"""GPU smoke on one recorded v3 sample; run each checkpoint in a fresh process."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from openwam.deploy.realrobot_v3 import (
    CAMERAS, V3Normalizer, camera_mosaic, cartesian_to_tcp, load_policy, rows_to_matrix,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--task", choices=("cup_stacking", "object_handover"), required=True)
    parser.add_argument("--mode", choices=("mono-left", "stereo"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "GPU smoke requires a CUDA allocation"
    root = Path(__file__).resolve().parents[1]
    assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    sample_dir = Path(args.sample_dir)
    sample = json.loads((sample_dir / "sample.json").read_text())
    assert sample["task"] == args.task
    checkpoint = Path(args.checkpoint)
    normalizer = V3Normalizer(json.loads((checkpoint.parent / "realrobot_stats.json").read_text()))
    state = np.asarray(sample["state20"], dtype=np.float32)
    np.testing.assert_allclose(normalizer.normalize(state), sample["normalized_proprio80"], atol=1e-5, rtol=1e-5)
    raw_target = normalizer.unnormalize(np.asarray(sample["normalized_action80"], dtype=np.float32))
    valid = np.asarray(sample["valid34"], dtype=bool)
    np.testing.assert_allclose(raw_target[valid], np.asarray(sample["relative34"])[valid], atol=1e-5, rtol=1e-5)
    # Build precisely the Cartesian ABI the AIRBOT client sends.
    arms = state.reshape(2, 10)
    quat = Rotation.from_matrix(rows_to_matrix(arms[:, 3:9])).as_quat()
    sdk_state = np.concatenate((arms[:, :3], quat, arms[:, 9:10]), axis=-1).reshape(16)
    np.testing.assert_allclose(cartesian_to_tcp(sdk_state), state, atol=1e-5, rtol=1e-5)
    observation = {"observation/cartesian_state": sdk_state, "prompt": sample["prompt"],
                   "observation_id": "recorded-sample", "observation/state_timestamp": 100.0,
                   "observation/camera_timestamps": np.full(6 if args.mode == "stereo" else 3, 100.0)}
    for name, mount in zip(CAMERAS, ("head", "left_wrist", "right_wrist")):
        observation["observation/" + name] = np.asarray(Image.open(sample_dir / (mount + "_left.png")).convert("RGB"))
        if args.mode == "stereo":
            observation["observation/right/" + name] = np.asarray(Image.open(sample_dir / (mount + "_right.png")).convert("RGB"))
    assert camera_mosaic(observation).size == (320, 384)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    policy = load_policy(checkpoint, args.task, args.mode, denoise_steps=10, actions_per_chunk=4)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    measurements = []
    for index in range(2):
        started = time.perf_counter()
        result = policy.infer(observation)
        torch.cuda.synchronize()
        actions = np.asarray(result["actions"])
        assert actions.shape == (4, 16) and np.isfinite(actions).all()
        assert result["first_target_offset_s"] == 0.05 and result["action_rate_hz"] == 20
        measurements.append(dict(seconds=time.perf_counter() - started, actions=actions.tolist()))
    report = dict(status="passed", checkpoint=str(checkpoint), task=args.task, mode=args.mode,
                  openwam_sha=sha, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  episode_index=sample["episode_index"], anchor=sample["anchor"],
                  loader_parity="passed", strict_load="passed", load_seconds=load_seconds,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), measurements=measurements,
                  scope="recorded-observation inference only; no robot or closed-loop success validation")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print("REALROBOT_V3_GPU_SMOKE_PASSED", args.task, args.mode, flush=True)


if __name__ == "__main__":
    main()
