"""Acceptance checks for the migrated cup dataset and a ten-update smoke."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def test_mug_right_arm_only(tmp_path, monkeypatch):
    """Exercise the reader with idle-arm offsets, both input modes and history padding."""
    import numpy as np
    import torch

    import openwam.dataloader.realrobot_v3 as reader

    states = np.zeros((40, 20), dtype=np.float32)
    states[:, [3, 7, 13, 17]] = 1  # identity rowRot6D
    states[:, 10] = np.arange(40) * 0.001
    states[:, 19] = 0.02
    np.save(tmp_path / "states.npy", states)
    stats = dict(state_mean=[0.] * 20, action_mean=[0.] * 20,
                 state_std=[1e-6] * 10 + [1.] * 10, action_std=[1e-6] * 10 + [1.] * 10)
    row = dict(states="states.npy", anchors=[0, 2], episode_index=0,
               instruction="hang the mug", videos={f"{m}_{e}": "unused.mp4"
               for m in ["head", "left_wrist", "right_wrist"] for e in ["left", "right"]})
    for task in ["mug_hanging", "cup_stacking", "object_handover"]:
        (tmp_path / f"{task}-catalog.json").write_text(json.dumps([row]))
        (tmp_path / f"{task}-stats.json").write_text(json.dumps(stats))
    monkeypatch.setattr(reader, "frames", lambda *args: np.zeros((17, 2, 2, 3), dtype=np.uint8))
    for mode in ["mono-left", "stereo"]:
        mug = reader.V3Dataset("mug_hanging", tmp_path, tmp_path, mode)
        cup = reader.V3Dataset("cup_stacking", tmp_path, tmp_path, mode)
        handover = reader.V3Dataset("object_handover", tmp_path, tmp_path, mode)
        for index in [0, 1]:
            sample, baseline = mug[index], cup[index]
            for key in ["action", "proprio", "action_mask", "proprio_mask"]:
                assert not sample[key][..., :10].any()
                assert torch.equal(sample[key][..., 10:], baseline[key][..., 10:])
                assert torch.equal(handover[index][key], baseline[key])
            valid = torch.arange(34) >= (1 if index == 0 else 0)
            assert torch.equal(sample["action_mask"][:, 34:44], valid[:, None].expand(-1, 10))
            assert sample["proprio_mask"][..., 34:44].all()
            assert (sample["right_video"] is not None) == (mode == "stereo")
            assert sample["prompt"] == baseline["prompt"]
        before = mug[0]
        # A different parked left pose/gripper must not change model inputs or right targets.
        mug.cache[0][:, :3] += 0.005
        mug.cache[0][:, 9] = 0.001
        after = mug[0]
        for key in ["proprio", "action", "proprio_mask", "action_mask"]:
            assert torch.equal(before[key], after[key])


def main():
    # Direct execution otherwise puts tests/dataloader ahead of Orion's dataloader.
    sys.path[0] = str(Path(__file__).resolve().parents[1])
    import numpy as np
    import torch
    from safetensors import safe_open

    from openwam.dataloader.realrobot_v3 import V3Dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--model", choices=["wan", "stereo"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.assets
    if args.run is None:
        dataset = V3Dataset("cup_stacking", root / "data", root / "preparation", "stereo")
        assert (len(dataset.rows), len(dataset)) == (222, 29070)
        offset = 0
        for row in dataset.rows:
            count = len(row["anchors"])
            for index in sorted({offset, offset + count // 2, offset + count - 1}):
                sample = dataset[index]
                assert sample["action"].shape == (34, 80)
                assert sample["proprio"].shape == (1, 80)
                assert all(frame.size == (320, 384) for frame in sample["video"] + sample["right_video"])
                assert len(sample["video"]) == 17
                assert torch.isfinite(sample["action"]).all()
            offset += count
        sample = dataset[0]
        previous = root.parent / "realrobot-v3-posttrain-20261111/smoke/cup_stacking/sample.json"
        reference = json.loads(previous.read_text())
        assert reference["episode_index"] == sample["episode_index"]
        np.testing.assert_allclose(sample["proprio"][0], reference["normalized_proprio80"], atol=1e-6)
        np.testing.assert_allclose(sample["action"], reference["normalized_action80"], atol=1e-6)
        print("V3_H100_LOADER_PASSED 222 episodes 29070 windows; original sample parity", flush=True)
        return

    result = json.loads((args.run / "result.json").read_text())
    assert result["status"] == "complete" and result["smoke"]
    assert result["model"] == args.model and result["optimizer_steps"] == 10
    out = Path(result["output"])
    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [row["opt_step"] for row in metrics] == list(range(1, 11))
    for row in metrics:
        assert all(math.isfinite(row[key]) for key in ["loss_total", "loss_action", "loss_video", "grad_norm"])
    source = root / "midtrain" / args.model / "checkpoint_step_50101.safetensors"
    saved = out / "checkpoint_step_10.safetensors"
    changed = []
    with (
        safe_open(str(source), framework="pt", device="cpu") as before,
        safe_open(str(saved), framework="pt", device="cpu") as after,
    ):
        assert set(before.keys()) == set(after.keys())
        for key in after.keys():
            assert before.get_slice(key).get_shape() == after.get_slice(key).get_shape()
            tensor = after.get_tensor(key)
            assert torch.isfinite(tensor).all(), key
            if "action" in key and len(changed) < 3 and not torch.equal(before.get_tensor(key), tensor):
                changed.append(key)
    assert changed, "No action parameter changed after optimizer updates"
    with saved.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = dict(
        status="passed",
        model=args.model,
        optimizer_steps=10,
        checkpoint=str(saved),
        checkpoint_bytes=saved.stat().st_size,
        sha256=checksum,
        changed_action_parameters=changed,
        first_loss=metrics[0]["loss_total"],
        last_loss=metrics[-1]["loss_total"],
    )
    (args.run / "smoke-verified.json").write_text(json.dumps(receipt, indent=2))
    print("V3_H100_TRAINING_SMOKE_PASSED", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
