"""Acceptance checks for the migrated cup dataset and a ten-update smoke."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

# Direct execution otherwise puts tests/dataloader ahead of Orion's dataloader.
sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np
import torch
from safetensors import safe_open

from openwam.dataloader.realrobot_v3 import V3Dataset


def main():
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
