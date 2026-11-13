"""Original v3 recorded-pose reader and preparation, with explicit task paths."""

import argparse
import json
import pathlib

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from dataloader.pretrain.action import dual_arm_tcp_target_relative_to_anchor

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.orion_midtrain import mosaic, scatter80


def targets(s, a):
    ix = torch.arange(a - 1, a + 33)
    valid = ix >= 0
    return s[a : a + 1], dual_arm_tcp_target_relative_to_anchor(s[a], s[ix.clamp_min(0)]), valid


def nearest(ts, grid):
    assert np.all(np.diff(ts) > 0), "Nonmonotonic native stream"
    hi = np.searchsorted(ts, grid).clip(0, len(ts) - 1)
    lo = (hi - 1).clip(0)
    ix = np.where(np.abs(ts[lo] - grid) <= np.abs(ts[hi] - grid), lo, hi)
    return ix, np.abs(ts[ix] - grid) <= 35000000


def rot6(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=1)
    assert np.all(np.abs(norm - 1) < 1e-3)
    x, y, z, w = (q / norm[:, None]).T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ],
        axis=1,
    )


def frames(path, indices):
    wanted = set(indices)
    out = {}
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        assert abs(float(stream.base_rate) - 10) < 1e-3
        c.seek(int(indices[0] / 10 / stream.time_base), stream=stream, backward=True)
        for f in c.decode(stream):
            i = round(float(f.pts * f.time_base) * 10)
            if i in wanted:
                assert i not in out
                out[i] = f.to_ndarray(format="rgb24")
            if i >= indices[-1]:
                break
    assert set(out) == wanted, (str(path), indices)
    return np.stack([out[i] for i in indices])


class V3Dataset(BaseDataset):
    action_dim = 80

    def __init__(self, task, data, prepared, mode="mono-left"):
        self.data = pathlib.Path(data)
        self.prepared = pathlib.Path(prepared)
        self.task = task
        self.mode = mode
        self.rows = json.loads((self.prepared / (task + "-catalog.json")).read_text())
        self.stats = json.loads((self.prepared / (task + "-stats.json")).read_text())
        self.tensors = {
            k: torch.tensor(self.stats[k], dtype=torch.float32)
            for k in ["state_mean", "state_std", "action_mean", "action_std"]
        }
        self.items = [(i, a) for i, r in enumerate(self.rows) for a in r["anchors"]]
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        i, a = self.items[index]
        r = self.rows[i]
        if i not in self.cache:
            if len(self.cache) >= 8:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = torch.from_numpy(np.load(self.prepared / r["states"], allow_pickle=False))
        state, action, valid = targets(self.cache[i], a)
        state = (state - self.tensors["state_mean"]) / self.tensors["state_std"]
        action = (action - self.tensors["action_mean"]) / self.tensors["action_std"]
        action, am = scatter80(action, valid[:, None].expand(-1, 20))
        state, sm = scatter80(state, torch.ones_like(state, dtype=torch.bool))
        views = {}
        for eye in ["left", "right"] if self.mode == "stereo" else ["left"]:
            views[eye] = mosaic(
                [
                    frames(self.data / r["videos"][mount + "_" + eye], list(range(a // 2, a // 2 + 17)))
                    for mount in ["head", "left_wrist", "right_wrist"]
                ]
            )
        return dict(
            video=views["left"],
            right_video=views.get("right"),
            first_frame_image=[views["left"][0]],
            vace_video=None,
            action=action,
            action_mask=am,
            proprio=state,
            proprio_mask=sm,
            video_mask=torch.ones(17, dtype=torch.bool),
            prompt="A video recorded from a robot's point of view executing the following instruction: "
            + r["instruction"],
            episode_index=r["episode_index"],
        )


def prepare(data, prepared, task, source_task):
    DATA = pathlib.Path(data)
    PREP = pathlib.Path(prepared)
    torch.set_num_threads(1)
    PREP.mkdir(exist_ok=False)
    info = json.loads((DATA / "meta/info.json").read_text())
    assert info["fps"] == 10
    rows = pq.read_table(DATA / "meta/episodes").to_pylist()
    rows.sort(key=lambda r: r["episode_index"])
    assert len(rows) == info["total_episodes"]
    desc = json.loads((DATA / "meta/task_descriptions.json").read_text())
    catalog = {task: []}
    selected = [r for r in rows if r["tasks"] == [source_task]]
    assert selected, "No episodes for selected source task"
    for row in selected:
        eid = row["episode_index"]
        table = pq.read_table(
            DATA / f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"
        )
        ts = np.asarray(table["timestamp_ns"].to_pylist(), dtype=np.int64)
        assert len(ts) == row["length"] and np.all(np.diff(ts) == 100000000)
        grid = ts[0] + np.arange(2 * len(ts) - 1, dtype=np.int64) * 50000000
        valid = np.ones(len(grid), dtype=bool)
        arms = []
        for side in ["left", "right"]:
            arrays = {}
            for kind in ["pose_state", "gripper__joint_state"]:
                native = pq.read_table(DATA / f"native/episode-{eid:06d}/arm__{side}__{kind}.parquet").to_pylist()
                nt = np.array([r["source_timestamp_ns"] for r in native], dtype=np.int64)
                ix, ok = nearest(nt, grid)
                valid &= ok
                payload = [json.loads(native[j]["payload_json"]) for j in ix]
                if kind == "pose_state":
                    xyz = [[v["pose"]["position"][k] for k in ["x", "y", "z"]] for v in payload]
                    q = [[v["pose"]["orientation"][k] for k in ["x", "y", "z", "w"]] for v in payload]
                    arrays["pose"] = np.concatenate([xyz, rot6(q)], axis=1)
                else:
                    arrays["g"] = np.array([v["position"][0] for v in payload])[:, None]
            arms.append(np.concatenate([arrays["pose"], arrays["g"]], axis=1))
        states = np.concatenate(arms, axis=1).astype(np.float32)
        assert np.isfinite(states).all()
        observed = np.array(table["observation.state"].to_pylist(), dtype=np.float32)
        assert np.allclose(states[::2], observed, atol=1e-5, rtol=1e-5), f"Native state parity failed {eid}"
        anchors = [a for a in range(0, len(states) - 32, 2) if valid[max(0, a - 1) : a + 33].all()]
        assert anchors, f"Zero valid windows {eid}"
        name = f"episode-{eid:06d}-states.npy"
        np.save(PREP / name, states, allow_pickle=False)
        videos = {}
        for mount in ["head", "left_wrist", "right_wrist"]:
            for eye in ["left", "right"]:
                k = f"videos/observation.images.cam_{mount}_{eye}"
                assert row[k + "/from_timestamp"] == 0
                videos[mount + "_" + eye] = (
                    f"{k}/chunk-{row[k + '/chunk_index']:03d}/file-{row[k + '/file_index']:03d}.mp4"
                )
        catalog[task].append(
            dict(
                episode_index=eid,
                states=name,
                anchors=anchors,
                videos=videos,
                instruction=desc[row["tasks"][0]]["en"],
                invalid_ticks=int((~valid).sum()),
            )
        )
        if eid % 25 == 0:
            print("PREPARED", eid, flush=True)
    for task, entries in catalog.items():
        (PREP / (task + "-catalog.json")).write_text(json.dumps(entries))
        counts = {k: 0 for k in ["state", "action"]}
        sums = {k: torch.zeros(20, dtype=torch.float64) for k in counts}
        sq = {k: torch.zeros(20, dtype=torch.float64) for k in counts}
        for r in entries:
            states = torch.from_numpy(np.load(PREP / r["states"], allow_pickle=False))
            for a in r["anchors"]:
                st, act, v = targets(states, a)
                for k, x in [("state", st), ("action", act[v])]:
                    x = x.double()
                    counts[k] += len(x)
                    sums[k] += x.sum(0)
                    sq[k] += (x * x).sum(0)
        stats = dict(
            episodes=len(entries),
            windows=sum(len(r["anchors"]) for r in entries),
            counts=counts,
            action_rate_hz=20,
            rgb_rate_hz=10,
            gripper_transform="empirical_zscore_no_clipping",
            supervision="recorded_pose_targets_not_joint_commands",
        )
        for k in counts:
            m = sums[k] / counts[k]
            sd = (sq[k] / counts[k] - m * m).clamp_min(0).sqrt().clamp_min(1e-6)
            stats[k + "_mean"] = m.tolist()
            stats[k + "_std"] = sd.tolist()
        (PREP / (task + "-stats.json")).write_text(json.dumps(stats, indent=2))
        ds = V3Dataset(task, DATA, PREP, "stereo")
        offset = 0
        for r in entries:
            for i in sorted({offset, offset + len(r["anchors"]) // 2, offset + len(r["anchors"]) - 1}):
                sample = ds[i]
                assert (
                    sample["action"].shape == (34, 80)
                    and len(sample["video"]) == 17
                    and len(sample["right_video"]) == 17
                )
                assert torch.isfinite(sample["action"]).all()
            offset += len(r["anchors"])
        print("TASK_LOADER_PASSED", task, stats["episodes"], stats["windows"], flush=True)
    (PREP / "result.json").write_text(
        json.dumps(
            dict(
                status="passed",
                episodes=len(selected),
                tasks={
                    t: dict(episodes=len(rs), windows=sum(len(r["anchors"]) for r in rs)) for t, rs in catalog.items()
                },
            )
        )
    )
    print("V3_PREPARATION_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source-task", required=True)
    args = parser.parse_args()
    prepare(args.data, args.prepared, args.task, args.source_task)
