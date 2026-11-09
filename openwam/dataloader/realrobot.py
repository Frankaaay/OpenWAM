"""Recorded cart-007 pose targets, with empirical raw-gripper normalization."""
import argparse
from bisect import bisect_right
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.orion_midtrain import mosaic, scatter80
from openwam.dataloader.utils import get_cfg
from dataloader.pretrain.action import dual_arm_tcp_target_relative_to_anchor


CONTRACT = 'cart007.recorded_pose.raw_gripper.v1'


@lru_cache(maxsize=8)
def episode_states(path):
    value = torch.tensor(pq.read_table(path, columns=['observation.state'])['observation.state'].to_pylist(), dtype=torch.float32)
    if value.ndim != 2 or value.shape[1] != 20 or not torch.isfinite(value).all():
        raise ValueError('Invalid recorded State20')
    return value


def window_targets(states, anchor):
    indices = torch.arange(anchor-1, anchor+33)
    valid = indices >= 0
    target = states[indices.clamp_min(0)]
    relative = dual_arm_tcp_target_relative_to_anchor(states[anchor], target)
    return states[anchor:anchor+1], relative, valid


def load_catalog(root):
    root = Path(root)
    info = json.loads((root/'meta/info.json').read_text())
    if info['ngad_extensions']['schema'] != CONTRACT or info['fps'] != 20:
        raise ValueError('Unexpected realrobot pose/gripper/timeline contract')
    rows = pq.read_table(root/'meta/episodes').to_pylist()
    rows.sort(key=lambda row: row['episode_index'])
    if len(rows) != info['total_episodes'] or sum(r['length'] for r in rows) != info['total_frames']:
        raise ValueError('Episode metadata totals mismatch')
    if [r['episode_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Noncontiguous episode identity')
    return info, rows


def data_path(root, row):
    return str(Path(root)/f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet")


def compute_stats(root, output):
    info, rows = load_catalog(root)
    # Exact population moments over the same 0.1s windows and valid action
    # substeps consumed by this reader. Gripper stays a scalar recorded target.
    counts = {'state': 0, 'action': 0}
    sums = {k: torch.zeros(20, dtype=torch.float64) for k in counts}
    squares = {k: torch.zeros(20, dtype=torch.float64) for k in counts}
    windows = 0
    for row in rows:
        states = episode_states(data_path(root, row))
        if len(states) != row['length']:
            raise ValueError('State/metadata length mismatch')
        for anchor in range(0, len(states)-32, 2):
            state, action, valid = window_targets(states, anchor)
            for key, value in [('state', state), ('action', action[valid])]:
                value = value.double()
                counts[key] += len(value)
                sums[key] += value.sum(0)
                squares[key] += value.square().sum(0)
            windows += 1
    result = dict(schema=CONTRACT, rgb_rate_hz=10, action_rate_hz=20, window_stride_seconds=0.1,
                  window_count=windows, episode_count=len(rows), counts=counts,
                  info_sha256=hashlib.sha256((Path(root)/'meta/info.json').read_bytes()).hexdigest(),
                  gripper_transform='empirical_zscore_no_clipping_no_physical_endpoints')
    for key in counts:
        if not counts[key]:
            raise ValueError('Empty statistics population')
        mean = sums[key] / counts[key]
        std = (squares[key] / counts[key] - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
        result[key+'_mean'], result[key+'_std'] = mean.tolist(), std.tolist()
    with Path(output).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return result


def read_view(path, indices):
    wanted = set(indices)
    frames = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if abs(float(stream.base_rate)-20) > 1e-3:
            raise ValueError('Video FPS mismatch')
        container.seek(int(indices[0]/20/stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            index = round(float(frame.pts * frame.time_base)*20)
            if index in wanted:
                if index in frames:
                    raise ValueError('Duplicate video timestamp')
                frames[index] = frame.to_ndarray(format='rgb24')
            if index >= indices[-1]:
                break
    if set(frames) != wanted:
        raise ValueError('Missing requested video frames')
    return np.stack([frames[i] for i in indices])


class RealRobotDataset(BaseDataset):
    action_dim = 80

    def __init__(self, dataset_dir, stats_path, split, input_mode='mono-left'):
        if split != 'train':
            raise ValueError('This all-data smoke reader has no held-out split')
        if input_mode not in ('mono-left', 'stereo'):
            raise ValueError(input_mode)
        self.input_mode = input_mode
        self._dataset_dir = Path(dataset_dir)
        self.info, self.rows = load_catalog(dataset_dir)
        stats = json.loads(Path(stats_path).read_text())
        digest = hashlib.sha256((self._dataset_dir/'meta/info.json').read_bytes()).hexdigest()
        if stats['schema'] != CONTRACT or stats['info_sha256'] != digest or stats['window_stride_seconds'] != 0.1:
            raise ValueError('Normalization does not match dataset/window contract')
        self.stats = {k: torch.tensor(stats[k], dtype=torch.float32) for k in ('state_mean','state_std','action_mean','action_std')}
        if any(v.shape != (20,) or not torch.isfinite(v).all() for v in self.stats.values()):
            raise ValueError('Invalid normalization statistics')
        if any((self.stats[k] <= 0).any() for k in ('state_std','action_std')):
            raise ValueError('Nonpositive normalization standard deviation')
        counts = [len(range(0, row['length']-32, 2)) for row in self.rows]
        if any(n <= 0 for n in counts):
            raise ValueError('Episode without complete RGB window')
        self._cum_n_starts = np.cumsum([0, *counts])
        self._eps_df = pd.DataFrame({'episode_index': [r['episode_index'] for r in self.rows]})
        if len(self) != stats['window_count']:
            raise ValueError('Normalization population mismatch')

    def __len__(self):
        return int(self._cum_n_starts[-1])

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = bisect_right(self._cum_n_starts, index)-1
        row = self.rows[episode]
        anchor = int(index-self._cum_n_starts[episode])*2
        states = episode_states(data_path(self._dataset_dir, row))
        proprio, action, valid = window_targets(states, anchor)
        action = (action-self.stats['action_mean'])/self.stats['action_std']
        proprio = (proprio-self.stats['state_mean'])/self.stats['state_std']
        action, action_mask = scatter80(action, valid[:, None].expand(-1, 20))
        proprio, proprio_mask = scatter80(proprio, torch.ones_like(proprio, dtype=torch.bool))
        views = {}
        for eye in (('left', 'right') if self.input_mode == 'stereo' else ('left',)):
            images = []
            for mount in ('head', 'left_wrist', 'right_wrist'):
                key = f'videos/observation.images.cam_{mount}_{eye}'
                path = self._dataset_dir/f"{key}/chunk-{row[key+'/chunk_index']:03d}/file-{row[key+'/file_index']:03d}.mp4"
                images.append(read_view(path, list(range(anchor, anchor+33, 2))))
            views[eye] = mosaic(images)
        return dict(video=views['left'], right_video=views.get('right'), first_frame_image=[views['left'][0]],
                    vace_video=None, action=action, action_mask=action_mask, proprio=proprio, proprio_mask=proprio_mask,
                    video_mask=torch.ones(17, dtype=torch.bool),
                    prompt="A video recorded from a robot's point of view executing the following instruction: "+row['tasks'][0],
                    episode_index=row['episode_index'])

    @classmethod
    def from_config(cls, cfg, split='train'):
        return cls(get_cfg(cfg,'dataset_dir'),get_cfg(cfg,'stats_path'),split,get_cfg(cfg,'input_mode','mono-left'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--stats-path', required=True)
    args = parser.parse_args()
    stats = compute_stats(args.dataset_dir, args.stats_path)
    dataset = RealRobotDataset(args.dataset_dir, args.stats_path, 'train')
    for start, end in zip(dataset._cum_n_starts[:-1], dataset._cum_n_starts[1:]):
        for i in sorted({int(start), int((start+end-1)//2), int(end-1)}):
            sample = dataset[i]
            if sample['action'].shape != (34,80) or sample['proprio'].shape != (1,80) or len(sample['video']) != 17:
                raise ValueError('Training sample shape mismatch')
            if not torch.isfinite(sample['action']).all() or not torch.isfinite(sample['proprio']).all():
                raise ValueError('Nonfinite normalized training sample')
    print(json.dumps(dict(loader_verified=True, windows=len(dataset), episodes=len(dataset.rows))), flush=True)
