"""DROID/ABC midtrain adapters using the pinned Orion state-target transforms.

ORION's ``src`` directory and StereoTok must be on PYTHONPATH. No recorded
commanded action is consumed; original source files remain read-only.
"""
from functools import lru_cache
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from PIL import Image
import torch

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.utils import get_cfg
from dataloader.pretrain.action import (
    dual_arm_tcp_target_relative_to_anchor,
    normalize_dual_arm_absolute_tcp,
    normalize_dual_arm_relative_tcp,
)
from dataloader.pretrain.canonical import ORIONPretrainDataset, interpolate_canonical_tcp
from dataloader.pretrain.windows import build_timeline_layout


LAYOUT = build_timeline_layout(((0, 16),), 2)
SLOTS = torch.tensor([*range(10), *range(34, 44)])
WINDOW_STRIDE = 8


class SparseABCOrionDataset(ORIONPretrainDataset):
    """Keep Orion's physical timeline while spacing episode anchors by 0.8s."""

    def _locate_window(self, index):
        episode, window = super()._locate_window(index)
        return episode, window * WINDOW_STRIDE


def scatter80(value, valid):
    result = torch.zeros((*value.shape[:-1], 80), dtype=value.dtype)
    mask = torch.zeros_like(result, dtype=torch.bool)
    result[..., SLOTS] = value * valid.to(value.dtype)
    mask[..., SLOTS] = valid
    return result, mask


def mosaic(views):
    """Real stereo views only; absent auxiliary tiles are explicitly black."""
    frames = []
    for t in range(17):
        canvas = Image.new('RGB', (320, 384))
        for images, size, xy in zip(views, [(320, 256), (160, 128), (160, 128)], [(0, 0), (0, 256), (160, 256)]):
            canvas.paste(Image.fromarray(images[t]).resize(size, Image.Resampling.BILINEAR), xy)
        frames.append(canvas)
    return frames


def abc_orion(root, stats_path, split):
    """Retain published source splits, then select the actual head stereo subset."""
    import pyarrow.parquet as pq
    root = Path(root)
    published = {}
    for path in sorted((root / 'meta/episodes/chunk-000').glob('*.parquet')):
        for row in pq.read_table(path, columns=['episode_index', 'source_split']).to_pylist():
            published[int(row['episode_index'])] = row['source_split']
    dataset = SparseABCOrionDataset(
        dataset_dirs=[dict(name='abc130k', path=str(root),
                           mask_and_mapping_path=str(root / 'dataset_configs/masks/mask_and_mapping_abc130k.json'))],
        normalization_stats_path=stats_path, rgb_rate_hz=10, action_steps_per_rgb_frame=2,
        frame_ranges=((0, 16),), source_phase_offsets=(0, 1, 2), validation_split=0, split='train',
    )
    before = len(dataset._episodes)
    # Orion can reject invalid task/short episodes during construction. Report,
    # rather than silently redefining the accepted publication.
    if before != len(published):
        raise ValueError(f'ABC Orion admission changed episode population: {before}/{len(published)}')
    requested = 'val' if split == 'val' else 'train'
    dataset._episodes = [e for e in dataset._episodes if published[e['episode_index']] == requested]
    dataset._episode_window_ends = np.cumsum([e['rgb_target_length'] for e in dataset._episodes]).tolist()
    dataset._length = dataset._episode_window_ends[-1] if dataset._episodes else 0
    dataset.require_camera_views((0, 1))
    dataset.select_camera_reads((0, 1))
    dataset._episode_window_ends = np.cumsum([
        (e['rgb_target_length'] + WINDOW_STRIDE - 1) // WINDOW_STRIDE
        for e in dataset._episodes]).tolist()
    dataset._length = dataset._episode_window_ends[-1] if dataset._episodes else 0
    if not len(dataset):
        raise ValueError(f'No ABC stereo windows in {split}')
    return dataset


class ABCOrionDataset(BaseDataset):
    action_dim = 80

    def __init__(self, root, stats_path, split, input_mode='stereo'):
        if input_mode not in ('stereo', 'mono-left'):
            raise ValueError(f'Unsupported input_mode: {input_mode}')
        self.input_mode = input_mode
        self._dataset_dir = Path(root)
        if stats_path and json.loads(Path(stats_path).read_text())['statistics'].get('window_stride_rgb') != WINDOW_STRIDE:
            raise ValueError('ABC normalization does not match the 0.8s window contract')
        self.dataset = abc_orion(root, stats_path, split)
        self._eps_df = pd.DataFrame({'episode_index': [e['episode_index'] for e in self.dataset._episodes]})
        self._cum_n_starts = np.array([0, *self.dataset._episode_window_ends], dtype=np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        raw = sample['action'].reshape(-1, 128)[:, :20]
        valid = sample['action_valid'].reshape(-1, 1) & sample['action_feature_mask'][:20]
        action, action_mask = scatter80(raw, valid)
        # Orion's first frame has substeps [-1,0]; substep1 is the anchor.
        proprio, proprio_mask = scatter80(sample['state'][0, 1, :20].unsqueeze(0),
                                          sample['state_feature_mask'][:20].unsqueeze(0))
        video = sample['video'].permute(1, 0, 3, 4, 2).numpy()
        left, right = mosaic([video[0]]), mosaic([video[1]])
        return dict(video=left, right_video=right if self.input_mode == 'stereo' else None, first_frame_image=[left[0]], vace_video=None,
                    action=action, action_mask=action_mask, proprio=proprio, proprio_mask=proprio_mask,
                    video_mask=sample['frame_valid'], prompt=sample['prompt'],
                    episode_index=sample['data_info']['episode_index'])

    @classmethod
    def from_config(cls, cfg, split='train'):
        return cls(get_cfg(cfg, 'dataset_dir'), get_cfg(cfg, 'stats_path'), split, get_cfg(cfg, 'input_mode', 'stereo'))


@lru_cache(maxsize=8)
def droid_states(episode_dir, trajectory_h5=None):
    """Keep the recorded robot-model EE frame; do not rename it to a TCP."""
    import h5py
    from scipy.spatial.transform import Rotation
    root = Path(episode_dir)
    metadata = json.loads((root / 'result.json').read_text())
    with h5py.File(trajectory_h5 or metadata['trajectory_h5'], 'r') as f:
        pose = np.asarray(f['observation/robot_state/cartesian_position'])
        closedness = np.asarray(f['observation/robot_state/gripper_position'])
        start = np.asarray(f['observation/timestamp/robot_state/read_start'], dtype=np.int64)
        end = np.asarray(f['observation/timestamp/robot_state/read_end'], dtype=np.int64)
    if pose.shape != (len(start), 6) or closedness.shape != (len(start),):
        raise ValueError('DROID state shape mismatch')
    if not np.isfinite(pose).all() or not np.isfinite(closedness).all():
        raise ValueError('DROID nonfinite state')
    if not (np.diff(start) > 0).all() or not (end >= start).all():
        raise ValueError('DROID invalid state read timestamps')
    if closedness.min() < -1e-4 or closedness.max() > 1.0001:
        raise ValueError('DROID measured gripper outside calibrated [0,1]')
    times = (start + end) * 500_000  # host milliseconds -> midpoint nanoseconds
    state = np.zeros((len(start), 2, 10), dtype=np.float32)
    state[:, 0, :3] = pose[:, :3]
    state[:, 0, 3:9] = Rotation.from_euler('xyz', pose[:, 3:]).as_matrix()[:, :2].reshape(-1, 6)
    state[:, 0, 9] = 1 - closedness
    state[:, 1, 3:9] = [1, 0, 0, 0, 1, 0]  # finite placeholder, never supervised
    return times, torch.from_numpy(state), metadata


def droid_targets(record):
    root = Path(record['table_root'])
    episode_dir = (root / record['camera_specs'][0]['lance_path']).parent.parent
    times, states, metadata = droid_states(str(episode_dir), record.get('trajectory_h5'))
    anchor_ns = int(record['target_ns'][0])
    offsets = LAYOUT.action_step_offsets.reshape(-1).numpy() * 50_000_000
    targets = anchor_ns + np.concatenate(([0], offsets))
    if not times[0] <= anchor_ns <= times[-1]:
        raise ValueError(f'DROID RGB anchor outside observed state timestamps: {metadata["id"]}')
    valid = (targets >= times[0]) & (targets <= times[-1])
    clamped = np.clip(targets, times[0], times[-1])
    hi = np.searchsorted(times, clamped, side='left').clip(0, len(times)-1)
    lo = np.maximum(hi-1, 0)
    fraction = np.divide(clamped-times[lo], times[hi]-times[lo],
                         out=np.zeros(len(targets)), where=times[hi]!=times[lo])
    grid = interpolate_canonical_tcp(states[lo], states[hi], torch.tensor(fraction, dtype=torch.float32))
    relative = dual_arm_tcp_target_relative_to_anchor(grid[0].reshape(1, 20), grid[1:].reshape(-1, 20))
    mask = torch.zeros((34, 20), dtype=torch.bool)
    mask[:, :10] = torch.from_numpy(valid[1:]).unsqueeze(-1)
    if not mask.any():
        raise ValueError('DROID window has no valid action targets')
    return grid, relative, mask, metadata


class DROIDOrionDataset(BaseDataset):
    action_dim = 80

    def __init__(self, index_path, stats_path, split, input_mode='stereo'):
        if input_mode not in ('stereo', 'mono-left'):
            raise ValueError(f'Unsupported input_mode: {input_mode}')
        self.input_mode = input_mode
        self.index_path = str(index_path)
        self._dataset_dir = Path(index_path).parent / 'droid'
        self.stats = json.loads(Path(stats_path).read_text()) if stats_path else None
        if self.stats and self.stats['statistics'].get('window_stride_s') != 0.8:
            raise ValueError('DROID normalization does not match the 0.8s window contract')
        with closing(sqlite3.connect(f'file:{self.index_path}?mode=ro', uri=True)) as con:
            rows = con.execute('select id, episode from windows where split=? order by episode,id', (split,)).fetchall()
        if not rows:
            raise ValueError(f'No DROID windows in {split}')
        self.ids = np.array([r[0] for r in rows], dtype=np.int64)
        episodes, counts = np.unique([r[1] for r in rows], return_counts=True)
        self._eps_df = pd.DataFrame({'episode_index': episodes})
        self._cum_n_starts = np.concatenate(([0], np.cumsum(counts)))
        self._connection = self._reader = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_connection=None, _reader=None)
        return state

    def __len__(self):
        return len(self.ids)

    def record(self, index):
        if self._connection is None:
            self._connection = sqlite3.connect(f'file:{self.index_path}?mode=ro', uri=True)
        return json.loads(self._connection.execute('select payload from windows where id=?', (int(self.ids[index]),)).fetchone()[0])

    def __getitem__(self, index):
        from stereotok.data.lance_reader import LanceReader
        record = self.record(index)
        grid, relative, valid, metadata = droid_targets(record)
        if self.stats is None:
            raise ValueError('DROID statistics must be prepared before training')
        s = {k: torch.tensor(v, dtype=torch.float32) for k, v in self.stats.items() if k in (
            'state_tcp_mean', 'state_tcp_std', 'action_tcp_mean', 'action_tcp_std',
            'gripper_open_value', 'gripper_closed_value')}
        action = normalize_dual_arm_relative_tcp(relative, s['action_tcp_mean'], s['action_tcp_std'],
                                                s['gripper_open_value'], s['gripper_closed_value'])
        proprio = normalize_dual_arm_absolute_tcp(grid[0].reshape(1, 20), s['state_tcp_mean'], s['state_tcp_std'],
                                                  s['gripper_open_value'], s['gripper_closed_value'])
        state_valid = torch.zeros((1, 20), dtype=torch.bool)
        state_valid[:, :10] = True
        action, action_mask = scatter80(action, valid)
        proprio, proprio_mask = scatter80(proprio, state_valid)
        if self._reader is None:
            self._reader = LanceReader()
        frames, _, _ = self._reader.read_droid(record)
        # Source order wrist/ext1/ext2 -> canvas head/ext1, wrist, ext2.
        left = mosaic([frames[i, 0] for i in (1, 0, 2)])
        right = mosaic([frames[i, 1] for i in (1, 0, 2)])
        episode_dir = (Path(record['table_root']) / record['camera_specs'][0]['lance_path']).parent.parent
        source_meta = json.loads((episode_dir / 'source_metadata.json').read_text())
        task = source_meta.get('current_task')
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f'DROID missing instruction: {metadata["id"]}')
        return dict(video=left, right_video=right if self.input_mode == 'stereo' else None, first_frame_image=[left[0]], vace_video=None,
                    action=action, action_mask=action_mask, proprio=proprio, proprio_mask=proprio_mask,
                    video_mask=torch.ones(17, dtype=torch.bool),
                    prompt="A video recorded from a robot's point of view executing the following instruction: " + task)

    @classmethod
    def from_config(cls, cfg, split='train'):
        return cls(get_cfg(cfg, 'index_path'), get_cfg(cfg, 'stats_path'), split, get_cfg(cfg, 'input_mode', 'stereo'))
