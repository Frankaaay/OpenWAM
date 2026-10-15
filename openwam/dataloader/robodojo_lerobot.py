"""Published RoboDojo v2.1 videos paired with the verified v3 EEF tables."""
from pathlib import Path
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.utils.video_io import decode_video_frames


class RoboDojoLeRobotSource:
    """Storage adapter only; the existing reader owns all action semantics."""

    CAMERA_KEYS = {
        'cam_head': 'observation.images.cam_high',
        'cam_left_wrist': 'observation.images.cam_left_wrist',
        'cam_right_wrist': 'observation.images.cam_right_wrist',
    }

    def __init__(self, root, eef_parquet_paths):
        self.root = Path(root)
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        if self.info['fps'] != 25 or self.info['features']['observation.state']['shape'] != [14]:
            raise ValueError('Expected the approved RoboDojo v2.1 joint14/25Hz source')
        self.episodes = [json.loads(line) for line in
                         (self.root / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
        if [e['episode_index'] for e in self.episodes] != list(range(self.info['total_episodes'])):
            raise ValueError('Episode metadata must cover the complete ordered release')
        self.lengths = np.array([e['length'] for e in self.episodes], dtype=np.int64)
        if (self.lengths < 2).any() or self.lengths.sum() != self.info['total_frames']:
            raise ValueError('Incomplete episode lengths')
        self.offsets = np.concatenate(([0], self.lengths.cumsum()))
        if not eef_parquet_paths:
            raise ValueError('Explicit verified EEF parquet files are required')
        columns = ['observation.state', 'frame_index', 'episode_index', 'index', 'timestamp', 'task_index']
        table = pa.concat_tables([pq.read_table(path, columns=columns) for path in eef_parquet_paths])
        if len(table) != int(self.offsets[-1]):
            raise ValueError('EEF tables do not cover every source frame')
        state = table['observation.state'].combine_chunks()
        if not pa.types.is_fixed_size_list(state.type) or state.type.list_size != 16:
            raise ValueError('EEF state must have 16 columns: dual xyz+wxyz+gripper')
        self.states = state.values.to_numpy().reshape(-1, 16)
        self.fields = {key: table[key].combine_chunks().to_numpy() for key in columns[1:]}
        if not np.isfinite(self.states).all():
            raise ValueError('Nonfinite EEF state')
        for quaternion in (self.states[:, 3:7], self.states[:, 11:15]):
            if not np.allclose(np.linalg.norm(quaternion, axis=1), 1., atol=1e-6, rtol=0):
                raise ValueError('Invalid EEF unit quaternion')
        for index, ep in enumerate(self.episodes):
            selection = slice(self.offsets[index], self.offsets[index + 1])
            if not np.all(self.fields['episode_index'][selection] == index):
                raise ValueError(f'EEF episode ordering mismatch: {index}')
            if not np.array_equal(self.fields['frame_index'][selection], np.arange(ep['length'])):
                raise ValueError(f'EEF frame ordering mismatch: {index}')
            if not np.allclose(self.fields['timestamp'][selection], np.arange(ep['length']) / 25,
                               atol=1e-4, rtol=0):
                raise ValueError(f'EEF timestamp mismatch: {index}')
            if len(np.unique(self.fields['task_index'][selection])) != 1:
                raise ValueError(f'EEF task changes within episode: {index}')
            if not ep['tasks'] or not all(isinstance(t, str) and t.strip() for t in ep['tasks']):
                raise ValueError(f'Missing instruction: {index}')
        self.tasks = sorted({str(int(self.fields['task_index'][start])) for start in self.offsets[:-1]})

    def data_path(self, index):
        return self.root / self.info['data_path'].format(
            episode_chunk=index // self.info['chunks_size'], episode_index=index)

    def read_states(self, index):
        """Validate release alignment before exposing achieved poses, never action."""
        native = pq.read_table(self.data_path(index), columns=[*self.fields, 'observation.state'])
        selection = slice(self.offsets[index], self.offsets[index + 1])
        for key, values in self.fields.items():
            if not np.array_equal(native[key].combine_chunks().to_numpy(), values[selection]):
                raise ValueError(f'v2.1/v3 mapping mismatch: episode {index}, {key}')
        raw = native['observation.state'].combine_chunks()
        joints = raw.values.to_numpy().reshape(-1, 14)
        states = self.states[selection]
        if not np.array_equal(joints[:, [6, 13]], states[:, [7, 15]]):
            raise ValueError(f'v2.1/v3 gripper mismatch: {index}')
        return {'state/left_ee_poses': states[:, :7],
                'state/right_ee_poses': states[:, 8:15],
                'state/left_ee_joint_states': states[:, 7:8],
                'state/right_ee_joint_states': states[:, 15:16]}

    def read_frames(self, index, frame_indices, cameras):
        frames = {}
        for camera in cameras:
            key = self.CAMERA_KEYS[camera]
            path = self.root / self.info['video_path'].format(
                episode_chunk=index // self.info['chunks_size'], episode_index=index, video_key=key)
            feature = self.info['features'][key]['info']
            frames[camera] = decode_video_frames(str(path), frame_indices,
                feature['video.height'], feature['video.width'])
        return frames
