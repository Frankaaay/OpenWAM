"""Validate the published storage adapter, shared stats and real video windows."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from openwam.dataloader.robodojo import MultiTaskRoboDojoDataset, read_calibrated_eef20
from openwam.dataloader.utils.stats_computation.robodojo_stats_computation import compute_robodojo_stats


def probe(root, eef_paths, output):
    start = time.time()
    output = Path(output)
    stats_path = output / 'openwam-robodojo-production-reader-stats-probe.npy'
    if stats_path.exists():
        raise FileExistsError(stats_path)
    stats = compute_robodojo_stats(root, lerobot_eef_paths=eef_paths)
    with stats_path.open('xb') as handle:
        np.save(handle, stats, allow_pickle=True)
    dataset = MultiTaskRoboDojoDataset(root, lerobot_eef_paths=eef_paths,
        normalization_stats_path=stats_path, unify_action_map=['0-9', '34-43'])
    reader = dataset._sub_datasets[0]
    source = reader._lerobot_source
    assert len(reader) == sum(source.lengths - 1)
    checks = []
    for index in (0, len(reader) // 2, len(reader) - 1):
        sample = reader[index]
        ep, offset = reader._window_index[index]
        raw = read_calibrated_eef20(source.read_states(ep), reader.calibration)
        restored = reader.denormalize_action(sample['action'].numpy())
        valid = min(reader.num_action_steps, len(raw) - offset - 1)
        error = float(np.abs(restored[:valid] - raw[offset + 1:offset + 1 + valid]).max())
        assert error < 1e-6, error
        assert sample['action_mask'].sum().item() == valid * 20
        assert torch.isfinite(sample['action']).all()
        assert len(sample['video']) == 9
        assert all(image.size == (320, 384) for image in sample['video'])
        checks.append(dict(window=index, episode=ep, valid_actions=valid, roundtrip_max_abs=error))
    report = dict(episodes=len(source.episodes), frames=int(source.offsets[-1]), windows=len(reader),
                  state_rows=stats['metadata']['state_rows'], action_rows=stats['metadata']['action_rows'],
                  all_numeric_mapping_verified=True, sampled_windows=checks,
                  stats_path=str(stats_path), full_video_decode=False, training=False,
                  seconds=time.time() - start)
    print(json.dumps(report), flush=True)
    return report
