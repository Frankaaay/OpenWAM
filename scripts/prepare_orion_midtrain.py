"""Prepare exact admitted DROID windows and Orion normalization for midtrain."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import shutil
import hashlib

import ijson
import numpy as np
import torch

from dataloader.compute_norm_stats import RunningMoments
from dataloader.pretrain.canonical import NGAD_CANONICAL_SCHEMA, CANONICAL_STATE_KEY, interpolate_canonical_tcp
from dataloader.pretrain.action import dual_arm_tcp_target_relative_to_anchor
from openwam.dataloader.orion_midtrain import abc_orion, droid_targets, DROIDOrionDataset, WINDOW_STRIDE


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def droid_episode_windows(record):
    """Reindex existing full-trajectory 10Hz Lance rows; never decode new video."""
    maps = []
    for camera in record['camera_specs']:
        path = Path(record['table_root']) / camera['sampling_path']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != camera['sampling_sha256']:
            raise ValueError(f'DROID sampling metadata changed: {path}')
        rows = json.loads(data)
        mapping = {int(row['target_ns']): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f'Duplicate DROID target timestamps: {path}')
        maps.append(mapping)
    first = int(record['target_ns'][0])
    last = min(max(mapping) for mapping in maps) - 1_600_000_000
    admitted, excluded = [], []
    for anchor in range(first, last + 1, WINDOW_STRIDE * 100_000_000):
        targets = [anchor + t * 100_000_000 for t in range(17)]
        cameras, stamps, reason = [], [], None
        for camera, mapping in zip(record['camera_specs'], maps):
            if any(t not in mapping for t in targets):
                reason = 'missing_camera_target'; break
            selected = [mapping[t] for t in targets]
            capture = np.array([r['capture_ns'] for r in selected], dtype=np.int64)
            source = np.array([r['source_frame'] for r in selected], dtype=np.int64)
            if (not (np.diff(capture) > 0).all() or not (np.diff(source) > 0).all()
                    or np.any(np.abs(capture - targets) > record['time_tolerance_ns'])):
                reason = 'camera_identity_or_time_tolerance'; break
            cameras.append(dict(camera, rows=[r['row'] for r in selected],
                                source_frames=source.tolist(), capture_ns=capture.tolist()))
            stamps.append(capture)
        if reason is None and np.any(np.ptp(stamps, axis=0) > record['cross_camera_tolerance_ns']):
            reason = 'cross_camera_time_tolerance'
        if reason:
            excluded.append(dict(episode=record['source_episode_id'], split=record['split'],
                                 anchor_ns=anchor, reason=reason))
            continue
        value = dict(record, target_ns=targets, camera_specs=cameras, start_stride=WINDOW_STRIDE,
                     episode_id=record['episode_id'].split('@')[0] + '@' + str(anchor))
        admitted.append(json.dumps(value))
    return admitted, excluded


def index_droid(manifest, audit_dir, output):
    summary = json.loads((audit_dir / 'droid-action-audit-summary.json').read_text())
    if not summary['full_action_schema_audited']:
        raise RuntimeError('DROID audit incomplete')
    audit_path = Path(summary.get('audit_path', audit_dir / 'droid-action-audit.jsonl'))
    audits = {r['id']: r for r in map(json.loads, audit_path.read_text().splitlines())}
    recovered = {r['id']: r for r in map(json.loads, (audit_dir / 'droid-state-recovery.jsonl').read_text().splitlines())}
    unresolved = [key for key, value in audits.items() if not value['ok'] and recovered.get(key, {}).get('status') not in ('recovered', 'exclude_missing_state')]
    if unresolved:
        raise RuntimeError(f'Unresolved DROID audit errors: {unresolved}')
    path = output / 'droid.sqlite'
    if path.exists():
        raise FileExistsError(path)
    counts, excluded, episode_ids, templates = Counter(), {}, {}, {}
    con = sqlite3.connect(path)
    con.execute('create table windows (id integer primary key, episode integer not null, split text not null, payload text not null)')
    with manifest.open('rb') as stream:
        for record in ijson.items(stream, 'episodes.item', use_float=True):
            if record.get('source') != 'droid':
                continue
            key = record['source_episode_id']
            audit = audits[key]
            recovery = recovered.get(key, {})
            if recovery.get('status') == 'exclude_missing_state':
                entry = excluded.setdefault(key, dict(recovery, windows=Counter()))
                entry['windows'][record['split']] += 1
                continue
            record['trajectory_h5'] = audit['trajectory'] if audit['ok'] else recovery['trajectory_h5']
            if key not in templates:
                templates[key] = record
    rejected_counts = Counter()
    with (output / 'droid-camera-window-exclusions.jsonl').open('x') as receipts, ProcessPoolExecutor(max_workers=8) as pool:
        for i, (record, (payloads, rejected)) in enumerate(zip(templates.values(), pool.map(droid_episode_windows, templates.values(), chunksize=8)), 1):
            episode = episode_ids.setdefault(record['source_episode_id'], len(episode_ids))
            con.executemany('insert into windows(episode,split,payload) values(?,?,?)',
                            ((episode, record['split'], payload) for payload in payloads))
            counts[record['split']] += len(payloads)
            for row in rejected:
                receipts.write(json.dumps(row) + '\n')
                rejected_counts[row['reason']] += 1
            if i % 500 == 0:
                con.commit()
                receipts.flush()
                print('DROID_INDEX', i, len(templates), dict(counts), 'rejected', dict(rejected_counts), flush=True)
    con.execute('create index windows_split_episode on windows(split,episode,id)')
    con.commit()
    con.close()
    write_json(output / 'droid-admission.json', dict(windows=counts, episodes=len(episode_ids), excluded=list(excluded.values()), original_windows=summary['windows'], window_stride_s=0.6, camera_rejected=rejected_counts))


def _droid_target_batch(indices):
    states, actions, masks, excluded = [], [], [], []
    for i in indices:
        record = _stats_dataset.record(i)
        try:
            grid, relative, valid, _ = droid_targets(record)
        except ValueError as exc:
            if not str(exc).startswith('DROID RGB anchor outside observed state timestamps:'):
                raise
            excluded.append(dict(window_id=int(_stats_dataset.ids[i]),
                                 episode=record['source_episode_id'],
                                 anchor_ns=int(record['target_ns'][0]), reason=str(exc)))
            continue
        states.append(grid[1:, :, :9].numpy())
        actions.append(relative.reshape(-1, 2, 10)[..., :9].numpy())
        masks.append(valid.reshape(-1, 2, 10)[..., :9].numpy())
    return ((np.stack(states), np.stack(actions), np.stack(masks)) if states else None), excluded


def droid_stats(output):
    global _stats_dataset
    _stats_dataset = dataset = DROIDOrionDataset(output / 'droid.sqlite', None, 'train')
    state, action = RunningMoments((2, 9)), RunningMoments((2, 9))
    required = torch.zeros((2, 9), dtype=torch.bool)
    required[0] = True
    # Separate processes avoid h5py's per-process global lock. Parent still
    # reduces every accepted target in deterministic index order using Orion.
    excluded_ids, excluded_counts = [], Counter()
    with (output / 'droid-window-exclusions.jsonl').open('x') as receipts:
        for split in ('train', 'val', 'test'):
            _stats_dataset = dataset = DROIDOrionDataset(output / 'droid.sqlite', None, split)
            batches = [range(i, min(i + 128, len(dataset))) for i in range(0, len(dataset), 128)]
            with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('fork')) as pool:
                for i, (arrays, excluded) in enumerate(pool.map(_droid_target_batch, batches), 1):
                    if arrays is not None and split == 'train':
                        absolute, relative, mask = arrays
                        state.update(torch.from_numpy(absolute), torch.from_numpy(mask))
                        action.update(torch.from_numpy(relative), torch.from_numpy(mask))
                    for row in excluded:
                        row['split'] = split
                        receipts.write(json.dumps(row) + '\n')
                        excluded_ids.append((row['window_id'],))
                        excluded_counts[split] += 1
                    if i % 8 == 0:
                        receipts.flush()
                        print('DROID_STATS', split, min(i * 128, len(dataset)), len(dataset), 'excluded', dict(excluded_counts), flush=True)
            if split == 'train':
                train_windows = len(dataset) - excluded_counts[split]
    # Preserve the original admitted index before removing explicitly receipted
    # windows whose current observation has no measured-state reference.
    backup = output / 'droid-before-window-filter.sqlite'
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copyfile(output / 'droid.sqlite', backup)
    with sqlite3.connect(output / 'droid.sqlite') as con:
        con.executemany('delete from windows where id=?', excluded_ids)
        counts = dict(con.execute('select split,count(*) from windows group by split'))
        episodes = con.execute('select count(distinct episode) from windows').fetchone()[0]
    admission_path = output / 'droid-admission.json'
    admission = json.loads(admission_path.read_text())
    write_json(output / 'droid-admission-before-window-filter.json', admission)
    admission.update(windows=counts, episodes=episodes, excluded_windows=dict(excluded_counts),
                     window_exclusions=str(output / 'droid-window-exclusions.jsonl'))
    admission_path.write_text(json.dumps(admission, indent=2))
    result = dict(schema_version=NGAD_CANONICAL_SCHEMA, gripper_open_value=[1., 1.], gripper_closed_value=[0., 0.],
                  statistics=dict(method='population_zscore_parallel_welford', train_windows=train_windows, rgb_rate_hz=10, action_rate_hz=20, window_stride_s=0.6))
    for name, moments in [('state', state), ('action', action)]:
        stats = moments.result(1e-5, required)
        for field in ('mean', 'std', 'min', 'max'):
            result[f'{name}_tcp_{field}'] = stats[field].tolist()
        result['statistics'][name + '_count'] = stats['count'].tolist()
    write_json(output / 'droid-stats.json', result)


# Orion 783b602d compute_zscore_statistics; only anchor selection is adapted.
def compute_zscore_statistics(
    dataset,
    *,
    std_floor: float = 1.0e-5,
    anchor_batch_size: int = 2048,
    episode_start: int = 0,
    episode_stop: int | None = None,
    max_episodes: int | None = None,
    max_anchors: int | None = None,
    progress_every: int = 100,
) -> dict:
    """按所有有效 anchor × horizon target 精确累计总体均值和方差。"""
    if std_floor <= 0 or anchor_batch_size <= 0:
        raise ValueError("std_floor and anchor_batch_size must be positive.")
    if episode_start < 0:
        raise ValueError("episode_start must be non-negative.")
    resolved_stop = len(dataset._episodes) if episode_stop is None else episode_stop
    if resolved_stop <= episode_start or resolved_stop > len(dataset._episodes):
        raise ValueError(
            "episode_stop must be greater than episode_start and no larger than "
            "the number of Dataset episodes."
        )
    if max_episodes is not None:
        resolved_stop = min(resolved_stop, episode_start + max_episodes)
    state_moments = RunningMoments((2, 9))
    action_moments = RunningMoments((2, 9))
    action_offsets = dataset.timeline_layout.action_step_offsets.reshape(-1)
    total_anchors = 0
    total_valid_targets = 0
    processed_episodes = 0
    required_state_features = torch.zeros((2, 9), dtype=torch.bool)

    for episode_number, episode in enumerate(
        dataset._episodes[episode_start:resolved_stop],
        start=episode_start,
    ):
        if max_anchors is not None and total_anchors >= max_anchors:
            break
        meta = dataset._root_meta[episode["root_index"]]
        state_feature_mask = meta["state_element_mask"].reshape(2, 10)[..., :9]
        required_state_features |= state_feature_mask

        source_indices = torch.arange(episode["length"], dtype=torch.long)
        state_only_mask = {key: False for key in meta["field_mask"]}
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            state_only_mask[key] = True
        state_only_mask[CANONICAL_STATE_KEY] = True
        rows = meta["table_backend"].read_rows(
            episode,
            source_indices,
            state_only_mask,
            meta["field_mapping"],
            dataset.camera_keys,
            torch.zeros_like(meta["camera_mask"]),
        )
        source_state = dataset._reshape_state_window(torch.tensor(
            [rows[index][CANONICAL_STATE_KEY] for index in range(episode["length"])],
            dtype=torch.float32,
        ))
        # 先插值到 action rate；State 和 Action 必须复用同一份绝对位姿网格。
        target_grid = torch.arange(episode["action_target_length"], dtype=torch.long)
        absolute_grids = []
        for phase_offset in dataset.source_phase_offsets:
            lower, upper, fraction = dataset._state_interpolation_indices(
                target_grid,
                meta["source_fps"],
                dataset.action_rate_hz,
                episode["length"],
                phase_offset=phase_offset,
            )
            absolute_grids.append(interpolate_canonical_tcp(
                source_state[lower], source_state[upper], fraction
            ))
        absolute_grids = torch.stack(absolute_grids)
        # __getitem__ 按全局窗口编号轮转相位，分片不能从相位 0 重新开始。
        episode_window_start = (
            dataset._episode_window_ends[episode_number - 1] if episode_number else 0
        )

        episode_anchor_count = (int(episode["rgb_target_length"]) + WINDOW_STRIDE - 1) // WINDOW_STRIDE
        if max_anchors is not None:
            episode_anchor_count = min(
                episode_anchor_count, max_anchors - total_anchors
            )
        for start in range(0, episode_anchor_count, anchor_batch_size):
            stop = min(start + anchor_batch_size, episode_anchor_count)
            window_indices = torch.arange(start, stop, dtype=torch.long)
            anchor_rgb = window_indices * WINDOW_STRIDE
            anchor_action = anchor_rgb * dataset.action_steps_per_rgb_frame
            raw_targets = anchor_action[:, None] + action_offsets[None, :]
            valid = (raw_targets >= 0) & (
                raw_targets < episode["action_target_length"]
            )
            target_indices = raw_targets.clamp(
                0, episode["action_target_length"] - 1
            )
            phase_indices = (
                episode_window_start + window_indices
            ) % len(dataset.source_phase_offsets)
            targets = absolute_grids[phase_indices[:, None], target_indices]
            anchors = absolute_grids[phase_indices, anchor_action]
            # Action 是 target 相对当前 anchor 的 fixed-anchor TCP。
            relative = dual_arm_tcp_target_relative_to_anchor(
                anchors.reshape(stop - start, 1, 20),
                targets.reshape(stop - start, action_offsets.numel(), 20),
            ).reshape(stop - start, action_offsets.numel(), 2, 10)
            feature_valid = valid[..., None, None].expand(-1, -1, 2, 9)
            feature_valid = feature_valid & state_feature_mask
            # Gripper 不做 z-score，仅统计每臂 XYZ(3) + Rot6D(6)。
            state_moments.update(targets[..., :9], feature_valid)
            action_moments.update(relative[..., :9], feature_valid)
            total_valid_targets += int(valid.sum())

        total_anchors += episode_anchor_count
        processed_episodes += 1
        if progress_every > 0 and (
            processed_episodes % progress_every == 0
            or processed_episodes == 1
        ):
            print(
                f"episodes={processed_episodes} anchors={total_anchors} "
                f"valid_targets={total_valid_targets}",
                flush=True,
            )

    state = state_moments.result(std_floor, required_state_features)
    action = action_moments.result(std_floor, required_state_features)
    return {
        "schema_version": NGAD_CANONICAL_SCHEMA,
        "state_tcp_mean": state["mean"].tolist(),
        "state_tcp_std": state["std"].tolist(),
        "state_tcp_min": state["min"].tolist(),
        "state_tcp_max": state["max"].tolist(),
        "action_tcp_mean": action["mean"].tolist(),
        "action_tcp_std": action["std"].tolist(),
        "action_tcp_min": action["min"].tolist(),
        "action_tcp_max": action["max"].tolist(),
        "statistics": {
            "method": "population_zscore_parallel_welford",
            "scope": "all admitted 0.6s-window canonical state/action targets",
            "window_stride_rgb": WINDOW_STRIDE,
            "std_floor": float(std_floor),
            "rgb_rate_hz": dataset.rgb_rate_hz,
            "action_rate_hz": dataset.action_rate_hz,
            "frame_offsets": dataset.timeline_layout.frame_offsets.tolist(),
            "source_phase_offsets": list(dataset.source_phase_offsets),
            "phase_assignment": "global_window_index_mod_phase_count",
            "episode_start": int(episode_start),
            "episode_stop": int(episode_start + processed_episodes),
            "episodes": processed_episodes,
            "anchors": total_anchors,
            "valid_targets": total_valid_targets,
            "state_count": state["count"].to(torch.int64).tolist(),
            "action_count": action["count"].to(torch.int64).tolist(),
            "state_raw_std": state["raw_std"].tolist(),
            "action_raw_std": action["raw_std"].tolist(),
            "state_std_floor_applied": state["std_floor_applied"].tolist(),
            "action_std_floor_applied": action["std_floor_applied"].tolist(),
        },
    }


def abc_stats(root, output):
    dataset = abc_orion(root, None, 'train')
    result = compute_zscore_statistics(dataset)
    endpoints = json.loads((root / 'dataset_configs/normalization/abc130k.json').read_text())
    for name in ('gripper_open_value', 'gripper_closed_value'):
        result[name] = endpoints[name]
    write_json(output / 'abc-stats.json', result)
    validation = abc_orion(root, str(output / 'abc-stats.json'), 'val')
    write_json(output / 'abc-admission.json', dict(train_windows=len(dataset), val_windows=len(validation),
               train_episodes=len(dataset._episodes), val_episodes=len(validation._episodes),
               camera_filter=dataset.camera_filter_stats))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['droid-index', 'droid-stats', 'abc-stats'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--abc-root', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.source == 'droid-index':
        index_droid(args.manifest, args.audit_dir, args.output)
    elif args.source == 'droid-stats':
        droid_stats(args.output)
    else:
        abc_stats(args.abc_root, args.output)
    print('MIDTRAIN_PREP_DONE', args.source, flush=True)
