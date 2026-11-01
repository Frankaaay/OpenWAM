"""Prepare exact admitted DROID windows and Orion normalization for midtrain."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import shutil

import ijson
import numpy as np
import torch

from dataloader.compute_norm_stats import RunningMoments, compute_zscore_statistics
from dataloader.pretrain.canonical import NGAD_CANONICAL_SCHEMA
from openwam.dataloader.orion_midtrain import abc_orion, droid_targets, DROIDOrionDataset


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


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
    counts, excluded, episode_ids = Counter(), {}, {}
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
            episode = episode_ids.setdefault(key, len(episode_ids))
            con.execute('insert into windows(episode,split,payload) values(?,?,?)', (episode, record['split'], json.dumps(record)))
            counts[record['split']] += 1
            if sum(counts.values()) % 10000 == 0:
                con.commit()
                print('DROID_INDEX', dict(counts), flush=True)
    con.execute('create index windows_split_episode on windows(split,episode,id)')
    con.commit()
    con.close()
    write_json(output / 'droid-admission.json', dict(windows=counts, episodes=len(episode_ids), excluded=list(excluded.values()), original_windows=summary['windows']))


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
                  statistics=dict(method='population_zscore_parallel_welford', train_windows=train_windows, rgb_rate_hz=10, action_rate_hz=20))
    for name, moments in [('state', state), ('action', action)]:
        stats = moments.result(1e-5, required)
        for field in ('mean', 'std', 'min', 'max'):
            result[f'{name}_tcp_{field}'] = stats[field].tolist()
        result['statistics'][name + '_count'] = stats['count'].tolist()
    write_json(output / 'droid-stats.json', result)


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
