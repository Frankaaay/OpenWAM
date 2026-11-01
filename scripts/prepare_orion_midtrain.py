"""Prepare exact admitted DROID windows and Orion normalization for midtrain."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

import ijson
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
    audits = {r['id']: r for r in map(json.loads, (audit_dir / 'droid-action-audit.jsonl').read_text().splitlines())}
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


def droid_stats(output):
    dataset = DROIDOrionDataset(output / 'droid.sqlite', None, 'train')
    state, action = RunningMoments((2, 9)), RunningMoments((2, 9))
    required = torch.zeros((2, 9), dtype=torch.bool)
    required[0] = True
    for i in range(len(dataset)):
        grid, relative, valid, _ = droid_targets(dataset.record(i))
        mask = valid.reshape(-1, 2, 10)[..., :9]
        # Same all-valid-target population as Orion; proprio uses this transform.
        state.update(grid[1:, :, :9], mask)
        action.update(relative.reshape(-1, 2, 10)[..., :9], mask)
        if (i + 1) % 1000 == 0:
            print('DROID_STATS', i + 1, len(dataset), flush=True)
    result = dict(schema_version=NGAD_CANONICAL_SCHEMA, gripper_open_value=[1., 1.], gripper_closed_value=[0., 0.],
                  statistics=dict(method='population_zscore_parallel_welford', train_windows=len(dataset), rgb_rate_hz=10, action_rate_hz=20))
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
