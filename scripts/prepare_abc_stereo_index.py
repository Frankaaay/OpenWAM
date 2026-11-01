"""Repair missing ABC right-video locators into a private metadata-only view."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq

LEFT = 'observation.images.cam_head_left'
RIGHT = 'observation.images.cam_head_right'
FIELDS = ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp')


def verify_video(item):
    path, length, fps = item
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames != length or abs(float(stream.average_rate) - fps) > 1e-6:
            raise ValueError(f'ABC right-video length/rate mismatch: {path}')
        if stream.start_time is None or abs(float(stream.start_time * stream.time_base)) > 1 / fps:
            raise ValueError(f'ABC right-video start timestamp mismatch: {path}')
        if stream.duration is None or abs(float(stream.duration * stream.time_base) - length / fps) > 1 / fps:
            raise ValueError(f'ABC right-video duration mismatch: {path}')
    return dict(path=str(path), frames=length, fps=fps)


def prepare(root, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for entry in root.iterdir():
        if entry.name != 'meta':
            (output / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (output / 'meta').mkdir()
    for entry in (root / 'meta').iterdir():
        if entry.name != 'episodes':
            (output / 'meta' / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    destination = output / 'meta/episodes/chunk-000'
    destination.mkdir(parents=True)
    info = json.loads((root / 'meta/info.json').read_text())
    fps = float(info['fps'])
    repaired = 0
    with (output / 'locator-repairs.jsonl').open('x') as receipt, ProcessPoolExecutor(max_workers=8) as pool:
        for path in sorted((root / 'meta/episodes/chunk-000').glob('*.parquet')):
            table = pq.read_table(path)
            rows = table.to_pylist()
            jobs, repairs = [], []
            for row in rows:
                if not row.get('availability/' + RIGHT):
                    continue
                if all(row.get('videos/' + RIGHT + '/' + field) is not None for field in FIELDS):
                    continue
                locator = {field: row['videos/' + LEFT + '/' + field] for field in FIELDS}
                if locator['from_timestamp'] != 0 or abs(locator['to_timestamp'] - row['length'] / fps) > 1 / fps:
                    raise ValueError('Missing right locator is not an episode-local video')
                video = root / info['video_path'].format(video_key=RIGHT, **locator)
                jobs.append((video, row['length'], fps))
                repairs.append(dict(episode_index=row['episode_index'], source_split=row['source_split'], source_metadata=str(path), locator=locator))
                for field, value in locator.items():
                    row['videos/' + RIGHT + '/' + field] = value
            for repair, proof in zip(repairs, pool.map(verify_video, jobs)):
                receipt.write(json.dumps(dict(repair, verification=proof)) + '\n')
                repaired += 1
            schema = table.schema
            for field in FIELDS:
                name = 'videos/' + RIGHT + '/' + field
                if name not in schema.names:
                    schema = schema.append(pa.field(name, pa.int64() if field.endswith('index') else pa.float64()))
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), destination / path.name)
            receipt.flush()
            print('ABC_LOCATOR_REPAIR', path.name, repaired, flush=True)
    result = dict(source_root=str(root), output_root=str(output), repaired_episodes=repaired,
                  contract='existing right MP4; episode-local left locator; equal frame count/rate/duration; no pixel changes')
    with (output / 'locator-repairs.jsonl').open('rb') as stream:
        result['repair_manifest_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (output / 'INDEX_READY.json').write_text(json.dumps(result, indent=2))
    print('ABC_INDEX_READY', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.output)
