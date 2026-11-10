"""Convert the reviewed cart-007 ROS2 recordings, preserving recorded poses.

Requires orion-data-pipeline/src on PYTHONPATH. This release uses raw gripper
values, NOT the PushCart canonical openness contract. No FK or idle trimming.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import time

import av
from mcap.reader import make_reader
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from orion_pipeline.umi.self_collected._episode_cdr import _CdrReader, _schema_sections, _EXPECTED_SCHEMAS
from orion_pipeline.common.geometry.rotation import quaternion_xyzw_to_matrix, matrix_to_rot6d
from orion_pipeline.common.video.frames import letterbox_rgb_array
from orion_pipeline.robot.pushcart.mapping import nearest


CAMERAS = [f'{mount}_{eye}' for mount in ('head', 'left_wrist', 'right_wrist') for eye in ('left', 'right')]
VIDEO_TOPICS = {f'/camera/{mount}/{eye}/compressed': f'{mount}_{eye}'
                for mount in ('head', 'left_wrist', 'right_wrist') for eye in ('left', 'right')}
SIGNALS = [f'/arm/{side}/{kind}' for side in ('left', 'right') for kind in ('pose_state', 'gripper/joint_state')]
FPS = 20


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def decode(schema, data):
    c = _CdrReader(data)
    if schema == 'geometry_msgs/msg/PoseStamped':
        header = c.header()
        value = [*c.vector().values(), *c.vector('xyzw').values()]
    elif schema == 'sensor_msgs/msg/JointState':
        header = c.header()
        names = [c.string() for _ in range(c.u32())]
        arrays = [[c.f64() for _ in range(c.u32())] for _ in range(3)]
        if names != ['gripper'] or len(arrays[0]) != 1:
            raise ValueError('Unexpected gripper names/shape')
        value = arrays[0]
    elif schema == 'foxglove_msgs/msg/CompressedVideo':
        header = dict(stamp=c.time(), frame_id=c.string())
        size = c.u32()
        c.require(size)
        value = bytes(c.data[c.pos:c.pos + size])
        c.pos += size
        if c.string() != 'h264':
            raise ValueError('Expected H264')
    else:
        raise ValueError(schema)
    if c.pos != len(c.data):
        padding = (-c.pos) % 4
        if not padding or c.data[c.pos:] != bytes(padding):
            raise ValueError('Non-padding trailing CDR bytes')
        c.pos += padding
    c.finish()
    stamp = header['stamp']
    return stamp['sec'] * 1_000_000_000 + stamp['nanosec'], header['frame_id'], value


def validate_schema(schema):
    lines = schema.data.decode().splitlines()
    if lines and lines[0].startswith('MSG: '):
        lines = lines[1:]
    text = re.sub(r'(?m)^(float64\s+[A-Za-z_][A-Za-z0-9_]*)\s+[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\s*$', r'\1', '\n'.join(lines))
    if schema.encoding != 'ros2msg' or _schema_sections(schema.name, text) != _EXPECTED_SCHEMAS[schema.name]:
        raise ValueError('ROS2 schema mismatch: ' + schema.name)


def convert(args):
    row, source_root, root, audit = args
    source = Path(source_root) / row['source_file']
    episode = row['ordinal']
    started = time.monotonic()
    if source.stat().st_size != row['source_bytes']:
        raise ValueError(f'Source size changed: {source}')
    signals = {k: [] for k in SIGNALS}
    images = {k: [] for k in VIDEO_TOPICS}
    codecs, schema_ids, frame_ids, geometry, decode_errors = {}, set(), {}, {}, {}

    def receive(topic, decoded):
        for frame in decoded:
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Decoded frame without source timestamp')
            stamp = round(frame.pts * frame.time_base * 1_000_000_000)
            rgb, _, geom = letterbox_rgb_array(frame.to_ndarray(format='rgb24'))
            if topic in geometry and geometry[topic] != geom:
                raise ValueError('Camera geometry changed within recording')
            geometry[topic] = geom
            images[topic].append((stamp, rgb))

    with source.open('rb') as f:
        reader = make_reader(f, validate_crcs=True)
        for schema, channel, message in reader.iter_messages(topics=[*SIGNALS, *VIDEO_TOPICS]):
            if channel.message_encoding != 'cdr':
                raise ValueError('Expected CDR message')
            if schema.id not in schema_ids:
                validate_schema(schema)
                schema_ids.add(schema.id)
            topic = channel.topic
            stamp, frame_id, value = decode(schema.name, message.data)
            frame_ids.setdefault(topic, set()).add(frame_id)
            if topic in signals:
                signals[topic].append((stamp, value))
            else:
                if topic not in codecs:
                    codecs[topic] = av.CodecContext.create('h264', 'r')
                    codecs[topic].thread_count = 1
                packet = av.Packet(value)
                packet.pts = packet.dts = stamp
                packet.time_base = Fraction(1, 1_000_000_000)
                try:
                    receive(topic, codecs[topic].decode(packet))
                except av.error.InvalidDataError:
                    # Initial P frames lack a reference; an error after decoded
                    # frames is a real integrity failure, not silently skipped.
                    if images[topic]:
                        raise
                    decode_errors[topic] = decode_errors.get(topic, 0) + 1
        for topic, codec in codecs.items():
            receive(topic, codec.decode(None))
    if any(len(v) != 1 or not next(iter(v)) for v in frame_ids.values()):
        raise ValueError('Empty or changing frame_id')
    # Decode in packet order, then align in presentation/capture timestamp
    # order. Source camera startup packets can emit late presentation frames.
    # Never sort the compressed packets or silently merge duplicate stamps.
    presentation_reorders = {}
    for topic, rows in images.items():
        presentation_reorders[topic] = sum(b[0] < a[0] for a,b in zip(rows, rows[1:]))
        rows.sort(key=lambda row: row[0])
    streams = {**signals, **images}
    stamps = {}
    for topic, rows in streams.items():
        if not rows:
            raise ValueError('Empty stream: ' + topic)
        stamps[topic] = np.array([r[0] for r in rows], dtype=np.int64)
        nearest(stamps[topic], stamps[topic][:1])
    start = max(t[0] for t in stamps.values())
    end = min(t[-1] for t in stamps.values())
    grid = np.arange(start, end + 1, 50_000_000, dtype=np.int64)
    if len(grid) < 34:
        raise ValueError('Fewer than 34 synchronized 20Hz samples')
    alignment, indices = {}, {}
    valid = np.ones(len(grid), dtype=bool)
    for topic, ts in stamps.items():
        ix, error = nearest(ts, grid)
        tolerance = 60_000_000 if topic in images else 20_000_000
        valid &= abs(error) <= tolerance
        indices[topic] = ix
        alignment[topic] = ts[ix]
    video_times = np.stack([alignment[t] for t in VIDEO_TOPICS])
    valid &= np.ptp(video_times, axis=0) <= 60_000_000
    good = np.flatnonzero(valid)
    if not len(good):
        raise ValueError('No synchronized interval: ' + str(source))
    first, last = int(good[0]), int(good[-1])+1
    if last-first < 34:
        raise ValueError('Fewer than 34 synchronized samples after boundary crop')
    boundary_crop = dict(leading_frames=first, trailing_frames=len(grid)-last)
    grid = grid[first:last]
    valid = valid[first:last]
    anchors = [a for a in range(0, len(grid)-32, 2) if valid[max(0, a-1):a+33].all()]
    indices = {k: v[first:last] for k,v in indices.items()}
    alignment = {k: v[first:last] for k,v in alignment.items()}
    states, poses = [], []
    for side in ('left', 'right'):
        pt, gt = f'/arm/{side}/pose_state', f'/arm/{side}/gripper/joint_state'
        pose = np.asarray([r[1] for r in signals[pt]], dtype=np.float64)[indices[pt]]
        grip = np.asarray([r[1] for r in signals[gt]], dtype=np.float64)[indices[gt]]
        if not np.isfinite(pose).all() or not np.isfinite(grip).all():
            raise ValueError('Nonfinite pose/gripper')
        if np.max(abs(np.linalg.norm(pose[:, 3:], axis=1) - 1)) > 0.01:
            raise ValueError('Nonunit source quaternion')
        rotation = quaternion_xyzw_to_matrix(pose[:, 3:])
        states.append(np.concatenate([pose[:, :3], matrix_to_rot6d(rotation), grip], axis=1))
        poses.append(pose)
    n = len(grid)
    state = np.concatenate(states, axis=1).astype(np.float32)
    data = pa.table({'frame_index': np.arange(n, dtype=np.int64), 'timestamp': np.arange(n) / FPS,
                     'source_timestamp_ns': grid,
                     'synchronized': valid,
                     'observation.state': pa.array(state.tolist(), type=pa.list_(pa.float32(), 20)),
                     'observation.source_pose': pa.array(np.concatenate(poses, axis=1).tolist(), type=pa.list_(pa.float64(), 14))})
    work = Path(audit) / f'episode-{episode:03d}'
    work.mkdir(parents=True, exist_ok=False)
    data_path = work / 'state.parquet'
    pq.write_table(data, data_path)
    if not pq.read_table(data_path).equals(data):
        raise ValueError('Parquet readback mismatch')
    video_paths = {}
    for topic, camera in VIDEO_TOPICS.items():
        path = Path(root) / f'videos/observation.images.cam_{camera}/chunk-000/file-{episode:03d}.mp4'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        with av.open(str(path), 'w') as output:
            stream = output.add_stream('libx264', rate=FPS)
            stream.width = stream.height = 256
            stream.pix_fmt = 'yuv420p'
            stream.options = {'preset': 'veryfast', 'crf': '20', 'g': '20', 'threads': '1'}
            for i, ix in enumerate(indices[topic]):
                frame = av.VideoFrame.from_ndarray(images[topic][ix][1], format='rgb24')
                frame.pts, frame.time_base = i, Fraction(1, FPS)
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        with av.open(str(path)) as inp:
            actual = list((f.width, f.height, float(f.pts * f.time_base)) for f in inp.decode(video=0))
            if len(actual) != n or any(w != 256 or h != 256 or abs(t - i / FPS) > 1e-5 for i, (w, h, t) in enumerate(actual)):
                raise ValueError('Video timestamp/frame readback mismatch')
        video_paths[camera] = str(path.relative_to(root)).replace('\\', '/')
    np.savez_compressed(work / 'alignment.npz', target_ns=grid, synchronized=valid, **alignment)
    result = dict(row, length=n, source_sha256=sha(source), state_path=str(data_path),
                  videos=video_paths, frame_ids={k: next(iter(v)) for k, v in frame_ids.items()},
                  geometry=geometry, initial_decode_errors=decode_errors, boundary_crop=boundary_crop,
                  presentation_reorders=presentation_reorders,
                  alignment_error_ns={k: int(abs(v-grid).max()) for k, v in alignment.items()},
                  window_anchors=anchors, invalid_ticks=int((~valid).sum()),
                  full_window_count=len(anchors), seconds=time.monotonic()-started)
    save(work / 'report.json', result)
    return result


def publish(records, root):
    root = Path(root)
    for topic in SIGNALS:
        if len({r['frame_ids'][topic] for r in records}) != 1:
            raise ValueError('Inconsistent source frame across episodes: ' + topic)
    tasks = sorted({r['task_text'] for r in records})
    episodes, offset = [], 0
    for eid, r in enumerate(records):
        n = r['length']
        table = pq.read_table(r['state_path'])
        for name, value in [('index', np.arange(offset, offset+n)), ('episode_index', np.full(n, eid)),
                            ('task_index', np.full(n, tasks.index(r['task_text'])))]:
            table = table.append_column(name, pa.array(value, type=pa.int64()))
        path = root / f'data/chunk-000/file-{eid:03d}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        if not pq.read_table(path).equals(table):
            raise ValueError('Published parquet mismatch')
        row = dict(episode_index=eid, length=n, dataset_from_index=offset, dataset_to_index=offset+n,
                   tasks=[r['task_text']], source_ordinal=r['ordinal'], source_sha256=r['source_sha256'],
                   source_file=r['source_file'], **{'data/chunk_index': 0, 'data/file_index': eid})
        row['window_anchors'] = r['window_anchors']
        for camera in CAMERAS:
            key = 'videos/observation.images.cam_' + camera
            row.update({key+'/chunk_index': 0, key+'/file_index': r['ordinal'],
                        key+'/from_timestamp': 0., key+'/to_timestamp': n/FPS})
        episodes.append(row)
        offset += n
    path = root/'meta/episodes/chunk-000/file-000.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episodes), path)
    pq.write_table(pa.table({'task_index': range(len(tasks)), 'task': tasks}), root/'meta/tasks.parquet')
    features = {key: dict(dtype='int64', shape=[1]) for key in ('index','episode_index','frame_index','task_index')}
    features.update({'timestamp': dict(dtype='float64', shape=[1]), 'observation.state': dict(dtype='float32', shape=[20]),
                     'synchronized': dict(dtype='bool', shape=[1])})
    features.update({'observation.images.cam_'+c: dict(dtype='video', shape=[256,256,3], video_info={'video.fps': FPS, 'video.codec': 'h264'}) for c in CAMERAS})
    save(root/'meta/info.json', dict(codebase_version='v3.0', robot_type='cart-007-recorded-pose', fps=FPS,
         total_episodes=len(records), total_frames=offset, total_tasks=len(tasks), total_videos=6*len(records), chunks_size=1000,
         data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
         video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4', features=features,
         ngad_extensions=dict(schema='cart007.recorded_pose.raw_gripper.v1', pose_source='pose_state',
              gripper='raw recorded position; no endpoint calibration or clipping', rot6d='first_two_rows_row_major',
              tcp='recorded source reference, physical offset unspecified', fk_applied=False, training_loader_verified=False,
              window_policy='continuous_valid_anchor_minus_1_through_plus_32_v1',
              window_index_sha256=hashlib.sha256(json.dumps([(r['ordinal'], r['window_anchors']) for r in records]).encode()).hexdigest())))
    save(root/'meta/source_manifest.json', records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding='utf-8'))['records']
    if not rows or len({r['ordinal'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate admission manifest')
    Path(args.output).mkdir(parents=True, exist_ok=False)
    Path(args.audit).mkdir(parents=True, exist_ok=False)
    save(Path(args.audit)/'input.json', dict(arguments=vars(args), manifest_sha256=sha(args.manifest), code_sha256=sha(__file__)))
    jobs = [(r, args.source_root, args.output, args.audit) for r in rows]
    records, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(convert, job): job[0] for job in jobs}
        for future in as_completed(pending):
            row = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                error = dict(ordinal=row['ordinal'], source_file=row['source_file'], error_type=type(exc).__name__, error=str(exc))
                errors.append(error)
                save(Path(args.audit)/f"error-{row['ordinal']:03d}.json", error)
                print(json.dumps(error), flush=True)
            else:
                records.append(result)
                print(json.dumps(dict(completed=len(records), errors=len(errors), total=len(rows), ordinal=result['ordinal'], frames=result['length'], seconds=result['seconds'])), flush=True)
    records.sort(key=lambda row: row['ordinal'])
    if errors:
        save(Path(args.audit)/'conversion_result.json', dict(successful_episodes=len(records), errors=errors, published=False))
        raise RuntimeError(f'{len(errors)} recordings failed; no dataset published; see conversion_result.json')
    publish(records, args.output)
    save(Path(args.audit)/'conversion_result.json', dict(episodes=len(records), frames=sum(r['length'] for r in records),
         windows=sum(r['full_window_count'] for r in records), zero_window_ordinals=[r['ordinal'] for r in records if not r['full_window_count']],
         video_readback_verified=True, training_loader_verified=False))


if __name__ == '__main__':
    main()
