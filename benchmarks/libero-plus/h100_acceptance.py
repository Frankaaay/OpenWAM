"""Bounded OpenWAM LIBERO-Plus acceptance: eight workers, fourteen episodes."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
ROOT = Path('/gpfs/jiuquyun/projects/Frank/openwam-libero-plus-20261116')
RUNTIME = Path('/gpfs/jiuquyun/projects/Frank/libero-rgbft-ppu15-20261114/eval-runtime')
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
BASE = Path('/mnt/workdir/frank')
CLIENT = BASE / 'envs/fastwam-baselines-20261014/bin/python'
SERVER = BASE / 'envs/openwam-baselines-20261014/bin/python'
PROJECTS = RUNTIME / 'frank/projects'


def selection(classification):
    assert sum(len(classification[s]) for s in SUITES) == 10030
    tasks = []
    for suite in SUITES:
        seen = set()
        for row in classification[suite]:
            category = row['category']
            if suite != 'libero_spatial' and category != 'Camera Viewpoints':
                continue
            if category not in seen:
                tasks.append((suite, int(row['id']) - 1, row['name'], category))
                seen.add(category)
        if suite == 'libero_spatial':
            noise_seen = set()
            for row in classification[suite]:
                if '_noise_' not in row['name']:
                    continue
                kind = (int(row['name'].rsplit('_noise_', 1)[1]) - 1) // 10
                if kind not in noise_seen:
                    item = (suite, int(row['id']) - 1, row['name'], 'Sensor Noise')
                    if item not in tasks:
                        tasks.append(item)
                    noise_seen.add(kind)
            assert noise_seen == set(range(5))
    assert len(tasks) == 14
    return tasks


def environment(rank, out, server=False, software_render=False):
    libs = BASE / 'envs/system-libs/prefix/usr/lib/x86_64-linux-gnu'
    python = SERVER if server else CLIENT
    paths = [REPO, REPO / 'third_party', PROJECTS / 'FastWAM-libero-eval-v2-20261019/src'] if server else [
        ROOT / 'plus', ROOT / 'deps', REPO,
        RUNTIME / 'frank/runs/libero-six-eval-20261022/client-deps',
        PROJECTS / 'libero-eval-20261022',
        PROJECTS / 'libero-eval-20261022/XPolicyLab/policy/LingBot_VA/lingbot_va']
    env = dict(os.environ, PYTHONPATH=':'.join(map(str, paths)),
        CUDA_VISIBLE_DEVICES=str(rank), MUJOCO_EGL_DEVICE_ID=str(rank),
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
        LIBERO_PLUS_PATH=str(ROOT / 'plus'), LIBERO_PLUS_CONFIG_ROOT=str(out / 'libero-config'),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
        DIFFSYNTH_SKIP_DOWNLOAD='true', WANDB_MODE='disabled',
        MAGICK_HOME=str(ROOT / 'system-libs/prefix/usr'),
        MAGICK_CONFIGURE_PATH=str(ROOT / 'system-libs/prefix/etc/ImageMagick-6'),
        MAGICK_CODER_MODULE_PATH=str(ROOT / 'system-libs/prefix/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.12/modules-Q16/coders'),
        TRITON_CACHE_DIR=str(out / 'triton'), TORCHINDUCTOR_CACHE_DIR=str(out / 'inductor'),
        LD_LIBRARY_PATH=':'.join(map(str, [ROOT / 'system-libs/prefix/usr/lib/x86_64-linux-gnu', python.parent.parent / 'lib/python3.12/site-packages/torch/lib',
            libs, libs / 'pulseaudio'])) + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(BASE / 'envs/system-libs/nvidia-egl-vendor.json'),
        PATH=str(BASE / 'envs/system-libs/prefix/usr/bin') + ':' + os.environ['PATH'])
    if software_render:
        assert not server
        env.update(__EGL_VENDOR_LIBRARY_FILENAMES=str(BASE / 'envs/system-libs/prefix/usr/share/glvnd/egl_vendor.d/50_mesa.json'),
            LIBGL_ALWAYS_SOFTWARE='1', GALLIUM_DRIVER='llvmpipe', LP_NUM_THREADS='1',
            LIBGL_DRIVERS_PATH=str(libs / 'dri'), CUDA_VISIBLE_DEVICES='')
        env.pop('MUJOCO_EGL_DEVICE_ID', None)
        probe = "from mujoco.egl import egl_ext as E\nfrom OpenGL import error\nfor i,d in enumerate(E.eglQueryDevicesEXT()):\n try:\n  p=E.eglGetPlatformDisplayEXT(E.EGL_PLATFORM_DEVICE_EXT,d,None)\n  if E.eglInitialize(p,None,None):\n   print(i);E.eglTerminate(p);break\n except error.GLError:pass\nelse:raise RuntimeError('No usable Mesa EGL device')"
        device = subprocess.check_output([str(CLIENT), '-c', probe], env=env, text=True).strip()
        assert device.isdigit(), device
        env['MUJOCO_EGL_DEVICE_ID'] = device
        print('MESA_EGL_DEVICE', device, flush=True)
    return env


def worker(variant, rank, run):
    out = run / f'rank{rank}'
    out.mkdir(parents=True, exist_ok=False)
    tasks = selection(json.loads((ROOT / 'plus/libero/libero/benchmark/task_classification.json').read_text()))
    (out / 'selection.json').write_text(json.dumps(tasks, indent=2))
    stereo = variant == 'stereotok'
    train = ROOT / 'frank/runs/eighteen-preparation-20261014/twelve-ready-u15853'
    checkpoint = train / ('train-openwam-libero-' + variant) / ('2026-10-21_22-24-28' if stereo else '2026-10-21_14-00-22')
    if variant == 'wan-rgbft':
        checkpoint = RUNTIME.parent / 'train-openwam-libero-rgbft/2026-11-13_15-24-40'
    weight = checkpoint / 'checkpoint_step_13230.safetensors'
    assert weight.stat().st_size == (28010055404 if stereo else 24813767464)
    original = yaml.safe_load((checkpoint / 'config.yaml').read_text())
    assert original['dataloader']['type'] == 'mixture'
    results = []
    for suite, task_id, name, category in tasks[rank::8]:
        current = out / f'{suite}-{task_id}'
        current.mkdir()
        cfg = copy.deepcopy(original)
        if variant == 'wan-rgbft':
            cfg['model']['video_backbone']['model_path'] = str(RUNTIME / 'assets/openwam')
            cfg['model']['video_backbone']['encoder']['model_path'] = str(RUNTIME / 'assets/openwam')
        datasets = [d for d in cfg['dataloader']['datasets'] if '/' + suite + '/' in d['dataset_dir']]
        assert len(datasets) == 1
        cfg['dataloader'] = datasets[0]
        assert bool(datasets[0].get('stereo', False)) == stereo
        assert datasets[0]['normalize_mode'] == 'min-max'
        stats = Path(datasets[0].get('normalization_stats_path') or str(Path(datasets[0]['dataset_dir']) / 'meta/libero_normalization_stats.npy'))
        raw = np.load(stats, allow_pickle=True).item()
        from openwam.dataloader.utils.normalization import materialize_eef_stats
        for key in ['eef', 'eef_state']:
            assert raw[key]['gripper_convention'] == 'minus1_closed_plus1_open'
            assert raw[key]['representation'] == 'native_delta_eef10'
            materialized = materialize_eef_stats(dict(raw[key]), 'min-max', dim=10,
                strict_minmax=False, force_rot6d_identity=True)
            for field in ['min', 'max']:
                assert np.asarray(raw[key][field]).shape == (10,)
                assert np.isfinite(raw[key][field]).all()
                np.testing.assert_array_equal(np.asarray(raw[key][field], dtype=np.float32), materialized[field])
        bundle = current / 'checkpoint'
        bundle.mkdir()
        links = [(weight.name, weight), ('tokenizer', checkpoint / 'tokenizer'), ('normalization_stats.npy', stats)]
        if variant == 'wan-rgbft':
            links[1] = ('tokenizer/google', RUNTIME / 'assets/openwam/google')
        if stereo:
            links.append(('stereotok.json', checkpoint / 'stereotok.json'))
        for link, target in links:
            assert target.exists(), target
            (bundle / link).parent.mkdir(parents=True, exist_ok=True)
            (bundle / link).symlink_to(target)
        (bundle / 'config.yaml').write_text(yaml.safe_dump(cfg))
        client = yaml.safe_load((REPO / 'benchmarks/libero/policy_config.yml').read_text())
        client.update(suite=suite, task_id=task_id, num_trials=1, trial_start=0, seed=42,
                      rng_mode='official_global', stereo=stereo, port=8848 + rank, result_dir=str(current))
        (current / 'client.yaml').write_text(yaml.safe_dump(client))
        started = time.monotonic()
        with (current / 'server.log').open('w') as log:
            process = subprocess.Popen([str(SERVER), str(REPO / 'scripts/deploy.py'), '--ckpt-dir', str(bundle),
                '--ckpt-name', weight.name, '--device', 'cuda:0', '--host', '127.0.0.1', '--port', str(8848 + rank)],
                cwd=REPO, env=environment(rank, out, True), stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 1200
                while 'WebSocket server started' not in (current / 'server.log').read_text(errors='replace'):
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f'Server did not become ready: {current}')
                    time.sleep(2)
                with (current / 'client.log').open('w') as clog:
                    subprocess.run([str(CLIENT), str(REPO / 'benchmarks/libero-plus/single_eval.py'),
                        '--config', str(current / 'client.yaml')], env=environment(rank, out, software_render=True), cwd=REPO,
                        stdout=clog, stderr=subprocess.STDOUT, check=True, timeout=5400)
                result = json.loads((current / 'results.json').read_text())
                assert result['task'] == name and len(result['trials']) == 1
                assert 'PLUS_RENDER_VERIFIED' in (current / 'client.log').read_text()
                result.update(category=category, seconds=time.monotonic() - started,
                              checkpoint=str(weight), variant=variant,
                              stats_sha256=hashlib.sha256(stats.read_bytes()).hexdigest())
                results.append(result)
                (out / 'episodes.json').write_text(json.dumps(results, indent=2))
                print('EPISODE_COMPLETE', variant, rank, suite, task_id, result['successes'], flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    (out / 'acceptance.json').write_text(json.dumps(dict(complete=True, episodes=results), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=['wan', 'stereotok', 'wan-rgbft'])
    parser.add_argument('--rank', type=int, choices=range(8))
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.rank is not None:
        worker(args.variant, args.rank, args.run)
        return
    assert os.environ['SLURM_GPUS_ON_NODE'] == '8'
    assert json.loads((ROOT / 'frank/runs/libero-plus-pro-20261116/assets-plus/ready.json').read_text())['complete']
    args.run.mkdir(parents=True, exist_ok=False)
    processes = []
    try:
        for rank in range(8):
            with (args.run / f'worker{rank}.log').open('w') as log:
                processes.append(subprocess.Popen([sys.executable, __file__, args.variant, '--rank', str(rank),
                    '--run', str(args.run)], stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('An acceptance worker failed; inspect worker and episode logs')
            time.sleep(5)
        assert all(p.returncode == 0 for p in processes)
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
    episodes = [e for rank in range(8) for e in json.loads((args.run / f'rank{rank}/acceptance.json').read_text())['episodes']]
    assert len(episodes) == 14
    (args.run / 'acceptance.json').write_text(json.dumps(dict(complete=True, variant=args.variant,
        episodes=episodes, integration_only=True), indent=2))
    print('OPENWAM_PLUS_ACCEPTANCE_COMPLETE', args.variant, len(episodes), flush=True)


if __name__ == '__main__':
    main()
