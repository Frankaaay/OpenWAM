"""H100 bindings for the existing full LIBERO-Plus scheduler."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import yaml

from h100_acceptance import ROOT, RUNTIME, REPO, SERVER, CLIENT, SUITES, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=['wan', 'wan-rgbft', 'stereotok'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    stereo = args.variant == 'stereotok'
    train = ROOT / 'frank/runs/eighteen-preparation-20261014/twelve-ready-u15853'
    checkpoint = train / ('train-openwam-libero-' + args.variant) / (
        '2026-10-21_22-24-28' if stereo else '2026-10-21_14-00-22')
    if args.variant == 'wan-rgbft':
        checkpoint = RUNTIME.parent / 'train-openwam-libero-rgbft/2026-11-13_15-24-40'
    weight = checkpoint / 'checkpoint_step_13230.safetensors'
    assert weight.stat().st_size == (28010055404 if stereo else 24813767464)
    original = yaml.safe_load((checkpoint / 'config.yaml').read_text())
    classification = json.loads((ROOT / 'plus/libero/libero/benchmark/task_classification.json').read_text())
    assert sum(len(classification[s]) for s in SUITES) == 10030
    os.environ.update(environment(0, args.run, server=True))
    spec = importlib.util.spec_from_file_location('full_launcher', REPO / 'benchmarks/libero-plus/run_all_suites.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    def client_env(base_env, *, libero_path, config_root, render_gpu):
        key = str(config_root)
        # NVIDIA 595.91.07 aborts in EGL readback on full policy trajectories.
        # Use the verified Mesa path with four raster threads per worker.
        env = environment(render_gpu, config_root.parent, software_render=True)
        env.update(LIBERO_PLUS_CONFIG_ROOT=key, LIBERO_CONFIG_PATH=key,
                   LP_NUM_THREADS='4', PYTHONFAULTHANDLER='1')
        return env

    launcher._native_client_env = client_env
    from openwam.dataloader.utils.normalization import materialize_eef_stats
    for suite in SUITES:
        bundle = args.run / 'bundles' / suite
        bundle.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(original)
        datasets = [d for d in cfg['dataloader']['datasets'] if '/' + suite + '/' in d['dataset_dir']]
        assert len(datasets) == 1
        cfg['dataloader'] = datasets[0]
        assert bool(datasets[0].get('stereo', False)) == stereo
        assert datasets[0]['normalize_mode'] == 'min-max'
        stats = Path(datasets[0].get('normalization_stats_path') or str(Path(datasets[0]['dataset_dir']) / 'meta/libero_normalization_stats.npy'))
        raw = np.load(stats, allow_pickle=True).item()
        for key in ['eef', 'eef_state']:
            assert raw[key]['representation'] == 'native_delta_eef10'
            assert raw[key]['gripper_convention'] == 'minus1_closed_plus1_open'
            checked = materialize_eef_stats(dict(raw[key]), 'min-max', dim=10, strict_minmax=False, force_rot6d_identity=True)
            for field in ['min', 'max']:
                np.testing.assert_array_equal(np.asarray(raw[key][field], dtype=np.float32), checked[field])
        links = [(weight.name, weight), ('normalization_stats.npy', stats), ('tokenizer', checkpoint / 'tokenizer')]
        if stereo:
            links.append(('stereotok.json', checkpoint / 'stereotok.json'))
        if args.variant == 'wan-rgbft':
            cfg['model']['video_backbone']['model_path'] = str(RUNTIME / 'assets/openwam')
            cfg['model']['video_backbone']['encoder']['model_path'] = str(RUNTIME / 'assets/openwam')
            links[-1] = ('tokenizer/google', RUNTIME / 'assets/openwam/google')
        for name, target in links:
            assert target.exists(), target
            link = bundle / name
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                assert link.resolve() == target.resolve()
            else:
                link.symlink_to(target)
        (bundle / 'config.yaml').write_text(yaml.safe_dump(cfg))
        policy = yaml.safe_load((REPO / 'benchmarks/libero-plus/policy_config.yml').read_text())
        policy['stereo'] = stereo
        policy_path = bundle / 'client.yaml'
        policy_path.write_text(yaml.safe_dump(policy))
        command = ['--ckpt-dir', str(bundle), '--ckpt-name', weight.name,
                   '--server-python', str(SERVER), '--libero-python', str(CLIENT),
                   '--libero-path', str(ROOT / 'plus'), '--policy-config', str(policy_path),
                   '--gpus', '0,1,2,3,4,5,6,7', '--replicas-per-gpu', '1',
                   '--suites', suite, '--num-trials', '1', '--seed', '10000',
                   '--output-dir', str(args.run / suite)]
        if args.dry_run:
            command.append('--dry-run')
        assert launcher.main(command) == 0, suite
    print('OPENWAM_PLUS_FULL_COMPLETE' if not args.dry_run else 'OPENWAM_PLUS_FULL_DRY_RUN_COMPLETE', args.variant, flush=True)


if __name__ == '__main__':
    main()
