"""Isolated paired H800 diagnostic; never writes into formal evaluation outputs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

B = Path('/mnt/workdir/frank')
F = B / 'projects/FastWAM-stage3-mono-734fe84'
OLD = B / 'projects/OpenWAM-release-9259a32'
CLIENT = B / 'projects/stage3-robotwin-OpenWAM-20261029'
NEW = Path(__file__).resolve().parents[1]
CK = B / 'runs/robotwin-u15853-mono-20261101/openwam/checkpoints/2026-11-01_22-36-55'
PY = B / 'envs/fastwam-stereotok-eval-cu128-20261009/bin/python'
SPY = B / 'envs/openwam-baselines-20261014/bin/python'
PORT = 29873
# Diagnostic selection from completed historical episodes: short success, failure,
# and native-limit long failure. Each arm uses the identical frozen plan entries.
CASES = [('open_laptop', 14), ('open_laptop', 1), ('open_microwave', 99)]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2)


def digest(value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if hasattr(value, '__array__') or hasattr(value, '__array_interface__'):
        a = np.asarray(value)
        return hashlib.sha256(str((a.shape, str(a.dtype))).encode() + a.tobytes()).hexdigest()
    if isinstance(value, dict):
        return {k: digest(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [digest(v) for v in value]
    return value


def evaluation():
    spec = importlib.util.spec_from_file_location('shared_eval', F / 'scripts/c2r/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def server(out):
    import torch
    from openwam.deploy import server as module
    original = module.build_server_from_config

    def build(*args, **kwargs):
        srv = original(*args, **kwargs)
        generate = srv.engine.generate
        stream = (out / 'generations.jsonl').open('x', buffering=1)

        def traced(conditions):
            inputs = digest(conditions)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = generate(conditions)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            actions = result['actions']
            if hasattr(actions, 'detach'):
                actions = actions.detach().cpu().numpy()
            stream.write(json.dumps(dict(inputs=inputs, actions=actions.tolist(), seconds=elapsed)) + '\n')
            return result

        srv.engine.generate = traced
        return srv

    module.build_server_from_config = build
    module.main(['--ckpt-dir', str(CK), '--ckpt-name', 'checkpoint_step_30000.safetensors',
                 '--device', 'cuda:0', '--host', '127.0.0.1', '--port', str(PORT)])


def client(out):
    import numpy as np
    from benchmarks.robotwin.openwam2robotwin_interface import SharedPlanPolicy
    e = evaluation()
    lock = e.shared_plan_lock()
    plans = e.load_shared_plans(dict(episodes=100, cases=[c for c in lock['cases'] if c['phase'] == 'clean']))
    official, device = e.setup_runtime()
    policy = SharedPlanPolicy(host='127.0.0.1', port=PORT, stereo=False)
    original_step = policy.step
    original_roundtrip = policy.client._client._roundtrip
    trace = []
    wire = []

    def roundtrip(message, **kwargs):
        start = time.perf_counter()
        response = original_roundtrip(message, **kwargs)
        wire.append(dict(seconds=time.perf_counter()-start, bytes=len(json.dumps(message)),
                         server_ms=response.get('latency_ms', 0)))
        return response

    def step(env, observation):
        original_take = env._env.take_action
        row = dict(action_index=env.take_action_cnt, observed=observation is not None)

        def take(action, **kwargs):
            row['action'] = np.asarray(action).tolist()
            start = time.perf_counter()
            result = original_take(action, **kwargs)
            row['simulation_seconds'] = time.perf_counter() - start
            return result

        env._env.take_action = take
        try:
            original_step(env, observation)
        finally:
            env._env.take_action = original_take
        row['rng'] = digest([env._python, env._numpy])
        trace.append(row)

    policy.step = step
    policy.client._client._roundtrip = roundtrip
    for task, ordinal in CASES:
        trace.clear()
        wire.clear()
        case = next(c for c in lock['cases'] if c['task'] == task and c['phase'] == 'clean')
        ep = plans[task + '--clean']['episodes'][ordinal]
        results = e.rollout_case(official, device, policy, case, [ep])
        put(out / f'{task}-{ordinal}.json', dict(result=results[0], trace=trace, wire=wire))
    policy.client._client.close()


def compare(root):
    import numpy as np
    rows = []
    failures = []
    for task, ordinal in CASES:
        name = f'{task}-{ordinal}.json'
        a, b = [json.loads((root / variant / name).read_text()) for variant in ('baseline', 'candidate')]
        for key in ['seed', 'instruction', 'success', 'actions', 'initial_scene_sha256', 'initial_observation_sha256']:
            if a['result'][key] != b['result'][key]:
                failures.append(f'{name}: {key}')
        aa, bb = [np.asarray([r['action'] for r in v['trace']]) for v in (a, b)]
        exact = aa.shape == bb.shape and np.array_equal(aa, bb)
        if not exact:
            failures.append(f'{name}: executed actions')
        rng_equal = [r['rng'] for r in a['trace']] == [r['rng'] for r in b['trace']]
        if not rng_equal:
            failures.append(f'{name}: environment RNG')
        rows.append(dict(task=task, ordinal=ordinal, exact_actions=exact, rng_equal=rng_equal,
                         max_action_error=float(np.max(np.abs(aa-bb))) if aa.shape == bb.shape else None,
                         baseline=a['result'], candidate=b['result'],
                         speedup=a['result']['seconds']/b['result']['seconds'],
                         baseline_simulation=sum(r['simulation_seconds'] for r in a['trace']),
                         candidate_simulation=sum(r['simulation_seconds'] for r in b['trace']),
                         baseline_wire=sum(r['seconds'] for r in a['wire']),
                         candidate_wire=sum(r['seconds'] for r in b['wire']),
                         baseline_bytes=sum(r['bytes'] for r in a['wire']),
                         candidate_bytes=sum(r['bytes'] for r in b['wire'])))
    generations = [[json.loads(l) for l in (root / v / 'generations.jsonl').read_text().splitlines()]
                   for v in ('baseline', 'candidate')]
    equal_inputs = [g['inputs'] for g in generations[0]] == [g['inputs'] for g in generations[1]]
    equal_chunks = [g['actions'] for g in generations[0]] == [g['actions'] for g in generations[1]]
    if not equal_inputs or not equal_chunks:
        failures.append('generation inputs/chunks')
    result = dict(passed=not failures, failures=failures, episodes=rows,
                  exact_generation_inputs=equal_inputs, exact_chunks=equal_chunks,
                  generation_counts=[len(v) for v in generations],
                  inference_seconds=[sum(g['seconds'] for g in v) for v in generations],
                  aggregate_speedup=sum(r['baseline']['seconds'] for r in rows)/sum(r['candidate']['seconds'] for r in rows))
    put(root / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise RuntimeError('Candidate consistency gate failed: ' + str(failures))


def run(root, baseline=None):
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,pci.device_id,compute_cap,memory.total', '--format=csv,noheader'], text=True)
    assert len(gpu.strip().splitlines()) == 1, gpu
    name, uuid, pci, capability, memory = [v.strip() for v in gpu.strip().split(',')]
    # This task's named "H800" PAI cluster reports an L20Y Hopper board.
    # Pin the independently probed identity, rather than infer SKU from its quota.
    assert (name, pci, capability) == ('NVIDIA L20Y', '0x26BA10DE', '9.0'), gpu
    assert 80000 <= int(memory.split()[0]) <= 82000, gpu
    if baseline is not None:
        previous = json.loads((baseline / 'contract.json').read_text())
        assert previous['gpu'] == gpu, 'Reusing baseline requires the identical physical GPU'
        assert previous['cases'] == [list(c) for c in CASES]
        assert previous['config_sha256'] == hashlib.sha256((CK/'config.yaml').read_bytes()).hexdigest()
        for task, ordinal in CASES:
            assert (baseline/'baseline'/f'{task}-{ordinal}.json').is_file()
        (root/'baseline').symlink_to((baseline/'baseline').resolve(), target_is_directory=True)
    for repo, sha in [(OLD, '9259a320a5945f15eab8167f1a888f5238971eb2'),
                      (CLIENT, '4a06f9a3c78a40605fb3a4f56d60dffb1251c220'),
                      (F, '734fe840b64a8f743fbf0c7447cc07c368720c97')]:
        assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == sha
        assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    assert not subprocess.check_output(['git', '-C', str(NEW), 'status', '--porcelain'], text=True).strip()
    assert (CK / 'checkpoint_step_30000.safetensors').stat().st_size == 28010055404
    put(root / 'contract.json', dict(gpu=gpu, job=os.environ.get('DLC_JOB_ID'), checkpoint=str(CK),
                                   candidate_sha=subprocess.check_output(['git', '-C', str(NEW), 'rev-parse', 'HEAD'], text=True).strip(),
                                   cases=CASES, baseline_source=str(baseline) if baseline else None,
                                   config_sha256=hashlib.sha256((CK/'config.yaml').read_bytes()).hexdigest()))
    libs = json.loads((B/'runs/fastwam-h800-smoke-20261008/system-libs-prepared.json').read_text())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0',
               LD_LIBRARY_PATH=libs['ld_library_path']+':'+os.environ.get('LD_LIBRARY_PATH', ''),
               PATH=libs['prefix']+'/usr/bin:/usr/local/cuda/bin:'+os.environ['PATH'],
               VK_ICD_FILENAMES=libs['vulkan_icd'], CUDA_HOME='/usr/local/cuda',
               __EGL_VENDOR_LIBRARY_FILENAMES=str(B/'envs/system-libs/nvidia-egl-vendor.json'),
               WANDB_MODE='disabled', DIFFSYNTH_SKIP_DOWNLOAD='true')
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(143))
    for variant in (['candidate'] if baseline is not None else ['baseline', 'candidate']):
        out = root / variant
        out.mkdir(parents=True)
        repo = OLD if variant == 'baseline' else NEW
        crepo = CLIENT if variant == 'baseline' else NEW
        senv = dict(env, PYTHONPATH=':'.join(map(str, [repo, repo/'third_party', F/'src'])), TORCHINDUCTOR_COMPILE_THREADS='1')
        cenv = dict(env, PYTHONPATH=':'.join(map(str, [B/'runs/shared-robotwin-acceptance-20261021/client-deps', F/'src', crepo, B/'projects'])))
        srv = None
        try:
            with (out/'server.log').open('x') as log:
                srv = subprocess.Popen([str(SPY), __file__, 'server', str(out)], cwd=repo, env=senv,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 1800
            while 'WebSocket server started' not in (out/'server.log').read_text(errors='replace'):
                if srv.poll() is not None:
                    raise RuntimeError((out/'server.log').read_text(errors='replace')[-6000:])
                if time.monotonic() > deadline:
                    raise TimeoutError('server startup')
                time.sleep(3)
            print('SERVER_READY', variant, flush=True)
            with (out/'client.log').open('x') as log:
                subprocess.run([str(PY), __file__, 'client', str(out)], cwd=F, env=cenv,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        finally:
            if srv is not None and srv.poll() is None:
                os.killpg(srv.pid, signal.SIGTERM)
                try:
                    srv.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(srv.pid, signal.SIGKILL)
                    srv.wait()
    compare(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'server', 'client', 'compare'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', type=Path, help='Completed baseline root; requires the same GPU UUID and contract')
    args = parser.parse_args()
    if args.mode == 'run':
        run(args.output, args.baseline)
    else:
        globals()[args.mode](args.output)
