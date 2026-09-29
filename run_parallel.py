#!/usr/bin/env python3
"""Run six isolated matching shards, merge them, and prepare the LFS dataset."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

D = Path(__file__).resolve().parent
ROOT = D.parent.parent
PYTHON = '/home/ben/.lichtfeld/plugins/densification/.venv/bin/python'
DRIVER = D / 'densify_parallel_gubbangen.py'
OUT = D / 'dense_output'
STATUS = D.parent / 'web/dense_state.json'
env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
if (D / 'HOLD.json').exists():
    raise RuntimeError((D / 'HOLD.json').read_text())
workers = []
logs = []
started = time.time()
config=json.loads((D/'config.json').read_text())

def status(stage, **extra):
    progress = []
    for i, p in enumerate(workers):
        content = (D / f'logs/worker_{i}.log').read_text(errors='replace')
        matches = re.findall(r'\[worker \d+\]\s+([\d.]+)% ([^\n]+)', content)
        progress.append({'shard': i, 'pid': p.pid, 'exit_code': p.poll(), 'percent': float(matches[-1][0]) if matches else 0, 'message': matches[-1][1] if matches else 'Loading matcher'})
    payload = dict(stage=stage, elapsed_seconds=round(time.time()-started), updated_at=time.time(), workers=progress, source_model=config['source_model'], **extra)
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(STATUS)

try:
    for i in range(6):
        log = open(D / f'logs/worker_{i}.log', 'w')
        logs.append(log)
        cmd = [PYTHON, str(DRIVER), 'worker', str(OUT), '--shard', str(i), '--num-shards', '6', '--roma', 'fast', '--nns', '4', '--certainty', '0.9', '--reproj', '2', '--matches', '10000']
        workers.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    while True:
        codes = [p.poll() for p in workers]
        status('matching')
        if any(c is not None and c != 0 for c in codes):
            raise RuntimeError(f'A matching worker failed: {codes}; inspect logs/worker_*.log')
        if all(c == 0 for c in codes):
            break
        time.sleep(5)
    status('merging')
    with open(D / 'logs/merge.log', 'w') as log:
        subprocess.run([PYTHON, str(DRIVER), 'merge', str(OUT), '--num-shards', '6', '--voxel', str(config['voxel_size_model_units'])], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    dataset = D / 'lfs_dataset'
    sparse = dataset / 'sparse/0'
    sparse.mkdir(parents=True, exist_ok=True)
    for name in ('cameras.bin', 'images.bin', 'rigs.bin', 'frames.bin'):
        shutil.copy2(D / 'perspective/sparse/0' / name, sparse / name)
    for target, source in [(dataset / 'images', D / 'perspective/images'), (dataset / 'masks', D / 'perspective/masks'), (sparse / 'points3D.ply', OUT / 'points3D_dense.ply'), (D.parent / 'web/dense_point_cloud.ply', OUT / 'points3D_dense.ply')]:
        if not target.exists():
            target.symlink_to(source)
    with open(OUT / 'points3D_dense.ply', 'rb') as f:
        header = f.read(512).decode('ascii', errors='ignore')
    points = int(re.search(r'element vertex (\d+)', header)[1])
    if points == 0:
        raise RuntimeError('Dense output is empty')
    status('validating', points=points)
    subprocess.run([str(ROOT/'.venv312/bin/python'),str(D/'finalize_dense.py')],check=True,cwd=ROOT)
    status('complete', points=points, dataset=str(dataset), ply='dense_point_cloud.ply', viewer='dense/')
    (D / 'completion.json').write_text(STATUS.read_text())
except Exception as exc:
    for p in workers:
        if p.poll() is None:
            p.terminate()
    for p in workers:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
    status('failed', error=str(exc))
    raise
finally:
    for log in logs:
        log.close()
