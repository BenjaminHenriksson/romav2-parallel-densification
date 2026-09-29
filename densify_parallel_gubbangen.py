#!/usr/bin/env python
"""Parallelized headless driver for the LichtFeld densification plugin.

Same algorithm, same quality settings, three fixes for scale:
1. prep: the plugin's reference selection (greedy visibility set-cover) is
   pure-Python O(k * n_imgs * pts_per_img) — hours at 4.8k images. Rewritten
   vectorized (same greedy, CSR + counting updates): minutes. Model parsing
   uses direct numpy binary readers instead of per-point Python loops.
2. worker: references are embarrassingly parallel (each ref matches only its
   own neighbors and triangulates independently) — N worker processes each
   run the plugin's own run_dense_pipeline() on an interleaved shard.
3. merge: concatenate shards (order is immaterial for a point cloud),
   write raw PLY, then apply the voxel cap.

Subcommands:
  prep   <sparse_dir> <images_dir> <out_dir> [--nns 4] [--num-refs 0.8]
  worker <out_dir> --shard I --num-shards N [--roma fast] [--certainty 0.9]
  merge  <out_dir> --num-shards N [--voxel 0.05]
"""
import argparse
import mmap
import pickle
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np

PLUGIN = Path(__file__).resolve().parent / 'plugin'


def _log(level):
    def f(msg):
        print(f'[{time.strftime("%H:%M:%S")}] [{level}] {msg}', flush=True)
    return f


def install_plugin_imports():
    sys.modules.setdefault('lichtfeld', types.SimpleNamespace(
        log=types.SimpleNamespace(info=_log('info'), warn=_log('warn'),
                                  error=_log('error'))))
    if 'densiplug' not in sys.modules:
        pkg = types.ModuleType('densiplug')
        pkg.__path__ = [str(PLUGIN)]
        sys.modules['densiplug'] = pkg


# ---------- fast COLMAP binary parsing (no per-point Python loops) ----------

def read_cameras_bin(path):
    cameras = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack('<iiQQ', f.read(24))
            assert model_id == 1, 'Dense matching requires rectified PINHOLE views'
            fx, fy, cx, cy = struct.unpack('<4d', f.read(32))
            cameras[cam_id] = (int(w), int(h), np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]))
    return cameras


def read_images_bin(path):
    """-> ids (N,), names list, qvecs (N,4), tvecs (N,3) — insertion order."""
    with open(path, 'rb') as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    off = 0
    n = struct.unpack_from('<Q', buf, off)[0]; off += 8
    ids = np.empty(n, np.int64)
    qvecs = np.empty((n, 4)); tvecs = np.empty((n, 3))
    names = []
    camera_ids = np.empty(n, np.int64)
    for i in range(n):
        ids[i] = struct.unpack_from('<I', buf, off)[0]; off += 4
        qvecs[i] = struct.unpack_from('<4d', buf, off); off += 32
        tvecs[i] = struct.unpack_from('<3d', buf, off); off += 24
        camera_ids[i] = struct.unpack_from('<I', buf, off)[0]; off += 4
        end = buf.find(b'\x00', off)
        names.append(buf[off:end].decode()); off = end + 1
        n2d = struct.unpack_from('<Q', buf, off)[0]; off += 8
        off += n2d * 24
    buf.close()
    return ids, names, qvecs, tvecs, camera_ids


def read_observations(path):
    """points3D.bin -> (obs_pt_row u32, obs_img_id u32)."""
    with open(path, 'rb') as f:
        raw = f.read()
    arr = np.frombuffer(raw, np.uint8)
    n = struct.unpack_from('<Q', raw, 0)[0]
    offsets = np.empty(n, np.int64); tlens = np.empty(n, np.int64)
    off = 8
    unpack = struct.unpack_from
    for i in range(n):
        offsets[i] = off
        tl = unpack('<Q', raw, off + 43)[0]
        tlens[i] = tl
        off += 51 + 8 * tl
    total = int(tlens.sum())
    starts = offsets + 51
    cum = np.concatenate(([0], np.cumsum(tlens)[:-1]))
    within = np.arange(total, dtype=np.int64) - np.repeat(cum, tlens)
    img_pos = np.repeat(starts, tlens) + 8 * within
    obs_img_id = arr[img_pos[:, None] + np.arange(4)].copy().view('<u4').ravel()
    obs_pt = np.repeat(np.arange(n, dtype=np.uint32), tlens)
    return obs_pt, obs_img_id


def qvec_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------- vectorized greedy visibility set-cover (same algorithm) ----------

def greedy_visibility_refs(obs_pt, obs_img_row, n_imgs, n_pts, k):
    key = obs_img_row.astype(np.uint64) << np.uint64(32) | obs_pt.astype(np.uint64)
    key = np.unique(key)
    u_img = (key >> np.uint64(32)).astype(np.int64)
    u_pt = (key & np.uint64(0xFFFFFFFF)).astype(np.int64)

    order = np.argsort(u_img, kind='stable')
    img_pts = u_pt[order]
    img_indptr = np.zeros(n_imgs + 1, np.int64)
    np.add.at(img_indptr, u_img + 1, 1)
    img_indptr = np.cumsum(img_indptr)

    order = np.argsort(u_pt, kind='stable')
    pt_imgs = u_img[order]
    pt_indptr = np.zeros(n_pts + 1, np.int64)
    np.add.at(pt_indptr, u_pt + 1, 1)
    pt_indptr = np.cumsum(pt_indptr)

    scores = (img_indptr[1:] - img_indptr[:-1]).astype(np.float64)
    covered = np.zeros(n_pts, bool)
    selected = np.empty(k, np.int64)
    for i in range(k):
        b = int(np.argmax(scores))
        selected[i] = b
        pts_b = img_pts[img_indptr[b]:img_indptr[b + 1]]
        new = pts_b[~covered[pts_b]]
        covered[new] = True
        if len(new):
            starts = pt_indptr[new]
            lens = pt_indptr[new + 1] - starts
            cum = np.concatenate(([0], np.cumsum(lens)[:-1]))
            idx = np.repeat(starts, lens) + (
                np.arange(int(lens.sum())) - np.repeat(cum, lens))
            np.subtract.at(scores, pt_imgs[idx], 1.0)
        scores[b] = -np.inf
    return np.sort(selected)


# ---------- subcommands ----------

def cmd_prep(a):
    install_plugin_imports()
    from densiplug.core.camera_models import CameraRecord
    # neighbor table now built from SfM covisibility (see cmd_prep)

    t0 = time.time()
    sparse, images_dir, out = Path(a.sparse_dir), Path(a.images_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cameras = read_cameras_bin(sparse / 'cameras.bin')
    ids, names, qvecs, tvecs, camera_ids = read_images_bin(sparse / 'images.bin')
    n = len(ids)
    order = np.argsort(ids)  # plugin sorts by image id
    print(f'[prep] {n} images, {len(cameras)} pinhole cameras', flush=True)

    records = []
    for j in order:
        w, h, K = cameras[int(camera_ids[j])]
        mask_path = str(Path(a.mask_dir) / names[j]) if a.mask_dir else None
        if mask_path: assert Path(mask_path).is_file(), mask_path
        R = qvec_to_R(qvecs[j]); t = tvecs[j]
        P = K @ np.hstack([R, t[:, None]])
        C = -R.T @ t
        records.append(CameraRecord(
            uid=int(ids[j]), image_path=str(images_dir / names[j]),
            width=w, height=h, K=K.copy(), R=R, t=t.copy(), P=P, C=C, mask_path=mask_path))

    obs_pt, obs_img_id = read_observations(sparse / 'points3D.bin')
    id2row = np.full(int(ids.max()) + 1, -1, np.int64)
    id2row[ids[order]] = np.arange(n)
    obs_img_row = id2row[obs_img_id]
    assert obs_img_row.min() >= 0
    n_pts = int(obs_pt.max()) + 1
    print(f'[prep] {n_pts:,} points, {len(obs_pt):,} obs '
          f'({time.time()-t0:.0f}s)', flush=True)

    k = int(round(a.num_refs * n)) if a.num_refs <= 1.0 else int(a.num_refs)
    refs = greedy_visibility_refs(obs_pt, obs_img_row, n, n_pts, max(1, k))
    print(f'[prep] {len(refs)} references selected ({time.time()-t0:.0f}s)', flush=True)

    # Neighbor selection by SfM covisibility, not camera position: with 79%
    # oblique imagery, co-located cameras can face opposite directions and
    # share no footprint (measured: position-based NNs give 24% of pairs
    # <10 shared points). Covisibility top-k pairs only genuinely
    # overlapping views; the pipeline's min-parallax filter handles the
    # short-baseline degenerate end.
    from scipy import sparse as sp
    A = sp.csr_matrix((np.ones(len(obs_pt), np.float32),
                       (obs_img_row, obs_pt.astype(np.int64))))
    A.data[:] = 1.0
    C = (A @ A.T).toarray()
    np.fill_diagonal(C, 0)
    # Virtual tiles from the same capture must not count as stereo baselines.
    frame_keys = np.array([Path(names[j]).stem for j in order])
    C[frame_keys[:, None] == frame_keys[None, :]] = 0
    nn = np.argsort(-C, axis=1)[:, :max(1, a.nns)].astype(np.int64)
    pairs = C[np.repeat(refs, nn.shape[1]), nn[refs].ravel()]
    print(f'[prep] covis neighbors: median {np.median(pairs):.0f} shared pts, '
          f'p10 {np.percentile(pairs, 10):.0f}', flush=True)
    with open(out / 'prep.pkl', 'wb') as f:
        pickle.dump({'records': records, 'refs': refs.tolist(), 'nn': nn}, f)
    print(f'[prep] done in {time.time()-t0:.0f}s -> {out}/prep.pkl', flush=True)


def cmd_worker(a):
    install_plugin_imports()
    from densiplug.core.config import DensePipelineConfig
    from densiplug.core.pipeline import run_dense_pipeline

    out = Path(a.out_dir)
    with open(out / 'prep.pkl', 'rb') as f:
        prep = pickle.load(f)
    shard_refs = prep['refs'][a.shard::a.num_shards]
    print(f'[worker {a.shard}] {len(shard_refs)} refs', flush=True)

    config = DensePipelineConfig(
        output_path=str(out / f'shard_{a.shard}.ply'),
        roma_setting=a.roma,
        nns_per_ref=a.nns,
        matches_per_ref=a.matches,
        certainty_thresh=a.certainty,
        reproj_thresh=a.reproj,
        sampson_thresh=5.0,
        min_parallax_deg=0.5,
        seed=1000 + a.shard,
        viz_interval=0,
        prefetch_packages=4,
        pack_workers=4,
    )

    def progress(pct, msg):
        print(f'[worker {a.shard}] {pct:5.1f}% {msg}', flush=True)

    res = run_dense_pipeline(prep['records'], shard_refs, prep['nn'], config,
                             progress_callback=progress)
    np.savez(out / f'shard_{a.shard}.npz', xyz=res.xyz, rgb=res.rgb, err=res.err)
    print(f'[worker {a.shard}] done: {res.xyz.shape[0]:,} pts, '
          f'{res.pairs_processed} pairs, {res.elapsed_seconds/60:.1f} min', flush=True)


def cmd_merge(a):
    install_plugin_imports()
    from densiplug.core.writers import write_ply
    from densiplug.densify import _voxel_downsample

    out = Path(a.out_dir)
    xs, cs = [], []
    for i in range(a.num_shards):
        d = np.load(out / f'shard_{i}.npz')
        xs.append(d['xyz']); cs.append(d['rgb'])
    xyz = np.concatenate(xs); rgb = np.concatenate(cs)
    del xs, cs
    if rgb.dtype != np.uint8:
        rgb8 = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    else:
        rgb8 = rgb
    write_ply(str(out / 'points3D_dense_raw.ply'), xyz.astype(np.float32), rgb8)
    print(f'[merge] raw: {len(xyz):,} points', flush=True)
    if a.voxel > 0:
        vx, vc = _voxel_downsample(xyz.astype(np.float32), rgb8.astype(np.float32), a.voxel)
        vc8 = np.clip(vc * 255.0, 0, 255).astype(np.uint8)
        write_ply(str(out / 'points3D_dense.ply'), vx, vc8)
        print(f'[merge] voxel {a.voxel}: {len(vx):,} points', flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep')
    p.add_argument('sparse_dir'); p.add_argument('images_dir'); p.add_argument('out_dir')
    p.add_argument('--nns', type=int, default=4)
    p.add_argument('--mask-dir')
    p.add_argument('--num-refs', type=float, default=0.8)
    w = sub.add_parser('worker')
    w.add_argument('out_dir')
    w.add_argument('--shard', type=int, required=True)
    w.add_argument('--num-shards', type=int, required=True)
    w.add_argument('--roma', default='fast')
    w.add_argument('--nns', type=int, default=4)
    w.add_argument('--certainty', type=float, default=0.9)
    # 0.8 was calibrated on per-tile solves (same-bundle pairs, sub-px
    # consistent); after the global BA the honest pairwise residual is
    # ~1.3-1.4 px, so pass ~2.0 to keep the same matches.
    w.add_argument('--reproj', type=float, default=0.8)
    w.add_argument('--matches', type=int, default=10000)
    m = sub.add_parser('merge')
    m.add_argument('out_dir')
    m.add_argument('--num-shards', type=int, required=True)
    m.add_argument('--voxel', type=float, default=0.05)
    a = ap.parse_args()
    return {'prep': cmd_prep, 'worker': cmd_worker, 'merge': cmd_merge}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
