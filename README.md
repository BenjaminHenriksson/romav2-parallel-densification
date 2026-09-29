# Parallel RoMaV2 densification

Source snapshot of the completed September 25, 2026 Insta360 X5 densification workflow. It preserves the parallel driver, project-local plugin fixes, and the exact local RoMaV2 Python source used by that workflow. The original processing workspace is unchanged.

## Workflow

1. Read a rectified COLMAP PINHOLE reconstruction, optionally with white-keep masks.
2. Select reference images with vectorized greedy visibility coverage and neighbors by sparse-track covisibility.
3. Partition references into independent, interleaved shards. The recorded run used six processes on one GPU, not six GPUs.
4. Match with RoMaV2, triangulate and filter, then merge raw shard points and optionally voxel-filter them.
5. For the Insta360 project, validate the result and prepare a LichtFeld Studio initialization dataset.

The local plugin fixes confidence filtering, sampling when masks leave few candidates, and normalized-grid pixel-center conversion. Its RoMaV2 source supports cached reference features through `match_from_features`. Keep the included source instead of substituting an unverified upstream checkout.

## Files

- `densify_parallel_gubbangen.py`: reusable `prep`, `worker`, and `merge` commands; retains its historical filename.
- `plugin/`: project-local matching, triangulation, sampling and writer code.
- `plugin/RoMaV2/`: vendored local matcher source, replacing the original absolute symlink.
- `prepare_views.py`, `run_parallel.py`, `finalize_dense.py`, `config.json`: original Insta360 scripts and run settings.
- `docs/INSTA360_WORKFLOW.md`: original workflow notes.
- `reports/insta360-2026-09-25/`: original completion and validation records.
- `docs/source-manifest.json`: original paths and SHA-256 hashes for copied files.

Images, videos, masks, reconstructions, point clouds, trained splats, checkpoints, weights and runtime logs are not included. The JSON records are historical results, not results from a fresh run of this repository.

## Environment

The recorded workstation uses Linux and Python 3.12. The matching environment currently has PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128. These observations are recorded after the run; they are not a complete historical environment lock.

On the existing workstation, the matching interpreter is:

```bash
PYTHON=/home/ben/.lichtfeld/plugins/densification/.venv/bin/python
"$PYTHON" densify_parallel_gubbangen.py --help
```

On another machine, create a Python 3.12 environment, install PyTorch/torchvision appropriate to its NVIDIA GPU and driver, then install `requirements.txt`. The bundled RoMaV2 source is imported directly; it does not need a separate package installation. The optional Insta360 preparation and preview scripts need `requirements-insta360.txt` and a compatible pycolmap rig API. Model weights are external and may be downloaded into the Torch cache on first use; they are not vendored.

## Reuse with an existing PINHOLE reconstruction

Run from this repository. Replace the three input/output paths and choose a voxel size in the reconstruction's own units. `0.023470080886203473` below is the recorded Insta360 value, not a recommended metric spacing for other datasets.

```bash
PYTHON=/path/to/python
SPARSE=/path/to/colmap/sparse/0
IMAGES=/path/to/rectified/images
OUT=/path/to/new/dense_output

"$PYTHON" densify_parallel_gubbangen.py prep "$SPARSE" "$IMAGES" "$OUT" --nns 4 --num-refs 0.8
# Add --mask-dir /path/to/masks when masks have the same relative names as images.

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
pids=()
for shard in 0 1 2 3 4 5; do
  "$PYTHON" densify_parallel_gubbangen.py worker "$OUT"     --shard "$shard" --num-shards 6 --roma fast --nns 4     --certainty 0.9 --reproj 2 --matches 10000 >"$OUT/worker_$shard.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "A worker failed; inspect the worker logs." >&2
  exit 1
fi
"$PYTHON" densify_parallel_gubbangen.py merge "$OUT" --num-shards 6 --voxel 0.023470080886203473
```

Use a fresh output directory. Each worker loads its own matcher, so worker count must fit GPU memory. `prep` writes a local Python pickle that workers consume; use preparation files you created or trust. With `--voxel 0`, merge writes only `points3D_dense_raw.ply`.

The driver supports rectified PINHOLE cameras only. It identifies tiles from the same capture by identical filename stems across view directories; other naming conventions need adaptation before treating this as a general rig workflow. The default worker reprojection threshold is 0.8 pixels; the recorded run explicitly overrides it to 2.

## Reproducing the Insta360 project

The three original orchestration scripts intentionally retain the original dataset layout and absolute interpreter/source paths. They are a preserved project recipe, not a portable one-command launcher. Review paths before running: they write logs, reports, output clouds and viewer state.

The expected parent reconstruction contains `upright/B/manhattan/sparse/0`, `upright/manifest.json`, `images`, `combined_masks`, and `web/state.json`. `prepare_views.py` generates the 1,240 masked perspective views. Run the driver's `prep` step against `perspective/sparse/0`, `perspective/images`, and `perspective/masks` to create `dense_output/prep.pkl`, then use `run_parallel.py` after adjusting its interpreter paths and `config.json` source path. `run_parallel.py` invokes `finalize_dense.py` and packages the initialization dataset; it does not train a splat. See the original notes for rig assumptions and validation checks.

## Recorded result

- Input: 310 native fisheye images, 155 rig poses, 1,240 rectified views.
- Six workers, RoMaV2 fast, four neighbors, 80% reference selection.
- Confidence 0.9; maximum reprojection error 2 pixels; minimum parallax 0.5 degrees; 10,000 requested samples per reference.
- 7,539,298 raw points, 2,429,456 after voxel filtering.
- Matching, merging and validation completed in 218 seconds according to the run record; image preparation is separate.
- All points finite and source model hashes unchanged in the saved validation.
- The dense initialization was subsequently used by the Insta360 MRNF and 3DGUT/PPISP splat runs.

## Provenance and licenses

The plugin derives from Shady Gmira's Lichtfeld-Densification-Plugin. Its GPL-3.0-or-later metadata and license are preserved in `docs/upstream-plugin-pyproject.toml` and `plugin/LICENSE`. RoMaV2 includes its MIT license and author notice in `plugin/RoMaV2/LICENSE`. These files preserve upstream attribution. The installed plugin came from an archive, so no verified upstream commit is claimed; exact copied source is identified by SHA-256 in the manifest.
