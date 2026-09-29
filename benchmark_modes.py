#!/usr/bin/env python3
"""Compare modes on the same evenly spaced prepared references, one GPU job at a time."""
import argparse
from dataclasses import replace
import hashlib
import json
import pickle
from pathlib import Path
import time

import numpy as np
import torch

from densify_parallel_gubbangen import install_plugin_imports


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep", type=Path, required=True, help="Trusted prep.pkl from this workflow")
    ap.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    ap.add_argument("--input-root", type=Path, default=Path.cwd(),
                    help="Base directory for image/mask paths stored relative to the original prep working directory")
    ap.add_argument("--refs", type=int, default=12)
    ap.add_argument("--neighbors", type=int, default=4)
    ap.add_argument("--modes", nargs="+", choices=["turbo", "fast", "base", "high", "precise"], default=["fast", "precise"])
    args = ap.parse_args()
    if args.refs < 1 or args.neighbors < 1:
        ap.error("refs and neighbors must be positive")
    if not torch.cuda.is_available():
        ap.error("This comparison requires an available CUDA GPU")
    install_plugin_imports()
    from densiplug.core.config import DensePipelineConfig
    from densiplug.core.pipeline import run_dense_pipeline
    from densiplug.core.writers import write_ply
    with args.prep.open("rb") as f:
        prep = pickle.load(f)
    def input_path(value):
        if not value:
            return value
        p = Path(value)
        return str((p if p.is_absolute() else args.input_root / p).resolve())
    # Resolve in memory only; the original prepared dataset is never rewritten.
    prep["records"] = [replace(record, image_path=input_path(record.image_path),
                               mask_path=input_path(record.mask_path)) for record in prep["records"]]
    if prep["nn"].shape[1] < args.neighbors:
        ap.error("Rebuild prep.pkl with enough neighbors before running the comparison")
    selected = np.linspace(0, len(prep["refs"]) - 1, min(args.refs, len(prep["refs"])), dtype=int)
    refs = [prep["refs"][i] for i in selected]
    used = set(refs) | {int(i) for ref in refs for i in prep["nn"][ref][:args.neighbors]}
    for i in used:
        record = prep["records"][i]
        for path in [record.image_path, record.mask_path]:
            if path and not Path(path).is_file():
                ap.error(f"Missing input: {path}; check --input-root")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"state": "running", "gpu": torch.cuda.get_device_name(),
              "prep": str(args.prep), "input_root": str(args.input_root.resolve()),
              "prep_sha256": hashlib.sha256(args.prep.read_bytes()).hexdigest(),
              "available_references": len(prep["refs"]), "reference_indices": refs,
              "reference_images": [prep["records"][i].image_path for i in refs],
              "neighbors": args.neighbors, "seed": 20260929, "runs": []}
    def save():
        tmp = args.output / "comparison.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.output / "comparison.json")
    save()
    try:
        for mode in args.modes:
            report["active_mode"] = mode
            save()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            config = DensePipelineConfig(output_path=str(args.output / f"{mode}.ply"),
                roma_setting=mode, nns_per_ref=args.neighbors, matches_per_ref=10000,
                certainty_thresh=.9, reproj_thresh=2., sampson_thresh=5., min_parallax_deg=.5,
                seed=report["seed"], viz_interval=0, prefetch_packages=1, pack_workers=1)
            started = time.monotonic()
            result = run_dense_pipeline(prep["records"], refs, prep["nn"], config,
                progress_callback=lambda pct, message: print(f"[{mode}] {pct:.1f}% {message}", flush=True))
            elapsed = time.monotonic() - started
            if not len(result.xyz) or not all(np.isfinite(a).all() for a in [result.xyz, result.rgb, result.err]):
                raise RuntimeError(f"Invalid or empty {mode} output")
            if result.err.max() > 2.00001:
                raise RuntimeError(f"{mode} output exceeds the reprojection gate")
            write_ply(config.output_path, result.xyz, np.clip(result.rgb * 255, 0, 255).astype(np.uint8))
            np.savez(args.output / f"{mode}.npz", xyz=result.xyz, rgb=result.rgb, err=result.err)
            metrics = {"mode": mode, "points": len(result.xyz), "wall_seconds": elapsed,
                "pipeline_seconds": result.elapsed_seconds, "references_with_points": result.pairs_processed,
                "reprojection_px": {"mean": float(result.err.mean()), "median": float(np.median(result.err)),
                                     "p95": float(np.percentile(result.err, 95)), "max": float(result.err.max())},
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "all_finite": True}
            report["runs"].append(metrics)
            save()
            print(json.dumps(metrics), flush=True)
        report["state"] = "complete"
        report.pop("active_mode", None)
    except Exception as exc:
        report["state"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
