#!/usr/bin/env python3
"""Add a completed mode trial or full Precise run to the existing point comparison."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--web-root", type=Path, required=True, help="Existing web/point-compare directory")
    ap.add_argument("--full-run", action="store_true", help="Publish a Precise run covering every prepared reference")
    args = ap.parse_args()
    report = json.loads(args.comparison.read_text())
    expected = {"precise"} if args.full_run else {"fast", "precise"}
    if report["state"] != "complete" or {r["mode"] for r in report["runs"]} != expected:
        ap.error("A completed Precise run is required" if args.full_run else "A completed fast/precise comparison is required")
    if args.full_run and len(set(report["reference_indices"])) != report["available_references"]:
        ap.error("A full run must include every prepared reference")
    manifest_path = args.web_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    center = np.asarray(manifest["center"], np.float64)
    axis = np.asarray(manifest["axis"], np.float64)
    scale = float(manifest["scale"])
    assert scale > 0 and np.isfinite(scale)
    dtype = np.dtype([("xyz", "<f4", (3,)), ("rgb", "u1", (3,)), ("pad", "u1")])
    for run in report["runs"]:
        mode = run["mode"]
        name = mode + ("_full" if args.full_run else "_trial")
        with np.load(args.comparison.parent / f"{mode}.npz") as cloud:
            xyz, rgb = cloud["xyz"], cloud["rgb"]
            assert len(xyz) == len(rgb) == run["points"]
            assert np.isfinite(xyz).all() and np.isfinite(rgb).all()
            points = np.empty(len(xyz), dtype=dtype)
            points["xyz"] = (xyz.astype(np.float64) - center) / scale * axis
            points["rgb"] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            points["pad"] = 255
        temporary = args.web_root / f"{name}.bin.tmp"
        points.tofile(temporary)
        temporary.replace(args.web_root / f"{name}.bin")
        shutil.copy2(args.comparison.parent / f"{mode}.ply", args.web_root / f"{name}.ply")
        manifest["clouds"][name] = {"count": len(points), "bytes": points.nbytes,
                                    "file": f"{name}.bin", "trial": not args.full_run}
    manifest["fullRun" if args.full_run else "modeTrial"] = {"referenceCount": len(report["reference_indices"]),
                             "neighbors": report["neighbors"], "runs": report["runs"],
                             "filtering": "No voxel or density filtering"}
    template = Path(__file__).resolve().parent / "viewer/point-compare"
    for name in ["app.js", "index.html"]:
        temporary = args.web_root / (name + ".tmp")
        shutil.copy2(template / name, temporary)
        temporary.replace(args.web_root / name)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps({"web_root": str(args.web_root), "clouds": manifest["clouds"]}, indent=2))


if __name__ == "__main__":
    main()
