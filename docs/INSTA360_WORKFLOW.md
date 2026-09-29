# Dense initialization from the accepted upright LoMa-B model

Source: ../upright/B/manhattan/sparse/0, 310 registered native fisheye images, 155 physical rig poses, 38,118 sparse points. Source binary hashes and unchanged-model validation are recorded in perspective_validation.json.

Four overlapping 110-degree 1536-square perspective views are rendered per lens, yielding 1,240 PINHOLE images. Views account for the 180-degree correction in the low-camera section. All carrier and edge masks are remapped and conservatively eroded. Sixteen virtual camera definitions encode four views per physical lens in each of the two image orientations; they share one rig, and eight views are present per capture. There are still only two physical lenses. The virtual reference sensor need not be observed in each frame. All rig transforms were numerically checked.

The parallel driver originates from the Gubbangen workflow at /home/ben/Desktop/skeppsholmen-kastellholmen-colmap-global/densification/densify_parallel_gubbangen.py. The scoped copy reads multiple PINHOLE cameras and passes masks. It selects 80% of views as references and their four highest-covisibility neighbors, excluding views from the same capture. Six disjoint shards run the cached RoMa v2 fast matcher at 512-square matching resolution, 0.9 confidence, 2-pixel maximum reprojection error in the perspective images, 0.5-degree minimum parallax, and 10,000 requested samples per reference.

The task-local plugin copy fixes confidence filtering (discard weak matches rather than raising their confidence), sampling when masks leave fewer nonzero candidates than requested, and the pixel-center conversion for normalized matching grids. The installed plugin is unchanged.

Raw shards and merged raw PLY are preserved. Voxel size is half the median nearest-neighbor spacing in the accepted sparse cloud (recorded in config.json). Units are arbitrary reconstruction units, not meters. The voxel cloud becomes the LFS initialization at lfs_dataset/sparse/0/points3D.ply. Rectified images and white-keep masks are linked into that dataset. No LFS training is launched.

The persistent user service insta360-dense-upright-b.service merges all six completed shards, checks finite values and reprojection bounds, verifies the source model hashes, and publishes a Manhattan-oriented viewer cloud. Progress is ../web/dense_state.json; final validation is validation.json. Dense and sparse clouds remain separately selectable in the web app.
