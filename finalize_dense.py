"""Validate the six dense shards and publish a Manhattan-oriented preview."""
from pathlib import Path
import hashlib
import json
import re
import time
import numpy as np
import pycolmap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=Path(__file__).resolve().parent
ROOT=D.parent
OUT=D/'dense_output'
WEB=ROOT/'web/dense'
WEB.mkdir(exist_ok=True)
config=json.loads((D/'config.json').read_text())
checks=json.loads((D/'perspective_validation.json').read_text())
for name,digest in checks['source_hashes'].items():
    assert hashlib.sha256((Path(config['source_model'])/name).read_bytes()).hexdigest()==digest
shards=[];total=0;error_sum=0.;maximum_error=0.
for i in range(6):
    with np.load(OUT/f'shard_{i}.npz') as data:
        xyz,rgb,err=data['xyz'],data['rgb'],data['err']
        assert len(xyz)>0 and len(xyz)==len(rgb)==len(err)
        assert np.isfinite(xyz).all() and np.isfinite(rgb).all() and np.isfinite(err).all()
        assert float(err.max())<=2.00001
        total+=len(xyz);error_sum+=float(err.astype(np.float64).sum());maximum_error=max(maximum_error,float(err.max()))
        shards.append(dict(shard=i,points=len(xyz),mean_max_view_error_px=float(err.mean())))

def read_ply(path):
    with open(path,'rb') as f:
        lines=[]
        while True:
            line=f.readline();assert line,'Truncated PLY header'
            lines.append(line)
            if line.strip()==b'end_header':break
        offset=f.tell()
    header=b''.join(lines).decode('ascii')
    assert 'format binary_little_endian 1.0' in header
    count=int(re.search(r'element vertex (\d+)',header)[1])
    dtype=np.dtype([('xyz','<f4',(3,)),('rgb','u1',(3,))])
    assert path.stat().st_size==offset+count*dtype.itemsize
    return np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=(count,))

raw=read_ply(OUT/'points3D_dense_raw.ply');assert len(raw)==total
cloud=read_ply(OUT/'points3D_dense.ply')
assert len(cloud)>checks['source_sparse_points'],'Dense initialization has fewer points than sparse input'
for i in range(0,len(cloud),1000000):assert np.isfinite(cloud['xyz'][i:i+1000000]).all()
rec=pycolmap.Reconstruction(config['source_model'])
frames=sorted((im for im in rec.images.values() if im.name.startswith('lens-0/')),key=lambda im:im.name)
trajectory=np.array([im.projection_center() for im in frames])
indices=np.linspace(0,len(cloud)-1,min(len(cloud),200000)).astype(int)
xyz=np.asarray(cloud['xyz'][indices],np.float64);rgb=cloud['rgb'][indices].astype(np.float64)/255
keep=np.all((xyz>=np.percentile(xyz,1,axis=0))&(xyz<=np.percentile(xyz,99,axis=0)),axis=1)
center=np.median(xyz[keep],axis=0);xyz-=center;trajectory-=center
extent=np.ptp(xyz[keep],axis=0);scale=float(extent.max()/2)
axis=np.diag([1.,-1.,-1.])
vertices=np.c_[xyz[keep]@axis/scale,rgb[keep]]
path=np.c_[trajectory@axis/scale,np.tile([1.,.38,.1],(len(trajectory),1))]
preview=dict(point_count=len(vertices),path_count=len(path),vertices=np.round(np.r_[vertices,path],5).ravel().tolist(),orientation='Manhattan axes; no PCA')
(WEB/'cloud.json').write_text(json.dumps(preview,separators=(',',':')))
fig=plt.figure(figsize=(14,6))
ax=fig.add_subplot(121,projection='3d');plot=xyz[:,[0,2,1]]*np.array([1,1,-1]);track=trajectory[:,[0,2,1]]*np.array([1,1,-1])
ax.scatter(*plot[keep].T,c=rgb[keep],s=.2,alpha=.6);ax.plot(*track.T,color='#d14900');ax.set(xlabel='X',ylabel='Z',zlabel='Height (−Y)',title='Dense Manhattan-aligned room');ax.set_box_aspect(np.maximum(extent[[0,2,1]],1e-3));ax.view_init(22,-65)
bx=fig.add_subplot(122);bx.scatter(xyz[keep,0],xyz[keep,2],c=rgb[keep],s=.2,alpha=.6);bx.plot(trajectory[:,0],trajectory[:,2],color='#d14900');bx.set_aspect('equal');bx.set(xlabel='X',ylabel='Z',title='Dense floor plan')
fig.suptitle(f'RoMa v2 · {len(cloud):,} dense points · Manhattan-aligned LoMa-B poses');fig.tight_layout();fig.savefig(WEB/'overview.png',dpi=150);plt.close(fig)
for name,source in [('point_cloud.ply',OUT/'points3D_dense.ply'),('raw_point_cloud.ply',OUT/'points3D_dense_raw.ply')]:
    (WEB/name).symlink_to(source)
report=dict(source_model=config['source_model'],raw_points=total,points=len(cloud),shards=shards,
    mean_max_view_error_px=error_sum/total,max_view_error_px=maximum_error,
    voxel_size_model_units=config['voxel_size_model_units'],all_points_finite=True,
    source_model_unchanged=True,lfs_dataset=str(D/'lfs_dataset'),
    source_registered_images=310,perspective_images=1240,registered_frames=155,
    alignment='Manhattan',masks='White keeps pixels; carrier and outer lens rim excluded')
(D/'validation.json').write_text(json.dumps(report,indent=2))
(WEB/'validation.json').write_text(json.dumps(report,indent=2))
(WEB/'version.json').write_text(json.dumps(dict(revision=str(time.time_ns()),alignment='Manhattan',dense=True)))
state=json.loads((ROOT/'web/state.json').read_text())
state.update(stage='Dense reconstruction complete',model=dict(dense=True,registered_images=310,registered_frames=155,points3D=len(cloud),mean_reprojection_error_px=error_sum/total))
(WEB/'state.json').write_text(json.dumps(state))
(D/'lfs_dataset/README.md').write_text('Dense initialization from the Manhattan-aligned upright LoMa-B reconstruction.\n\nLoad this directory in LichtFeld Studio as a COLMAP dataset. Images are rectified PINHOLE views with white-keep masks excluding the carrier and lens rim. The sparse/0/points3D.ply file is the dense initialization; source sparse tracks remain archived in the perspective reconstruction. Coordinates retain the original arbitrary scale and Manhattan alignment. No training has been started.\n')
print(json.dumps(report,indent=2),flush=True)
