"""Render masked perspective tiles while preserving the calibrated X5 rig geometry."""
from pathlib import Path
import json,collections,concurrent.futures,hashlib
import cv2,numpy as np,pycolmap
from scipy.spatial.transform import Rotation
D=Path(__file__).resolve().parent;ROOT=D.parent;OUT=D/'perspective';SP=OUT/'sparse/0';SP.mkdir(parents=True,exist_ok=True)
N=1536;FOV=110.;f=N/(2*np.tan(np.deg2rad(FOV/2)));K=np.array([[f,0,N/2],[0,f,N/2],[0,0,1.]])
angles=[(-40,-40),(40,-40),(-40,40),(40,40)]
rots=[Rotation.from_euler('YX',a,degrees=True).as_matrix().T for a in angles]
cv2.setNumThreads(1)
SOURCE=ROOT/'upright/B/manhattan/sparse/0'
src=pycolmap.Reconstruction(str(SOURCE))
orientation={r['name']:int(r['rotation_degrees']==180) for r in json.loads((ROOT/'upright/manifest.json').read_text())}
Q=np.diag([-1.,-1.,1.])
ref_poses={}
for im in src.images.values():
 if im.name.startswith('lens-0/'):
  T=np.eye(4);T[:3]=im.cam_from_world().matrix();A=np.eye(4);A[:3,:3]=rots[0];ref_poses[im.frame_id]=A@T
source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.bin')}
y,x=np.mgrid[:N,:N];view_rays=np.c_[((x.ravel()+.5-N/2)/f),((y.ravel()+.5-N/2)/f),np.ones(N*N)]
maps={};coverage={}
for cid,cam in src.cameras.items():
 for flip in [0,1]:
  for v,baseR in enumerate(rots):
   R=baseR@(Q if flip else np.eye(3))
   rays=view_rays@R
   uv=cam.img_from_cam(rays)
   uv[~np.isfinite(uv)]=-100
   maps[cid,flip,v]=((uv[:,0]-.5).reshape(N,N).astype(np.float32),(uv[:,1]-.5).reshape(N,N).astype(np.float32))
 xx,yy=np.meshgrid(np.arange(15,3840,30),np.arange(15,3840,30));pix=np.c_[xx.ravel(),yy.ravel()]
 c=np.array(cam.params[2:4]);pix=pix[np.linalg.norm(pix-c,axis=1)<1689.6]
 norm=cam.cam_from_img(pix);rays=np.c_[norm,np.ones(len(norm))];covered=np.zeros(len(norm),bool)
 for R in rots:
  q=rays@R.T;uv=q[:,:2]/q[:,2,None]
  covered|=(q[:,2]>0)&np.all(np.abs(uv)<np.tan(np.deg2rad(FOV/2)),axis=1)
 coverage[str(cid)]=float(covered.mean());assert covered.mean()>.999
print('Perspective coverage:',coverage,flush=True)
for lens in range(2):
 for flip in [0,1]:
  for v in range(4):
   for folder in ['images','masks']:(OUT/f'{folder}/lens-{lens}_orientation-{flip}_view-{v}').mkdir(parents=True,exist_ok=True)
views=[]
def process(im):
 lens=int(im.name.split('/')[0][-1]);name=Path(im.name).name
 rgb=cv2.imread(str(ROOT/'images'/im.name));mask=cv2.imread(str(ROOT/'combined_masks'/Path(im.name).parent/(name+'.png')),0)
 assert rgb is not None and mask is not None
 points=[p for p in im.points2D if p.has_point3D()]
 ids=np.array([p.point3D_id for p in points],dtype=np.int64)
 norm=src.cameras[im.camera_id].cam_from_img(np.array([p.xy for p in points]));rays=np.c_[norm,np.ones(len(norm))]
 pose=np.eye(4);pose[:3]=im.cam_from_world().matrix()
 result=[];flip=orientation[im.name]
 for v,baseR in enumerate(rots):
  R=baseR@(Q if flip else np.eye(3))
  mx,my=maps[im.camera_id,flip,v]
  tile=cv2.remap(rgb,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
  keep=cv2.remap(mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
  keep=cv2.erode(keep,np.ones((5,5),np.uint8))
  rel=Path(f'lens-{lens}_orientation-{flip}_view-{v}')/name
  assert cv2.imwrite(str(OUT/'images'/rel),tile,[cv2.IMWRITE_PNG_COMPRESSION,3])
  assert cv2.imwrite(str(OUT/'masks'/rel),keep)
  q=rays@R.T;xy=q[:,:2]/q[:,2,None]*f+N/2
  valid=(q[:,2]>0)&np.isfinite(xy).all(axis=1)&(xy>=0).all(axis=1)&(xy<N).all(axis=1)
  good=np.flatnonzero(valid);ij=xy[good].astype(int);good=good[keep[ij[:,1],ij[:,0]]>0]
  T=np.eye(4);T[:3,:3]=R;T=T@pose
  result.append({'id':(im.image_id-1)*4+v+1,'camera_id':lens*4+v+1+8*flip,'frame_id':im.frame_id,'source_image_id':im.image_id,'name':str(rel),'pose':T,'xy':xy[good],'pids':ids[good],'valid_fraction':float((keep>0).mean())})
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for n,result in enumerate(pool.map(process,sorted(src.images.values(),key=lambda im:im.image_id)),1):
  views.extend(result)
  if n%10==0:print(f'Rendered {n}/{len(src.images)} source images ({n*4} perspective views)',flush=True)
tracks=collections.defaultdict(list)
for v in views:
 for i,pid in enumerate(v['pids']):tracks[int(pid)].append((v['id'],i))
# Remove points without enough retained observations, keeping indices consistent.
valid_pids={pid for pid,tr in tracks.items() if len(tr)>=2};tracks=collections.defaultdict(list)
for v in views:
 good=np.array([int(pid) in valid_pids for pid in v['pids']]);v['xy']=v['xy'][good];v['pids']=v['pids'][good]
 for i,pid in enumerate(v['pids']):tracks[int(pid)].append((v['id'],i))
byframe=collections.defaultdict(list)
for v in views:byframe[v['frame_id']].append(v)
for frame in byframe.values():assert len(frame)==8
cameras=''.join(f'{i} PINHOLE {N} {N} {f:.17g} {f:.17g} {N/2} {N/2}\n' for i in range(1,17))
(SP/'cameras.txt').write_text(cameras)
def pose_values(T):return [*Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True),*T[:3,3]]
with (SP/'images.txt').open('w') as out:
 for v in sorted(views,key=lambda v:v['id']):
  out.write(f"{v['id']} "+' '.join(f'{x:.17g}' for x in pose_values(v['pose']))+f" {v['camera_id']} {v['name']}\n")
  out.write(' '.join(f'{xy[0]:.12g} {xy[1]:.12g} {pid}' for xy,pid in zip(v['xy'],v['pids']))+'\n')
with (SP/'points3D.txt').open('w') as out:
 for pid,tr in sorted(tracks.items()):
  p=src.points3D[pid]
  out.write(f'{pid} '+' '.join(f'{x:.17g}' for x in p.xyz)+' '+' '.join(str(int(x)) for x in p.color)+f' {p.error:.12g} '+' '.join(f'{im} {idx}' for im,idx in tr)+'\n')
extr={}
for fid,vs in byframe.items():
 for v in vs:extr.setdefault(v['camera_id'],v['pose']@np.linalg.inv(ref_poses[fid]))
assert len(extr)==16
(SP/'rigs.txt').write_text('1 16 CAMERA 1 '+' '.join(f'CAMERA {cid} 1 '+' '.join(f'{x:.17g}' for x in pose_values(extr[cid])) for cid in range(2,17))+'\n')
maxrig=0
with (SP/'frames.txt').open('w') as out:
 for fid,vs in sorted(byframe.items()):
  vs.sort(key=lambda v:v['camera_id']);T=ref_poses[fid]
  for v in vs:maxrig=max(maxrig,float(np.max(np.abs(v['pose']-extr[v['camera_id']]@T))))
  out.write(f'{fid} 1 '+' '.join(f'{x:.17g}' for x in pose_values(T))+' 8 '+' '.join(f"CAMERA {v['camera_id']} {v['id']}" for v in vs)+'\n')
assert maxrig<1e-10
rec=pycolmap.Reconstruction(str(SP));assert rec.num_reg_images()==1240 and len(rec.rigs)==1 and len(rec.frames)==155
rec.write(str(SP))
for p in SOURCE.glob('*.bin'):assert hashlib.sha256(p.read_bytes()).hexdigest()==source_hashes[p.name]
manifest=[{k:v[k] for k in ['id','camera_id','frame_id','source_image_id','name','valid_fraction']} for v in views]
(OUT/'view_manifest.json').write_text(json.dumps(manifest,indent=2))
report={'source_images':310,'perspective_images':len(views),'rigs':1,'frames':len(byframe),'virtual_cameras':16,'source_model':str(SOURCE),'rotation_corrected_views':True,'resolution':[N,N],'field_of_view_degrees':FOV,'views_per_lens':4,'virtual_orientations_yaw_pitch':angles,'valid_lens_coverage':coverage,'retained_sparse_points':len(tracks),'source_sparse_points':src.num_points3D(),'max_rig_transform_discrepancy':maxrig,'source_model_unchanged':True,'source_hashes':source_hashes}
(D/'perspective_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
