"""Data loading, deterministic cached geometry labels and synchronized D4 augmentation."""
from pathlib import Path
import csv, hashlib, os, tempfile, math
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import torch
from torch.utils.data import Dataset

FIELDS=('mask','centerline','direction','radius','junction','thin','normal','boundary','fov','direction_valid','radius_valid')
VERSION='geom_v1.0'
NEIGH=[(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]

def geometry_targets(mask,fov=None,thin_width=3.,node_radius=3,near=2,walk=4):
    mask=np.asarray(mask,dtype=bool); fov=np.ones_like(mask) if fov is None else np.asarray(fov,dtype=bool)
    mask=mask&fov; sk=skeletonize(mask).copy(); H,W=mask.shape
    # zero-pad EDT so border foreground is handled consistently
    edt=ndi.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
    padded=np.pad(sk,1); b=np.stack([padded[1+dy:1+dy+H,1+dx:1+dx+W] for dy,dx in NEIGH]); cn=np.abs(b.astype(np.int8)-np.roll(b.astype(np.int8),1,axis=0)).sum(0)/2
    nodes=sk&(cn>=3); junction=ndi.binary_dilation(nodes,iterations=node_radius)&fov if nodes.any() else nodes
    # exclude truncated image-border nodes and FOV boundary neighborhoods
    safe=ndi.binary_erosion(fov,iterations=walk,border_value=0)
    junction &= safe
    dirs=np.zeros((8,H,W),np.float32); valid=np.zeros_like(sk)
    points=set(map(tuple,np.argwhere(sk)))
    for y,x in points:
        # trace each immediate branch separately; do not jump to nearby disconnected vessels
        for dy,dx in NEIGH:
            seed=(y+dy,x+dx)
            if seed not in points: continue
            seen={(y,x),seed}; frontier={seed}; last=frontier
            for _ in range(walk-1):
                nxt={(yy+dy2,xx+dx2) for yy,xx in frontier for dy2,dx2 in NEIGH if (yy+dy2,xx+dx2) in points and (yy+dy2,xx+dx2) not in seen}
                if not nxt: break
                seen|=nxt; frontier=nxt; last=nxt
            for yy,xx in last:
                if (yy,xx)==(y,x): continue
                a=int(np.floor((math.atan2(y-yy,xx-x)%(2*math.pi))/(math.pi/4)+.5))%8
                dirs[a,y,x]=1; valid[y,x]=True
    radius=np.zeros_like(edt,dtype=np.float32); direction_valid=np.zeros_like(mask); radius_valid=np.zeros_like(mask)
    if sk.any():
        dist,idx=ndi.distance_transform_edt(~sk,return_indices=True); iy,ix=idx
        radius=edt[iy,ix].astype(np.float32); radius_valid=mask&(radius>0); direction_valid=(mask|(dist<=near))&valid[iy,ix]&fov
        dirs=dirs[:,iy,ix]*direction_valid[None]; radius*=radius_valid
    thin=mask&((2*radius-1)<=thin_width)&radius_valid
    boundary=(ndi.binary_dilation(mask)^ndi.binary_erosion(mask))&fov
    out=dict(mask=mask,centerline=sk,direction=dirs,radius=radius,junction=junction,thin=thin,normal=mask&~thin,boundary=boundary,fov=fov,direction_valid=direction_valid,radius_valid=radius_valid)
    return {k:np.asarray(v,dtype=np.float32) if k=='direction' else np.asarray(v,dtype=np.float32)[None] for k,v in out.items()}

def augment(image,target,rot=0,flip_h=False,flip_v=False):
    # directions d: (cos theta,-sin theta) in x/y coordinates
    def spatial(x):
        x=np.rot90(x,rot,axes=(-2,-1));
        if flip_h: x=np.flip(x,-1)
        if flip_v: x=np.flip(x,-2)
        return x
    image=spatial(image); out={k:spatial(v) for k,v in target.items()}
    mapping=(np.arange(8)+2*rot)%8
    if flip_h: mapping=(4-mapping)%8
    if flip_v: mapping=(-mapping)%8
    d=np.empty_like(out['direction']); d[mapping]=out['direction']; out['direction']=d
    return np.ascontiguousarray(image),{k:np.ascontiguousarray(v) for k,v in out.items()}

def read_mask(path,threshold=None):
    x=np.asarray(Image.open(path).convert('L')); t=(0 if x.max()<=1 else 127) if threshold is None else threshold
    return x>t

def read_manifest(path):
    path=Path(path).resolve(); rows=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'image','mask'}<=set(reader.fieldnames or []): raise ValueError('CSV requires image,mask columns (optional fov,id)')
        for i,r in enumerate(reader):
            def resolve(k): return str((path.parent/r[k]).resolve()) if r.get(k) else None
            item=dict(image=resolve('image'),mask=resolve('mask'),fov=resolve('fov'),id=r.get('id') or Path(r['image']).stem)
            for k in ('image','mask','fov'):
                if item[k] and not Path(item[k]).is_file(): raise FileNotFoundError(item[k])
            rows.append(item)
    if not rows: raise ValueError(f'empty manifest {path}')
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('duplicate sample ids in manifest')
    return rows

class VesselDataset(Dataset):
    def __init__(self,manifest,in_channels=3,size=256,training=False,cache_dir='geometry_cache',thin_width=3.,resize=False,mask_threshold=None):
        self.rows=read_manifest(manifest); self.channels=in_channels; self.size=size; self.training=training; self.cache=Path(cache_dir) if cache_dir else None; self.thin_width=thin_width; self.resize=resize; self.threshold=mask_threshold
        if in_channels not in (1,3): raise ValueError('in_channels must be 1 or 3')
        if self.cache: self.cache.mkdir(parents=True,exist_ok=True)
    def __len__(self): return len(self.rows)
    def _load(self,i):
        row=self.rows[i]; im=Image.open(row['image']).convert('L' if self.channels==1 else 'RGB'); m=read_mask(row['mask'],self.threshold); f=read_mask(row['fov']) if row['fov'] else np.ones_like(m)
        if m.shape!=f.shape or im.size!=(m.shape[1],m.shape[0]): raise ValueError(f"image/mask/FOV shape mismatch: {row['id']}")
        if im.size!=(self.size,self.size):
            if not self.resize: raise ValueError(f"{row['id']}: expected {self.size}x{self.size}; set data.resize=true only if intentional")
            im=im.resize((self.size,self.size),Image.Resampling.BILINEAR)
            m=np.asarray(Image.fromarray(m.astype(np.uint8)*255).resize(im.size,Image.Resampling.NEAREST))>0
            f=np.asarray(Image.fromarray(f.astype(np.uint8)*255).resize(im.size,Image.Resampling.NEAREST))>0
        arr=np.asarray(im,dtype=np.float32)/255.; arr=arr[None] if self.channels==1 else arr.transpose(2,0,1)
        return arr,m,f
    def _target(self,m,f):
        key=hashlib.sha256(VERSION.encode()+repr((m.shape,self.thin_width)).encode()+m.tobytes()+f.tobytes()).hexdigest()
        path=self.cache/(key+'.npz') if self.cache else None
        if path and path.exists():
            try:
                with np.load(path,allow_pickle=False) as z: return {k:z[k].astype(np.float32) for k in FIELDS}
            except (OSError,ValueError,KeyError): pass
        t=geometry_targets(m,f,self.thin_width)
        if path:
            fd,tmp=tempfile.mkstemp(suffix='.npz',dir=self.cache); os.close(fd)
            try: np.savez_compressed(tmp,**t); os.replace(tmp,path)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
        return t
    def __getitem__(self,i):
        img,m,f=self._load(i); t=self._target(m,f)
        if self.training:
            img,t=augment(img,t,int(np.random.randint(4)),bool(np.random.rand()<.5),bool(np.random.rand()<.5))
            if np.random.rand()<.5: img=np.clip(img*np.random.uniform(.85,1.15)+np.random.uniform(-.08,.08),0,1)
        return torch.from_numpy(np.ascontiguousarray((img-.5)/.5)),{k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in t.items()},self.rows[i]['id']

def coarse_targets(t):
    import torch.nn.functional as F
    out={k:F.max_pool2d(v,2) for k,v in t.items() if k!='radius'}
    out['radius']=F.avg_pool2d(t['radius']*t['radius_valid'],2)/F.avg_pool2d(t['radius_valid'],2).clamp_min(1e-6)
    return out
