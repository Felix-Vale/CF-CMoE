"""Inference on unlabelled 256x256 samples. No masks or geometry labels needed."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models import CFCMoE
from train import amp_ctx

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--images',required=True);p.add_argument('--output',default='predictions');p.add_argument('--device',default='auto');p.add_argument('--threshold',type=float,default=.5);a=p.parse_args();device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=ck['config'];m=CFCMoE(**cfg['model']).to(device).eval();m.load_state_dict(ck['model']);c=cfg['model']['in_channels'];size=cfg['data']['size'];sc=ck.get('schedule',{'eta':1.,'route_mix':1.});root=Path(a.images);paths=[root] if root.is_file() else sorted(q for q in root.rglob('*') if q.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.gif'});out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if not paths:raise FileNotFoundError('no images')
    with torch.inference_mode():
        for i,pth in enumerate(paths):
            im=Image.open(pth).convert('L' if c==1 else 'RGB');
            if im.size!=(size,size):
                if not cfg['data'].get('resize',False):raise ValueError(f'{pth}: expected {size}x{size}')
                im=im.resize((size,size),Image.Resampling.BILINEAR)
            x=np.asarray(im,dtype=np.float32)/255.;x=x[None] if c==1 else x.transpose(2,0,1);x=torch.from_numpy(np.ascontiguousarray((x-.5)/.5))[None].to(device)
            with amp_ctx(device,cfg['training'].get('amp',True)):z=m(x,sc['eta'],sc['route_mix']); pr=z.float().sigmoid()[0,0].cpu().numpy()
            Image.fromarray(np.round(pr*255).astype(np.uint8)).save(out/f'{i+1:05d}_{pth.stem}_prob.png');Image.fromarray((pr>=a.threshold).astype(np.uint8)*255).save(out/f'{i+1:05d}_{pth.stem}_mask.png')
    print(f'{len(paths)} images -> {out}')
if __name__=='__main__':main()
