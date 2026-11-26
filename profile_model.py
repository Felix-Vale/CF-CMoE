"""Synthetic full-model training profile; reports measured time and GPU allocation."""
import argparse,time,json
import numpy as np
import torch
from train import load_config,amp_ctx
from models import CFCMoE
from learning import JointLoss
from data import geometry_targets

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.yaml');p.add_argument('--device',default='auto');p.add_argument('--in-channels',type=int,choices=[1,3]);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--steps',type=int,default=3);a=p.parse_args();cfg=load_config(a.config);dev=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device);ch=a.in_channels or cfg['model']['in_channels'];cfg['model']['in_channels']=ch;size=cfg['data']['size'];B=a.batch_size;m=CFCMoE(**cfg['model']).to(dev).train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4);amp=dev.type=='cuda' and cfg['training'].get('amp',True);scaler=torch.amp.GradScaler('cuda',enabled=amp)
    label=np.zeros((size,size),bool);label[size//2-2:size//2+2,:]=1;label[2:size//2,size//2]=1;label[size//2:,size//2+4]=1
    t={k:torch.from_numpy(v)[None].repeat(B,1,1,1).to(dev) for k,v in geometry_targets(label).items()};criterion=JointLoss(cfg['training'],cfg['model']['region']);x=torch.randn(B,ch,size,size,device=dev);elapsed=[]
    for step in range(a.steps):
        if dev.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();opt.zero_grad(set_to_none=True)
        with amp_ctx(dev,amp):out=m(x,return_details=True);loss,stats=criterion(m,out,t,dict(eta=1.,route_mix=1.,utility=1.,perturb=1.,warm=False))
        scaler.scale(loss).backward();scaler.step(opt);scaler.update()
        if dev.type=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-start;elapsed.append(seconds);print(json.dumps(dict(step=step+1,seconds=seconds,loss=float(loss.detach()),pert_rois=stats.get('pert_rois',0),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if dev.type=='cuda' else None)),flush=True)
        del out,loss
    print(json.dumps(dict(device=str(dev),channels=ch,batch=B,size=size,params=sum(p.numel() for p in m.parameters()),mean_seconds=float(np.mean(elapsed[1:] or elapsed)))))
if __name__=='__main__':main()
