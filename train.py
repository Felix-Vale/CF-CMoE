import argparse,csv,json,random,time,os,math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from models import CFCMoE
from data import VesselDataset
from learning import JointLoss,schedule
from metrics import metrics_one,aggregate



def load_config(path):
    with open(path,encoding='utf-8') as f:return yaml.safe_load(f)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_dataset(cfg,split,training=False):
    d=cfg['data']; return VesselDataset(d[split+'_csv'],cfg['model']['in_channels'],d['size'],training,d.get('cache_dir'),d.get('thin_width',3.),d.get('resize',False),d.get('mask_threshold'))

def make_loader(ds,cfg,shuffle=False):
    return DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=shuffle,num_workers=cfg['data'].get('workers',0),pin_memory=torch.cuda.is_available(),drop_last=False)

def amp_ctx(device,enabled): return torch.autocast(device_type=device.type,dtype=torch.float16,enabled=enabled and device.type=='cuda')
def append_json(path,row):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        return None if isinstance(x,float) and not math.isfinite(x) else x
    with open(path,'a',encoding='utf-8') as f:f.write(json.dumps(clean(row),ensure_ascii=False)+'\n')

@torch.no_grad()
def validate(model,loader,device,amp,eta=1.,route_mix=1.,threshold=.5,save_dir=None):
    from PIL import Image
    import re
    model.eval(); rows=[]; routes=[]; scales=[]; expert_scores=[]; coarse_recalls=[]; elapsed=0.; n=0
    if save_dir: Path(save_dir).mkdir(parents=True,exist_ok=True)
    for x,t,ids in loader:
        x=x.to(device,non_blocking=True)
        if device.type=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        with amp_ctx(device,amp):out=model(x,eta=eta,route_mix=route_mix,return_details=True)
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed+=time.perf_counter()-start; n+=x.shape[0]; p=out['logits'].float().sigmoid().cpu().numpy()
        if 'route_weights' in out['fine']:
            routes.append(torch.stack([out[s]['route_weights'].float().mean((0,2,3)).cpu() for s in ('coarse','fine')]).numpy())
            scales.append(torch.stack([out[s]['scale_weights'].float().mean((0,3,4)).cpu() for s in ('coarse','fine')]).numpy())
        if 'expert_logits' in out['fine']:
            from data import coarse_targets
            ct=coarse_targets(t)
            for ib in range(x.shape[0]):
                yc=ct['mask'][ib,0].numpy()>0; fc=ct['fov'][ib,0].numpy()>0
                pc=out['coarse']['logits'][ib,0].float().sigmoid().cpu().numpy()>=threshold
                if (yc&fc).any():coarse_recalls.append(float((pc&yc&fc).sum()/(yc&fc).sum()))
                y=t['mask'][ib,0].numpy()>0; ff=t['fov'][ib,0].numpy()>0
                aux=out['fine']['expert_logits'][ib].float().sigmoid().cpu().numpy()>=threshold; ps=aux&ff[None]
                target_y=y&ff; den=ps.sum((1,2))+target_y.sum(); expert_scores.append(np.where(den>0,2*(ps&target_y).sum((1,2))/np.maximum(den,1),1.))
        for i,name in enumerate(ids):
            target={k:v[i].numpy() for k,v in t.items()}; row=metrics_one(p[i],target,threshold); rows.append({'id':name,**row})
            if save_dir:
                name=re.sub(r'[^a-zA-Z0-9_.-]+','_',name); pred=(p[i,0]>=threshold)&(target['fov'][0]>0)
                Image.fromarray(pred.astype(np.uint8)*255).save(Path(save_dir)/f'{len(rows):05d}_{name}_mask.png')
                Image.fromarray(np.round(p[i,0]*255).astype(np.uint8)).save(Path(save_dir)/f'{len(rows):05d}_{name}_prob.png')
    summary=aggregate(rows); summary['ms_per_image']=1000*elapsed/max(n,1)
    if routes:summary['route_mean_coarse_fine']=np.mean(routes,axis=0).tolist(); summary['scale_mean_coarse_fine']=np.mean(scales,axis=0).tolist()
    if expert_scores:summary['expert_fine_dice_NJT']=np.mean(expert_scores,axis=0).tolist()
    if coarse_recalls:summary['coarse_presence_recall']=float(np.mean(coarse_recalls))
    return summary,rows

def save_atomic(obj,path):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def main():
    parser=argparse.ArgumentParser();
    parser.add_argument('--config',default='config_OCTA_3M.yaml');

    parser.add_argument('--resume');
    parser.add_argument('--device',default='auto');
    parser.add_argument('--in-channels',type=int,choices=[1,3]);
    parser.add_argument('--output-dir');
    parser.add_argument('--batch-size',type=int);
    parser.add_argument('--epochs',type=int);
    parser.add_argument('--accum-steps',type=int);
    parser.add_argument('--full-model',action='store_true');
    args=parser.parse_args()
    cfg=load_config(args.config); tr=cfg['training']
    if args.in_channels:cfg['model']['in_channels']=args.in_channels
    for k,v in [('output_dir',args.output_dir),('batch_size',args.batch_size),('epochs',args.epochs),('accum_steps',args.accum_steps)]:
        if v is not None:tr[k]=v
    if args.full_model:tr['full_model']=True
    accum=max(1,int(tr.get('accum_steps',1)))
    seed_all(cfg['seed']); device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    amp=bool(tr.get('amp',True) and device.type=='cuda'); run=Path(tr['output_dir']); run.mkdir(parents=True,exist_ok=True)
    if (run/'last.pt').exists() and not args.resume: raise FileExistsError(f'{run} already has last.pt; use --resume or a new output directory')
    train_ds=make_dataset(cfg,'train',True); val_ds=make_dataset(cfg,'val')
    common={r['image'] for r in train_ds.rows}&{r['image'] for r in val_ds.rows}
    if common:raise ValueError(f'Train/val share {len(common)} image paths; fix split')
    train_loader=make_loader(train_ds,cfg,True); val_loader=make_loader(val_ds,cfg)
    model=CFCMoE(**cfg['model']).to(device); criterion=JointLoss(tr,cfg['model']['region'])
    opt=torch.optim.AdamW(model.parameters(),lr=tr['lr'],weight_decay=tr['weight_decay']); schedlr=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=tr['epochs'],eta_min=tr['min_lr']); scaler=torch.amp.GradScaler('cuda',enabled=amp)
    start_epoch=0;best=-1.;best_full=-1.
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)
        if ck['config']['model']!=cfg['model']:raise ValueError('resume model config differs')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);schedlr.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler']);start_epoch=ck['epoch']+1;best=ck['best'];best_full=ck.get('best_full',-1.)
        if 'rng' in ck:
            random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy']);torch.set_rng_state(ck['rng']['torch']);
            if device.type=='cuda' and ck['rng']['cuda'] is not None:torch.cuda.set_rng_state_all(ck['rng']['cuda'])
    with (run/'config_used.yaml').open('w',encoding='utf-8') as f:yaml.safe_dump(cfg,f,sort_keys=False,allow_unicode=True)
    print(f'Device={device}, AMP={amp}, params={sum(p.numel() for p in model.parameters()):,}, train={len(train_ds)}, val={len(val_ds)}',flush=True)
    for epoch in range(start_epoch,tr['epochs']):
        model.train(); sc=schedule(epoch,tr); sums={}; count=0; begin=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        for j,(x,t,_) in enumerate(train_loader):
            x=x.to(device,non_blocking=True);t={k:v.to(device,non_blocking=True) for k,v in t.items()}
            group_start=(j//accum)*accum; group_size=min(accum,len(train_loader)-group_start)
            with amp_ctx(device,amp):out=model(x,eta=sc['eta'],route_mix=sc['route_mix'],return_details=True); loss,stats=criterion(model,out,t,sc)
            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss epoch={epoch} iter={j}')
            scaler.scale(loss/group_size).backward()
            if (j+1)%accum==0 or j+1==len(train_loader):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),tr['clip_grad']);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            stats['total']=float(loss.detach());count+=1
            for k,v in stats.items():sums[k]=sums.get(k,0.)+v
            if (j+1)%tr.get('log_interval',10)==0 or j==0:print(f'epoch {epoch+1}/{tr["epochs"]} iter {j+1}/{len(train_loader)} loss {stats["total"]:.4f} eta={sc["eta"]:.2f} pert={sc["perturb"]:.2f}',flush=True)
            del out,loss
        peak=torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else None; train_seconds=time.perf_counter()-begin
        val,_=validate(model,val_loader,device,amp,sc['eta'],sc['route_mix'],tr.get('threshold',.5));lr=opt.param_groups[0]['lr'];schedlr.step()
        improved=val['dice']>best;best=max(best,val['dice']);full=all(sc[k]>=1. for k in ('eta','route_mix','utility','perturb'));improved_full=full and val['dice']>best_full
        if improved_full:best_full=val['dice']
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedlr.state_dict(),scaler=scaler.state_dict(),epoch=epoch,best=best,best_full=best_full,config=cfg,schedule=sc,rng=rng)
        save_atomic(ck,run/'last.pt')
        if tr.get('save_every_epoch',True):save_atomic(ck,run/f'epoch_{epoch+1:03d}.pt')
        if improved:save_atomic(ck,run/'best.pt')
        if improved_full:save_atomic(ck,run/'best_full.pt')
        row=dict(epoch=epoch+1,lr=lr,train={k:v/count for k,v in sums.items()},val=val,schedule=sc,train_seconds=train_seconds,peak_vram_gib=peak);append_json(run/'history.jsonl',row)
        print(f'epoch {epoch+1}: dice={val["dice"]:.4f}, clDice={val["cldice"]:.4f}, best={best:.4f}, train_seconds={train_seconds:.1f}, peak GiB={peak}',flush=True)
if __name__=='__main__':main()
