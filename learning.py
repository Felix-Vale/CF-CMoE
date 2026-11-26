"""Segmentation/geometry/expert losses, sampled utility and feedback intervention."""
import torch
from torch import nn
from torch.nn import functional as F
from models.decision import CANDIDATES
from data import coarse_targets

def masked_mean(x,w):
    w=w.expand_as(x); return (x*w).sum()/w.sum().clamp_min(1)

def bce(z,y): return F.binary_cross_entropy_with_logits(z.float(),y.float().expand_as(z),reduction='none')

def seg(z,y,fov,w=None):
    z=z.float(); p=z.sigmoid(); w=fov if w is None else fov*w
    loss=masked_mean(bce(z,y),w)
    dims=(-2,-1); inter=(p*y*fov).sum(dims); denom=((p+y)*fov).sum(dims)
    valid=fov.sum(dims)>0
    dice=1-(2*inter+1)/(denom+1)
    return loss+masked_mean(dice,valid.float())

def geometry_loss(o,t):
    g=o['geometry_logits'].float(); f=t['fov']; dr=t['direction_valid']*f; rr=t['radius_valid']*f
    # BCE+Dice for sparse geometry; FOV excludes unlabelled pixels
    return .4*seg(g[:,:1],t['centerline'],f)+.25*masked_mean(bce(g[:,1:9],t['direction']),dr)+.2*masked_mean(F.smooth_l1_loss(g[:,9:10],t['radius'].clamp_min(.5).log(),reduction='none'),rr)+.15*seg(g[:,10:11],t['junction'],f)

def hardneg(o,t):
    y=t['mask']; far=F.max_pool2d(y,5,1,2)==0
    return ((o['expert_logits'][:,2:3].detach().sigmoid()>.6)&far&(t['fov']>0)).float()

def gate_loss(o,t):
    z=o['base_logits'].detach().float().sigmoid(); raw=o['raw_delta'].detach(); y=t['mask']; f=t['fov']
    # focus baseline uncertain or wrong AND meaningful proposed residual
    need=((z-y).abs()>.25).float()*f; pos=(raw>1e-5)*need; neg=(raw< -1e-5)*need
    a=bce(o['gate_positive_logits'],y)
    b=bce(o['gate_negative_logits'],1-y)
    return (masked_mean(a,pos)+masked_mean(b,neg))*.5

def expert_loss(o,t,warm_candidate=False):
    y,f=t['mask'],t['fov']; hard=hardneg(o,t)
    ws=[1+2*t['normal']+t['boundary'],1+2*t['junction'],1+2*t['thin']+hard]
    loss=sum(seg(o['expert_logits'][:,i:i+1],y,f,ws[i]) for i in range(3))/3+.25*gate_loss(o,t)
    if warm_candidate: loss=loss+.1*sum(seg(o['candidate_logits'][:,i:i+1],y,f) for i in range(1,7))/6
    return loss

def gather_regions(x,ids,r=4):
    # aligned nonoverlapping tiles; ids [B,N], spatial output [B*N,C,r,r]
    B,C,H,W=x.shape; N=ids.shape[1]; nr=H//r; nc=W//r
    tiles=x.reshape(B,C,nr,r,nc,r).permute(0,2,4,1,3,5).reshape(B,nr*nc,C,r,r)
    return tiles[torch.arange(B,device=x.device)[:,None],ids].reshape(B*N,C,r,r)

def gather_scores(x,ids):
    B,C,H,W=x.shape; return x.flatten(2).gather(2,ids[:,None].expand(-1,C,-1)).permute(0,2,1).reshape(-1,C)

@torch.no_grad()
def select_regions(o,t,n=16,r=4):
    # Sample without replacement, keeping fixed shape; if FOV insufficient, fill and mask in losses.
    def presence(v): return F.max_pool2d(v,r).flatten(1)>0
    valid=presence(t['fov']); thin=presence(t['thin']*t['fov']); j=presence(t['junction']*t['fov']); hn=presence(hardneg(o,t)); n=min(n,valid.shape[1]); all_ids=[]
    for b in range(valid.shape[0]):
        avail=torch.ones_like(valid[b]); picks=[]
        for mask,k in ((thin,6),(j,4),(hn,4),(valid,2)):
            inds=torch.where(mask[b]&valid[b]&avail)[0]; inds=inds[torch.randperm(len(inds),device=inds.device)[:min(k,n-sum(x.numel() for x in picks))]]; picks.append(inds); avail[inds]=False
        needed=n-sum(x.numel() for x in picks)
        for pool in (valid[b]&avail,avail):
            inds=torch.where(pool&avail)[0]; inds=inds[torch.randperm(len(inds),device=inds.device)[:needed]]; picks.append(inds); avail[inds]=False; needed-=len(inds)
        all_ids.append(torch.cat(picks))
    return torch.stack(all_ids)

def roi_targets(t,ids,r): return {k:gather_regions(v,ids,r) for k,v in t.items()}

@torch.no_grad()
def actual_utility(base,candidates,t):
    # balanced class BCE per ROI; score in nats, Null exactly zero
    y=t['mask']; f=t['fov']; fg=y*f; bg=(1-y)*f
    diff=bce(base,y)-bce(candidates,y); axes=(-2,-1)
    n1=fg.sum(axes); n0=bg.sum(axes)
    u=((diff*fg).sum(axes)/n1.clamp_min(1)+(diff*bg).sum(axes)/n0.clamp_min(1))/((n1>0).float()+(n0>0).float()).clamp_min(1)
    u[:,0]=0
    return u

def utility_loss(scores,u,valid,complement=True):
    a=masked_mean(F.smooth_l1_loss(scores[:,1:].float(),u[:,1:],reduction='none'),valid[:,None])
    if complement:
        def extra(z): return torch.stack([z[:,k]-torch.maximum(z[:,i+1],z[:,j+1]) for k,(i,j) in enumerate(CANDIDATES[4:],4)],1)
        a=a+.25*masked_mean(F.smooth_l1_loss(extra(scores.float()),extra(u),reduction='none'),valid[:,None])
    return a

def probe(o,t,ids,r):
    tt=roi_targets(t,ids,r); u=actual_utility(gather_regions(o['base_logits'],ids,r),gather_regions(o['candidate_logits'],ids,r),tt)
    score=gather_scores(o['utility_scores'],ids); valid=(tt['fov'].sum((1,2,3))>0).float()
    return u,score,tt,valid

@torch.no_grad()
def dependency(u,a):
    vals=[]
    for i in range(3):
        ans=torch.zeros_like(u[:,0])
        for k,s in enumerate(CANDIDATES):
            if i in s:
                subset=tuple(e for e in s if e!=i); j=CANDIDATES.index(subset); ans+=a[:,k]*(u[:,k]-u[:,j]).relu()
        vals.append(ans)
    return torch.stack(vals,1)

class UtilityFeedbackLearning(nn.Module):
    def __init__(self,region=4,prob=.5,ratio=.25,strength=(.3,.6),value_guided=True,hard_negative=True,reestimate=True,consistency=True):
        super().__init__(); self.r=region; self.prob=prob; self.ratio=ratio; self.strength=strength; self.value_guided=value_guided; self.hard_negative=hard_negative; self.reestimate=reestimate; self.consistency=consistency
    def forward(self,model,o,t,ids,u,tt,eta=1.,route_mix=1.):
        r=self.r; ctx={k:gather_regions(v,ids,r) for k,v in o['decision_context'].items()}; msg=[gather_regions(v,ids,r) for v in o['expert_messages']]; support=gather_regions(o['path_support'],ids,r)
        a=gather_scores(o['route_weights'],ids).detach(); hn=gather_regions(hardneg(o,t),ids,r)>0; y=tt['mask']; f=tt['fov']; B,C,H,W=msg[0].shape
        with torch.no_grad():
            contribution=dependency(u,a); active=torch.rand(B,device=y.device)<self.prob
            positive=((tt['thin']+tt['junction'])*y*f).sum((1,2,3))>0; neg=hn.any(dim=(1,2,3))&~positive&self.hard_negative
            e=contribution.argmax(1) if self.value_guided else torch.randint(3,(B,),device=y.device)
            good=contribution.max(1).values>1e-5 if self.value_guided else torch.ones_like(active)
            pos=active&positive&good; neg=active&neg; e=torch.where(neg,torch.full_like(e,2),e)
            region_mask=torch.where(neg[:,None,None,None],hn,(y*f)>0).float()
            # perturb all valid pixels in a positive ROI to disturb spatial evidence, background only for hard negatives
            region_mask=torch.where(pos[:,None,None,None],f,region_mask)
            alpha=torch.empty(B,1,1,1,device=y.device).uniform_(*self.strength); mods=[]
            for i,m in enumerate(msg):
                energy=(m.detach().float().abs()*region_mask).sum((-2,-1)); nch=max(1,round(C*self.ratio)); idx=energy.topk(nch,1).indices
                ch=torch.zeros_like(energy).scatter_(1,idx,1)[:,:,None,None]; sign=neg.float()-pos.float(); mod=1+alpha*sign[:,None,None,None]*(e==i)[:,None,None,None]*ch*region_mask; mods.append(mod.to(m.dtype))
        pert=model.decision(ctx,[m*s for m,s in zip(msg,mods)],support,model.heads[1],eta,route_mix)
        # verify hard-negative intervention actually increases pre-gate foreground residual T
        with torch.no_grad():
            old=gather_regions(o['raw_delta'][:,2:3],ids,r); inc=((pert['raw_delta'][:,2:3].detach()-old)*hn).sum((1,2,3))>1e-6
            chosen=pos|(neg&inc); valid=chosen.float(); f2=f*valid[:,None,None,None]
            # if failed, excluded from losses; no artificial target change
            ut=actual_utility(ctx['base'],pert['candidate_logits'],tt)
            clean=gather_regions(o['logits'],ids,r).float().sigmoid(); alt=pert['candidate_logits'].detach().float().sigmoid(); best=((alt>=.5)==(y>0)).any(1,keepdim=True)
            weight=f2*((clean>=.5)==(y>0))*(torch.maximum(clean,1-clean)>=.7)*best
        if not bool(chosen.any()): return o['logits'].sum()*0,{'pert_rois':0,'route_shift':0.}
        loss=seg(pert['logits'],y,f2)
        if self.reestimate: loss=loss+.5*utility_loss(pert['utility_scores'][:,:,0,0],ut,valid)
        if self.consistency: loss=loss+.5*masked_mean((pert['logits'].float().sigmoid()-clean).square(),weight)
        loss=loss+.1*gate_loss(pert,{**tt,'fov':f2})
        shift=((pert['route_weights'][:,:,0,0].detach()-a).abs().sum(1)*valid).sum()/valid.sum().clamp_min(1)
        return loss,{'pert_rois':int(chosen.sum()),'route_shift':float(shift)}

def schedule(epoch,cfg):
    if cfg.get('full_model',False): return dict(eta=1.,route_mix=1.,utility=1.,perturb=1.,warm=False)
    e0,e1,e2,e3=[cfg.get(k,v) for k,v in [('expert_start',10),('router_start',20),('utility_start',30),('perturb_start',50)]]
    ramp=lambda start,length: max(0.,min(1.,(epoch-start+1)/length)) if epoch>=start else 0.
    return dict(eta=ramp(e0,10),route_mix=ramp(e1,10),utility=ramp(e2,5),perturb=ramp(e3,10),warm=epoch<e2)

class JointLoss(nn.Module):
    def __init__(self,cfg,region=4):
        super().__init__(); self.cfg=cfg; self.r=region; self.upl=UtilityFeedbackLearning(region=region,**cfg.get('upl',{})); self.n=cfg.get('utility_rois',16)
    def forward(self,model,out,t,sched):
        c,f=out['coarse'],out['fine']; tc=coarse_targets(t); zero=f['logits'].sum()*0
        ls=seg(f['logits'],t['mask'],t['fov'])+.5*seg(c['logits'],tc['mask'],tc['fov'])
        if sched['eta']>0: ls=ls+.2*seg(f['base_logits'],t['mask'],t['fov'])+.1*seg(c['base_logits'],tc['mask'],tc['fov'])
        lg=geometry_loss(f,t)+.5*geometry_loss(c,tc); le=lu=lp=zero; stats={}
        if sched['eta']>0: le=expert_loss(f,t,sched['warm'])+.5*expert_loss(c,tc,sched['warm'])
        if sched['utility']>0 or sched['perturb']>0:
            ic=select_regions(c,tc,self.n,self.r); ix=select_regions(f,t,self.n,self.r)
            uc,sc,_,vc=probe(c,tc,ic,self.r); uf,sf,tf,vf=probe(f,t,ix,self.r)
            lu=utility_loss(sf,uf,vf)+.5*utility_loss(sc,uc,vc)
            stats['utility_mae']=float(masked_mean((sf.detach()-uf).abs(),vf[:,None]))
            stats['pair_gain']=float(masked_mean(torch.stack([uf[:,k]-torch.maximum(uf[:,i+1],uf[:,j+1]) for k,(i,j) in enumerate(CANDIDATES[4:],4)],1),vf[:,None]))
            if sched['perturb']>0: lp,pstats=self.upl(model,f,t,ix,uf,tf,sched['eta'],sched['route_mix']); stats.update(pstats)
        losses={'seg':ls,'geo':lg,'expert':le,'utility':lu,'pert':lp}
        w=self.cfg.get('weights',{}); total=ls+w.get('geo',.2)*lg+w.get('expert',.2)*le+w.get('utility',.1)*sched['utility']*lu+w.get('pert',.3)*sched['perturb']*lp
        return total,{**{k:float(v.detach()) for k,v in losses.items()},**stats}
