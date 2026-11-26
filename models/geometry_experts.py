"""Geometry field, shared contrastive sampling and three evidence experts."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .backbone import up, gn, LiteBlock


def conv(ci,co): return nn.Conv2d(ci,co,1)


def decode_geometry(z):
    # radius in input pixel units; bounded log radius for safe sampling
    return torch.cat((z[:,:9].sigmoid(), z[:,9:10].float().clamp(math.log(.5),math.log(24.)).exp().to(z.dtype),z[:,10:11].sigmoid()),1)


def geo_norm(g):
    return torch.cat((g[:,:9],g[:,9:10]/8.,g[:,10:11]),1)


class GeometryField(nn.Module):
    def __init__(self,c=16,b=8,z=4):
        super().__init__()
        self.net=nn.Sequential(conv(c+b+11+z,c),nn.Conv2d(c,c,3,padding=1,groups=c),gn(c),nn.SiLU(),conv(c,11))
    def forward(self,h,shallow,stage,prev=None):
        size=h.shape[-2:]
        shallow=F.adaptive_avg_pool2d(shallow,size) if shallow.shape[-2:]!=size else shallow
        old=h.new_zeros(h.shape[0],11,*size) if prev is None else up(prev,size)
        delta=self.net(torch.cat((h,shallow,old,stage),1))
        logits=old+delta
        return logits,decode_geometry(logits)


def sample_map(feature, coords, image_size):
    """coords [B,H,W,P,2] in original input pixel coordinates, x/y order.
    Returns [B,C,H,W,P] and [B,1,H,W,P]. FP32 interpolation for AMP safety.
    """
    B,H,W,P,_=coords.shape; ih,iw=image_size
    grid=torch.stack((2*(coords[...,0]+.5)/iw-1,2*(coords[...,1]+.5)/ih-1),-1)
    valid=((coords[...,0]>=0)&(coords[...,0]<=iw-1)&(coords[...,1]>=0)&(coords[...,1]<=ih-1)).unsqueeze(1)
    with torch.autocast(device_type=feature.device.type,enabled=False):
        v=F.grid_sample(feature.float(),grid.reshape(B,H,W*P,2).float(),align_corners=False,padding_mode='border',mode='bilinear')
    return v.reshape(B,feature.shape[1],H,W,P).to(feature.dtype)*valid,valid


def masked_mean(x,m,dim=-1):
    return (x*m).sum(dim)/m.sum(dim).clamp_min(1)


class ContrastivePathConv(nn.Module):
    def __init__(self,c=16,b=8,checkpoint_sampling=True,outer_ratio=1.5):
        super().__init__(); self.b=b; self.checkpoint_sampling=checkpoint_sampling; self.outer_ratio=outer_ratio
        self.curvature=conv(c+11,8); nn.init.zeros_(self.curvature.weight); nn.init.zeros_(self.curvature.bias)
        theta=torch.arange(8)*math.pi/4
        self.register_buffer('directions',torch.stack((theta.cos(),-theta.sin()),-1),persistent=False)
        self.n_op=nn.Sequential(conv(5*b,b),nn.SiLU(),conv(b,b))
        # path convolution implemented on last (ordered sample) dimension
        self.t_dw=nn.Conv1d(b,b,3,padding=1,groups=b,bias=False); self.t_pw=nn.Conv1d(b,b,1)
        self.t_op=nn.Sequential(conv(3*b,b),nn.SiLU(),conv(b,b))
        self.arm=nn.Sequential(conv(2*b,b),nn.SiLU(),conv(b,b))
        self.j_op=nn.Sequential(conv(4*b,b),nn.SiLU(),conv(b,b))
        self.axis_score=nn.ModuleList(conv(b+2,b//2) for _ in range(2)); self.axis_out=nn.ModuleList(conv(b//2,1) for _ in range(2))
        self.scale_score=nn.ModuleList(nn.Sequential(conv(c+11+b+4+3,8),nn.SiLU(),conv(8,1)) for _ in range(3))
    def build_paths(self,h,g,image_size,stage):
        B,_,H,W=h.shape; ih,iw=image_size; K=2 if stage==0 else 1; step=4. if stage==0 else 2.
        y=(torch.arange(H,device=h.device,dtype=torch.float32)+.5)*ih/H-.5
        x=(torch.arange(W,device=h.device,dtype=torch.float32)+.5)*iw/W-.5
        yy,xx=torch.meshgrid(y,x,indexing='ij'); p=torch.stack((xx,yy),-1).expand(B,H,W,2)
        D=g[:,1:9].float().permute(0,2,3,1); u=self.directions.float()
        t=F.normalize(sum((D.roll(-j,-1)+1e-3).unsqueeze(-1)*u.roll(-j,0) for j in (-1,0,1)),dim=-1)
        n=torch.stack((-t[...,1],t[...,0]),-1)
        bend=self.curvature(torch.cat((h,geo_norm(g)),1)).float().tanh().permute(0,2,3,1)
        k=torch.arange(1,K+1,device=h.device,dtype=torch.float32)
        off=step*k.view(1,1,1,1,K,1)*t.unsqueeze(-2)+.5*step*bend[...,None,None]*(k/K).square().view(1,1,1,1,K,1)*n.unsqueeze(-2)
        q=p[...,None,None,:]+off # B H W 8 K 2
        geom,vm=sample_map(g, q.flatten(3,4), image_size)
        geom=geom.reshape(B,11,H,W,8,K); vm=vm.reshape(B,1,H,W,8,K)
        ds=[]
        for d in range(8):
            aligned=torch.maximum(geom[:,1+d,:,:,d],geom[:,1+(d+4)%8,:,:,d])
            ds.append(masked_mean(geom[:,0,:,:,d]*aligned,vm[:,0,:,:,d],-1))
        support=torch.stack(ds,1)
        # axis sequences signed from negative to positive; center and nearest sections
        paths=[]
        for d in range(4):
            seq=torch.cat((q[...,d+4,:,:].flip(-2),p.unsqueeze(-2),q[...,d,:,:]),-2)
            ids=[K-1,K,K+1] if stage==0 else [K]
            centers=seq[...,ids,:]
            # normal estimated using adjacent path points, orientation along sequence
            tans=[]
            for j in ids:
                tang=F.normalize(seq[...,min(j+1,2*K),:]-seq[...,max(j-1,0),:],dim=-1)
                tans.append(torch.stack((-tang[...,1],tang[...,0]),-1))
            normal=torch.stack(tans,-2)
            rr,_=sample_map(g[:,9:10],centers,image_size); rr=rr[:,0].float().clamp(.5,24)
            # low-confidence geometry blends radius with 1.5px, remains a soft prior
            conf,_=sample_map(g[:,:1],centers,image_size); rr=rr*conf[:,0].float()+1.5*(1-conf[:,0].float())
            offsets=rr[...,None]*normal
            sections=torch.stack((centers,centers+offsets,centers-offsets,centers+self.outer_ratio*offsets,centers-self.outer_ratio*offsets),-2)
            # reuse already sampled center points instead of sampling them twice
            coords=torch.cat((seq,sections[...,1:,:].flatten(-3,-2)),-2)
            paths.append(coords)
        return torch.stack(paths,3),q[..., -1,:],support
    def _axis(self,feature,coords,image_size,K):
        v,mask=sample_map(feature,coords,image_size); B,C,H,W,_=v.shape; L=2*K+1
        seq=v[...,:L]; m=mask[...,:L]; ids=[K-1,K,K+1] if K==2 else [K]; A=len(ids)
        side=v[...,L:].reshape(B,C,H,W,A,4); sm=mask[...,L:].reshape(B,1,H,W,A,4)
        vc=seq[...,ids]; cm=m[...,ids]
        vb=masked_mean(side[...,:2],sm[...,:2]); vo=masked_mean(side[...,2:],sm[...,2:]); asym=(side[...,2]-side[...,3]).abs()*(sm[...,2]&sm[...,3]); diff=(vc-vo)*(sm[...,2:].any(-1))
        n_in=torch.cat([masked_mean(x,cm) for x in (vc,vb,vo,diff,asym)],1); n=self.n_op(n_in)
        flat=seq.permute(0,2,3,1,4).reshape(-1,C,L)
        t=self.t_pw(self.t_dw(flat)).reshape(B,H,W,C,L).permute(0,3,1,2,4)
        mean=masked_mean(t,m); mx=t.masked_fill(~m,-1e4).amax(-1); mx=torch.where(m.any(-1),mx,torch.zeros_like(mx))
        change=masked_mean((seq[...,1:]-seq[...,:-1]).abs(),m[...,1:]&m[...,:-1]); t=self.t_op(torch.cat((mean,mx,change),1))
        plus=self.arm(torch.cat((seq[...,K],masked_mean(seq[...,K+1:],m[...,K+1:])),1)); minus=self.arm(torch.cat((seq[...,K],masked_mean(seq[...,:K],m[...,:K])),1))
        return n,t,plus,minus
    def _scale(self,feature,paths,support,image_size,K):
        ns=[]; ts=[]; plus=[]; minus=[]
        for d in range(4):
            args=(feature,paths[:,:,:,d],image_size,K)
            if self.training and self.checkpoint_sampling and torch.is_grad_enabled():
                n,t,a,b=checkpoint(self._axis,*args,use_reentrant=False)
            else: n,t,a,b=self._axis(*args)
            ns.append(n); ts.append(t); plus.append(a); minus.append(b)
        results=[]
        for i,items in enumerate((ns,ts)):
            scores=torch.stack([self.axis_out[i](F.silu(self.axis_score[i](torch.cat((v,support[:,d:d+1],support[:,d+4:d+5]),1)))) for d,v in enumerate(items)],1).softmax(1)
            results.append((scores*torch.stack(items,1)).sum(1))
        arms=torch.stack(plus+minus,1); weighted=arms*(.25+.75*support.unsqueeze(2))
        interactions=[(weighted*weighted.roll(-m,1)).mean(1) for m in (1,2,3)]
        j=self.j_op(torch.cat((arms.mean(1),*interactions),1))
        return results[0],j,results[1]
    def forward(self,bank,h,g,stage_vec,image_size,stage):
        paths,endpoints,support=self.build_paths(h,g,image_size,stage)
        K=2 if stage==0 else 1
        evidence=[self._scale(f,paths,support,image_size,K) for f in bank]
        fused=[]; weights=[]; B,_,H,W=h.shape
        for e in range(3):
            scores=[]
            for l in range(3):
                ident=h.new_zeros(B,3,H,W); ident[:,l]=1
                scores.append(self.scale_score[e](torch.cat((h,geo_norm(g),evidence[l][e],stage_vec,ident),1)))
            a=torch.cat(scores,1).softmax(1); weights.append(a)
            fused.append(sum(a[:,l:l+1]*evidence[l][e] for l in range(3)))
        return fused,endpoints,support,torch.stack(weights,1)


class ComplementaryExperts(nn.Module):
    def __init__(self,c=16,b=8):
        super().__init__()
        self.experts=nn.ModuleList(LiteBlock(c+b,c) for _ in range(3)); self.heads=nn.ModuleList(conv(c,1) for _ in range(3))
        self.msg_proj=nn.ModuleList(conv(c,4) for _ in range(3)); self.msg_score=nn.ModuleList(nn.Sequential(conv(c+4+1,8),nn.SiLU(),conv(8,1)) for _ in range(3)); self.msg_out=nn.ModuleList(conv(c+4,c) for _ in range(3))
    def forward(self,evidence,h,previous,endpoints,support,image_size):
        states=[]
        for i in range(3):
            s=self.experts[i](torch.cat((h,evidence[i]),1)); s=s if previous is None else s+up(previous[i],h.shape[-2:]); states.append(s)
        aux=torch.cat([head(s) for head,s in zip(self.heads,states)],1)
        sampled,valid=sample_map(torch.cat([p(s) for p,s in zip(self.msg_proj,states)],1),endpoints,image_size)
        messages=[]
        for i,s in enumerate(states):
            v=sampled[:,i*4:i*4+4]; scores=torch.cat([self.msg_score[i](torch.cat((s,v[...,d],support[:,d:d+1]),1)) for d in range(8)],1)
            scores=scores.masked_fill(~valid[:,0].permute(0,3,1,2),-1e4).softmax(1)
            agg=(scores.permute(0,2,3,1).unsqueeze(1)*v).sum(-1)
            messages.append(self.msg_out[i](torch.cat((s,agg),1)))
        return states,messages,aux
