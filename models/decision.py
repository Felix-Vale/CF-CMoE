"""Parallel conditional utility QKV and signed evidence update. No spatial
normalization or >1x1 operators in ROI-replayed decision path."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .geometry_experts import conv,geo_norm

CANDIDATES=((),(0,),(1,),(2,),(0,1),(0,2),(1,2))
NAMES=('null','N','J','T','NJ','NT','JT')

class ExpertCombination(nn.Module):
    def __init__(self,c=16):
        super().__init__(); self.a=conv(c,c//2); self.o=conv(c,c)
    def forward(self,m):
        a=[self.a(x) for x in m]
        return list(m)+[(m[i]+m[j])/math.sqrt(2)+self.o(torch.cat((a[i]*a[j],a[i]-a[j]),1)) for i,j in CANDIDATES[4:]]

class BidirectionalEvidenceGate(nn.Module):
    def __init__(self,c=16):
        super().__init__(); self.h=conv(c,c); self.m=conv(c,c); self.q=conv(c,c); self.g=conv(8,c); self.v=conv(c,c); self.gates=conv(c,2)
        nn.init.normal_(self.v.weight,std=.02); nn.init.zeros_(self.v.bias)
    def context(self,h,q,g): return self.h(h)+self.q(q)+self.g(g)
    def forward(self,ctx,m,head):
        t=F.silu(ctx+self.m(m)); delta=self.v(t); r=F.conv2d(delta,head.weight,None); gp_log,gn_log=self.gates(t).chunk(2,1); gp,gn=gp_log.sigmoid(),gn_log.sigmoid()
        value=torch.where(r>=0,gp,gn)*delta
        return value,r,gp,gn,gp_log,gn_log

class ConditionalUtilityAttention(nn.Module):
    def __init__(self,c=16,region=4,temperature=.5,heads=2):
        super().__init__(); self.c=c; self.region=region; self.temperature=temperature; self.heads=heads
        assert c%heads==0
        self.geo=conv(11,8); self.query=conv(2*c+8+4+2,c); self.key=conv(c+8+8+3,c)
        self.qpool=conv(2*c,c); self.kpool=conv(2*c,c)
        self.combo=ExpertCombination(c); self.gate=BidirectionalEvidenceGate(c)
        ids=torch.tensor([[float(i in a) for i in range(3)] for a in CANDIDATES[1:]])
        self.register_buffer('ids',ids,persistent=False)
    def pool(self,x,proj):
        r=self.region
        return proj(torch.cat((F.avg_pool2d(x,r),F.max_pool2d(x,r)),1))
    def make_context(self,h,difference,g,stage_vec,old_p,head):
        base=head(h); ge=self.geo(geo_norm(g)); q=self.query(torch.cat((h,difference,ge,stage_vec,base.sigmoid(),old_p),1))
        return {'h':h,'ge':ge,'q':q,'base':base}
    def forward(self,context,messages,support,head,eta=1.,route_mix=1.):
        h,ge,q,base=(context[k] for k in ('h','ge','q','base')); B,C,H,W=h.shape; r=self.region
        assert H%r==0 and W%r==0
        qr=self.pool(q,self.qpool).reshape(B,self.heads,C//self.heads,H//r,W//r)
        combos=self.combo(messages); keys=[]; values=[]; logits=[base]; raw=[]; gps=[]; gns=[]; gpls=[]; gnls=[]
        ctx=self.gate.context(h,q,ge)
        for i,m in enumerate(combos):
            ident=self.ids[i].to(h.dtype).view(1,3,1,1).expand(B,-1,H,W)
            kr=self.pool(self.key(torch.cat((m,ge,support,ident),1)),self.kpool).reshape_as(qr)
            score=(qr.float()*kr.float()).sum(2).mean(1)/math.sqrt(C//self.heads); keys.append(score)
            v,rr,gp,gn,gpl,gnl=self.gate(ctx,m,head); values.append(v); raw.append(rr); gps.append(gp); gns.append(gn); gpls.append(gpl); gnls.append(gnl)
            logits.append(base+eta*F.conv2d(v,head.weight,None))
        scores=torch.stack([torch.zeros_like(keys[0]),*keys],1); a=scores.div(self.temperature).softmax(1)
        if route_mix!=1.: a=route_mix*a+(1-route_mix)/len(CANDIDATES)
        # exact piecewise-constant regional weights, same operation in ROI replay
        ap=a.repeat_interleave(r,-2).repeat_interleave(r,-1).to(h.dtype)
        fused=h+eta*sum(ap[:,i+1:i+2]*v for i,v in enumerate(values))
        return {'base_state':h,'fused_state':fused,'base_logits':base,'logits':head(fused),'utility_scores':scores,'route_weights':a,'candidate_logits':torch.cat(logits,1),'raw_delta':torch.cat(raw,1),'gate_positive':torch.cat(gps,1),'gate_negative':torch.cat(gns,1),'gate_positive_logits':torch.cat(gpls,1),'gate_negative_logits':torch.cat(gnls,1)}
