import torch
from torch import nn
from torch.nn import functional as F
from .backbone import LightweightBackbone,up
from .geometry_experts import GeometryField,ContrastivePathConv,ComplementaryExperts,conv
from .decision import ConditionalUtilityAttention

class CFCMoE(nn.Module):
    def __init__(self,in_channels=3,channels=(24,48,96,192),bank_dim=8,expert_dim=16,region=4,temperature=.5,heads=2,checkpoint_sampling=True,outer_ratio=1.5):
        super().__init__(); assert in_channels in (1,3); assert expert_dim%2==0 and bank_dim>=2
        self.in_channels=in_channels; self.region=region
        self.backbone=LightweightBackbone(in_channels,channels,bank_dim)
        self.adapters=nn.ModuleList([conv(channels[1],expert_dim),conv(channels[0],expert_dim)])
        self.feedback=nn.Conv2d(expert_dim,channels[1],1,bias=False)
        self.stage_embedding=nn.Embedding(2,4)
        self.geometry=GeometryField(expert_dim,bank_dim)
        self.ccpc=ContrastivePathConv(expert_dim,bank_dim,checkpoint_sampling,outer_ratio)
        self.experts=ComplementaryExperts(expert_dim,bank_dim)
        self.decision=ConditionalUtilityAttention(expert_dim,region,temperature,heads)
        self.heads=nn.ModuleList(conv(expert_dim,1) for _ in range(2))
        self.new_detail=conv(bank_dim,expert_dim); self.old_detail=conv(expert_dim,expert_dim)
    def forward_stage(self,x,bank,image_size,stage,prev,eta,route_mix):
        h=self.adapters[stage](x); B,C,H,W=h.shape
        z=self.stage_embedding.weight[stage].to(h.dtype).view(1,4,1,1).expand(B,-1,H,W)
        gl,g=self.geometry(h,bank[0],z,None if prev is None else prev['geometry_logits'])
        old_p=h.new_zeros(B,1,H,W) if prev is None else up(prev['logits'].sigmoid(),(H,W))
        diff=torch.zeros_like(h) if prev is None else self.new_detail(bank[0])-self.old_detail(up(prev['fused_state'],(H,W)))
        ctx=self.decision.make_context(h,diff,g,z,old_p,self.heads[stage])
        if eta<=0:
            base=ctx['base']; return {'base_state':h,'fused_state':h,'base_logits':base,'logits':base,'geometry_logits':gl,'geometry':g,'expert_states':None}
        evidence,endpoints,support,scale_weights=self.ccpc(bank,h,g,z,image_size,stage)
        states,msg,aux=self.experts(evidence,h,None if prev is None else prev['expert_states'],endpoints,support,image_size)
        out=self.decision(ctx,msg,support,self.heads[stage],eta,route_mix)
        out.update(geometry_logits=gl,geometry=g,expert_states=states,expert_messages=msg,expert_logits=aux,decision_context=ctx,path_support=support,scale_weights=scale_weights)
        return out
    def forward(self,x,eta=1.,route_mix=1.,return_details=False):
        if x.ndim!=4 or x.shape[1]!=self.in_channels: raise ValueError(f'expected BCHW with {self.in_channels} channels')
        if any(n%8 or n%(2*self.region) for n in x.shape[-2:]): raise ValueError('H,W must be divisible by 8 and 2*region (default 8)')
        e=self.backbone.encode(x); d2,bank=self.backbone.decode_coarse(e)
        c=self.forward_stage(d2,bank,x.shape[-2:],0,None,eta,route_mix)
        d1=self.backbone.d1(d2+self.feedback(c['fused_state']-c['base_state']),e[0]); f=self.forward_stage(d1,bank,x.shape[-2:],1,c,eta,route_mix)
        return {'coarse':c,'fine':f,'logits':f['logits']} if return_details else f['logits']
