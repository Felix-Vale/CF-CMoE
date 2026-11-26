import torch
from torch import nn
from torch.nn import functional as F


def up(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


def gn(c):
    return nn.GroupNorm(max(g for g in (1,2,4,8) if c % g == 0), c)


class LiteBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ci,co,1,bias=False), nn.Conv2d(co,co,3,padding=1,groups=co,bias=False),gn(co),nn.SiLU(),nn.Conv2d(co,co,1,bias=False),gn(co))
        self.skip = nn.Identity() if ci==co else nn.Conv2d(ci,co,1,bias=False)
    def forward(self,x):
        return F.silu(self.body(x)+self.skip(x))


class DecoderBlock(nn.Module):
    def __init__(self, ci, skip, co):
        super().__init__()
        self.block=LiteBlock(ci+skip,co)
    def forward(self,x,skip):
        return self.block(torch.cat((up(x,skip.shape[-2:]),skip),1))


class LightweightBackbone(nn.Module):
    def __init__(self,in_channels=3,channels=(24,48,96,192),bank_dim=8):
        super().__init__()
        a,b,c,d=channels
        self.e1=nn.Sequential(nn.Conv2d(in_channels,a,3,padding=1,bias=False),gn(a),nn.SiLU(),LiteBlock(a,a))
        self.e2=LiteBlock(a,b); self.e3=LiteBlock(b,c); self.e4=LiteBlock(c,d)
        self.d3=DecoderBlock(d,c,c); self.d2=DecoderBlock(c,b,b); self.d1=DecoderBlock(b,a,a)
        self.bank=nn.ModuleList(nn.Sequential(nn.Conv2d(i,bank_dim,1,bias=False),gn(bank_dim)) for i in (a,b,c))
    def encode(self,x):
        e1=self.e1(x); e2=self.e2(F.max_pool2d(e1,2)); e3=self.e3(F.max_pool2d(e2,2)); e4=self.e4(F.max_pool2d(e3,2))
        return e1,e2,e3,e4
    def decode_coarse(self,e):
        e1,e2,e3,e4=e; d3=self.d3(e4,e3); d2=self.d2(d3,e2)
        return d2,tuple(p(x) for p,x in zip(self.bank,(e1,e2,d3)))
