import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
import numbers

# Optimization-Inspired Cross-Attention Transformer for Compressive Sensing (CVPR 2023)

# q (CVPR 2023)

# https://github.com/songjiechong/OCTUF/blob/493c74ae450db440e50bcea20c4e11db676a2ce3/model_octuf.py


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



# Define ISCA block
class Atten(torch.nn.Module):
    def __init__(self, channels):
        super(Atten, self).__init__()

        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')

        # self.blockNL = blockNL(self.channels)

        self.conv_q = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
            # nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, groups=self.channels, bias=True)
        )
        self.conv_kv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1,
                      groups=self.channels * 2, bias=True)

        )
        self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1,
                                  bias=True)


        ###位置嵌入1×1
        self.pos_emb = nn.Sequential(
        nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, bias=False,
                  groups=self.channels),
        nn.GELU(),
        nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, bias=False,
                  groups=self.channels),
    )


    def forward(self, pre, cur):
        # m  s
        # pre是光流   cur是起始帧
        b, c, h, w = pre.shape
        pre_ln = self.norm1(pre)
        cur_ln = self.norm2(cur)
        #q应该是S，引导M学习
        q = self.conv_q(cur_ln)
        q = q.view(b, c, -1)
        k, v = self.conv_kv(pre_ln).chunk(2, dim=1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        att = torch.matmul(q, k.permute(0, 2, 1))
        att = self.softmax(att)
        out = torch.matmul(att, v).view(b, c, h, w)
        ########## dual 原来的是这个
        # out = self.conv_out(out) + pre


        ########## 进行替换
        # out = out + pre
        # out = torch.mul(out,pre)
        # out = torch.cat((out,pre),1)
        # out = torch.cat((self.conv_out(out) , pre), 1)


        ####用的是这个
        out = torch.cat((pre,self.conv_out(out) ), 1)

        # out = torch.mul(self.conv_out(out) , pre)
        # out = torch.add(self.conv_out(out), pre)



        # out = torch.cat([out, pre],1)
        # out = torch.cat([pre, out], 1)
        # out = self.conv_out(out)




        # out = out+ pre
        # out = torch.mul(self.conv_out(out),pre)
        # out = torch.cat((self.conv_out(out), pre),dim=1)






        return out


