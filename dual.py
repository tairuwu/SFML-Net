

from model.part4 import MANet
from model.shvit_modify import shvit_s1
import torch.nn as nn
import torch

from timm.models.layers import DropPath, trunc_normal_



from self_guide import Atten





class DualInception(nn.Module):
     def __init__(self, num_classes = 5):
        super(DualInception, self).__init__()
        self.manet = MANet()
        self.shvit = shvit_s1()
        # self.shvit = ViT()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fusion = Atten(512)

        self.head = nn.Linear(1024, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
             if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def forward(self, x):

        m = self.manet(x)
        # print(m.szie())
        s = self.shvit(x)
        # print(s.szie())

        output = self.fusion(m,s)

        output = self.avgpool(output)  # B C 1
        output = torch.flatten(output, 1)
        output = self.head(output)

        return output