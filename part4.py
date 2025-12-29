import torch

import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MANet(nn.Module):

    def __init__(self, block_b=BasicBlock,
                                 layers=[1, 1, 1, 1], num_classes=1000, zero_init_residual=False):
                 # layers=[1, 1, 1, 1], num_classes=3, zero_init_residual=False):
                 # layers=[1, 1, 1, 1], num_classes=5, zero_init_residual=False):
        super(MANet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_b, 64, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block_b, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_b, 128, 256, layers[1], stride=2)

        # # branch 1
        self.layer4_1_p1 = self._make_layer(block_b, 256, 512, layers[3], stride=1)
     

        self.layer4_1_p2 = self._make_layer(block_b, 256, 512, layers[3], stride=1)
        
        self.layer4_1_p3 = self._make_layer(block_b, 256, 512, layers[3], stride=1)
     
        self.layer4_1_p4 = self._make_layer(block_b, 256, 512, layers[3], stride=1)
        self.layer4_2_all = self._make_layer(block_b, 512, 512, layers[3], stride=2)

   
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_1 = nn.Linear(512, num_classes)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block_b):
                    nn.init.constant_(m.bn2.weight, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        splited = torch.chunk(x, dim=1, chunks=2)
        x = splited[0]  # (8,3,224,224)
        of = x
        of = self.conv1(of)
        of = self.bn1(of)
        of = self.relu(of)
        of = self.maxpool(of)    
        of = self.layer1(of)
        of = self.layer2(of)
        of = self.layer3(of)


        # # # branch 1-1 ############################################
        #
        left_eye_of = of[:, :, 0:7, 0:7]
        right_eye_of = of[:, :, 0:7, 7:14]
        left_lip_of = of[:, :, 7:14, 0:7]
        right_lip_of = of[:, :, 7:14, 7:14]
        
        left_eye_of_out = self.layer4_1_p1(left_eye_of)

        
        right_eye_of_out = self.layer4_1_p2(right_eye_of)

        
        left_lip_of_out = self.layer4_1_p3(left_lip_of)

        
        right_lip_of_out = self.layer4_1_p4(right_lip_of)

        eye_of_1 = torch.cat([left_eye_of_out, right_eye_of_out], dim=3)  # torch.Size([8, 512, 7, 14])
        lips_of_1 = torch.cat([left_lip_of_out, right_lip_of_out], dim=3)

        of_out_1 = torch.cat([eye_of_1, lips_of_1], dim=2)
        of_out_1 = self.layer4_2_all(of_out_1)  ### torch.Size([8, 512, 14, 14])

        #
        output = of_out_1   #14
       

        return output

    def forward(self, x):
        return self._forward_impl(x)


