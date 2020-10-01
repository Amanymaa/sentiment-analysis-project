import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum, IntEnum
from ..utils.helpers import *
from ..utils.options import *


class _Fn(IntEnum):
    Conv = 0
    BatchNorm = 1
    MaxPool = 2
    AdaptiveAvgPool = 3
    AvgPool = 4


class _Dim(Enum):
    ResNet1d = {
        _Fn.Conv: nn.Conv1d,
        _Fn.BatchNorm: nn.BatchNorm1d,
        _Fn.MaxPool: nn.MaxPool1d,
        _Fn.AdaptiveAvgPool: nn.AdaptiveAvgPool1d,
        _Fn.AvgPool: nn.AvgPool1d,
    }
    ResNet2d = {
        _Fn.Conv: nn.Conv2d,
        _Fn.BatchNorm: nn.BatchNorm2d,
        _Fn.MaxPool: nn.MaxPool2d,
        _Fn.AdaptiveAvgPool: nn.AdaptiveAvgPool2d,
        _Fn.AvgPool: nn.AvgPool2d,
    }
    ResNet3d = {
        _Fn.Conv: nn.Conv3d,
        _Fn.BatchNorm: nn.BatchNorm3d,
        _Fn.MaxPool: nn.MaxPool3d,
        _Fn.AdaptiveAvgPool: nn.AdaptiveAvgPool3d,
        _Fn.AvgPool: nn.AvgPool3d,
    }

    def get_type(index):
        return [_Dim.ResNet1d, _Dim.ResNet2d, _Dim.ResNet3d][index - 1]


def conv3x3(in_planes, out_planes, fn, stride=1):
    """3x3 convolution with padding"""
    return fn(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dim, stride=1, downsample=None, Nonlinearity=nn.ReLU):
        super(BasicBlock, self).__init__()
        self.dim = dim

        self.conv1 = conv3x3(inplanes, planes, self.dim.value[_Fn.Conv], stride)
        self.bn1 = self.dim.value[_Fn.BatchNorm](planes)
        self.relu = Nonlinearity(inplace=True)
        self.conv2 = conv3x3(planes, planes, self.dim.value[_Fn.Conv])
        self.bn2 = self.dim.value[_Fn.BatchNorm](planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class NetworkBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, dim, stride=1, downsample=None, Nonlinearity=nn.ReLU):
        super(NetworkBlock, self).__init__()
        self.dim = dim

        self.conv1 = self.dim.value[_Fn.Conv](inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = self.dim.value[_Fn.BatchNorm](planes)
        self.conv2 = self.dim.value[_Fn.Conv](planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = self.dim.value[_Fn.BatchNorm](planes)
        self.conv3 = self.dim.value[_Fn.Conv](planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self.dim.value[_Fn.BatchNorm](planes * 4)
        self.relu = Nonlinearity(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, cfg):
        self.cfg = cfg

        dim = get(self.cfg, ResNetOptions.RESNET_DIMENSIONS.value, default=2)
        self.dim = _Dim.get_type(dim)

        if has(self.cfg, 'resnet_type'):
            resnet_type = get(self.cfg, ResNetOptions.RESNET_TYPE.value)

            if resnet_type == 18:
                block = BasicBlock
                layers = [2, 2, 2, 2]
            elif resnet_type == 34:
                block = BasicBlock
                layers = [3, 4, 6, 3]
            elif resnet_type == 50:
                block = NetworkBlock
                layers = [3, 4, 6, 3]
            elif resnet_type == 101:
                block = NetworkBlock
                layers = [3, 4, 23, 3]
            elif resnet_type == 152:
                block = NetworkBlock
                layers = [3, 8, 36, 3]
        else:
            block = get(self.cfg, ResNetOptions.RESNET_BLOCK.value, default=NetworkBlock)
            layers = get(self.cfg, ResNetOptions.RESNET_LAYERS.value)

        in_channels = get(self.cfg, ResNetOptions.IN_CHANNELS.value)
        out_channels = get(self.cfg, ResNetOptions.OUT_CHANNELS.value)
        Nonlinearity = get(self.cfg, ResNetOptions.RESNET_NONLINEARITY.value, default=nn.ReLU)

        self.inplanes = 64
        super(ResNet, self).__init__()

        if has(self.cfg, 'resnet_inblock'):
            self.conv1 = self.dim.value[_Fn.Conv](
                in_channels,
                64,
                kernel_size=get(self.cfg, ResNetOptions.RESNET_INBLOCK.value, 'conv_kernel_size', default=7),
                stride=get(self.cfg, ResNetOptions.RESNET_INBLOCK.value, 'conv_stride', default=2),
                padding=get(self.cfg, ResNetOptions.RESNET_INBLOCK.value, 'conv_padding', default=3),
                bias=False
            )
            self.bn1 = self.dim.value[_Fn.BatchNorm](64)
            self.relu = Nonlinearity(inplace=True)
            self.maxpool = self.dim.value[_Fn.MaxPool](
                kernel_size=get(self.cfg, ResNetOptions.RESNET_INBLOCK.value, 'maxpool_kernel_size', default=3),
                stride=get(self.cfg, ResNetOptions.RESNET_INBLOCK.value, 'maxpool_stride', default=2),
                padding=get(self.cfg, ResNetOptions.RESNET_INBLOCK.value, 'maxpool_padding', default=1)
            )
        else:
            self.conv1 = self.dim.value[_Fn.Conv](in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = self.dim.value[_Fn.BatchNorm](64)
            self.relu = Nonlinearity(inplace=True)
            self.maxpool = self.dim.value[_Fn.MaxPool](kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], self.dim, Nonlinearity=Nonlinearity)
        self.layer2 = self._make_layer(block, 128, layers[1], self.dim, stride=2, Nonlinearity=Nonlinearity)
        self.layer3 = self._make_layer(block, 256, layers[2], self.dim, stride=2, Nonlinearity=Nonlinearity)
        self.layer4 = self._make_layer(block, 512, layers[3], self.dim, stride=2, Nonlinearity=Nonlinearity)

        self.no_fc = get(self.cfg, ResNetOptions.RESNET_NO_FC.value, default=False)
        if self.no_fc:
            self.avgpool = self.fc = None
        else:
            if has(self.cfg, 'resnet_outblock'):
                use_adaptive = get(self.cfg, ResNetOptions.RESNET_OUTBLOCK.value, 'adaptive_avgpool', default=False)
                if use_adaptive:
                    self.avgpool = self.dim.value[_Fn.AdaptiveAvgPool](1)
                else:
                    self.avgpool = self.dim.value[_Fn.AvgPool](
                        get(self.cfg, ResNetOptions.RESNET_OUTBLOCK.value, 'avgpool_kernel_size', default=7),
                        stride=get(self.cfg, ResNetOptions.RESNET_OUTBLOCK.value, 'avgpool_stride', default=1)
                    )
            else:
                self.avgpool = self.dim.value[_Fn.AvgPool](7, stride=1)

            self.fc = nn.Linear(512 * block.expansion, out_channels)

    def _make_layer(self, block, planes, blocks, dim, stride=1, Nonlinearity=nn.ReLU):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.dim.value[_Fn.Conv](
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False
                ),
                self.dim.value[_Fn.BatchNorm](planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, dim, stride, downsample, Nonlinearity=Nonlinearity))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dim, Nonlinearity=Nonlinearity))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.no_fc:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x
