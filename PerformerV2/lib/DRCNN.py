import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange, repeat
from functools import partial

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Element-wise addition
        out = F.relu(out)
        return out


class DilatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super(DilatedResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Element-wise addition
        out = F.relu(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        # Assuming the input from each block is concatenated along the channel dimension
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # Concatenate along channel dimension
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BiPathResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dilation=2, use_dilate_conv=True):
        super(BiPathResBlock, self).__init__()
        # Define two ResBlocks and two DilatedResBlocks in sequence for each path
        self.resblock = ResBlock(in_channels, mid_channels)
        self.dilated_resblock = DilatedResBlock(in_channels, mid_channels, dilation=dilation)
        # Define the Fusion Block
        self.fusionblock = FusionBlock(2 * mid_channels, out_channels)
        self.use_dilate_conv = use_dilate_conv

    def forward(self, x):
        res_out = self.resblock(x)
        dilated_res_out = self.dilated_resblock(x)
        if self.use_dilate_conv:
            x = self.fusionblock(res_out, dilated_res_out)
        else:
            x = self.fusionblock(res_out, res_out)
        return x


class DRCNN(nn.Module):
    def __init__(self, use_dilate_conv=True):
        super(DRCNN, self).__init__()

        # Define channel transitions from the input to the deepest layer
        channels = [64, 128, 256, 512, 1024]
        self.layers = nn.ModuleList()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        for idx in range(1, len(channels)):
            self.layers.append(nn.MaxPool2d(2))

            self.layers.append(BiPathResBlock(channels[idx - 1], channels[idx], channels[idx], use_dilate_conv=use_dilate_conv))
            #if idx !=1:
            # self.layers.append(nn.MaxPool2d(2))

    def forward(self, x):

        x = self.conv1(x)
        # print(out.shape)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape)
        #x = self.maxpool(x)
        #print(x.shape)

        features = []
        for layer in self.layers:
            x = layer(x)
            #print(x.shape)
            if isinstance(layer, BiPathResBlock):  # Conditionally append feature maps following DoubleResBlock layers
                features.append(x)
        # # Include the final feature map post application of MaxPool2d layer for completeness of the hierarchical representations
        # features.append(x)

        return features

if __name__ == '__main__':
    x = torch.randn(16, 3, 224, 224)
    model = DRCNN()
    out1,out2,out3,out4= model(x)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)
