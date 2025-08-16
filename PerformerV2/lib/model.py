import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin import swin_B, swin_L
from .LGFF import LGFF
from .DRCNN import DRCNN
from .PGA import PGA
from .PMF import PMF


up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class PerformerV2(nn.Module):
    def __init__(self, out_planes=9, encoder='swin_B'):
        super(PerformerV2, self).__init__()
        self.encoder = encoder
        if self.encoder == 'swin_L':
            mutil_channel = [192, 384, 768, 1536]
            self.backbone = swin_L()
        elif self.encoder == 'swin_B':
            mutil_channel = [128, 256, 512, 1024]
            self.backbone = swin_B()
        self.DRCNN = DRCNN()


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.PMF = PMF(mutil_channel, up_kwargs=up_kwargs)

        self.fu1 = LGFF(ch_1=mutil_channel[0], ch_2=mutil_channel[0], ch_int=mutil_channel[0],
                        ch_out=mutil_channel[0])
        self.fu2 = LGFF(ch_1=mutil_channel[1], ch_2=mutil_channel[1], ch_int=mutil_channel[1],
                        ch_out=mutil_channel[1])
        self.fu3 = LGFF(ch_1=mutil_channel[2], ch_2=mutil_channel[2], ch_int=mutil_channel[2],
                        ch_out=mutil_channel[2])
        self.fu4 = LGFF(ch_1=mutil_channel[3], ch_2=mutil_channel[3], ch_int=mutil_channel[3],
                        ch_out=mutil_channel[3])

        self.PGA4 = PGA(mutil_channel[3], mutil_channel[2], is_bottom=True)
        self.PGA3 = PGA(mutil_channel[2], mutil_channel[1])
        self.PGA2 = PGA(mutil_channel[1], mutil_channel[0])
        self.PGA1 = PGA(mutil_channel[0], mutil_channel[0] // 2)

        self.decoder3 = BasicConv2d(mutil_channel[3], mutil_channel[2], 3, padding=1)
        self.decoder2 = BasicConv2d(mutil_channel[2], mutil_channel[1], 3, padding=1)
        self.decoder1 = BasicConv2d(mutil_channel[1], mutil_channel[0], 3, padding=1)
        self.outimage4 = nn.Conv2d(mutil_channel[3], out_planes, kernel_size=1, stride=1)
        self.outimage3 = nn.Conv2d(mutil_channel[2], out_planes, kernel_size=1, stride=1)
        self.outimage2 = nn.Conv2d(mutil_channel[1], out_planes, kernel_size=1, stride=1)
        self.outimage1 = nn.Conv2d(mutil_channel[0], out_planes, kernel_size=1, stride=1)


    def forward(self, x):
        #print(x.shape)
        x_tr1, x_tr2, x_tr3, x_tr4 = self.backbone(x)
        x_down1, x_down2, x_down3, x_down4 = self.DRCNN(x)

        x1 = self.fu1(x_down1, x_tr1, None)
        x2 = self.fu2(x_down2, x_tr2, x1)
        x3 = self.fu3(x_down3, x_tr3, x2)
        x4 = self.fu4(x_down4, x_tr4, x3)

        y1, y2, y3, y4 = self.PMF(x1, x2, x3, x4)

        d4 = self.PGA4(y4, y4)
        outimage4 = F.interpolate(self.outimage4(d4), scale_factor=32, mode='bilinear')
        d3 = self.decoder3(self.PGA3(self.decoder3(self.upsample(d4)), y3))
        outimage3 = F.interpolate(self.outimage3(d3), scale_factor=16, mode='bilinear')
        d2 = self.decoder2(self.PGA2(self.decoder2(self.upsample(d3)), y2))
        outimage2 = F.interpolate(self.outimage2(d2), scale_factor=8, mode='bilinear')
        d1 = self.decoder1(self.PGA1(self.decoder1(self.upsample(d2)), y1))
        outimage1 = F.interpolate(self.outimage1(d1), scale_factor=4, mode='bilinear')

        outimage=outimage1+outimage2+outimage3+outimage4

        return outimage


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


