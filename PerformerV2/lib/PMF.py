import torch
import torch.nn as nn
import torch.nn.functional as F
# from LPA_Block import LPA

class PMF(nn.Module):
    def __init__(self, in_channels, up_kwargs=None):
        super(PMF, self).__init__()

        self.up_kwargs = up_kwargs
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels[3], in_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[2], in_channels[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[2], in_channels[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[1], in_channels[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True)
            )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[1], in_channels[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True)
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[0], in_channels[0], kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[0], in_channels[0], kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True)
        )


    def forward(self, x1, x2, x3, x4):
        x4_1 = x4

        x4_2 = self.conv4_2(F.interpolate(x4_1, scale_factor=2, **self.up_kwargs))
        x3_1 = x4_2 * self.conv3_1(x3)

        x3_2 = self.conv3_2(F.interpolate(x3_1, scale_factor=2, **self.up_kwargs))
        x2_1 = x3_2 * self.conv2_1(x2)


        x2_2 = self.conv2_2(F.interpolate(x2_1, scale_factor=2, **self.up_kwargs))
        x1_1 = x2_2 * self.conv1_1(x1)

        return x1_1, x2_1, x3_1, x4_1


if __name__ == '__main__':
    up_kwargs = {'mode': 'bilinear', 'align_corners': True}
    x = torch.randn(16, 96, 56, 56)
    y = torch.randn(16, 192, 28, 28)
    z = torch.randn(16, 384, 14, 14)
    m = torch.randn(16, 768, 7, 7)

    model = PMF([96, 192, 384, 768], up_kwargs=up_kwargs)

    output1, output2, output3, output4 = model(x, y, z, m)
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)
    print(output4.shape)
