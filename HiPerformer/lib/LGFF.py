import torch
from torch import nn
import torch.nn.functional as F
# Hierachical Feature Fusion Block
class LGFF(nn.Module):
    def __init__(self, ch_1, ch_2, ch_int, ch_out):
        super(LGFF, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.Updim = Conv(ch_int//2, ch_int, 1, bn=True, relu=True)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)
        self.CAM = CAM(in_dim=ch_int)
        self.SAM=Spatial(in_channel=ch_int,ratio=4)

        self.gelu = nn.GELU()

        self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)

    def forward(self, l, g, f):

        W_local = self.W_l(l)   # local feature from Local Feature Block
        W_global = self.W_g(g)   # global feature from Global Feature Block
        if f is not None:
            W_f = self.Updim(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W_local, W_global], 1)
            X_f = self.norm1(X_f)
            X_f = self.W3(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm2(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)

        l=self.SAM(l)

        g = self.CAM(g)

        fuse = torch.cat([g, l, X_f], 1)
        fuse = self.norm3(fuse)
        fuse = self.residual(fuse)
        fuse = shortcut + fuse
        return fuse


class Spatial(nn.Module):
    def __init__(self, in_channel, ratio, kernel_size=7):
        super(Spatial, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d( in_channel, in_channel // ratio, kernel_size=7, padding=padding)
        self.bn = nn.BatchNorm2d(in_channel // ratio)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel // ratio, in_channel, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sig = nn.Sigmoid()
    def forward(self, x):
         conv1 = self.act(self.bn(self.conv1(x)))
         conv2 = self.bn1(self.conv2(conv1))
         output = self.sig(conv2)
         return x * output



class CAM(nn.Module):
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

#### Inverted Residual MLP
class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


if __name__ == '__main__':
    x = torch.randn(4, 96, 56, 56)
    y = torch.randn(4, 96, 56, 56)
    z = torch.randn(4, 192, 28, 28)
    m = torch.randn(4, 192, 28, 28)
    HFF_dp = 0.
    fu1 = LGFF(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96)
    fu2 = LGFF(ch_1=192, ch_2=192, r_2=16, ch_int=192, ch_out=192)
    fu3 = LGFF(ch_1=384, ch_2=384, r_2=16, ch_int=384, ch_out=384)
    fu4 = LGFF(ch_1=768, ch_2=768, r_2=16, ch_int=768, ch_out=768)
    # x_f_1 = self.fu1(x_c_1, x_s_1, None)
    # x_f_2 = self.fu2(x_c_2, x_s_2, x_f_1)
    # x_f_3 = self.fu3(x_c_3, x_s_3, x_f_2)
    # x_f_4 = self.fu4(x_c_4, x_s_4, x_f_3)
    x_f_1 = fu1(x, y, None)
    x_f_2 =fu2(z, m, y)
    # x_f_3 = self.fu3(x_c_3, x_s_3, x_f_2)
    # x_f_4 = self.fu4(x_c_4, x_s_4, x_f_3)
    # out = x_f_1(x, y)
    print(x_f_2.shape)

    # self.fu1 = HFF_block(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96, drop_rate=HFF_dp)
    # self.fu2 = HFF_block(ch_1=192, ch_2=192, r_2=16, ch_int=192, ch_out=192, drop_rate=HFF_dp)
    # self.fu3 = HFF_block(ch_1=384, ch_2=384, r_2=16, ch_int=384, ch_out=384, drop_rate=HFF_dp)
    # self.fu4 = HFF_block(ch_1=768, ch_2=768, r_2=16, ch_int=768, ch_out=768, drop_rate=HFF_dp)