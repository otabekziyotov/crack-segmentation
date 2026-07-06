import torch
from torch.nn import *


class UNetBlock(Module):
    def __init__(self, in_chs, out_chs, ks=3, p=1):
        super().__init__()

        self.ks, self.p = ks, p

        self.block_1 = self.get_conv_block(in_chs=in_chs, out_chs=out_chs)
        self.block_2 = self.get_conv_block(in_chs=out_chs, out_chs=out_chs)

    def get_conv_block(self, in_chs, out_chs):
        return Sequential(
            Conv2d(in_channels=in_chs, out_channels=out_chs, kernel_size=self.ks, padding=self.p),
            BatchNorm2d(num_features=out_chs),
            ReLU(inplace=True)
        )

    def forward(self, inp): return self.block_2(self.block_1(inp))


class DownSampling(Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.downsample_block = Sequential(
            MaxPool2d(kernel_size=2, stride=2),
            UNetBlock(in_chs=in_chs, out_chs=out_chs)
        )

    def forward(self, inp): return self.downsample_block(inp)


class UpSampling(Module):
    def __init__(self, in_chs, out_chs, mode):
        super().__init__()
        assert mode in ['bilinear', 'nearest', 'tr_conv']
        if mode in ['bilinear', 'nearest']:
            self.upsample = Upsample(scale_factor=2, mode=mode)
        elif mode == 'tr_conv':
            self.upsample = ConvTranspose2d(in_channels=in_chs, out_channels=in_chs // 2, kernel_size=2, stride=2)
        self.conv = UNetBlock(in_chs=in_chs, out_chs=out_chs)

    def forward(self, inp1, inp2):
        inp1 = self.upsample(inp1)
        pad_y = inp2.size()[2] - inp1.size()[2]
        pad_x = inp2.size()[3] - inp1.size()[3]

        pad_xx, pad_yy = pad_x // 2, pad_y // 2

        inp1 = torch.nn.functional.pad(inp1, [pad_xx, pad_x - pad_xx, pad_yy, pad_y - pad_yy])
        concat = torch.cat([inp1, inp2], dim=1)

        return self.conv(concat)


class FinalConv(Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.conv = Conv2d(in_channels=in_chs, out_channels=out_chs, kernel_size=1)

    def forward(self, inp): return self.conv(inp)


class UNet(Module):
    def __init__(self, in_chs, out_chs, n_cls, up_method):
        super().__init__()

        self.init_conv = UNetBlock(in_chs=in_chs, out_chs=out_chs, ks=3, p=1)
        factor = 2 if up_method in ['bilinear', 'nearest'] else 1

        ### Encoding ###
        self.enc_block_1 = DownSampling(out_chs, out_chs * 2)
        self.enc_block_2 = DownSampling(out_chs * 2, out_chs * 4)
        self.enc_block_3 = DownSampling(out_chs * 4, out_chs * 8)
        self.enc_block_4 = DownSampling(out_chs * 8, out_chs * 16 // factor)

        ### Decoding ###
        self.dec_block_1 = UpSampling((out_chs * 16), (out_chs * 8 // factor), up_method)
        self.dec_block_2 = UpSampling((out_chs * 8), (out_chs * 4 // factor), up_method)
        self.dec_block_3 = UpSampling((out_chs * 4), (out_chs * 2 // factor), up_method)
        final_out_chs = (out_chs // factor) * 2 if up_method in ["bilinear", "nearest"] else out_chs // factor
        self.dec_block_4 = UpSampling((out_chs * 2), final_out_chs, up_method)

        self.classifier = FinalConv(out_chs, n_cls)

    def forward(self, inp):
        init_conv = self.init_conv(inp)
        enc_1 = self.enc_block_1(init_conv)
        enc_2 = self.enc_block_2(enc_1)
        enc_3 = self.enc_block_3(enc_2)
        enc_4 = self.enc_block_4(enc_3)

        dec_1 = self.dec_block_1(inp1=enc_4, inp2=enc_3)
        dec_2 = self.dec_block_2(inp1=dec_1, inp2=enc_2)
        dec_3 = self.dec_block_3(inp1=dec_2, inp2=enc_1)
        dec_4 = self.dec_block_4(inp1=dec_3, inp2=init_conv)

        out = self.classifier(dec_4)

        return out


if __name__ == "__main__":
    inp = torch.rand(1, 3, 224, 224)

    m1 = UNet(in_chs=3, out_chs=64, n_cls=2, up_method="tr_conv")
    print(m1(inp).shape)  # torch.Size([1, 2, 224, 224])
