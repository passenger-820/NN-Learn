import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# TemporalBlock(in_channels=1,out_channels=25,kernel_size=7,stride=1,dilation=1,padding=(7-1)*1=6,dropout=0.05)
# TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=2,padding=(7-1)*2=12,dropout=0.05)
# TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=4,padding=(7-1)*4=24,dropout=0.05)
# TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=8,padding=(7-1)*8=48,dropout=0.05)
# TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=16,padding=(7-1)*16=96,dropout=0.05)
# TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=32,padding=(7-1)*32=192,dropout=0.05)
# TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=64,padding=(7-1)*64=384,dropout=0.05)
# TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=128,padding=(7-1)*128=768,dropout=0.05)

"""
x1 shape: torch.Size([64, 1, 790])
x1_out shape: torch.Size([64, 1, 784])

x2 shape: torch.Size([64, 1, 796])
x2_out shape: torch.Size([64, 1, 784])

x3 shape: torch.Size([64, 1, 808])
x3_out shape: torch.Size([64, 1, 784])

x4 shape: torch.Size([64, 1, 832])
x4_out shape: torch.Size([64, 1, 784])

x5 shape: torch.Size([64, 1, 880])
x5_out shape: torch.Size([64, 1, 784])

x6 shape: torch.Size([64, 1, 976])
x5_out shape: torch.Size([64, 1, 784])

x7 shape: torch.Size([64, 1, 1168])
x7_out shape: torch.Size([64, 1, 784])

x8 shape: torch.Size([64, 1, 1552])
x8_out shape: torch.Size([64, 1, 784])
"""

conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7,stride=1, padding=6, dilation=1)
chomp1 = Chomp1d(6)

x = torch.randn(64,1,784)
"""L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor"""
# ( input_size - dilation * (kernel_size - 1) - 1 + 2 * padding ) / stride + 1
x1 = conv1(x)
print('x1 shape:', x1.shape)
x1_out = chomp1(x1)
print('x1_out shape:', x1_out.shape)

conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7,stride=1, padding=12, dilation=2)
chomp2 = Chomp1d(12)
x2 = conv2(x1_out)
print('x2 shape:', x2.shape)
x2_out = chomp2(x2)
print('x2_out shape:', x2_out.shape)

conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7,stride=1, padding=24, dilation=4)
chomp3 = Chomp1d(24)
x3 = conv3(x2_out)
print('x3 shape:', x3.shape)
x3_out = chomp3(x3)
print('x3_out shape:', x3_out.shape)

conv4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7,stride=1, padding=48, dilation=8)
chomp4 = Chomp1d(48)
x4 = conv4(x3_out)
print('x4 shape:', x4.shape)
x4_out = chomp4(x4)
print('x4_out shape:', x4_out.shape)

conv5 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7,stride=1, padding=96, dilation=16)
chomp5 = Chomp1d(96)
x5 = conv5(x4_out)
print('x5 shape:', x5.shape)
x5_out = chomp5(x5)
print('x5_out shape:', x5_out.shape)

conv6 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7,stride=1, padding=192, dilation=32)
chomp6 = Chomp1d(192)
x6 = conv6(x5_out)
print('x6 shape:', x6.shape)
x6_out = chomp6(x6)
print('x5_out shape:', x6_out.shape)

conv7 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7,stride=1, padding=384, dilation=64)
chomp7 = Chomp1d(384)
x7 = conv7(x6_out)
print('x7 shape:', x7.shape)
x7_out = chomp7(x7)
print('x7_out shape:', x7_out.shape)

conv8 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7,stride=1, padding=768, dilation=128)
chomp8 = Chomp1d(768)
x8 = conv8(x7_out)
print('x8 shape:', x8.shape)
x8_out = chomp8(x8)
print('x8_out shape:', x8_out.shape)




