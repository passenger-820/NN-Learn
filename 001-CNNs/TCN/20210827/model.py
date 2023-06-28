import torch.nn.functional as F
from torch import nn
from tcn import TemporalConvNet

# TCN的主网络
# num_inputs就是输入数据的通道数，一般就是1；
# num_channels
# 整个TCN模型包含两个TemporalBlock，整个模型共有4个卷积层，第一个TemporalBlock的两个卷积层的膨胀系数dilation=2^0=1
# 第二个TemporalBlock的两个卷积层的膨胀系数是dilation=2^1=2
class TCN(nn.Module):
    # input_size=1, output_size=10=n_classes, num_channels=[25,25,25,25,25,25,25,25]=channel_sizes, kernel_size=7, dropout=0.05
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        # TemporalConvNet(input_size=1, num_channels=[25,25,25,25,25,25,25,25], kernel_size=7, dropout=0.05)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # Linear(25,10)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        # y1[64,25,784]    inputs[64,1,784]
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        # y1[:, :, -1] => torch.Size([64, 25])
        # o[64,10]  抽出了最上面一层丢到FC里面？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)