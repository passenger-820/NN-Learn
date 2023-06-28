import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# 用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        [  all, all, [0,-padding)  ] 真就是把加的padding个全都裁掉
        """
        return x[:, :, :-self.chomp_size].contiguous() # contiguous让新数据在内存中连续，方便使用，如配合view()、permute()等
    # 第一个数据到倒数第chomp_size的数据，这个chomp_size就是padding的值。比方说输入数据是5，padding是1，那么会产生6个数据没错吧，那么就是保留前5个数字。

# 这个就是TCN的基本模块，包含8个部分，两个（卷积+修剪+relu+dropout）
# 里面提到的downsample就是下采样，其实就是实现残差链接的部分。不理解的可以无视这个
class TemporalBlock(nn.Module):
    # TemporalBlock(n_inputs=1=in_channels,n_outputs=25=out_channels,kernel_size=7,stride=1,dilation=1,padding=(7-1)*1=6,dropout=0.05)
    # ......
    # TemporalBlock(n_inputs=25=in_channels,n_outputs=25=out_channels,kernel_size=7,stride=1,dilation=128,padding=(7-1)*128=768,dropout=0.05)
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block
        n_inputs: int, 输入通道数
        n_outputs: int, 输出通道数
        kernel_size: int, 卷积核尺寸
        stride: int, 步长，一般为1
        dilation: int, 膨胀系数
        padding: int, 填充系数
        dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        # Conv1d(n_inputs=1, n_outputs=25, kernel_size=7,stride=1, padding=6, dilation=1)
        # ......
        # Conv1d(n_inputs=25, n_outputs=25, kernel_size=7,stride=1, padding=768, dilation=128)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Conv1d(n_inputs=25, n_outputs=25, kernel_size=7,stride=1, padding=6, dilation=1)
        # ......
        # Conv1d(n_inputs=25, n_outputs=25, kernel_size=7,stride=1, padding=768, dilation=128)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # if n_inputs != n_outputs 就进行下采样，也就第0层不同，所以只有第0层下采样，这就是残差块右边的那个 1x1框框！！
        # Conv1d(n_inputs=1, n_outputs=25, kernel_size=1)  (784 - 1*(1-1) - 1 + 2*0)/1 + 1 = 784 没改变尺寸，只改变维度！！！
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        param x: size of (Batch, input_channel, seq_len)
        """
        # out[64,25,784]
        out = self.net(x)
        # 用于短路的那条线，确保最后res 和 out 的 shape 一致
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    # TemporalConvNet(num_inputs=1=input_size, num_channels=[25, 25, 25, 25, 25, 25, 25, 25], kernel_size=7, dropout=0.05)
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        # num_levels = len([25, 25, 25, 25, 25, 25, 25, 25]) = 8
        num_levels = len(num_channels)
        for i in range(num_levels): # [0,8)
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
                # 结果就是，除了第0层为1，其余都是在[25, 25, 25, 25, 25, 25, 25, 25]选 i=1~7 i-1=0~6，结果都是25！！！！！
            out_channels = num_channels[i]  # 确定每一层的输出通道数
                # 结果也还是在[25, 25, 25, 25, 25, 25, 25, 25]选 i=0~7 结果都是25！
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
                # TemporalBlock(in_channels=1,out_channels=25,kernel_size=7,stride=1,dilation=1,padding=(7-1)*1=6,dropout=0.05)
                # TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=2,padding=(7-1)*2=12,dropout=0.05)
                # TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=4,padding=(7-1)*4=24,dropout=0.05)
                # TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=8,padding=(7-1)*8=48,dropout=0.05)
                # TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=16,padding=(7-1)*16=96,dropout=0.05)
                # TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=32,padding=(7-1)*32=192,dropout=0.05)
                # TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=64,padding=(7-1)*64=384,dropout=0.05)
                # TemporalBlock(in_channels=25,out_channels=25,kernel_size=7,stride=1,dilation=128,padding=(7-1)*128=768,dropout=0.05)


        # 形参——单个*号代表这个位置接收任意多个非关键字参数，转化成元组方式。
        # 实参——如果*号加在了是实参上，代表的是将输入迭代器拆成一个个元素。如下：
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)