import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    """
    from    [b, ch_in,  h, w]
    to      [b, ch_out, h, w]
    """

    # 需要知道x是什么，才能把维度对应好
    def __init__(self, ch_in, ch_out):

        super(ResBlock, self).__init__()

        """两个卷积"""
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            # 笔记里曾经提到，BN是在channel上做的
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out,ch_out, kernel_size=3, padding=1),
            # 笔记里曾经提到，BN是在channel上做的
            nn.BatchNorm2d(ch_out)
            # 这里加不加relu看自己咯
        )

        """预防ch_in != ch_out，先建一个空的，相等，就相当于啥都没发生，不相等，就要加内容了"""
        self.extra = nn.Sequential()
        if ch_in != ch_out:
            # [b,ch_in,h,w] => [b,ch_out,h,w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1),
                nn.BatchNorm2d(ch_out)
            )


    def forward(self, x):
        """

        :param x:  [b,ch_in,h,w]
        :return: [b,ch_out,h,w]
        """
        out = self.conv1(x)
        out = self.conv2(out)

        """短接"""
        # element-wise add [b,ch_in,h,w] with [b,ch_out,h,w]
        # 但是这样得保证 ch_in, ch_out一样
        # 如果不一样咋办？额外加一个单元！看上面init里的if
        out = self.extra(x) + out

        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # followed by 4 blocks
        # [b, 64, h, w] => [b, 128, h ,w]
        self.blk1 = ResBlock(64, 128)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlock(128, 256)
        # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlock(256, 512)
        # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4 = ResBlock(512, 1024)

        self.outlayer = nn.Linear(1024*32*32, 10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        # [b, 3, h, w] => [b, 64, h, w]
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        print('after conv:',x.shape)

        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        print('after linear:', x.shape)
        return x

def test():

    blk = ResBlock(64,128)
    fake_data = torch.randn(2, 64, 32, 32)
    out = blk(fake_data)
    print(out.shape)

    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)
    print(out.shape)
if __name__ == '__main__':
    test()

