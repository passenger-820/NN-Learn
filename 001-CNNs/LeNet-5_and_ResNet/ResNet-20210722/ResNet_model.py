import torch
from torch import nn
from torch.nn import functional as F

"""
基于现实情况，如果最后变成[b,1024,h,w]这个参数量将非常之多
因此
    一方面不建议把channel放得太大，一般≤512
    另一方面，可以使用stride缩小图片
最后
    这不是真正的ResNet18
    图片尺寸就不是24*24，而是用的cifar10的32*32
    等等
"""

class ResBlock(nn.Module):
    """
    from    [b, ch_in,  h, w]
    to      [b, ch_out, h, w]
    """

    # 需要知道x是什么，才能把维度对应好
    def __init__(self, ch_in, ch_out, stride=1): # stride默认值为1,具体优先级为 指定值>默认值

        super(ResBlock, self).__init__()

        """两个卷积"""
        # 第一个卷积改变了channel和图片尺寸
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            # 笔记里曾经提到，BN是在channel上做的
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        # 第二个卷积没改变
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out,ch_out, kernel_size=3, stride=1, padding=1),
            # 笔记里曾经提到，BN是在channel上做的
            nn.BatchNorm2d(ch_out)
            # 这里加不加relu看自己咯
        )

        """预防ch_in != ch_out，先建一个空的，相等，就相当于啥都没发生，不相等，就要加内容了"""
        self.extra = nn.Sequential()
        if ch_in != ch_out:
            # [b,ch_in,h,w] => [b,ch_out,h,w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, stride=stride, kernel_size=1), # 因为需要保证图片一样大，所以指定stride
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

        # [2,3,32,32] => [2,64,10,10]       (32-3+0)/3 + 1 =10
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )

        # [2,64,10,10] => [2,128,5,5]       (10-3+2)/2 + 1 = 5
        self.blk1 = ResBlock(64, 128, stride=2)
        # [2,128,5,5] => [2,256,3,3]        (5-3+2)/2 + 1 = 3
        self.blk2 = ResBlock(128, 256, stride=2)
        # [2,256,3,3] => [2,512,2,2]        (3-3+2)/2 + 1 = 2
        self.blk3 = ResBlock(256, 512, stride=2)
        # [2,512,2,2] => [2,512,1,1]        (2-3+2)/2 + 1 = 1
        # 但现实不是这样的，而是 =>[2,512,2,2] 不知道怎么算来的？？？？？？？？？？？？？？
        self.blk4 = ResBlock(512, 512, stride=2)


        # 变成[2,512*1*1]
        self.outlayer = nn.Linear(512, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # [2, 512, 2, 2] => [2, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])

        # 变成[2,512*1*1]
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x

# def test():
#
#     blk = ResBlock(64, 128, stride=4)
#     tmp = torch.randn(2, 64, 32, 32)
#     out = blk(tmp)
#     print('ResBlock:', out.shape)
#
#     # 注释里的shape变换均从[2, 3, 32, 32]开始
#     x = torch.randn(2, 3, 32, 32)
#     model = ResNet18()
#     out = model(x)
#     print('ResNet:', out.shape)
#
#
# if __name__ == '__main__':
#     test()

