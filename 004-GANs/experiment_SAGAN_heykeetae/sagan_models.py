


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        # 三个卷积操作，分别命名为如下：     # //是整除
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)

        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        # gamma初始化为0 后面会变更
        self.gamma = nn.Parameter(torch.zeros(1))
        # 当dim=0时， 是对每一维度相同位置的数值进行softmax运算
        # 当dim = 1时， 是对某一维度的 列 进行softmax运算
        # 当dim=2时， 是对某一维度的 行 进行softmax运算
        # 当dim=-1时， 是对某一维度的 行 进行softmax运算 从这里可以发现，dim=-1和dim=2的结果是一样的
        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( Batchsize * Channel * Width * Height)
            returns :
                out : self attention value + input feature 
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        # fp经过query卷积层得到f(x)；将此tensor转换成 B*C*N 其中，N=W*H；再经过permute转化成 B*N*C 在此转置是为了下面的计算
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        # fp经过key卷积层得到g(x)；将此tensor转换成 B*C*N 其中，N=W*H；                           N*C*C*N=N*N
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height)
        # 矩阵相乘 f^T * g得到transpose check  形状为B*N*N
        energy = torch.bmm(proj_query,proj_key)
        # 再经由softmax得到attention map  形状仍为B*N*N
        attention = self.softmax(energy)
        # 之后，fp经过query卷积层得到h(x)；将此tensor转换成 B*C*N 其中，N=W*H；
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)
        # h * attention    C*N*N*N = C*N  这里attention为什么还要转置？？？？？？？？、
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        # tensor还原为 B*C*W*N
        out = out.view(m_batchsize,C,width,height)
        # 公式 y_i = γ*O_i + X_i   这个gamma初始化为0的，后面通过不断的学习会发生改变，再加上最初的fp，就能得到新的fp Y
        out = self.gamma*out + x
        return out,attention

class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=128, conv_dim=64): # 原本z_dim是100，我改成128了 其实可以都不要 =?，本来传过来的就给了值了
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3 # log2(64) - 3 = 6 - 3 = 3
        mult = 2 ** repeat_num # 2^3 = 8
        # 普范数归一化(转置卷积)操作        # output = (input-1)*stride + outputpadding -2*padding + kernel_size
        #         in_channels: 100 , 输入图片的通道数
        #         out_channels: 64 * 8 = 512,   输出图片的通道数    这些只是通道数，具体图片尺寸是用上面output公式算的！
        #         kernel_size: 4,
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        # BatchNorm操作
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        # 激活函数ReLU
        layer1.append(nn.LeakyReLU(0.1))
        # 当前维度 512
        curr_dim = conv_dim * mult

        #         in_channels: 512,
        #         out_channels: 256,
        #         kernel_size: 4,
        #         stride: 2,
        #         padding: 1,
        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.LeakyReLU(0.1))
        # 当前维度 256
        curr_dim = int(curr_dim / 2)

        #         in_channels: 256,
        #         out_channels: 128,
        #         kernel_size: 4,
        #         stride: 2,
        #         padding: 1,
        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.LeakyReLU(0.1))

        if self.imsize == 64:
            layer4 = []
            # 当前维度 128
            curr_dim = int(curr_dim / 2)
            #         in_channels: 128
            #         out_channels: 64,
            #         kernel_size: 4,
            #         stride: 2,
            #         padding: 1,
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            # 当前维度 64
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        # 最后一层，利用转置卷积操作
        #         in_channels: 64
        #         out_channels: 3,
        #         kernel_size: 4,
        #         stride: 2,
        #         padding: 1,
        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        # Tanh 激活函数
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        # 经过上面定义的Self_Atten 函数
        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')
    # 将他们相连
    def forward(self, z):

        z = z.view(z.size(0), z.size(1), 1, 1)      # 64，128----》64，128，1，1        # 相当于64个，128通道数的1*1的图片
        out=self.l1(z)                              # 64，128，1，1----》64，512，4，4         # 4 = output = (1-1)*1 + 0 - 2*0 + 4
        out=self.l2(out)                            # 64，512，4，4----》64，256，8，8         # 8 = output = (4-1)*2 + 0 - 2*1 + 4
        out=self.l3(out) # l3 输出维度是128           # 64，256，8，8----》64，128，16，16       # 16 = output = (8-1)*2 + 0 - 2*1 + 4
        out,p1 = self.attn1(out) # out = self.gamma*out + x； attention = self.softmax(energy)   # 还是64，128，16，16  # p1是64，256，256
        out=self.l4(out) # l4 输出维度是64            # 64，128，16，16----》64，64，32，32      # 32 = output = (16-1)*2 + 0 - 2*1 + 4
        out,p2 = self.attn2(out)                    # 还是64，64，32，32 # p1是64，256，256  # p2是64，1024，1024
        out=self.last(out)                          # 64，64，32，32----》64，3，64，64        # 64 = output = (32-1)*2 + 0 - 2*1 + 4

        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []
        # 类似，只是这里用卷积而不是转置卷积 # (input_size - kernel_size + 2*padding)/stride  +  1
        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        # 没有BatchNorm操作，操作激活函数LReLU
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        # 最后一层，利用卷积
        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)
        # 经过上面定义的Self_Atten 函数
        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    # 将他们相连
    def forward(self, x):               # 64，3，64，64
        out = self.l1(x)                # 64，3，64，64----》64，64，32，32
        out = self.l2(out)              # 64，64，32，32----》64，128，16，16
        out = self.l3(out)              # 64，128，16，16----》64，256，8，8
        out,p1 = self.attn1(out)        # 64，256，8，8 p1:64,64,64
        out=self.l4(out)                # 64，256，8，8----》64，512，4，4
        out,p2 = self.attn2(out)        # 64，512，4，4 p1:64,16,16
        out=self.last(out)              # 64，512，4，4----》64,1,1,1
        # 把shape中为1的维度去掉  https://blog.csdn.net/qq_38675570/article/details/80048650
        return out.squeeze(), p1, p2    # 64
