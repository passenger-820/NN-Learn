"""
代码来自：https://blog.csdn.net/qq_40608730/article/details/110546612
"""

#from datasets import *

import torch.nn as nn
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.utils.data
from scipy.stats import entropy
from torchvision.models.inception import inception_v3

from IS_data_loader import ISImageDataset

# 读取文件的路径
path = "./samples/sagan_10_project/02/sagan_10_100000"

# 统计一共有多少图片
count = 0
for root,dirs,files in os.walk(path):    #遍历统计
      for each in files:
             count += 1   #统计文件夹下文件个数
print(count)

# 一批喂的张数
batch_size = 32
"""
transforms.Resize:
    Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
transforms.ToTensor(): 0~255的shape为高*宽*通道 ————》 0.0~1.0的shape为通道*高*宽
    Converts a PIL(Python Image Library) Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
transforms.Normalize: 把0-1变换到(-1,1)
    对每个通道而言，Normalize执行以下操作：
        image=(image-mean)/std
        其中mean和std分别通过(0.5,0.5,0.5)和(0.5,0.5,0.5)进行指定。原来的0-1最小值0则变成(0-0.5)/0.5=-1，而最大值1则变成(1-0.5)/0.5=1.

    Normalize a tensor image with mean and standard deviation.
        Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n`` channels, 
        this transform will normalize each channel of the input ``torch.*Tensor`` i.e.,
        ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
        .. note::
            This transform acts out of place, i.e., it does not mutate the input tensor.
        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation in-place.
"""
transforms_ = [
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # we should use same mean and std for inception v3 model in training and testing process
    # reference web page: https://pytorch.org/hub/pytorch_vision_inception_v3/
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # mean[1]=0.485,std[1]=0.229; ...
    # 计算结果-2.118~2.249; -2.036~2.429; -1.804~2.640
]

# Set up dataloader
val_dataloader = DataLoader(
    ISImageDataset(path, transforms_=transforms_),
    batch_size = batch_size,
)

# Set up dtype
cuda = True if torch.cuda.is_available() else False
print('cuda: ',cuda)
tensor = torch.cuda.FloatTensor

# Load inception model
inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
inception_model.eval()
up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).cuda()

def get_pred(x):
    if True:
        x = up(x)  # ori_x shape [batch_size,3,256,256]; new_x/up(x) shape [batch_size,3,299,299]
    x = inception_model(x) # inception 原本输出的就是 1000
    return F.softmax(x, dim=1).data.cpu().numpy() # 所以这里shape是(batch_size,1000)

# Get predictions using pre-trained inception_v3 model
print('Computing predictions using inception v3 model')
# preds 为 count行 1000列的 0矩阵
preds = np.zeros((count, 1000))

"""
对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
    如果对一个列表，既要遍历索引又要遍历元素时，首先可以这样写：
        list1 = ["这", "是", "一个", "测试"]
        for i in range (len(list1)):
        print i ,list1[i]
    上述方法有些累赘，利用enumerate()会更加直接和优美：
        list1 = ["这", "是", "一个", "测试"]
        for index, item in enumerate(list1):
        print index, item
        >>>
        0 这
        1 是
        2 一个
        3 测试
https://blog.csdn.net/churximi/article/details/51648388
"""
# 所以 i是val_dataloader的索引，data是val_dataloader每个索引项对应的元素
for i, data in enumerate(val_dataloader):
    # 转换成tensor类型这个tensor来自这儿
    # cuda = True if torch.cuda.is_available() else False
    # print('cuda: ',cuda)
    # tensor = torch.cuda.FloatTensor  是这种tensor
    data = data.type(tensor) # shape [batch_size,3,256,256]
    # data.size()={Size:4}torch.Size([4（不够整除剩余的量）/batch_size, 3, 256, 256])
    # 所以 data.size()[0] = batch_size或 不够整除剩余的量
    batch_size_i = data.size()[0]
    # 原本pred为100*1000的0矩阵，现在把这[batch_size,3,256,256]丢到up里去，变成[batch_size,3,259,259]
    # （batch_size=8时）第1维度的第i=0:0~7，i=1:8~15，16~23，...，i=11(batch_size_i=8):88~95，i=12(batch_size_i=4):96~99行依次赋值
    preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(data)

# Now compute the mean KL Divergence
# print('Computing KL Divergence')
split_scores = []
splits=10
N = count
for k in range(splits):
    # 取第 “k * (N // splits)”行到第“(k + 1) * (N // splits)-1”行的所有列
    part = preds[k * (N // splits): (k + 1) * (N // splits), :] # split the whole data into several parts
    py = np.mean(part, axis=0)  # marginal probability
    scores = []
    for i in range(part.shape[0]):
        pyx = part[i, :]  # conditional probability
        scores.append(entropy(pyx, py))  # compute divergence
    split_scores.append(np.exp(np.mean(scores)))

# 证明过程来自：https://blog.csdn.net/qq_27261889/article/details/86483505
# IS值越高，图片质量和多样性则好
# std 标准差 看数据分布是否集中
mean, std  = np.mean(split_scores), np.std(split_scores)
print('IS is %.4f' % mean)
print('The std is %.4f' % std)
