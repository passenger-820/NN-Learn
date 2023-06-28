import numpy as np
import random
import visdom
# python -m visdom.server
import matplotlib.pyplot as plt
import torchvision
import torch
from torch import nn, optim, autograd
from torch.nn import functional as f

h_dim =400
batchsz = 512
viz = visdom.Visdom()

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # z:[b, 2] => [b, 2]
            nn.Linear(2, h_dim),
            nn.ReLU(inplace=True), # 进行原地操作,节省内存
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2) # 为了能在2维平面可视化
        )

    def forward(self, z):
        out = self.net(z)
        return out

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid() # D输出一个0~1的probablity
        )

    def forward(self, x):
        out = self.net(x)
        return out

def dataset_generator():
    """
    8-gaussion mixture models
    """
    scale = 2.
    # centers是单位圆与坐标轴的4个交点 + 这4个点中，每相邻两个点连线的中点，共8个点
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    # all 坐标 *2
    centers = [(scale * x, scale * y) for x, y in centers]
    # 迭代器
    while True:
        # 数据存于此list中
        dataset = []
        # 一次迭代一个batch，即一次循环往dataset里放512个点
        for i in range(batchsz):
            # 从标准正态分布sample一个点
            point = np.random.randn(2) * 0.02
            # 随机在上面8个center中选择一个作为本次center
            center = random.choice(centers)
            # 在此center加上variance，相当于在这个点很近的附近随机找个点
            point[0] += center[0]
            point[1] += center[1]
            # 于是形成了以本此center为mean的那么一个distribution
            dataset.append(point)
        # 一个batch size后，这8个center附近都有了不少sample出的点
        # 转换成numpy array
        dataset = np.array(dataset, dtype='float32')
        # 放缩一下
        dataset /= 1.414 # stdev
        # 运行至此，会返回数据，并保存当前状态，下次调用迭代器时，会从这次的状态继续走下去
        yield dataset

def generate_image(D, G, xr, epoch):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    # points [128,128,2]
    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    # -3~3，切分成128份,再从[128]=>[128,1],然后赋给[128,128];即这128行1列的数复制到128行128列，也就是那一列的值copy128遍
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    # -3~3，切分成128份,再从[128]=>[1,128],然后赋给[128,128];即这1行128列的数复制到128行128列，也就是那一行的值copy128遍
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    # [128,128,2] => [16384, 2]
    points = points.reshape((-1, 2))


    # draw contour
    with torch.no_grad():
        points = torch.Tensor(points).cuda() # [16384, 2]
        disc_map = D(points).cpu().numpy() # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)
    # plt.colorbar()


    # draw samples
    with torch.no_grad():
        z = torch.randn(batchsz, 2).cuda() # [b, 2]
        samples = G(z).cpu().numpy() # [b, 2]
    plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    viz.matplot(plt, win='contour', opts=dict(title='p(x):%d'%epoch))

def main():
    # 养成好习惯，设置一下种子，这样可以让以后有着同样的随机性，不至于结果每次都不一样
    torch.manual_seed(23)
    np.random.seed(23)

    # data_iter = dataset_generator()
    # x = next(data_iter)
    # print(x.shape) # (512, 2)

    G = Generator().cuda()
    D = Discriminator().cuda()
    optim_G = optim.Adam(G.parameters(), lr=5e-4)
    optim_D = optim.Adam(D.parameters(), lr=5e-4)
    # 这里不用写crteon，因为不用额外找loss function，直接用pred作loss即可
    # print(G)
    # print(D)

    # 可视化
    # [[0, 0]]:第一个曲线是loss_D 第二个曲线是loss_G
    # [0]: 横坐标[epoch]
    viz.line([[0,0]], [0], win='loss', opts=dict(title='loss', legend=['loss_D','loss_G']))

    """train"""
    for epoch in range(50000):
        """train D for k steps, remember to fix G"""
        for _ in range(5):
            # 1. train on real data
            x_real = next(dataset_generator())
            x_real = torch.from_numpy(x_real).cuda()
            # [512, 2] => [512, 1]
            pred_real = D(x_real)
            # we need to maximize pred_real
            # so we let its mean to be negtive, thus we can use gradient descent
            # 样符合原公式，但却没有使用 loss_real = 1 - (pred_real.mean())
            loss_real = - pred_real.mean()

            # 2. train on fake data
            z = torch.randn(batchsz, 2).cuda()
            x_fake = G(z).detach() # 梯度闸门，不会往前传梯度了，因而G is fixed；类似于 tf.stop_gradient()
            pred_fake = D(x_fake)
            # we need to minimize pred_fake, so we use positive
            loss_fake = pred_fake.mean()

            # aggregate all
            loss_D = loss_real + loss_fake

            # bp
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        """train G, remember to fix D"""
        z = torch.randn(batchsz, 2).cuda()
        x_fake = G(z)
        pred_fake = D(x_fake) # 不能在这.detach()，否则你断了bp路，怎么优化上面的G！！！！
            # 那就让他算D的梯度呗，反正咱们在train D的时候，有　optim_D.zero_grad()，清零了，不会影响D的训练
        # we need G to successfully fool D, so wo have to maximize pred_fake, thus to be negtive to use GD
        # 这样符合原公式，但却没有使用 loss_G = 1 - (pred_fake.mean())
        loss_G = - pred_fake.mean()

        # bp
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch %100 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
            print(loss_D.item(), loss_G.item())

            generate_image(D, G, x_real.cpu(), epoch)

if __name__ == '__main__':
    main()