import torch
from torch import nn

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        # [b,784] => [b, 20]
        # mu: [b,10]
        # sifma: [b,10]
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        #  [b, 20] => [b,10] => [b, 784]
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            # 要压缩到0~1
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: [b, 1, 28 ,28]
        :return:
        """
        batchsz = x.size(0)
        # Flatten
        x = x.view(batchsz, 784)
        # encoder  [b, 20]
        h_ = self.encoder(x)


        # one trick: u + sigma * distribution
        # => [b, 20]   h_ includes mean and sigma
        # 在第1维度拆分  [b, 20] => [b, 10] and [b, 10]
        mu, sigma = h_.chunk(2, dim=1)
        # reparameterize trick, epison~N(0, 1)  size like sigma
        h = mu + sigma * torch.randn_like(sigma) # [b, 10] + [b, 10] = [b, 10]




        # decoder
        x_hat = self.decoder(h)
        # reshape
        x_hat = x_hat.view(batchsz, 1, 28, 28)



        # 计算KL，1e-8是为了让结果不出现-∞
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (batchsz * 28 * 28) # 这样除一下，可以比较好地平衡左右两个loss

        return  x_hat, kld

