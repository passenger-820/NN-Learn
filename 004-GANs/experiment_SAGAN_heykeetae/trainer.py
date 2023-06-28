
import os
import time
import torch
import datetime
import visdom

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from sagan_models import Generator, Discriminator
from utils import *

import matplotlib.pyplot as plt





class Trainer(object):

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # 设置随机数种子
    setup_seed(2010)



    def __init__(self, data_loader, config):

        # history
        # self.history_d_loss_fake = []
        # self.history_d_out_real = []
        self.history_d_loss = []
        self.history_g_loss_fake = []

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model                        # [sagan]
        self.adv_loss = config.adv_loss                  # [hinge]

        # Model hyper-parameters
        self.imsize = config.imsize                      # [64]
        self.g_num = config.g_num                        # 5
        self.z_dim = config.z_dim                        # 128
        self.g_conv_dim = config.g_conv_dim              # 64
        self.d_conv_dim = config.d_conv_dim              # 64
        self.parallel = config.parallel                  # type=str2bool, default=False  是否使用多个GPU

        self.lambda_gp = config.lambda_gp                # 10
        self.total_step = config.total_step              # [100000]
        self.d_iters = config.d_iters                    # 5
        self.batch_size = config.batch_size              # 64
        self.num_workers = config.num_workers            # 2
        self.g_lr = config.g_lr                          # 0.001
        self.d_lr = config.d_lr                          # 0.004
        self.lr_decay = config.lr_decay                  # 0.95
        self.beta1 = config.beta1                        # 0.0  这两个β就是 《lhy 2020.08-09 Optimization》 第16页的那 2 个参数
        self.beta2 = config.beta2                        # 0.9
        self.pretrained_model = config.pretrained_model  # [25536]

        self.dataset = config.dataset                    # [celeb]
        self.use_tensorboard = config.use_tensorboard    # type=str2bool, default=False
        self.image_path = config.image_path              # ./data
        self.log_path = config.log_path                  # ./log
        self.model_save_path = config.model_save_path    # ./models
        self.sample_path = config.sample_path            # ./samples
        self.log_step = config.log_step                  # [10]
        self.sample_step = config.sample_step            # [400]
        self.model_save_step = config.model_save_step    # [1.0]
        self.version = config.version                    # [sagan_10]

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.build_model()                               # 下面的 build_model(self)方法

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()


    def train(self):
        # Data iterator 数据迭代器
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)  # 21551/64 = 336
        model_save_step = int(self.model_save_step * step_per_epoch) # 1*336 = 336

        # Fixed input for debugging [64个这个 → [128个这个 → 标准正态分布出来的数] ]
        # fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim)) # 转成 cuda() 数据

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()
        # Visulization
        viz = visdom.Visdom() # python -m visdom.server
        viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D_loss', 'G_loss']))
        for step in range(start, self.total_step):

            # ================== Train D ================== #

            # model.train()
            # 启用 Batch Normalization 和 Dropout。
            # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
            # model.train()是保证BN层能够用到每一批数据的均值和方差。
            # 对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
            # model.eval()则是不启用
            self.D.train()
            self.G.train()

            try:
                # 64,3,64,64
                real_images, _ = next(data_iter) # 如果可以，用下一波数据训练
            except:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)

            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            real_images = tensor2var(real_images) # 转成cuda数据
            d_out_real,dr1,dr2 = self.D(real_images) # 让D认识真图
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':          # 使用铰链损失函数 得到D对真实图片的loss
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            # z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            # real_images.size() torch.Size([64,3,64,64])
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim)) # 生成cuda随机噪声
            fake_images,gf1,gf2 = self.G(z) # 用于产生假图
            d_out_fake,df1,df2 = self.D(fake_images) # 让D去分辨这些假图

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':          # 使用铰链损失函数 得到D对虚假图片的loss
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake  # D的总loss
            self.reset_grad()                   # 梯度归零
            d_loss.backward()                   # 自动计算梯度
            self.d_optimizer.step()             # 然后优化


            # ================== Train G and gumbel ================== #
            # Create random noise
            # z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images,_,_ = self.G(z)

            # Compute loss with fake images
            g_out_fake,_,_ = self.D(fake_images)  # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            # # 记录一些history data
            # # self.history_d_loss_fake.append(d_loss_fake.item())
            # # self.history_d_out_real.append(d_loss_real.item())
            # self.history_d_loss.append(d_loss.item())
            # self.history_g_loss_fake.append(g_loss_fake.item())

            # Print out log info
            if (step + 1) % self.log_step == 0:
                viz.line([[d_loss.item(), g_loss_fake.item()]], [step], win='loss', update='append')
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                # print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                #       " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                #       format(elapsed, step + 1, self.total_step, (step + 1),
                #              # IndexError: invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number
                #              # d_loss_real.data[0]---->d_loss_real.item()  再下面也是同理
                #              self.total_step , d_loss_real.item(),
                #              self.G.attn1.gamma.mean().item(), self.G.attn2.gamma.mean().item() ))
                print("Elapsed [{}], step [{}/{}], d_out_real: {:.4f}, "
                      " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".format(
                                                                            elapsed, step + 1,
                                                                            self.total_step,
                                                                            d_loss_real.item(),
                                                                            self.G.attn1.gamma.mean().item(),
                                                                            self.G.attn2.gamma.mean().item()))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images,_,_= self.G(fixed_z)
                save_image(denorm(fake_images.data[:64]),
                           os.path.join(self.sample_path, '{}_fake.jpg'.format(step + 1)))
                # # 独立小图
                # for i in range(0, fake_images.size(0)):
                #     save_image(denorm(fake_images.data[i]),os.path.join(self.sample_path, '{}_fake{}.png'.format(step + 1, i + 1)))
                # # 一张大图  即64小图
                # # 移到上面去了

            # # 我自己加的
            # if (step + 1) % self.total_step == 0:
            #     for j in range(1000):
            #         z_new = tensor2var(torch.randn(self.batch_size, self.z_dim))
            #         fake_images, _, _ = self.G(z_new)
            #         save_image(denorm(fake_images.data),os.path.join(self.sample_path, f'{step + 1}_{j}fake.jpg'))
            #         # 独立小图
            #         for i in range(0, fake_images.size(0)):
            #             save_image(denorm(fake_images.data[i]),os.path.join(self.sample_path, f'{step + 1}_{j}_fake{i + 1}.png'))


            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

        # # 绘制loss   .cpu().detach().numpy()
        # # plt.plot(self.history_d_out_real, color='?', linewidth=1.0, linestyle='-', label='d_loss_real')
        # #plt.plot(self.history_d_loss_fake, color='?', linewidth=1.0, linestyle='-', label='d_loss_fake')
        # plt.plot(self.history_d_loss, color='blue', linewidth=1.0, linestyle='-', label='d_loss')
        # #plt.plot(self.history_g_loss_fake, color='?', linewidth=1.0, linestyle='-', label='g_loss_fake')
        # plt.xlabel('steps')
        # plt.ylabel('Loss')
        # # plt.ylim(0.75, 1)
        # plt.legend(loc='best')
        # plt.title('D_Loss at Different Step')
        # plt.show()
        #
        # # plt.plot(self.history_d_out_real, color='red', linewidth=1.0, linestyle='-', label='d_loss_real')
        # #plt.plot(self.history_d_loss_fake, color='blue', linewidth=1.0, linestyle='-', label='d_loss_fake')
        # #plt.plot(self.history_d_loss, color='yellow', linewidth=1.0, linestyle='-', label='d_loss')
        # plt.plot(self.history_g_loss_fake, color='green', linewidth=1.0, linestyle='-', label='g_loss_fake')
        # plt.xlabel('steps')
        # plt.ylabel('Loss')
        # # plt.ylim(0.75, 1)
        # plt.legend(loc='best')
        # plt.title('G_Loss at Different Step')
        # plt.show()

    def build_model(self):

        self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).cuda()
        if self.parallel:                                # 因为 Flase ，所以不使用多个GPU
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        """根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了"""
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    # 这个data_iter作用机理还没摸明白，以后研究
    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))

