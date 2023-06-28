# bug 2 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# rusume是否使用预训练模型继续训练,问号处输入模型的编号
resume = True  # 是继续训练，否重新训练
datasets = 'mnist'  # 选择cifar10, cifar100, mnist, fashion_mnist,STL10,Anime

if datasets == 'cifar10' or datasets == 'cifar100' or datasets == 'STL10' or datasets == 'Anime':
    nc = 3  # 图片的通道数
elif datasets == 'mnist' or datasets == 'fashion_mnist':
    nc = 1
else:
    print('数据集选择错误')

batch_size = 128
nz = 100  # 噪声向量的维度
ndf = 64
ngf = 64
real_label = 1
fake_label = 0
start_epoch = 0

# 定义模型


# 生成器                             #(N,nz, 1,1)
netG = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                     nn.Tanh()  # (N,nz, 128,128)

                     )

# 判别器             #(N,nc, 128,128)
netD = nn.Sequential(nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # (N,1,1,1)
                     nn.Flatten(),  # (N,1)
                     nn.Sigmoid()
                     )


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


netD.apply(weights_init)
netG.apply(weights_init)

# 加载数据集
apply_transform1 = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

apply_transform2 = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

if datasets == 'cifar100':
    train_dataset = torchvision.datasets.CIFAR100(root='../data/cifar100', train=False, download=True,
                                                  transform=apply_transform1)
elif datasets == 'cifar10':
    train_dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, download=True,
                                                 transform=apply_transform1)
elif datasets == 'STL10':
    train_dataset = torchvision.datasets.STL10(root='../data/STL10', split='train', download=True,
                                               transform=apply_transform1)
elif datasets == 'mnist':
    train_dataset = torchvision.datasets.MNIST(root='../data/mnist', train=False, download=True,
                                               transform=apply_transform2)
elif datasets == 'fashion_mnist':
    train_dataset = torchvision.datasets.FashionMNIST(root='../data/fashion_mnist', train=False, download=True,
                                                      transform=apply_transform2)
elif datasets == 'Anime':
    train_dataset = torchvision.datasets.ImageFolder(root='../data/Anime', transform=apply_transform1)

else:
    print('数据集不存在')
# bug 1 DataLoader worker (pid(s) 11268, 1384, 10744, 11168) exited unexpectedly
# dataloader=tud.DataLoader(data,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
# DataLoader worker (pid(s) 11404, 1168, 15692, 13308) exited unexpectedly
# dataloader=tud.DataLoader(data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
# 因为windows中不能使用多进程
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 定义损失函数
criterion = torch.nn.BCELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setup optimizer
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 显示16张图片

if datasets == 'Anime':
    image, label = next(iter(train_loader))
    image = (image * 0.5 + 0.5)[:16]
elif datasets == 'mnist' or datasets == 'fashion_mnist':
    image = next(iter(train_loader))[0]
    image = image[:16] * 0.5 + 0.5

elif datasets == 'STL10':
    image = torch.Tensor(train_dataset.data[:16] / 255)
else:
    image = torch.Tensor(train_dataset.data[:16] / 255).permute(0, 3, 1, 2)
plt.imshow(torchvision.utils.make_grid(image, nrow=4).permute(1, 2, 0))

# 训练和保存模型
# 如果继续训练，就加载预训练模型
if resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/GAN_%s_best.pth' % datasets)
    netG.load_state_dict(checkpoint['net_G'])
    netD.load_state_dict(checkpoint['net_D'])
    start_epoch = checkpoint['start_epoch']
print('netG:', '\n', netG)
print('netD:', '\n', netD)

print('training on:   ', device, '   start_epoch', start_epoch)

netD, netG = netD.to(device), netG.to(device)
# 固定生成器，训练判别器
for epoch in range(start_epoch, 300):
    for batch, (data, target) in enumerate(train_loader):
        batch_size = data.size(0)
        label = torch.full((batch_size, 1), real_label).to(device)

        # （1）训练判别器
        # training real data
        netD.zero_grad()
        data = data.to(device)
        output = netD(data)
        loss_D1 = criterion(output, label)
        loss_D1.backward()

        # training fake data
        noise_z = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_data = netG(noise_z)
        label = torch.full((batch_size, 1), fake_label).to(device)
        output = netD(fake_data.detach())
        loss_D2 = criterion(output, label)
        loss_D2.backward()

        # 更新判别器
        optimizerD.step()

        # （2）训练生成器
        netG.zero_grad()
        label = torch.full((batch_size, 1), real_label).to(device)
        output = netD(fake_data)
        lossG = criterion(output, label)
        lossG.backward()

        # 更新生成器
        optimizerG.step()

        if batch % 100 == 0:
            print('epoch: %4d, batch: %4d, discriminator loss: %.4f, generator loss: %.4f'
                  % (epoch, batch, loss_D1.item() + loss_D2.item(), lossG.item()))

    # 每2个epoch保存图片
    if epoch % 2 == 0:

        # 如果是单通道图片，那么就转成三通道进行保存
        if nc == 1:
            fake_data = torch.cat((fake_data, fake_data, fake_data), dim=1)  # fake_data（N,1,H,W）->(N,3,H,W)
        # 保存图片
        data = fake_data.detach().cpu().permute(0, 2, 3, 1)
        data = np.array(data)

        # 保存单张图片，将图片归一化到(0,1)
        data = (data * 0.5 + 0.5)

        plt.imsave('./generated_fake/%s/epoch_%d.png' % (datasets, epoch), data[0])
        torchvision.utils.save_image(fake_data[:16],
                                     filename='./generated_fake/%s/epoch%d_grid.png' % (datasets, epoch), nrow=4,
                                     normalize=True)

    # 保存模型
    state = {
        'net_G': netG.state_dict(),
        'net_D': netD.state_dict(),
        'start_epoch': epoch + 1
    }
    torch.save(state, './checkpoint/GAN_%s_best.pth' % (datasets))
    torch.save(state, './checkpoint/GAN_Anime_best_copy.pth' % (datasets))





