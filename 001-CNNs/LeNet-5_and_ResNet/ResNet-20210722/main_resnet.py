import torch
import torchvision
from torchvision import transforms

from ResNet_model import ResNet18

def main():

    cifar10_bs = 32
    # 这个类一次智能加载完所有图片
    cifar_train = torchvision.datasets.CIFAR10(r'F:\PycharmProjects\NeuralNetwork\data\cifar10',
                             train=True,
                             transform=transforms.Compose([
                                 # Size should be int or sequence
                                 transforms.Resize((32,32)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]), download=True)
    # 然后按照batch_size加载
    cifar_train = torch.utils.data.DataLoader(cifar_train,batch_size=cifar10_bs, shuffle=True)
    """-------------------------------比较复杂的网络，最好把图片正则化，这样可以加快训练速度-------------------------------"""
    cifar_test = torchvision.datasets.CIFAR10(r'F:\PycharmProjects\NeuralNetwork\data\cifar10',
                                              train=False,
                                              transform=transforms.Compose([
                                                  transforms.Resize((32, 32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])
                                              ]), download=True)
    cifar_test = torch.utils.data.DataLoader(cifar_test, batch_size=cifar10_bs, shuffle=True)

    # x, label = iter(cifar_train).next()
    # # x:  torch.Size([32, 3, 32, 32]) label:  torch.Size([32])
    # print('x: ',x.shape,'label: ',label.shape)

    """---train---"""
    # use GPU
    device = torch.device('cuda')
    """-----------------------这里换个model就行了，很方便-----------------------"""
    # use model
    model = ResNet18().to(device)
    # loss function
    criteon = torch.nn.CrossEntropyLoss().to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # print model
    print(model)

    for epoch in range(1000):
        for batchidx, (x, label) in enumerate(cifar_train):
            # to tensor
            # x : [b, 3, 32, 32]
            # label: [10]
            x, label = x.to(device), label.to(device)
            # logits: [b, 10], label: [10]
            logits = model(x)

            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每完成一个epoch
        print(epoch, loss.item())
        """
        
        """


        """每完成一个epoch，进行 test"""
        total_correct = 0
        total_num = 0
        for batchidx, (x, label) in enumerate(cifar_test):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            # [10]
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        acc = total_correct/total_num
        print(epoch,'test acc: ',acc)
        """
        0 1.429471492767334
        0 test acc:  0.5783
        
        1 0.8001211285591125
        1 test acc:  0.642
        """



if __name__ == '__main__':
    main()