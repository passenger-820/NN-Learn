import torch
import torchvision
from torchvision import transforms

from LeNet5_model import Lenet5

def main():

    cifar10_bs = 32

    """
        SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 18-19: malformed \ N character escape
        引起这个错误的原因就是转义的问题
        1、在路径前面加r，即保持字符原始值的意思。
        2、替换为双反斜杠
        3、替换为正斜杠
    """
    # 这个类一次智能加载完所有图片
    cifar_train = torchvision.datasets.CIFAR10(r'F:\PycharmProjects\NeuralNetwork\data\cifar10',
                             train=True,
                             transform=transforms.Compose([
                                 # Size should be int or sequence
                                 transforms.Resize((32,32)),
                                 transforms.ToTensor()
                             ]), download=True)
    # 然后按照batch_size加载
    cifar_train = torch.utils.data.DataLoader(cifar_train,batch_size=cifar10_bs, shuffle=True)

    cifar_test = torchvision.datasets.CIFAR10(r'F:\PycharmProjects\NeuralNetwork\data\cifar10', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = torch.utils.data.DataLoader(cifar_test, batch_size=cifar10_bs, shuffle=True)

    # x, label = iter(cifar_train).next()
    # # x:  torch.Size([32, 3, 32, 32]) label:  torch.Size([32])
    # print('x: ',x.shape,'label: ',label.shape)

    """---train---"""
    # use GPU
    device = torch.device('cuda')
    # use model
    model = Lenet5().to(device)
    """-----------------------------加载checkpoint-----------------------------"""
    # model.load_state_dict(torch.load('ckpt.mdl'))

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
        0 1.1374456882476807
        1 1.2553949356079102
        2 1.0216774940490723
        3 0.4976608157157898
        4 0.9076613187789917
        5 0.3592182993888855
        6 0.6362616419792175
        """

        """-----------------------------保存checkpoint-----------------------------"""
        # torch.save(model.state_dict(), 'ckpt.mdl')

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
        0 1.3506735563278198
        0 test acc:  0.4997
        
        1 1.4095042943954468
        1 test acc:  0.5478
        
        2 1.4588080644607544
        2 test acc:  0.583
        """



if __name__ == '__main__':
    main()