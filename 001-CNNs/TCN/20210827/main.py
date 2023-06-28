import torch
from parameter import *
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys

sys.path.append("../../")
from data_loader import get_data
from model import TCN
import numpy as np


args = get_parameters()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


n_classes = 10
input_channels = 1
seq_length = int(784 / input_channels)
epochs = args.epochs
steps = 0

print(args)
train_loader, test_loader = get_data(args.root, args.batch_size)

# 将某个对象转置成 [784]
permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
# [25]*8 => [25,25,25,25,25,25,25,25]    python list乘法!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
channel_sizes = [args.nhid] * args.levels
# TCN[input_channels=1, n_classes=10, channel_sizes=[25,25,25,25,25,25,25,25], kernel_size=7, dropout=0.05]
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=args.ksize, dropout=args.dropout)

if args.cuda:
    model.cuda()
    permute = permute.cuda()

# 2e-3
lr = args.lr
# getattr在(torch.nn.)optim中根据名称args.optim即"Adam"返回对象"torch.nn.optim.Adam()"
# 相当于 optimizer = torch.nn.optim.Adam(model.parameters(), lr=lr)
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(ep):
    global steps
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        # [64,1,784]
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            # [:,:,784]
            data = data[:, :, permute]
        # [64,1,784], [64]
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        # [64,10]
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * args.batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval, steps))
            train_loss = 0


def evaluate():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss


if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch)
        evaluate()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr