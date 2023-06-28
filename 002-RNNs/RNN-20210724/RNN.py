
import  numpy as np
import  torch
import  torch.nn as nn
import  torch.optim as optim
from    matplotlib import pyplot as plt


num_time_steps = 50
input_size = 1
hidden_size = 16
output_size = 1
lr=0.01



class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size, 			# 1
            hidden_size=hidden_size,		# 16
            num_layers=1,
            batch_first=True,				# [batch,seq_len,feature_len]
        )
        for p in self.rnn.parameters():
          nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size) # 输出层了，只要1个数字

    def forward(self, x, hidden_prev): # x, h0

       out, hidden_prev = self.rnn(x, hidden_prev)
       		# out:[b, seq, h] ht:[b,1,h]
       # out打平送入下一层进行输出 [b, seq, h]=> [b*seq, h]=[seq, h]
       out = out.view(-1, hidden_size)
	   # 得到输出的数值 [seq, h]=>[seq, 1]
       out = self.linear(out)
       # 在0维度前插入一个新维度[seq, 1]=>[1,seq, 1]
       # 因为要和y做误差的比较(用的MSE) y是[b,seq,1]即[1,seq, 1]
       out = out.unsqueeze(dim=0)
       return out, hidden_prev




model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr) # lr=0.01

"""-----------------train-----------------"""
# 新建h0 [b,layer,h] = [1,1,16]
hidden_prev = torch.zeros(1, 1, hidden_size)

for iter in range(6000):
    """sample数据"""
    # [0,3)随机找一个
    start = np.random.randint(3, size=1)[0]
    # 等量切分，双向闭合，切成50份
    time_steps = np.linspace(start, start + 10, num_time_steps)
    # 将这些点取sin
    data = np.sin(time_steps)
    # [50]=>[50,1]
    data = data.reshape(num_time_steps, 1)
    # 一直x是第0~48的曲线
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
    # 去预测y的1~49的曲线，这个y是真正的y，用于和预测的比较误差
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

    """送去train"""
    # 把x和h0送入RNN得到数值
    output, hidden_prev = model(x, hidden_prev)
    hidden_prev = hidden_prev.detach()

    # 计算MSELoss
    loss = criterion(output, y)
    # 更新参数 Whh_l0和Wih_l0
    model.zero_grad()
    loss.backward()
    # for p in model.parameters():
    #     print(p.grad.norm())
    # torch.nn.utils.clip_grad_norm_(p, 10)
    optimizer.step()

    if iter % 100 == 0:
        # 打印loss
        print("Iteration: {} loss {}".format(iter, loss.item()))

"""-----------------test-----------------"""
"""sample数据"""
start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

# 先把预测置空
predictions = []
# x是[1,seq,1],咱就取一个[1,1]做样本
input = x[:, 0, :]
for _ in range(x.shape[1]):
  # view一下这个点
  input = input.view(1, 1, 1) 	  # [1,1]=>[1,1,1]
  # 根据train出的memory：hidden_prev的到预测值，并更新memory
  pred, hidden_prev = model(input, hidden_prev)
  input = pred
  predictions.append(pred.detach().numpy().ravel()[0])

x = x.data.numpy().ravel()
y = y.data.numpy()
plt.scatter(time_steps[:-1], x.ravel(), s=90)
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[1:], predictions)
plt.show()