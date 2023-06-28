import torch
from torch import nn


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # [b, 3, 32, 32] => [b, 3, 28, 28]
            nn.Conv2d(3, 6, kernel_size=5),
            # [b, 3, 28, 28] => [b, 3, 14, 14]
            nn.MaxPool2d(kernel_size=2),
            # [b, 3, 14, 14] => [b, 3, 10, 10]
            nn.Conv2d(6, 16, kernel_size=5),
            # [b, 3, 10, 10] => [b, 3, 5, 5]
            nn.MaxPool2d(kernel_size=2)
        )

        # Flatten

        self.fc_unit_logits = nn.Sequential(
            # 来自 "拍平为[b,16*5*5]"
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        """
        :param x: [b, 3, 32, 32]
        :return: logits
        """
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)

        # 拍平 [b, 16, 5, 5] => [b, 16*5*5]
        # x.size(0) 等价于 x.shape[0] 001-?.md的笔记中有
        x = x.view(x.size(0), -1)

        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit_logits(x)
        return logits

#     def test_conv_unit(self):
#         fake_data = torch.randn(2, 3, 32, 32)
#         out = self.conv_unit(fake_data)
#         # torch.Size([2, 16, 5, 5]) 所以拍平为[b,16*5*5]
#         print('conv out: ', out.shape)
#
#
#
# if __name__ == '__main__':
#     l5 = LeNet5()
#     l5.test_conv_unit()