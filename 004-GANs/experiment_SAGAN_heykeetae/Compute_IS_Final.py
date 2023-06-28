import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
import matplotlib.pyplot as plt


class ComputeIS():
    def __init__(self, path, count, batch_size):
        self.mean = []
        self.std = []
        self.path = path
        self.count = count
        self.batch_size = batch_size

    def compute(self):
        transforms_ = [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        # Set up dataloader
        val_dataloader = DataLoader(
            ISImageDataset(self.path, transforms_=transforms_),
            batch_size = self.batch_size,
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
                x = up(x)
            x = inception_model(x)
            return F.softmax(x, dim=1).data.cpu().numpy()

        # Get predictions using pre-trained inception_v3 model
        print('Computing predictions using inception v3 model')
        preds = np.zeros((self.count, 1000))

        for i, data in enumerate(val_dataloader):
            data = data.type(tensor)
            batch_size_i = data.size()[0]
            if i % 8 == 0 :
                for j in range(8):
                    preds[j * self.batch_size:j * self.batch_size + batch_size_i] = get_pred(data)
                # Now compute the mean KL Divergence
                #print('Computing KL Divergence')
                split_scores = []
                splits = 8 # 50/10 = 5
                N = self.count # 50
                for k in range(splits):
                    part = preds[k * (N // splits): (k + 1) * (N // splits), :]  # split the whole data into several parts
                    py = np.mean(part, axis=0)  # marginal probability
                    scores = []
                    for i in range(part.shape[0]):
                        pyx = part[i, :]  # conditional probability
                        scores.append(entropy(pyx, py))  # compute divergence
                    split_scores.append(np.exp(np.mean(scores)))
                self.mean.append(np.mean(split_scores))
                self.std.append(np.std(split_scores))

        #self.drawer(self.mean, self.std)
        self.drawer(self.mean)

    #def drawer(self, y1, y2):
    def drawer(self, y1):
        x = np.arange(64)
        plt.plot(x, y1)
        #plt.plot(x, y2)

        #plt.title('IS & STD')
        plt.title('Inception Score')
        plt.xlabel('Batch')
        plt.ylabel('Score')
        plt.show()



if __name__ == '__main__':
    computeIS = ComputeIS("./samples/sagan_10", 64, 8)
    computeIS.compute()
