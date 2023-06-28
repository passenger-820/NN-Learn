
import torch
import torchvision.datasets as dsets
from torchvision import transforms


class Data_Loader():
    # 此处dataset赋值仅让程序员看明白，具体值在下方load处确定数据集
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(64)) # 70
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform


    def load_animefaces(self):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.ImageFolder('C:/DataSets/DeepLearning/animefaces/', transform=transforms) # E:/PyCharmProject/NeuronNetwork/data/CelebA/
        return dataset

    def loader(self):
        dataset = self.load_animefaces()
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=4, # For fast training
                                             pin_memory=True, # For fast training
                                              drop_last=True)
        return loader

