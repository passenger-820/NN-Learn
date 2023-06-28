import glob
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ISImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        # self.files.sort(key=lambda x: (int(x.split('_')[0]),  # 依据step(400~20000: +400)升序
        #                           int(x.split('_fake')[1].split('.')[0])  # 再依据i(1~64: +1) 升序
        #                           ))

        self.files = sorted(glob.glob(os.path.join(root) + "/*.png"),key=lambda x: (int(x.split('\\')[1].split('_')[0]),  # 依据step(400~20000: +400)升序
                                   int(x.split('\\')[1].split('_fake')[1].split('.')[0])  # 再依据i(1~64: +1) 升序
                                  ))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        item_image = self.transform(img)
        return item_image

    def __len__(self):
        return len(self.files)


