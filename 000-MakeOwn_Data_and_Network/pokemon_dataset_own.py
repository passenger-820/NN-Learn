import torch
import os,glob
import random,csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Pokemon(Dataset):

    """
    root: 文件所在目录
    resize: 指定输出尺寸
    mode: train，validation，test等等
    """
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        self.root = root
        self.resize = resize

        """-------------给每一个类宝可梦做映射：皮卡丘、妙蛙种子.....--------------"""
        self.name2label = {}  # dic 类型
        # listdir每次返回的顺序可能不同，因此排一下序，甭管是啥顺序，只要排了就行
        for name in sorted(os.listdir(os.path.join(root))):  # 方式为遍历root目录，拿到每个子文件夹的name
            # 不进行过滤的话，会包含目录名和文件名，因此要过滤掉文件
            if not os.path.isdir(os.path.join(root, name)):
                continue

            # 对于key，value这样的键值对
            # 每个key：name对应的value为当前key类别的总数，因此最终按照某一顺序对应0，1，2，3，4
            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label) # {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}


        """
        希望拿到一个对象，形为(image,label) 
        此处image用的是image_path
            读取csv文件，有则读，无则建
        """
        self.images, self.labels = self.load_csv('images.csv')

        """
        split dataset
            train，validation，test = 6:2:2
        """
        """
        咱们在004.md中学过：
            print( 'train:', len(train_db), 'test:', len(test_db))
            # 把数据集划分一下 50k+10k
            train_db,val_db = torch.utils.data.random_split(train_db,[50000，10000])
            print('db1:', len(train_db), 'db2:', len(val_db))
            
            train_loader = torch.utils.data.DataLoader(
                train_db,
                batch_size=batch_size, shuffle=True)
            
            val_loader = torch.utils.data.DataLoader(
                val_db,
                batch_size=batch_size, shuffle=True)
            # 咱们之前不是有个属性叫做train=True嘛，那就是默认全部都作为同一部分的数据集
            
            # 这时候就具备了train_loader，val_ loader和test_loader
        但是这里不能这样，因为image,label分别在两个变量里，各自random就乱了，所以得按照下面的方式，用同样的法子
        """
        if mode=='train': # 60%
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode=='val': # 20% = 60%->80%
            self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        else: # 20% = 80%->100%
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]

    def load_csv(self, filename):
        """-------------------"""
        """---------如果images.csv不存在，创建--------"""
        if not os.path.exists(os.path.join(self.root, filename)):
            images =[]
            for name in self.name2label.keys():
                # 把当前文件夹下所有文件都保存下来
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            # 这样其实并没有存下来label，至于label判定，后面可由path推断出来  pokemon\\bulbasaur\\00000000.png
            # print(len(images), images) # 1167 '.\\data\\pokemon\\bulbasaur\\00000000.png'

            """
            ------把对应关系保存到csv文件里-----------
            """
            # 先shuffle一下
            random.shuffle(images)
            # 然后保存
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: #'.\\data\\pokemon\\bulbasaur\\00000000.png'
                    # 用 \\ 分隔，取倒数第二个
                    # 而Linux和Windows分隔斜杠不同，这里统一用os.sep就不会有差别了
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]

                    writer.writerow([img, label])
                print('write into csv file:', filename)

        """---------如果文件存在，直接读取images.csv----------"""
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader: # 'pokemon\\bulbasaur\\00000000.png', 0
                img, label = row
                # label 转成int型
                label = int(label)

                images.append(img)
                labels.append(label)

        """---------保证这两个玩意儿长度一致----------"""
        assert len(images) == len(labels)

        return images, labels





    def __len__(self):
        return len(self.images)




    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # normalize ： x_hat = (x-mean)/std
        # denormalize ： x = x_hat*std = mean

        # 注意保证shape一致再计算
        # x.shape: [c, h, w]
        # 而每个x只有一个mean.shape: [3] => [3, 1, 1]
        # std同理
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0

        img, label = self.images[idx], self.labels[idx]

        """-----因为img存的是path，咱们这希望直接能返回图片-----"""
        transf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path= > image data
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.RandomRotation(15), # 旋转太多会增加计算量
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = transf(img)
        label = torch.tensor(label)

        return img, label




def old_test():
    import visdom
    # 控制台启动   python -m visdom.server
    import time

    viz = visdom.Visdom()
    db = Pokemon('data\pokemon', 224, 'train')

    x,y = next(iter(db))
    print('sample:', x.shape, y.shape, y) # sample: torch.Size([3, 224, 224]) torch.Size([]) tensor(1)

    # 因为原本数据normalize了，但可视化时不需要，因此就de回去
    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=16,shuffle=True,num_workers=2)  # 原本是按照idx取的，因此shuffle一下 可以不用多线程
    for x,y in loader:
        viz.images(db.denormalize(x), nrow=4, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(10)

def new_test():
    import visdom
    import time
    import torchvision

    tf = transforms.Compose([
                    transforms.Resize((64,64)),
                    transforms.ToTensor(),
    ])
    """------------------------torchvision.datasets.ImageFolder(root,transform)---------------------"""
    # 如果数据集严格按照上述二级目录的方式存储，那么这样就不用去地写init，len，getitem那些了
    db = torchvision.datasets.ImageFolder(root='data\pokemon', transform=tf)
    loader = DataLoader(db, batch_size=16, shuffle=True)

    # 查看编码
    print(db.class_to_idx) # {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}

    viz = visdom.Visdom()
    for x,y in loader:
        viz.images(x, nrow=4, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(10)


if __name__ == '__main__':

    new_test()
    # old_test()
