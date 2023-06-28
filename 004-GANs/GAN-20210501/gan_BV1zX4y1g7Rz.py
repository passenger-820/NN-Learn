
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Input, Flatten
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

"""
GAN  原始论文链接 ：  https://arxiv.org/pdf/1406.2661.pdf
 
研究现状
1.DL领域研究最成功的还是判别器模型，由于“最大似然估计”等许多复杂的“概率计算”问题，导致生成器模型发展缓慢。
2.Goodfellow等人提出了一个新的生成模型估计程序，避开了这些难题，给生成器模型带来了蓬勃发展。
    D：一般熟知的带标签分类、回归等监督学习都属于判别器模型。学习的是某种概率分布下的条件概率p(y|x)。
    G：聚类、自动编码器等无监督学习模型。学习的是联合分布概率p(x,y)。通过模型学到最优的p(x|y)。

对抗过程
    G生成器：假币制造者；D判别器：检察员；
    G1造假钱（0）；D1看着真钱（1），看着假钱（0），知道假钱就是假钱；至此，D1会鉴定假钱了
    G1不服气，知道了D1懂得技术后，精益求精继续制造假钱，制造出足以骗过D1的假钱（1）；至此，G1进化为G2
    D1也不服气，不断研究G2造的假钱（1？0？），仔细比对真钱（1），继续学习高深技术，终于又可以鉴定出G2制造的假钱（1——>0）；至此D1进化为D2
    ........如此往复，带到动态均衡（D再也识别不了G）
"""

"""
框架：keras + tensorflow
"""

class GAN():
    def __init__(self):   #这就是在定义GAN这个类的全局变量，以便其他方法调用 （CSDN）
#1bg
        # 输入维度
        self.latent_dim = 100
        # 图像尺寸
        self.img_rows = 28
        self.img_cols = 28
        self.chanel = 1
        # 图像形状
        self.img_shape = (self.img_rows, self.img_cols, self.chanel)
#1end

#5bg
        # 构建判别器
        self.discriminator = self.build_discriminator()
#5end
        optimizer = Adam(0.0002, 0.5)
        self.discriminator.compile(loss='binary_crossentropy', # loss为二进制交叉熵损失函数
                                   optimizer=optimizer, # 激活函数为Adam
                                   metrics=['accuracy']) # metrics选择['accuracy'] （y和y_label都是数值时，选此）（CSND）
#3bg
        # 构建生成器 （这里就不用写compile了，G的compile会融合进combined的model里面）
        self.generator = self.build_generator() # 这个G可以完成由noise到img的生成
#3end
#7bg
        # 先将D可训练设为否，即联合训练中只训练G，不训练D
        self.discriminator.trainable = False

        # 输入形状： 输入的维度为latent_dim 【input参数见“博客园”，shape这里为何有个“，”，我猜是为了确切表示这里只是1维】
        z = Input(shape=(self.latent_dim,))
        # 这里的img 是由噪声z预测产生出的图像
        img = self.generator(z)
        # 判别结果：由D依据输入的img产生的 概率
        validity = self.discriminator(img)
        # 联合模型相当于 z到validity 这么一个过程
        self.combined = Model(z, validity)
#7end

#9bg
        # 联合模型训练过程的编译  G的优化也是在这里面了！
        self.combined.compile(loss=['binary_crossentropy'], # loss为二进制交叉熵损失函数
                                   optimizer=optimizer) # 激活函数为Adam
#9end


     # 使用keras下的Sequential 编写生成器
    def build_generator(self):
#2bg
        model = Sequential()
        # 1st FC
        model.add(Dense(256, input_dim=self.latent_dim)) # 输入噪声是1维含有100个元素的张量
            # 常见的一种用法：只提供了input_dim=32，说明输入是一个32维的向量，相当于一个一阶、拥有32个元素的张量，它的shape就是（32，）。因此，input_shape=(32, )
        model.add(LeakyReLU(alpha=0.2)) # 在Relu的基础上，给 x < 0 也赋予了一个非零斜率（xi/ai ai是（1，+∞）内固定的参数），这个alpha就是那个1/ai （CSDN）
        model.add(BatchNormalization(momentum=0.8)) # (CSDN) BN层，目前只要知道这也是一层网络层，而且很牛，可以更快速的让机器去学习！！
        # 2nd FC
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 3rd FC
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # The last FC 作为输出
        model.add(Dense(np.product(self.img_shape), activation='tanh')) # 神经元个数（输出维度） = 图像尺寸乘积（图像尺寸拉直后的维度），这样才能reshape成 28*28*1的图像
        model.add(Reshape(self.img_shape)) # 把输出的形状[28*28*1]转换成图片的形状[28,28,1]
        # 记录各层参数情况 （Param计算过程见CSDN）
        model.summary()


        # 定义输入：噪声
        noise = Input(shape=(self.latent_dim,)) # Input 形状 为self.latent_dim（和1st FC 对应  此为100维）
        # 定义输出：生成的image
        img = model(noise)  # 就是模型将noise输入进去得到的输出
        # 输入输出的返回值   返回一个模型
        return Model(noise, img) # 这个模型：以noise为输入，img为输出
#2end

    # 使用keras下的Sequential 编写判别器
    def build_discriminator(self): # 需要输入图片，然后判别出真假。即img--->label的转化
#4bg
        model = Sequential()
        # 1st 平坦层 ： 先将输入的图片拉平
        model.add(Flatten(input_shape=self.img_shape)) # 将输入型状为self.img_shape的图片拉平
        # 1st FC  不使用BN层
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        #2nd FC 也不使用BN层
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        # The last FC 输出可能性，用 sigmoid激活
        model.add(Dense(1, activation='sigmoid')) # 输出结果落入（0，1）的概率区间
        # 记录各层参数情况
        model.summary()

        # 定义输入：img
        img = Input(shape=self.img_shape) # Input 形状为self.img_shape [28,28,1]
        # 定义输出：可能性
        validity = model(img) #  把img输入后，预测出输出概率
        # 输入输出的返回值   返回一个模型
        return Model(img, validity) # 这个模型：以img为输入，validity为输出   （由图像生成可能性）
#4end

#6bg
    # 定义训练函数 （回数，一批样本数，打印间隔）
    def train(self, epochs, batch_size, sample_interval):
        # 获取mnist数据集，加载的数据集 shape为 60000*28*28  每个pixel为（0，255）
        (x_train, _), (_, _) = mnist.load_data() # 只要x_train作为输入图像，不需要标签，也不要测试集
        # 将灰度图（0，255）转化为（-1，1）
        x_train = x_train / 127.5 - 1
        # 将60000*28*28的图像 拓展成3维图像 60000*28*28*1 【60000是第0维，28为第1维，28为第2维，1为第3维】??????????????????
        x_train = np.expand_dims(x_train, axis=3) #拓展对象为x_train，拓展成的维度为3维
        # 定义 真实 标签
        valid = np.ones((batch_size, 1)) # 2维的标签 一共batch_size行1列 （结果就是为这batch_size张图像进行标记，都标为1）
        # 定义 虚假 标签
        fake = np.zeros((batch_size, 1)) # 2维的标签 一共batch_size行1列 （结果就是为这batch_size张图像进行标记，都标为0）

        # 开始训练
        for epoch in range(epochs):
            # -------------------
            # 先训练D（需要将真实数据和虚假数据都输入）（给真实图像标1，给G生成的图像标0）
            # -------------------
            # 将真实数据和虚假数据，以及对应的标签输入D，来训练D，同时记录D的loss损失
            # 从样本中 随机抽取 batch_size 个图片的方式进行输入
            idx = np.random.randint(0, x_train.shape[0], batch_size)  # 索引 = np产生的bacth_size个随机整数 0~60000
            # 我们的输入为一批img 0~60000张图片中的bacth_size张
            imgs = x_train[idx]  # 从上可见idx是个1维向量，每张图片均为28*28*1，共batch_size张   真实样本
            # 由真实数据产生的loss（用batch组真实图像imgs输入，输出的是valid）
            d_loss_real =self.discriminator.train_on_batch(imgs, valid)

            # 输入到G的维度 numpy.random.normal(loc=0.0, scale=1.0, size=None)
                # loc:float 概率分布的均值，对应着整个分布的中心center    即 μ
                # scale:float 概率分布的标准差，对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高    即 sigma^2
                # size:int or tuple of ints 输出的shape，默认为None，只输出一个值
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))  # 从标准正态分布里随机生成batch_size组维度为latent_dim的噪声输入进G
            # 使用G，将noise生成gen_imgs
            gen_imgs = self.generator.predict(noise)  # 通过noise预测出的images（完成了噪声生成图像，也就是"虚假图像"，虚假图像标签是 0）
            # 由虚假图像产生的loss（用batch组虚假图像gen_imgs输入，输出的是fake）
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            # 整个D总的loss用2个loss的平均值来代替
            d_loss = 0.5 * np.add(d_loss_fake,d_loss_real)
            # -----以上完成了对D的训练-----

            # -------------------
            # 再训练G
            # -------------------
            # 使用新的噪声（从标准正态分布里随机生成batch_size组维度为latent_dim的噪声输入进combined网络）
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim)) # 这里的噪声维度为 batch_size * 100
            # G期待的输出是valid（能够骗过D，让D认为G生成的是真的valid）
            # 这时我们需要训练的是G+D，因为G没有判别能力
#6end
#8bg
            # 将联合模型进行batch训练，输入为noise，期待输出不再是虚假的fake，而是真实的valid
            g_loss = self.combined.train_on_batch(noise, valid)
#8end

#11bg
            # 绘制进度图
            print("%d [D loss: %f, acc: %.2f%% ] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            # 每sample_interval轮打印生成的图片
            if epoch % sample_interval == 0 :
                self.sample_images(epoch)
    # 保存图像
    def sample_images(self, epoch):
        # 行和列   即想保存25张图
        r, c = 5, 5
        # 从标准正态分布里随机生成25组维度为latent_dim的噪声
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        # 使用G，将noise生成gen_imgs array形状为[0~24,0~27,0~27,0]
        gen_imgs = self.generator.predict(noise) # 25*28*28*1
        # Rescale images 0 - 1(反归一化，放缩到0~1大小)
        gen_imgs = 0.5 * gen_imgs + 0.5
        # fig, ax = plt.subplots(1,3),其中参数1和3分别代表子图的行数和列数，一共有 1x3 个子图像。函数返回一个figure图像和子图ax的array列表
        # fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作。
        # fig, ax = plt.subplots()等价于：
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)  （CSDN）
        fig, ax = plt.subplots(r, c)
        # 用cnt来计数
        cnt = 0
        for i in range(r):
            for j in range(c):
                # 第cnt张，所有（行，列）的（值）组成的图，按灰度图显示 （CSDN  numpy数组的冒号）
                ax[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                # 关闭所有坐标轴线、刻度标记和标签
                ax[i, j].axis('off')
                cnt += 1
        # 调用保存图片的方法
        fig.savefig("BV1zX4y1g7Rz_images/%d.png" % epoch)
        # cla() Clear axis即清除当前图形中的当前活动轴。其他轴不受影响。
        # clf() Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
        # close() Close a figure window  据说画完一个figure后close，这是个好习惯
        plt.close()

#10bg
if __name__ == '__main__':
    gan = GAN()
    # 调用train函数，总共30000轮，一批128个，每隔200轮打印一次
    gan.train(epochs=30000, batch_size=128, sample_interval=200)
#10end
