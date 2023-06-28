from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Flatten, multiply, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
"""
CGAN  原始论文链接 ：  https://arxiv.org/pdf/1411.1784.pdf
  通过为数据增加label(y)进行构造，在G和D的输入上都增加了label.
  然后做了两个有条件的边缘的实验，也就是给定条件y,结合随机分布，生成符合条件y的样本

  1.使用minist数据及基于给定lable生产特定数字的模型试验
  2.对于multi-model model学习,生成不属于训练标签的描述性标记

前言
1.对于原始GAN，生成的数据是不可控的。
2.对于one-to-many mapping模型，比如image tag问题，一张图像可能不止一个tag，传统模型无法解决，
  因此可以使用条件生成概率，将输入图像视为conditional variable使用条件预测分布去获取一对多的映射关系

流程
1.D判别器的输入除了图像之外加入了y标签，y这个特征维度与x相同，是因为在模型内部进行了embedding处理
2.G生成器的输入除了噪声之外，也加入了y标签。维度关系同上。然后将生成的图像作为一半输入与y结合输入到D判别器

目标函数
区别在于CGAN引入了条件y,在已知y条件的情况下计算样本x的特征分布
D(x|y) D((G(z|y)))
"""

class CGAN():
    def __init__(self):

        # 写入输入维度
        self.img_rows = 28
        self.img_cols = 28
        self.img_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        self.num_classes = 10  # 类别数 0~9 共十个
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.discriminator.trainable = False
        noise = Input(shape=(100,))
        label = Input(shape=(1,))

        img = self.generator([noise, label])

        valid = self.discriminator([img, label])

        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.product(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()  # 记录参数情况
        # 输入的噪声的维度   总输入为 batch_size * 100
        noise = Input(shape=(self.latent_dim,))
        # 标签的维度
        label = Input(shape=(1,), dtype='int32') # 1维int型
        # 将输入X和Y的维度变成一致的--》使用embedding层
        # 输入维度，即输入类别数，也即种类数；输出维度就是我们想要达到的latent_dim,和噪声同样的维度；使用的变量为label？？？？？？？？？？？？？？？？？？？？？？？？？？;最后把它加入平坦层？？？？？？？？？
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))  # 将10个种类的（词向量种类）的label映射到latent_dim维度？？？？？？？？？？？？？？？？
            # 将100维转化为(None,100)， 这里None会随着batch而改变。
        # noise和label进行合并（对应位置相乘），作为模型输入
        model_input = multiply([noise, label_embedding])
        # 由模型输入产生的img
        img = model(model_input)
        # 返回Model模型，由输入映射到输出
        # 输入就只按照noise 和 label形式，至于它们的合并，则由内部完成【label_embedding到model_input】？？？？？？？？？？
        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(512, input_dim=np.product(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        # 输入的图片形状
        img = Input(shape=self.img_shape)
        # 标签的维度
        label = Input(shape=(1,), dtype='int32') # 1维int型
        # label与img的shape是不同的，因此这里像生成器一样，让他们维度相同
        # 把它加入平坦层；输入维度，即输入类别数，也即种类数；输出维度就是我们想要达到的图像的尺寸拉成一维的样子；指定的变量为label;
        label_embedding = Flatten()(Embedding(self.num_classes, np.product(self.img_shape))(label))
            # label_embedding的shape为 (None, 784)
        # 接下来吧img也转换成这个shape；借用Flatten层，第二个维度？？？？？？？？？？？？？？？？？？？？？
        flat_img = Flatten()(img)
            # 至此就获得了 img 和 label的待合并变量
        # img和label进行合并（对应位置相乘），作为模型输入
        model_input = multiply([flat_img, label_embedding])  # 完成了对应元素相乘 shape为(None, 784)
        # 获取模型的输出概率结果，即将model_input输入到模型中获取得到的概率结果
        validity = model(model_input)
        # 返回Model模型，由输入映射到输出
        # 输入就像生成器里那样，不是合并之后的输入，因为合并等操作是在判别器模型内部完成的，需要输入的是[img, label]，输出是valiidity
        return Model([img, label], validity)  # 注意： 合并和维度操作是由模型内部完成 【label_embedding到model_input】？？？？？？？？？？

    def train(self, epochs, batch_size, sample_interval):

        # 获取数据集
        (X_train, Y_train, ), (_, _) = mnist.load_data()

        # 将获取到的图像转化为-1 到 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
            # 将 60000*28*28维度的图像扩展为 60000*28*28*1
        # 将Y_train reshape成 60000*1
        Y_train = Y_train.reshape(-1, 1)  # -1自动计算第0维它的维度空间数?????????????????????

        # 写入 真实输出 与 虚假输出
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ----------------
            # 训练判别器
            # ----------------
            # 从0-6w中随机获取batch_size个索引数
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], Y_train[idx]
                # 完成了随机获取batch_size个图像以及对应的标签。
                # imgs shape 为(batch_size, 28, 28 ,1)
                # labels shape 为(32, 1)
            # 使用真实图片和真实标签送入判别器，给这情况标1
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            # 从标准正态分布里随机生成batch_size组维度为latent_dim的噪声输入进G
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # 符合正态分布， shape为（batch_size , 100)
            # 使用虚假图片和真实标签送入判别器，给这情况标0
            gen_imgs = self.generator.predict([noise, labels])
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            # 总loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #----------------
            # 训练生成器 固定鉴别器，训练生成器——在联合模型中
            # ----------------
            # 随机生成一些label
            sampled_label = np.random.randint(0, 10, batch_size).reshape(-1, 1) #0，10？？？？？？？？reshape？？？？？？？？
            # 联合模型进行训练，希望给用噪声和随机标签生成的假图，这种情况标1
            g_loss = self.combined.train_on_batch([noise, sampled_label], valid)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            # 绘制进度图

            if epoch % sample_interval == 0 :
                self.sample_images(epoch)
            # 完成图像保存

    def sample_images(self, epoch):
        r, c = 2, 5  # 输出 2行5列的10张指定图像

        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                # 给每个小图添加标题
                axs[i, j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("BV1Mi4y1P7ta_images/%d.png" % epoch)  # images 文件路径和代码文件同目录的
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=20001, batch_size=128, sample_interval=200)












