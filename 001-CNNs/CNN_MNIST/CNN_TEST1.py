
import numpy as np
import matplotlib.pylab as plt
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# draw bar
def draw_bar(y_numbers):
    # digits是从0 ~ y_number数组长度-1 的数组
    digits = range(len(y_numbers))
    # 转成字符型
    digits = [str(i) for i in digits]
    # x轴的值由 digits 串表示 ， y值就是y_number数组
    plt.bar(digits, y_numbers)
    # x轴取名digits ， 颜色blue
    plt.xlabel('digits', color='b')
    # y轴取名numbers of digits ， 颜色blue
    plt.ylabel('numbers of digits', color='b')
    # 遍历x，y ：x右偏0.05，y上移0.05，数据显示格式，横向居中，纵向贴底
    for x, y in zip(range(len(y_numbers)), y_numbers):
        plt.text(x + 0.05, y + 0.05, '%.2d' % y, ha='center', va='bottom')
    # show
    plt.show()

# print image
def image(images, row, col):
    # 所有图像用im存起来
    im = images[:col * row]
    # 横向堆叠
    hs = np.hstack(im)
    # 竖向切割 分成 row 份
    sp = np.split(hs, row, axis=1)
    # 再将这 row 份 竖向堆叠
    show_image = np.vstack(sp)
    # 结果以灰度图展示
    plt.imshow(show_image, cmap='gray')
    # 不显示坐标轴
    plt.axis("off")
    # show 最终将这100张灰度图以10*10的形式拼在一张图上
    plt.show()



# Normalization 归一化 ###
def normalize(x):
    x = x.astype('float32') / 255.0
    return x








# Prepare mnist dada 加载mnist原始数据
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


""" 打印 训练集 和 测试集 的标签情况 """
print('--------训练集 和 测试集 的标签情况--------')
# Y_train 和 Y_test 去重 再 顺排
temp1 = np.unique(Y_train)
temp2 = np.unique(Y_test)
# 定义空数组 备存数据
Y_train_numbers = []
Y_test_numbers = []
# 统计 Y_train 和 Y_test 里各个标签的数量
for i in temp1:
    Y_train_numbers.append(np.sum(Y_train == i))
for i in temp2:
    Y_test_numbers.append(np.sum(Y_test == i))
# figure1 name
plt.title('The number of each digit in the train data')
# draw it
draw_bar(Y_train_numbers)
# figure2 name
plt.title('The number of each digit in the test data')
# draw it
draw_bar(Y_test_numbers)


"""输出各数据集的形状"""
print('--------各数据集的形状--------')
print('x_train:', X_train.shape, '\ny_train:', Y_train.shape, '\nx_test:', X_test.shape, '\ny_test:', Y_test)



"""  显示出训练集数字图像  """
print('--------训练集数字图像--------')
# 在一张图中 显示前 100 个训练集的图像 具体细节见函数说明
image(X_train, 10, 10)
print('--------训练集中第一个数字图像--------')
plt.imshow(X_train[0], cmap='binary')
plt.axis('off')
plt.show()



print('--------正式开始CNN--------')


"""全连接层只能处理一个维度的向量,因此有必要先处理一下数据"""
# Setup data shape  设置数据形状
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
# Normalization
X_train = normalize(X_train)
X_test = normalize(X_test)

"""轮到创建模型了"""
# Setup model
model = Sequential()

# 1st Con2D layer 第一个卷积层
"""
这里用8个卷积核
每个都是5*5的
padding 填充 可以粗略的理解为same--padding=2    valid--padding=0  使用same，原图filter之后得到的feature map尺寸不变
可参考 https://ezyang.github.io/convolution-visualizer/index.html
第一次输入需要确定输入值得形状（之后的都会自动计算出来 ）
激活函数 relu （已存到Typora----Relu激活函数）
"""
model.add(Conv2D(
    filters=8,
    kernel_size=(5, 5),
    padding='same',
    input_shape=(28, 28, 1),
    activation='relu'
))

# 1st Maxpooling layer 第一个最大池化层
"""
使用2*2的
最大池化
"""
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Con2D layer 第二个卷积层
"""
这次使用16个kernel
每个都是5*5的
padding = 2
激活函数 relu
"""
model.add(Conv2D(
    filters=16,
    kernel_size=[5, 5],
    padding='same',
    activation='relu'
))

# 2nd Maxpooling layer 第二个最大池化层
"""
使用2*2的
最大池化
"""
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten 扁平层 把第二个池化层的结果拉平
model.add(Flatten())

# 1st Fully Connected Dense 第一个全连接层
"""
100个神经元
relu激活
"""
model.add(Dense(100, activation='relu'))
# Dropout，用来放弃一些权值，防止过拟合
model.add(Dropout(0.25))

# 2nd Fully Connected Dense 第二个全连接层
"""
10个神经元
softmax （已存到Typora----Softmax简述）
"""
model.add(Dense(10, activation='softmax'))

# Compile
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

# Run network
history = model.fit(x=X_train,  # 原始数据  iamge 1
          y=Y_train,  # 对应原始数据标签 label 1
          validation_split=0.2,
          epochs=10,  #为了方便测试，这里只用4轮  演示时用10轮
          batch_size=200,
          verbose=2,
          #validation_data=(X_test, Y_test),
          )

"""
开始着手CNN Visualization
   plot单独显示设置
   File->Settings->Tools->Python Scientific
   取消掉Show plots in toolwindow 选项
"""
# Accuracy
# figure1_1 add x,y value
fig1 = plt.plot(history.history['accuracy'], color='r', linewidth=1.0, linestyle='-', label='Training accuracy')
# figure1_2 add x,y value
fig2 = plt.plot(history.history['val_accuracy'], color='b', linewidth=1.0, linestyle='-', label='Validation accuracy')
# set x,y label
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# set y limitation
plt.ylim(0.75, 1)
# add legend
"""
plt.legend(handles=[fig1, fig2], labels=['Trainning accuracy'', 'Validation accuracy'], loc='best') 不用了这种了 labels放到plot里
"""
plt.legend(loc='best')
# set figure title
plt.title('Accuracy of Different Data Set at Different Epoch')
# show
plt.show()

"""
原始参考代码
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.75, 1])
plt.legend(loc='lower right')
plt.show()
看来是 默认直线，自动分配颜色
"""

# Loss
# figure2_1 add x,y value
fig1 = plt.plot(history.history['loss'], color='r', linewidth=1.0, linestyle='-', label='Training loss')
# figure1_2 add x,y value
fig2 = plt.plot(history.history['val_loss'], color='b', linewidth=1.0, linestyle='-', label='Validation loss')
# set x,y label
plt.xlabel('Epochs')
plt.ylabel('Loss')
# set y limitation
plt.ylim(0, 0.6)
# add legend
plt.legend(loc='best')
# set figure title
plt.title('Loss of Different Data Set at Different Epoch')
# show
plt.show()
"""
原始参考代码
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0, 0.6])
plt.legend(loc='upper right')
plt.show()
"""

