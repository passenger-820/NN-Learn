import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def image(images, row, col):
    im = images[:col * row]
    hs = np.hstack(im)
    sp = np.split(hs, row, axis=1)
    show_image = np.vstack(sp)
    plt.imshow(show_image, cmap='gray')
    plt.axis("off")
    plt.show()


def normalize(x):
    x = x.astype(np.float32) / 255.0
    return x


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


def draw_bar(y_numbers):
    digits = range(len(y_numbers))
    digits = [str(i) for i in digits]
    plt.bar(digits, y_numbers)
    plt.xlabel('digits', color='b')
    plt.ylabel('numbers of digits', color='b')
    for x, y in zip(range(len(y_numbers)), y_numbers):
        plt.text(x + 0.05, y + 0.05, '%.2d' % y, ha='center', va='bottom')
    plt.show()


"""  画柱形图  """
temp1 = np.unique(y_train)
temp2 = np.unique(y_test)
y_train_numbers = []
y_test_numbers = []
for v in temp1:
    y_train_numbers.append(np.sum(y_train == v))
for v in temp2:
    y_test_numbers.append(np.sum(y_test == v))
plt.title("The number of each digit in the train data")
draw_bar(y_train_numbers)
plt.title("The number of each digit in the test data")
draw_bar(y_test_numbers)

# x_train = x_train[:30000, ]
# y_train = y_train[:30000]
print('x_train:', x_train.shape, '\ny_train:', y_train.shape, '\nx_test:', x_test.shape, '\ny_test:', y_test)

"""  显示出训练集数字图像  """
image(x_train, 10, 10)
plt.imshow(x_train[0], cmap='binary')
plt.axis('off')
plt.show()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
x_train = normalize(x_train)
x_test = normalize(x_test)

# 创建模型，输入28*28*1个神经元，输出10个神经元
model = Sequential()
# 第一个卷积层
# input_shape输入平面
# kernel——size 卷积窗口大小
# padding padding方法 same or valid
# activation 激活函数
model.add(Conv2D(filters=8, kernel_size=(5, 5), padding='same',
                 input_shape=(28, 28, 1), activation='relu'))
# 第一个池化层
model.add(MaxPooling2D(pool_size=(2, 2)))  # 出来的特征图为14*14大小
# 第二个卷积层 16个滤波器（卷积核），卷积窗口大小为5*5
# 经过第二个卷积层后，有16个feature map，每个feature map为14*14
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # 出来的特征图为7*7大小

# 平坦层，把第二个池化层的输出扁平化为1维 16*7*7
model.add(Flatten())
# 第一个全连接层
model.add(Dense(100, activation='relu'))
# Dropout，用来放弃一些权值，防止过拟
model.add(Dropout(0.25))
# 第二个全连接层，由于是输出层，所以使用softmax做激活函数
model.add(Dense(10, activation='softmax'))
# 打印模型
print(model.summary())

# 训练配置
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# 开始训练
history = model.fit(x=x_train, y=y_train, validation_split=0.2,
                    epochs=10, batch_size=200, verbose=2)

plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.75, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0, 0.6])
plt.legend(loc='upper right')
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("test loss - ", round(test_loss, 3), " - test accuracy - ", round(test_acc, 3))
