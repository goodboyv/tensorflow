"""
    通过输入管道将图片文件加载进来
"""


import tensorflow as tf
from tensorflow import keras
import os

# 解决“Failed to get convolution algorithm. This is probably because cuDNN failed to initialize”错误
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# 加载图片文件
def input_picture():
    # 在这个文件夹下面包含了所有图片文件
    data_path = "C:/Users/Goodboy/.keras/datasets/flower_photos/"
    # 每个类别的图片文件的路径
    daisy_path = os.path.join(data_path, "daisy")
    dandelion_path = os.path.join(data_path, "dandelion")
    roses_path = os.path.join(data_path, "roses")
    sunflowers_path = os.path.join(data_path, "sunflowers")
    tulips_path = os.path.join(data_path, "tulips")
    # print("各个图片分类文件夹的路径：", daisy_path, dandelion_path, roses_path, sunflowers_path, tulips_path)
    # 查看每个分类文件夹下有多少张图片
    num_daisy = len(os.listdir(daisy_path))
    num_dandelion = len(os.listdir(dandelion_path))
    num_roses = len(os.listdir(roses_path))
    num_sunflowers = len(os.listdir(sunflowers_path))
    num_tulips = len(os.listdir(tulips_path))
    # print("每个分类文件夹下的图片数量：", num_daisy, num_dandelion, num_roses, num_sunflowers, num_tulips)

    # 构建输入管道
    # 管道参数
    batch_size = 32
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    # 实例化管道
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
    # 导入图片
    data_generator = image_generator.flow_from_directory(batch_size=batch_size, directory=data_path, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode="binary")
    # print(type(data_generator))
    # 查看data_generator
    images, labels = next(data_generator)
    # print(images.shape)
    # print(images[0])
    # print(labels)
    return data_generator


# 建立模型：卷积神经网络
def model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=(150, 150, 3)),  # 150, 150, 3-->150, 150, 16
        tf.keras.layers.MaxPool2D(2, 2, padding="valid"),  # 150, 150, 16 --> 75, 75, 16
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),  # 75, 75, 16 --> 75, 75, 32
        tf.keras.layers.MaxPool2D(2, 2),  #  75, 75, 32 --> 37, 37, 32
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),  # 37, 37, 32 --> 37, 37, 64
        tf.keras.layers.MaxPool2D(2, 2),  # 37, 37, 64 -->18, 18, 64
        tf.keras.layers.Flatten(),  # 20736
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax")
    ])
    print("模型结构：", model.summary())
    # 编译模型
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# 训练模型
def train(model, data_generator):
    history = model.fit_generator(data_generator, steps_per_epoch=3670//32, epochs=20)


if __name__ == "__main__":
    data_generator = input_picture()
    model = model()
    train(model, data_generator)

