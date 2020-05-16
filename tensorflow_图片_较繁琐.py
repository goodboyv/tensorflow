"""
    这种方式相对于输入管道来说，就太复杂了
"""


import tensorflow as tf
import random
import pathlib
from tensorflow import keras


# 加载图片：获取所有图片的路径以及标签
def input_data():
    # 加载数据集
    # data_path = tf.keras.utils.get_file(origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz", fname="flower_photos", untar=True)  # 下载文件，返回下载到本地的文件路径
    data_path = "C:/Users/Goodboy/.keras/datasets/flower_photos/"
    # 使用pathlib将一个字符串路径，转成一个对象路径
    data_path = pathlib.Path(data_path)
    # 导入图片路径
    all_images_path = list(data_path.glob("*/*"))  # Windows path
    # print(all_images_path[:10])
    # 将对象路径转成字符串路径
    all_images_path = [str(path) for path in all_images_path]  # str path
    # 查看总共有多少张图片
    image_count = len(all_images_path)
    print("图片数量：", image_count)

    # 为每张图片确定标签
    label_names = sorted(item.name for item in data_path.glob("*/") if item.is_dir())
    # print("label_names", label_names)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    # print(label_to_index)
    all_images_label = [label_to_index[pathlib.Path(path).parent.name] for path in all_images_path]

    return all_images_path, all_images_label


#  处理图片
def preprocess(image):
    image = tf.image.decode_jpeg(image, channels=3)  # 解码
    image = tf.image.resize(image, [150, 150])  # 统一图片大小
    image = image/255.0  # 归一化
    return image


# 读入图片
def load(path):
    image = tf.io.read_file(path)
    return preprocess(image)


def load_preprocess(path, label):
    return load(path), label


# 加载图片：将图片读入进来，转成dataset
def read_input(all_iamges_path, all_iamges_label):
    dataset = tf.data.Dataset.from_tensor_slices((all_images_path, all_images_label))
    image_label_ds = dataset.map(load_preprocess)
    image_label_ds = image_label_ds.shuffle(4000).batch(32)
    return image_label_ds


# 建立模型
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
def train(model, image_label_ds):
    history = model.fit(image_label_ds, epochs=20)


if __name__ == "__main__":
    all_images_path, all_images_label = input_data()
    image_label_ds = read_input(all_images_path, all_images_label)
    model = model()
    train(model, image_label_ds)
