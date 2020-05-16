"""
    将导入进来的numpy数组转成datasets
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


# 加载数据集并划分数据集
def input_data():
    # 加载数据集
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print("数据类型：", type(train_images))
    # 划分数据集
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25)
    print("训练集：", train_images.shape, train_labels.shape)
    print("验证集：", val_images.shape, val_labels.shape)
    print("测试集：", test_images.shape, test_labels.shape)
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


# 了解数据集
def learning_datasets(images, labels, names, k):
    # 显示一张图片
    if k == 0:
        plt.figure()
        plt.imshow(images[0])
        plt.colorbar()
        plt.grid(False)
        plt.show()
    else:
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.imshow(images[i], cmap=plt.cm.binary)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.xlabel(names[labels[i]])
        plt.show()


# 数据预处理：归一化
def preprocessing(train_images, val_images, test_images):
    train_images = train_images/255.0
    val_images = val_images/255.0
    test_images = test_images/255.0
    return train_images, val_images, test_images


# 将numpy数组转成datasets
def ndarray_datasets(train_images, train_labels, val_images, val_labels, test_images, test_labels):
    # 将numpy数组转成dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    # print(train_dataset)
    # print(val_dataset)
    # 打乱以及批次化
    train_dataset = train_dataset.shuffle(50000).batch(32)
    val_dataset = val_dataset.batch(32)
    test_dataset = test_dataset.batch(32)
    # print(train_dataset)
    # print(val_dataset)
    return train_dataset, val_dataset, test_dataset


# 建立模型
def model():
    model = tf.keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    print(model.summary())
    return model


# 编译、训练模型
def train(model, train_dataset):
    # 编译模型
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # 训练模型
    model.fit(train_dataset, epochs=10)
    return model


# 评估模型
def evaluate(model, val_dataset):
    val_loss, val_acc = model.evaluate(val_dataset, verbose=2)
    print("Loss:", val_loss)
    print("Accuracy:", val_acc)


# 模型预测
def predict(model, test_image, test_labels, class_names):
    prediction = model.predict(test_images)
    print("第一张图片的预测值：", prediction[0])
    print("第一张图片的预测值对应的下标：", np.argmax(prediction[0]))
    print("第一张图片的预测值对应的标签：", class_names[np.argmax(prediction[0])])
    print("第一张图片的真实标签：", class_names[test_labels[0]])


if __name__ == "__main__":
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = input_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    learning_datasets(train_images, train_labels, class_names, 1)
    train_images, val_images, test_images = preprocessing(train_images, val_images, test_images)
    train_dataset, val_dataset, test_dataset = ndarray_datasets(train_images, train_labels, val_images, val_labels, test_images, test_labels)
    model = model()
    model = train(model, train_dataset)
    evaluate(model, val_dataset)
    predict(model, test_images, test_labels, class_names)