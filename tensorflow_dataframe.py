"""
    当导入进来的是pandas的dataframe数组
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


# 加载数据集
def input_data():
    URL = "E:/Desktop/机器学习_新/数据集/heart.csv"
    cancer = pd.read_csv(URL)
    pd.set_option("display.max_columns", 100)
    print(type(cancer))
    print(cancer.head(5))
    return cancer


# 划分数据集
def split_dataset(cancer):
    train, test = train_test_split(cancer, test_size=0.25)
    train, val = train_test_split(train, test_size=0.25)
    print("训练集：", train.shape)
    print("验证集：", val.shape)
    print("测试集：", test.shape)
    return train, val, test


# 将dataframe转成datasets
def df_to_dataset(dataframe, shuffle=True, batch_size=32, num_epochs=10):
    # 取出特征和目标
    target = dataframe.pop("target")
    feature = dataframe
    dataset = tf.data.Dataset.from_tensor_slices((dict(feature), target))
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32).repeat(num_epochs)
    return dataset


# 了解数据集
def learn_dataset(dataset):
    for feature, target in dataset.take(1):
        print("打印特征值的名字：", list(feature.keys()))
        print("年龄：", feature["age"])
        print("目标值：", target)


# 几种特征处理方式
def feature_handle(dataset):
    # 取出dataset中的一个样本，用来做示例
    example_batch = next(iter(dataset))[0]
    # print(example_batch)

    # 特征列：数字列
    age = tf.feature_column.numeric_column("age")  # 产生数字特征列
    age_layer = tf.keras.layers.DenseFeatures(age)  # 将特征列转成一个层次
    print(age_layer(example_batch).numpy())

    # 分桶列
    # 就是将数字列分成几段：比如：小于18,18-25,,25-30,30-35,35-40,40-45,45-50,50-55,55-60,60-65,大于65
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])  # 产生分桶特征列
    age_buckets_layer = tf.keras.layers.DenseFeatures(age_buckets)  # 将特征列转成一个层次
    print(age_buckets_layer(example_batch).numpy())

    # 分类列
    thal = tf.feature_column.categorical_column_with_vocabulary_list("thal", ["fixed", "normal", "reversible"])  # 产生分类特征列
    thal_one_hot = tf.feature_column.indicator_column(thal)
    thal_layer = tf.keras.layers.DenseFeatures(thal_one_hot)
    print(thal_layer(example_batch).numpy())

    # 嵌入列
    # 当分类列的取值非常多的时候，如果使用one-hot编码，则不可行，这时候可以使用嵌入列
    thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)  # 产生嵌入列，dimension限制one_hot编码之后的列数
    thal_embedding_layer = tf.keras.layers.DenseFeatures(thal_embedding)
    print(thal_embedding_layer(example_batch).numpy())

    # 经过哈希处理过的特征列(了解）
    # 也是用来处理分类列的分类值特别多的情况
    thal_hash = tf.feature_column.categorical_column_with_hash_bucket("thal", hash_bucket_size=10)
    thal_hash_one_hot = tf.feature_column.indicator_column(thal_hash)
    thal_hash_layer = tf.keras.layers.DenseFeatures(thal_hash_one_hot)
    print(thal_hash_layer(example_batch).numpy())

    # 组合列（了解）
    crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=10)
    crossed_feature_one_hot = tf.feature_column.indicator_column(crossed_feature)
    crossed_feature_layer = tf.keras.layers.DenseFeatures(crossed_feature_one_hot)
    print(crossed_feature_layer(example_batch).numpy())


# 构建特征列以及特征层
def feature():
    feature_columns = []
    # 数字列
    for header in ['trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
        feature_columns.append(tf.feature_column.numeric_column(header))

    # 分桶列
    age = tf.feature_column.numeric_column("age")
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    feature_columns.append(age_buckets)

    # 分类列
    thal = tf.feature_column.categorical_column_with_vocabulary_list("thal", ['fixed', 'normal', 'reversible'])
    thal_one_hot = tf.feature_column.indicator_column(thal)
    feature_columns.append(thal_one_hot)

    # 创建层次
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    return feature_layer


# 建立模型
def model(feature_layer):
    model = tf.keras.Sequential([
        feature_layer,
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    # 编译模型
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# 训练模型
def model_fit(model, train_dataset, val_dataset):
    model.fit(train_dataset, validation_data=val_dataset, epochs=100)


# 测试模型
def model_evaluate(model, test_dataset):
    loss, accuracy = model.evaluate(test_dataset)
    print("损失：", loss)
    print("准确率：", accuracy)


if __name__ == "__main__":
    cancer = input_data()
    train, val, test = split_dataset(cancer)
    train_dataset = df_to_dataset(train)
    val_dataset = df_to_dataset(val, shuffle=False)
    test_dataset = df_to_dataset(test, shuffle=False)
    # learn_dataset(train_dataset)
    # feature_handle(train_dataset)
    feature_layer = feature()
    model = model(feature_layer)
    model_fit(model, train_dataset, val_dataset)
    model_evaluate(model, test_dataset)


