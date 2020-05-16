""""
    图片增强：通过一些变换，扩充数据集，达到增强模型性能的目的
"""


import tensorflow as tf
import PIL.Image
import matplotlib.pyplot as plt


# 显示一张图片
image_path = "C:/Users/Goodboy/.keras/datasets/flower_photos/daisy/3640845041_80a92c4205_n.jpg"
image = PIL.Image.open(image_path)
plt.imshow(image)
plt.show()


# 加载一张图片
image_string = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image_string)

def visualize(original, augmented):
    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Origin image")
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title("Augmented image")
    plt.imshow(augmented)

    plt.show()


# 图片增强
# 翻转图片
flipped = tf.image.flip_left_right(image)
visualize(image, flipped)

# 灰度图片
gray_image = tf.image.rgb_to_grayscale(image)
visualize(image, tf.squeeze(gray_image))
plt.colorbar()

# 饱和图片
saturation = tf.image.adjust_saturation(image, 3)
visualize(image, saturation)

# 更改图片亮度
bright_image = tf.image.adjust_brightness(image, 0.4)
visualize(image, bright_image)

# 旋转图片
rot_image = tf.image.rot90(image)
visualize(image, rot_image)

# 中心裁剪图像
crop_image = tf.image.central_crop(image, central_fraction=0.5)
visualize(image, crop_image)
