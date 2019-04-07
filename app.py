# coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import backward
import  forward
import app_num_str


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y = forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


# 图片预处理
# 反相 + 去噪声 (训练集中的样本为黑底白字)
# 单行列切片 - 将一个字符串/数字串切成多个单字符/数字
# 白色为255 黑色为0
def capture_num_str(picName):
    img = Image.open(picName)
    im_arr = np.array(img.convert('L'))
    cur_capture = im_arr[:, left_cut: right_cut]
    cur_capture = Image.fromarray(cur_capture)  # 转为图片
    cur_detect = cur_detect.resize((28, 28), Image.ANTIALIAS)  # 图片resize
    cur_detect = np.array(cur_detect.convert('L'))
    cur_detect = cur_detect.reshape([1, 784])  # 图片reshape
    cur_detect = cur_detect.astype(np.float32)
    cur_detect = np.multiply(cur_detect, 1.0 / 255.0)
    img_ready.append(cur_detect)  # 加入结果list
    return img_ready


def application():
    print("Running app.py...")
    picNumber = (int)(input("input the number of pictures:"))
    for i in range(picNumber):
        img_name = input("input the path of pictures:")
        cut_images = pre_pic(img_name)
        print("The predict number is:")
        for img in cut_images:
            detect_number = restore_model(img)
            print(detect_number, end=' ')
        print("\n")


def main():
    application()


if __name__ == '__main__':
    main()
