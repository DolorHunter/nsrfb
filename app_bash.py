# coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import backward
import forward


def restore_model(test_img_arr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y = forward.forward(x, None)
        pre_value = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                pre_value = sess.run(pre_value, feed_dict={x: test_img_arr})
                return pre_value
            else:
                print("No checkpoint file found")
                return -1


# 捕捉日期切割点位置 (比例)
def cut_date(date):
    date.append(250 / 3255)
    date.append(670 / 3255)
    date.append(4190 / 6200)
    date.append(6100 / 6200)


# 捕捉金额切割点位置 (比例)
def cut_amount(amount):
    amount.append(1455 / 3255)
    amount.append(1780 / 3255)
    amount.append(4645 / 6200)
    amount.append(6030 / 6200)


# 捕捉日期或金额
def capture(img_name, cap_obj):
    img = Image.open(img_name)
    im_arr = np.array(img.convert('L'))
    cut_pos = []  # 存储切割点位置
    # 判断切割对象
    if cap_obj == 'date':
        cut_date(cut_pos)  # 切割日期
    elif cap_obj == 'amount':
        cut_amount(cut_pos)  # 切割金额
    else:
        print('ERROR: Capture Error!!')
        return -1
    up_cut = int(cut_pos[0] * im_arr.shape[0])
    down_cut = int(cut_pos[1] * im_arr.shape[0])
    left_cut = int(cut_pos[2] * im_arr.shape[1])
    right_cut = int(cut_pos[3] * im_arr.shape[1])
    # 捕捉后存入im_arr
    im_arr = im_arr[up_cut: down_cut, left_cut: right_cut]
    threshold = 50  # 噪点控制
    im_arr_num = []  # 偶数位储存字符开始位置, 奇数位储存字符结束位置
    for i in range(im_arr.shape[0]):
        for j in range(im_arr.shape[1]):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    black = True  # 黑白检测模式切换(先是黑检测)
    for col in range(im_arr.shape[1]):
        # 对第col列求和，如果该列全部为0则为全白，否则有黑色。
        if black and sum(im_arr[:, col]) != 0:
            im_arr_num.append(col)
            black = False
        elif not black and sum(im_arr[:, col]) == 0:
            im_arr_num.append(col)
            black = True
    if len(im_arr_num) % 2 == 1:
        im_arr_num.append(len(im_arr.shape[1]))

    # 每个数字进行截取, resize, reshape并返回img_ready
    img_ready = []  # 存放多个数字
    assert len(im_arr_num) % 2 == 0  # 开始结束成对出现：im_arr_num是偶数，否则错误
    # im_arr:原始截取图像
    # im_arr_num:存放原图切片位置. 单数字符由偶数开始, 奇数结束(eg.第一个数字开始在0，结束在1)
    for num in range(0, len(im_arr_num), 2):  # 中点切片
        # 开始:nm_arr_char[i]; 结束:nm_arr_char[i+1]+1
        if num == 0:
            left_cut = 0
        else:
            left_cut = int((im_arr_num[num - 1] + im_arr_num[num]) / 2)

        if num+2 == len(im_arr_num):
            right_cut = im_arr.shape[1]
        else:
            right_cut = int((im_arr_num[num + 1] + im_arr_num[num + 2]) / 2)

        cur_detect = im_arr[:, left_cut: right_cut]
        cur_detect = Image.fromarray(cur_detect)  # 转为图片
        cur_detect = cur_detect.resize((28, 28), Image.ANTIALIAS)  # 图片resize
        cur_detect = np.array(cur_detect.convert('L'))
        cur_detect = cur_detect.reshape([1, 784])  # 图片reshape
        cur_detect = cur_detect.astype(np.float32)
        cur_detect = np.multiply(cur_detect, 1.0 / 255.0)
        img_ready.append(cur_detect)  # 加入结果list
    return img_ready


def application():
    print("Running app.py...")
    img_name = input("input the path of pictures:")
    cut_images = capture(img_name, 'date')
    print("The predict date is:", end=' ')
    for img in cut_images:
        detect_number = restore_model(img)
        print(detect_number, end=' ')
    print("\n")
    cut_images = capture(img_name, 'amount')
    print("The predict amount is:", end=' ')
    for img in cut_images:
        detect_number = restore_model(img)
        print(detect_number, end=' ')
    print("\n")


def main():
    application()


if __name__ == '__main__':
    main()
