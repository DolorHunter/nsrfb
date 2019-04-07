# coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import backward
import forward


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

'''
# 捕捉日期
def capture_date(picName):
    img = Image.open(picName)
    im_arr = np.array(img.convert('L'))
    # 捕捉日期 (计算比例)
    up_cut = int((250 / 3255) * im_arr.shape[0])
    down_cut = int((670 / 3255) * im_arr.shape[0])
    left_cut = int((4190 / 6200) * im_arr.shape[1])
    right_cut = int((6100 / 6200) * im_arr.shape[1])
    cur_capture = im_arr[up_cut: down_cut, left_cut: right_cut]
    cur_capture = Image.fromarray(cur_capture)  # 转为图片
    return cur_capture


# 捕捉金额
def capture_amount(picName):
    img = Image.open(picName)
    im_arr = np.array(img.convert('L'))
    # 捕捉金额 (计算比例)
    up_cut = int((1455/3255)*im_arr.shape[0])
    down_cut = int((1780/3255)*im_arr.shape[0])
    left_cut = int((4645/6200)*im_arr.shape[1])
    right_cut = int((6030/6200)*im_arr.shape[1])
    cur_capture = im_arr[up_cut: down_cut, left_cut: right_cut]
    cur_capture = Image.fromarray(cur_capture)  # 转为图片
    return cur_capture
'''

# 捕捉日期
# 反相 + 去噪声 (训练集中的样本为黑底白字)
# 单行列切片 - 将一个字符串/数字串切成多个单字符/数字
# 白色为255 黑色为0
def capture_date(picName):
    img = Image.open(picName)
    im_arr = np.array(img.convert('L'))
    # 捕捉金额 (按比例计算)
    up_cut = int((250 / 3255) * im_arr.shape[0])
    down_cut = int((670 / 3255) * im_arr.shape[0])
    left_cut = int((4190 / 6200) * im_arr.shape[1])
    right_cut = int((6100 / 6200) * im_arr.shape[1])
    im_arr = im_arr[up_cut: down_cut, left_cut: right_cut]  # 捕捉后存入im_arr
    threshould = 50  # 噪点控制
    im_arr_num = []  # 偶数位储存字符开始位置, 奇数位储存字符结束位置
    for i in range(im_arr.shape[0]):
        for j in range(im_arr.shape[1]):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshould:
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
    # im_arr:原图像
    # im_arr_num:存放切片位置, 单数字符由偶数开始, 奇数结束(eg.第一个数字开始在0，结束在1)
    for num in range(0, len(im_arr_num), 2):  # 中点切片
        # 开始：nm_arr_char[i]; 结束：nm_arr_char[i+1]+1 因为切片是开区间
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


# 捕捉金额
# 反相 + 去噪声 (训练集中的样本为黑底白字)
# 单行列切片 - 将一个字符串/数字串切成多个单字符/数字
# 白色为255 黑色为0
def capture_amount(picName):
    img = Image.open(picName)
    im_arr = np.array(img.convert('L'))
    # 捕捉金额 (按比例计算)
    up_cut = int((1455 / 3255) * im_arr.shape[0])
    down_cut = int((1780 / 3255) * im_arr.shape[0])
    left_cut = int((4645 / 6200) * im_arr.shape[1])
    right_cut = int((6030 / 6200) * im_arr.shape[1])
    im_arr = im_arr[up_cut: down_cut, left_cut: right_cut]  # 捕捉后存入im_arr
    threshould = 50  # 噪点控制
    im_arr_num = []  # 偶数位储存字符开始位置, 奇数位储存字符结束位置
    for i in range(im_arr.shape[0]):
        for j in range(im_arr.shape[1]):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshould:
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
    # im_arr:原图像
    # im_arr_num:存放切片位置, 单数字符由偶数开始, 奇数结束(eg.第一个数字开始在0，结束在1)
    for num in range(0, len(im_arr_num), 2):  # 中点切片
        # 开始：nm_arr_char[i]; 结束：nm_arr_char[i+1]+1 因为切片是开区间
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
    picNumber = int(input("input the number of pictures:"))
    for i in range(picNumber):
        img_name = input("input the path of pictures:")
        cut_images = capture_date(img_name)
        print("The predict date is:", end=' ')
        for img in cut_images:
            detect_number = restore_model(img)
            print(detect_number, end=' ')
        print("\n")
        cut_images = capture_amount(img_name)
        print("The predict amount is:", end=' ')
        for img in cut_images:
            detect_number = restore_model(img)
            print(detect_number, end=' ')
        print("\n")


def main():
    application()


if __name__ == '__main__':
    main()
