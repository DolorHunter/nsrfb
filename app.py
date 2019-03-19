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


# 图片预处理
# 反相 + 去噪声 (训练集中的样本为黑底白字)
# 单行列切片 - 将一个字符串/数字串切成多个单字符/数字
# 白色为255 黑色为0
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshould = 50  # 噪点控制
    im_arr_num = []  # 偶数位储存字符开始位置, 奇数位储存字符结束位置
    count = 0  # arr_num的存储位置
    space = 0  # 存储空白像素
    black = True  # 黑白检测模式切换(先是黑检测)

	# 图片转化为黑白二色, 并去除噪点(50) + 反相
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshould:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    # 假定每个数字间有空隙, 开始出现黑色即为出现字符, 记录位置; 全列空白即为字符结束, 记录位置
    # 单行列切片
    for row in range(28):
        for i in range(28):
            if black:
                # 黑检测模式(检测是否出现数字)
                # 出现黑色即为出现字符
                if im_arr[i][row] == 0:
                    im_arr_num.append(row)
                    count = count + 1
                    black = False  # 切换白检测模式
                    continue
            else:
                # 白检测模式(数字结束)
                if im_arr[i][row] == 255:
                    space = space + 1
        if space == 28:
            im_arr_num.append(row)
            count = count + 1
            black = True  # 切换黑模式
        # space归零
        space = 0

    # 每个数字进行截取, resize, reshape并返回img_ready
    img_ready = []  # 存放多个数字

    assert len(im_arr_num) % 2 == 0  # 开始结束成对出现：im_arr_num是偶数，否则错误

    # im_arr:原图像
    # 第一个数字开始在0，结束在1
    # 每个数字步长为2
    for num in range(0, len(im_arr_num), 2):
        # 开始：nm_arr_char[i]
        # 结束：nm_arr_char[i+1]+1 因为切片是开区间
        cur_detect = im_arr[:, im_arr_num[num]:im_arr_num[num + 1] + 1]
        cur_detect = Image.fromarray(cur_detect)  # 转为图片
        cur_detect = cur_detect.resize((28, 28), Image.ANTIALIAS)  # 图片resize

################################################################

	cur_arr = np.array(cur_detect.comvert('L'))
	nm_arr = cur_arr.reshape([1, 784])
	nm_arr = nm-arr.astype(np.float32)	
	cur_ready = np.multiply(cur_detect, 1.0/255.0)

################################################################

        img_ready.append(cur_ready)  # 加入结果list
    return img_ready


def application():
    print("Running app.py...")
    img_name = '18.jpg'
    cut_images = pre_pic(img_name)
    print("The predict number is:")
    for img in cut_images:
        detect_number = restore_model(img)
        print(detect_number, end=' ')



def main():
    application()


if __name__ == '__main__':
    main()
