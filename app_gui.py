# coding:utf-8

# This is a test gui

import tkinter as tk
from PIL import Image, ImageTk


def center_window(w=300, h=200):
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    root.geometry("%dx%d+%d+%d" % (w, h, x, y))

    # 打开图片
    img_open = tk.PhotoImage(file="pic/hk_check.gif")
    label_img = tk.Label(root, image=img_open)
    label_img.pack()


root = tk.Tk(className="基于卷积神经网络识别金融票据中的数字串")
center_window(500, 300)
root.mainloop()