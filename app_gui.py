# coding:utf-8


import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import app


class App:
    def __init__(self, master):
        self.frame = Frame(master)
        self.frame.pack()
        # TEST!!
        self.file_path = StringVar()     # 文件夹路径
        self.file_name = StringVar()     # 文件名
        self.img_label = StringVar()     # 图片
        self.date_label = StringVar()    # 预测的日期
        self.amount_label = StringVar()  # 预测的金额
        # TEST!!
        self.path_label = Label(self.frame, text="目标路径:")
        self.path_label.pack(side=LEFT)
        self.path_entry = Entry(self.frame, textvariable=self.file_path)
        self.path_entry.pack(side=LEFT)
        self.path_button = Button(self.frame, text="浏览文件", command=self.select_file)
        self.path_button.pack(side=LEFT)
        self.path_button = Button(self.frame, text="选择路径", command=self.select_path)
        self.path_button.pack(side=LEFT)
        self.quit_button = Button(self.frame, text="  QUIT  ", command=quit)
        self.quit_button.pack(side=RIGHT)

    def center_window(self, w=300, h=200):
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        root.geometry("%dx%d+%d+%d" % (w, h, x, y))

    def select_file(self):
        self.file_name = filedialog.askopenfile()

    def select_path(self):
        self.file_path = filedialog.askdirectory()
        self.file_path.set(self.file_path)

    def open_image(self):
        open_img = Image.open(self.file_name)
        img = ImageTk.PhotoImage(image=open_img)  # 处理格式
        self.img_label = Label(self.frame, text="Image", image=img)
        self.img_label.pack(side=LEFT)

    def prt_date(self):
        cut_images = app.capture_date(self.file_name)
        date = []
        for img in cut_images:
            detect_number = app.restore_model(img)
            date.append(detect_number)
        self.date_label = Label(self.frame, text=date)
        self.date_label.pack(side=LEFT)

    def prt_amount(self):
        cut_images = app.capture_amount(self.file_name)
        amount = []
        for img in cut_images:
            detect_number = app.restore_model(img)
            amount.append(detect_number)
        self.amount_label = Label(self.frame, text=amount)
        self.amount_label.pack(side=LEFT)


root = Tk(className="基于卷积神经网络识别金融票据中的数字串")
app = App(root)
app.center_window(500, 300)
root.mainloop()
