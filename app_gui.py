# coding:utf-8


import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import PIL


class App:
    path = "abc"
    # path = StringVar()  # 文件夹路径
    #file_name =
    #img = self.show_img(file_name)

    def __init__(self, master):
        frame = Frame(master)
        frame.pack()
        self.path = StringVar()  # 文件夹路径
        self.path_label = Label(frame, text="目标路径:")
        self.path_label.pack(side=LEFT)
        self.path_entry = Entry(frame, textvariable=self.path)
        self.path_entry.pack(side=LEFT)
        self.path_button = Button(frame, text="浏览文件", command=self.select_file)
        self.path_button.pack(side=LEFT)
        self.path_button = Button(frame, text="选择路径", command=self.select_path)
        self.path_button.pack(side=LEFT)

        # self.img_label = Label(frame, image=self.tkImage)
        # self.path_button.pack()

        self.quit_button = Button(frame, text="  QUIT  ", command=quit)
        self.quit_button.pack(side=RIGHT)

    def center_window(self, w=300, h=200):
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        root.geometry("%dx%d+%d+%d" % (w, h, x, y))

    def select_file(self):
        file_name = filedialog.askopenfile()
        return file_name

    def select_path(self):
        img_path = filedialog.askdirectory()
        self.path.set(img_path)

'''
    def show_img(self, file_name):
        # img = self.select_file()
        img = Image.open("pic/0.png")
        tkImage = ImageTk.PhotoImage(image=img)
        return tkImage
'''

root = Tk(className="基于卷积神经网络识别金融票据中的数字串")
app = App(root)
app.center_window(500, 300)
root.mainloop()
