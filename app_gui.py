# coding:utf-8


from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import app


class app_gui:
    def __init__(self, master):
        self.frame = Frame(master)
        self.frame.pack()
        self.file_name = None     # 目标路径
        self.img = None           # 图片
        self.date = []            # 预测的日期
        self.amount = []          # 预测的金额
        self.path_label = Label(self.frame, text="目标路径:").grid(row=0, column=0)
        self.path_entry = Entry(self.frame, textvariable=self.file_name).grid(row=0, column=1)
        self.file_button = Button(self.frame, text="    浏览文件    ", command=self.analysis_image).grid(row=0, column=2)
        self.help_button = Button(self.frame, text="    帮助    ", command=self.help).grid(row=0, column=3)
        self.quit_button = Button(self.frame, text="退出", command=quit).grid(row=0, column=4)

    def center_window(self, w=300, h=200):
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        root.geometry("%dx%d+%d+%d" % (w, h, x, y))

    def help(self):
        print("NOT GONNA HELP!!")

    def select_file(self):
        self.file_name = filedialog.askopenfilename()
        print("file_name:", self.file_name)

    def open_image(self):
        image = Image.open(self.file_name)
        # 图像缩小显示，scale:缩放比例
        scale = max(300 / image.size[0], 200 / image.size[1])
        width = int(image.size[0] * scale)
        height = int(image.size[1] * scale)
        image = image.resize((width, height), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(image=image)  # 处理格式
        # 不写有几率会炸
        self.img.image = self.img
        Label(self.frame, image=self.img).grid(row=1, column=0, columnspan=6)
        print("img:", self.img)

    def print_date(self):
        cut_images = app.capture_date(self.file_name)
        for img in cut_images:
            detect_number = app.restore_model(img)
            self.date.append(detect_number)
        Label(self.frame, text="(预测)日期为:").grid(row=2, column=0, columnspan=2)
        Label(self.frame, text=self.date).grid(row=2, column=2, columnspan=2)
        print("date[]:", self.date)

    def print_amount(self):
        cut_images = app.capture_amount(self.file_name)
        for img in cut_images:
            detect_number = app.restore_model(img)
            self.amount.append(detect_number)
        Label(self.frame, text="(预测)金额为:").grid(row=3, column=0, columnspan=2)
        Label(self.frame, text=self.amount).grid(row=3, column=2, columnspan=2)
        print("amount[]:", self.amount)

    def analysis_image(self):
        self.select_file()
        self.open_image()
        self.print_date()
        self.print_amount()


root = Tk(className="基于卷积神经网络识别金融票据中的数字串")
app_gui = app_gui(root)
app_gui.center_window(500, 300)
root.mainloop()
