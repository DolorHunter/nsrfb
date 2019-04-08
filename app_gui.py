# coding:utf-8


from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import app


class App:
    def __init__(self, master):
        self.frame = Frame(master)
        self.frame.pack()
        self.file_path = None     # 文件夹路径
        self.file_name = None     # 文件名
        self.img_label = None     # 图片
        self.date = []            # 预测的日期
        self.amount = []          # 预测的金额
        self.path_label = Label(self.frame, text="目标路径:").grid(row=0, column=0)
        self.path_entry = Entry(self.frame, textvariable=self.file_path).grid(row=0, column=1)
        self.path_button = Button(self.frame, text="    浏览文件    ", command=self.analysis_image).grid(row=0, column=2)
        self.path_button = Button(self.frame, text="    选择路径    ", command=self.select_path).grid(row=0, column=3)
        self.quit_button = Button(self.frame, text="退出", command=quit).grid(row=0, column=4)
        self.image_label = Label(self.frame, image=self.img_label).grid(row=1, column=0)
        self.date_label = Label(self.frame, text=self.date).grid(row=2, column=0)
        self.amount_label = Label(self.frame, text=self.amount).grid(row=3, column=0)

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
        # 不写会炸, 写了还是炸的神奇语句..
        self.img_label.image = img

    def print_date(self):
        cut_images = app.capture_date(self.file_name)
        for img in cut_images:
            detect_number = app.restore_model(img)
            self.date.append(detect_number)

    def print_amount(self):
        cut_images = app.capture_amount(self.file_name)
        for img in cut_images:
            detect_number = app.restore_model(img)
            self.amount.append(detect_number)

    def analysis_image(self):
        self.select_file()
        self.open_image()
        self.print_date()
        self.print_amount()


root = Tk(className="基于卷积神经网络识别金融票据中的数字串")
app = App(root)
app.center_window(500, 300)
root.mainloop()
