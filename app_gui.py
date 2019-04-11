# coding:utf-8


from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import app_bash


class app_gui:
    def __init__(self, master):
        self.frame = Frame(master)
        self.frame.pack()
        self.file_name = StringVar()     # 目标路径
        self.img = StringVar()           # 图片
        self.date = StringVar()          # 预测的日期
        self.amount = StringVar()        # 预测的金额
        self.path_label = Label(self.frame, text="目标路径:")
        self.path_label.grid(row=0, column=0)
        self.path_entry = Entry(self.frame, textvariable=self.file_name, state="normal")
        self.path_entry.grid(row=0, column=1)
        self.file_button = Button(self.frame, text="    浏览文件    ", command=self.analysis_image)
        self.file_button.grid(row=0, column=2)
        self.help_button = Button(self.frame, text="    帮助    ", command=self.help)
        self.help_button.grid(row=0, column=3)
        self.quit_button = Button(self.frame, text="退出", command=sys.exit)
        self.quit_button.grid(row=0, column=4)

    def center_window(self, w=300, h=200):
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        root.geometry("%dx%d+%d+%d" % (w, h, x, y))

    def help(self):
        messagebox.showwarning('Need Help?', 'NOT GONNA HELP!!')
        print("NOT GONNA HELP!!")
        # EXTREMELY IMPORTANT DONT DELETE
        self.help()

    def select_file(self):
        self.path_entry.delete(0, END)
        self.file_name = filedialog.askopenfilename()
        self.path_entry.insert(0, self.file_name)

    def open_image(self):
        image = Image.open(self.file_name)
        # 图像缩小显示，scale:缩放比例
        scale = max(300 / image.size[0], 210 / image.size[1])
        width = int(image.size[0] * scale)
        height = int(image.size[1] * scale)
        image = image.resize((width, height), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(image=image)  # 处理格式
        # 不写有几率会炸
        self.img.image = self.img
        Label(self.frame, image=self.img).grid(row=1, column=0, columnspan=5)

    def print_date(self):
        self.date = []
        cut_images = app_bash.capture_date(self.file_name)
        for img in cut_images:
            detect_number = app_bash.restore_model(img)
            self.date.append(detect_number)
        Label(self.frame, text="(预测)日期为:").grid(row=2, column=0, columnspan=2)
        Label(self.frame, text=self.date).grid(row=2, column=2, columnspan=2)

    def print_amount(self):
        self.amount = []
        cut_images = app_bash.capture_amount(self.file_name)
        for img in cut_images:
            detect_number = app_bash.restore_model(img)
            self.amount.append(detect_number)
        Label(self.frame, text="(预测)金额为:").grid(row=3, column=0, columnspan=2)
        Label(self.frame, text=self.amount).grid(row=3, column=2, columnspan=2)

    def analysis_image(self):
        self.select_file()
        print("file_name:", self.file_name)
        self.open_image()
        print("img:", self.img)
        self.print_date()
        print("date[]:", self.date)
        self.print_amount()
        print("amount[]:", self.amount)


root = Tk(className="基于卷积神经网络识别金融票据中的数字串")
app_gui = app_gui(root)
app_gui.center_window(500, 300)
root.mainloop()
