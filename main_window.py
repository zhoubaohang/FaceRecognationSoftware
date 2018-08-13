# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:42:36 2018

@author: 周宝航
"""

import cv2
import numpy
from PIL import Image
from PIL import ImageTk
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog


class TopMenu():
    '''菜单类'''
    
    def __init__(self, root):
        '''初始化菜单'''
        self.root = root
        self.menubar = tk.Menu(self.root) # 创建菜单栏
        
        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="导入", command=self.file_import)
        filemenu.add_separator()
        filemenu.add_command(label="退出", command=self.root.quit)
        
        helpmenu = tk.Menu(self.menubar, tearoff=0)
        helpmenu.add_command(label="关于", command=self.help_about)
        
        self.menubar.add_cascade(label="功能", menu=filemenu)
        self.menubar.add_cascade(label="帮助", menu=helpmenu)
        
        # 最后再将菜单栏整个加到窗口 root
        self.root.config(menu=self.menubar)
        
        self.path = tk.StringVar()
        
    def file_import(self):
        self.window = tk.Toplevel(self.root)
        tk.Label(self.window,text = "目标路径:").grid(row = 0, column = 0)
        tk.Entry(self.window, textvariable = self.path).grid(row = 0, column = 1)
        tk.Button(self.window, text = "路径选择", command = self.__selectPath).grid(row = 0, column = 2)
        tk.Button(self.window, text = "开始导入", command =  self.__import).grid(row = 0, column = 3)
        self.progressBar = ttk.Progressbar(self.window, length = 200, mode = "determinate", orient = tk.HORIZONTAL)
        self.progressBar.grid(row = 1, column = 1)
        self.progressBar["maximum"] = 100
        self.progressBar["value"] = 0
        self.lb_progress = tk.Label(self.window, text = '导入进度')
        self.lb_progress.grid(row = 1, column = 0)
    
    def __import(self):
        padW, padH = self.root.fr.padding
        path = self.path.get()
        if path:
            import os
            imgs = os.listdir(path)
            num_imgs = len(imgs)
            
            for i in range(num_imgs):
                self.progressBar["value"] = (i + 1) / num_imgs * self.progressBar["maximum"]
                self.lb_progress['text'] = '{0}/{1}'.format((i+1), num_imgs)
                self.window.update()
                
                image_path = "{0}\\{1}".format(path, imgs[i])
                image, gray, faces, _ = self.root.fr.findFaces(cv2.cvtColor(numpy.asarray(Image.open(image_path)), cv2.COLOR_RGB2BGR))
                if len(faces):
                    face_area, maxface = 0, faces[0]
                    for face in faces:
                        _, _, w, h = face
                        if (w*h) > face_area:
                            maxface = face
                            face_area = w*h
                    (x, y, w, h) = maxface
                    image = Image.fromarray(gray).crop((x-padW, y-padH, x+padW+w, y+padH+h)).resize(self.root.fr.img_size)
                    image.save('{0}{1}.bmp'.format(self.root.user_image_path, imgs[i].split('.')[0]))
                else:
                    messagebox.showinfo('警告', 'image file: {0} \n find no faces in the image, please check'.format(imgs[i]))

            self.root.fr.load_user_images()
        else:
            messagebox.showinfo("提示", "请选择导入人像路径")
        
        
    def __selectPath(self):
        path_ = filedialog.askdirectory()
        self.path.set(path_)

    def help_about(self):
        messagebox.showinfo('关于', '作者：周宝航 \n verion 1.0 \n 感谢您的使用！ \n zbh12306@163.com ')


class MainWindow(tk.Tk):
    
    def __init__(self, faceRecognation):
        
        super().__init__()
        # init gui
        self.__init_gui()
        # camera running flag
        self.running_flag = False
        # user face images path
        self.user_image_path = "data\\user\\"
        # face recognation class
        self.fr = faceRecognation
        
    
    def __init_gui(self):
        
        self.wm_title('人脸识别')
        self.config(background = '#FFFFFF')
        self.protocol("WM_DELETE_WINDOW", self.__closeWindow)              
                           
        self.lb_name = tk.Label(self, text = "识别结果")
        self.lb_name.grid(row = 0, column = 0)
        
        self.canvas = tk.Canvas(self, width = 400, height = 400)
        self.canvas.grid(row = 1, column = 0)

        self.fm_control = tk.Frame(self, width = 400, height = 100, background = '#FFFFFF')
        self.fm_control.grid(row = 2, column = 0, padx = 10, pady = 2)
        
        self.btn_control = tk.Button(self.fm_control, text = '开启', command = self.__controlCamera)
        self.btn_control.grid(row = 0, column = 0, padx = 10, pady = 2)
        
        self.btn_catch = tk.Button(self.fm_control, text = '拍照', command = self.__catchPhotos)
        self.btn_catch.grid(row = 0, column = 1, padx = 10, pady = 2)
        
        tk.Label(self.fm_control, text="姓名:").grid(row = 0, column = 2, padx = 10, pady = 2)
        
        self.tv_name = tk.Entry(self.fm_control, textvariable=tk.StringVar())
        self.tv_name.grid(row = 0, column = 3, padx = 10, pady = 2)
        
        TopMenu(self)
        
    def __openCamera(self):
        # camera capture
        self.cap = cv2.VideoCapture(0)
        while (self.running_flag):
            # get a frame
            ret, frame = self.cap.read()
            # get faces
            image, _, _, names = self.fr.findFaces(frame)
            if len(names):
                self.lb_name['text'] = ','.join(names)                        
            else:
                self.lb_name['text'] = '未检测到人脸'
            # show a frame
            self.__imageUpdate(image)
    
    def __closeCamera(self):
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __imageUpdate(self, image):
        
        image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(200, 200, image = image)
        
        self.update_idletasks()
        self.update()
    
    def __catchPhotos(self):
        padW, padH = self.fr.padding
        if self.running_flag:
            name = self.tv_name.get()
            if name != '':
                ret, frame = self.cap.read()
                image, gray, faces, _ = self.fr.findFaces(frame)
                if len(faces):
                    (x, y, w, h) = faces[0]
                    image = Image.fromarray(gray).crop((x-padW, y-padH, x+padW+w, y+padH+h)).resize(self.fr.img_size)
                    image.save('{0}{1}.bmp'.format(self.user_image_path, name))
                    self.fr.load_user_images()
                else:
                    messagebox.showinfo('提示', '未收集到人像, 请重新拍照')
            else:
                messagebox.showinfo("提示","请输入保存用户姓名")
        else:
            messagebox.showinfo("提示","请打开摄像头")
    
    def launch(self):
        
        self.mainloop()
    
    def __closeWindow(self, *event):
        
        if self.running_flag:
            self.__closeCamera()
        self.quit()
        
    def __controlCamera(self):
        
        self.running_flag = not self.running_flag
        self.btn_control['text'] = "关闭" if self.running_flag else "开启"
        if self.running_flag:
            self.__openCamera()
        else:
            self.__closeCamera()
    
