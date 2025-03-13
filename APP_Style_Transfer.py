"""
任务: 实现图片风格迁移的APP, 包括静态图片的风格迁移和动态视频帧的风格迁移
      完成相关GUI界面的设计,实现对应模式的切换
      图像文件的导入和保存, 实现风格迁移的过程展示, 实现迁移后的图片的展示
时间: 2024/11/15-Redal
"""
import cv2
import os
import torch
import numpy as np

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import threading 
import queue
from PIL import Image, ImageTk
import torchvision.models as models
import torchvision.transforms as transforms
from models.VGG_Fixed_Style_Transfer import VGG_Fixed_Style_Transfer
from models.MetaNet_Random_Style_Transfer import VGG,TransformNet, MetaNet


class ImageStyleTransferApp(tk.Frame):
    """
    主要针对图片风格迁移: 
    固定风格固定内容的风格迁移——>输入图片和风格图片，输出迁移后的图片;
    任意风格任意内容的极速迁移——>输入图片,实时对视频帧进行风格迁移;
    epoches: 固定风格迁移的程度的定义
    """
    def __init__(self, root=None):
        super().__init__(root)
        self.root = root
        self.root.title("Image Style Transfer App-Sub Redal")
        self.root.geometry("600x400")
        self.__set_widgets__()

        self.video_cap = cv2.VideoCapture(0)
        self.frame = None 
        self.content_image = None
        self.style_image = None
        self.output_image = None
        self.epoches_gotten = None
        # 加载模型以及参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg16 = VGG(vgg16.features[:23]).to(self.device).eval()
        self.transform_net = TransformNet(base=32).to(self.device).eval()
        self.transform_net.load_state_dict(torch.load('state_dict/metanet_base32_style50_tv1e-06_tagnohvd_transform_net.pth', 
                                                      map_location=self.device, weights_only=True))
        self.meta_net = MetaNet(self.transform_net.get_param_dict()).to(self.device).eval()
        self.meta_net.load_state_dict(torch.load('state_dict/metanet_base32_style50_tv1e-06_tagnohvd.pth', 
                                               map_location=self.device, weights_only=True))
        
        self.toggle_fixed_style_transfer_active = False
        self.toggle_dynamic_style_transfer_active = False
        
        self.thread = threading.Thread(target = self.__update_video__)
        self.thread.daemon = True
        self.thread.start()
        
    def __set_widgets__(self):
        """定义界面控件"""
        self.frame_label = tk.Label(self.root, text="Image Style Transfer")
        self.frame_label.place(x=0, y=0, width=400, height=400)
        self.frame_initial_image = cv2.resize(cv2.cvtColor(cv2.imread(
            r"content_image\content_5.jpg"), cv2.COLOR_BGR2RGB), (400,400))
        self.frame_initial_image = ImageTk.PhotoImage(Image.fromarray(self.frame_initial_image))
        self.frame_label.config(image=self.frame_initial_image)
        self.frame_label.image = self.frame_initial_image
        # 导入内容图片展示
        self.original_content_image_label = tk.Label(self.root, text="等待导入内容图片...", font=("仿宋", 12))
        self.original_content_image_label.place(x=400, y=200, width=200, height=200)
        
        # 固定迁移程度定义label和entry
        self.system_name_label = tk.Label(self.root, text="风格迁移", font=("仿宋", 16))
        self.system_name_label.place(x=400, y=0, width=200, height=40)
        self.entry_epoches_gotten = tk.Entry(self.root, width=10)
        self.entry_epoches_gotten.place(x=480, y=60)
        self.entry_epoches_label = tk.Label(self.root, text="迁移程度", font=("仿宋", 12))
        self.entry_epoches_label.place(x=410, y=55, width=70, height=30)

        self.import_content_image_button = tk.Button(self.root, text="导入内容", command=self.__import_content_image__)
        self.import_content_image_button.place(x=410, y=90, width=60, height=30)
        self.import_style_image_button = tk.Button(self.root, text="导入风格", command=self.__import_style_image__)
        self.import_style_image_button.place(x=470, y=90, width=60, height=30)
        self.exit_button = tk.Button(self.root, text="退出", command=lambda: self.root.quit())
        self.exit_button.place(x=530, y=90, width=60, height=30)
        
        # 固定风格迁移按钮
        self.fixed_style_transfer_button = tk.Button(self.root, text="固定迁移", command=self.__toggle_fixed_style_transfer__)
        self.fixed_style_transfer_button.place(x=410, y=130, width=60, height=30)
        # 动态风格迁移按钮
        self.dynamic_style_transfer_button = tk.Button(self.root, text="动态迁移", command=self.__toggle_dynamic_style_transfer__)
        self.dynamic_style_transfer_button.place(x=470, y=130, width=60, height=30)
        # 保存按钮
        self.button_save = tk.Button(self.root, text='保存', command=self.__save_output_image__)
        self.button_save.place(x=530, y=130, width=60, height=30)
        self.show_output_img_button = tk.Button(self.root, text="清除显示", command=self.__clear__showed_image__)
        self.show_output_img_button.place(x=530, y=170, width=60, height=30)
    
    
    def __update_video__(self):
        """实时视频帧更新"""
        while self.video_cap.isOpened():
            flag, frame = self.video_cap.read()
            if flag:

                # 固定风格迁移
                if self.toggle_fixed_style_transfer_active:
                    # 固定风格迁移初始化界面显示
                    self.frame_label.config(image=self.frame_initial_image)
                    self.frame_label.image = self.frame_initial_image
                    self.original_content_image_label.config(text="请先填写迁移程度...")
                    
                    if self.content_image is not None and self.style_image is not None:
                        # 获取程度参数
                        self.epoches_gotten = self.entry_epoches_gotten.get()
                        if self.epoches_gotten != '':
                            self.fixed_style_transfer_func = VGG_Fixed_Style_Transfer(self.content_image, 
                                                self.style_image, self.device, epoches=int(self.epoches_gotten))
                            output = self.fixed_style_transfer_func.run_style_transfer()
                            # 保存输出图片
                            self.output_image = (output[0].cpu().detach().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
                            img_tk = Image.fromarray(cv2.resize(self.output_image, (400,400)))
                            img_tk = ImageTk.PhotoImage(img_tk)
                            self.frame_label.config(image=img_tk)
                            self.frame_label.image = img_tk
                            
                            # 固定风格迁移完成后，按钮恢复初始状态
                            self.original_content_image_label.config(text="等待导入内容图片...")
                            img_tk = ImageTk.PhotoImage(self.content_image.resize((200,200)),Image.Resampling.LANCZOS)
                            self.original_content_image_label.config(image=img_tk)
                            self.original_content_image_label.image = img_tk
                            self.entry_epoches_label.config(text="迁移程度")
                            self.toggle_fixed_style_transfer_active = False
                            self.style_image, self.content_image = None, None # 风格图片和内容图片清空
                        else:
                            self.entry_epoches_label.config(text="填写参数")

                # 动态风格迁移
                if self.toggle_dynamic_style_transfer_active:
                    self.original_content_image_label.config(text="等待导入风格图片...")
                    self.frame = cv2.flip(cv2.resize(cv2.cvtColor(frame, 
                                        cv2.COLOR_BGR2RGB), (400,400)), 1)
                    # 进行实时风格迁移
                    if self.style_image is not None:    
                        self.frame = self.preprocess_image(Image.fromarray(self.frame))
                        with torch.no_grad():
                            self.frame = self.transform_net(self.frame)
                        # 显示视频帧
                        image_pil = Image.fromarray(self.postprocess_image(self.frame)).resize((400,400))
                        self.frame = ImageTk.PhotoImage(image_pil, Image.Resampling.LANCZOS)
                        self.frame_label.config(image=self.frame)
                        self.frame_label.image = self.frame
                    else: self.__video_show__()
            else: break
    
    def __video_show__(self):
        """实时视频帧显示"""
        img_array = Image.fromarray(self.frame)
        img_tk = ImageTk.PhotoImage(img_array)
        self.frame_label.config(image=img_tk)
        self.frame_label.image = img_tk
        self.after(33)
        
    def __toggle_fixed_style_transfer__(self):
        """固定风格迁移按钮激活"""
        self.original_content_image_label.config(text="等待导入内容图片...")
        self.toggle_dynamic_style_transfer_active = False
        self.toggle_fixed_style_transfer_active = not self.toggle_fixed_style_transfer_active
        self.frame_label.config(image=self.frame_initial_image)
        self.frame_label.image = self.frame_initial_image
        if not self.toggle_dynamic_style_transfer_active:
            self.frame_label.config(image=self.frame_initial_image)
            self.frame_label.image = self.frame_initial_image
    def __toggle_dynamic_style_transfer__(self):
        """动态风格迁移按钮激活"""
        self.original_content_image_label.config(text="等待导入内容图片...")
        self.toggle_fixed_style_transfer_active = False
        self.toggle_dynamic_style_transfer_active = not self.toggle_dynamic_style_transfer_active
        if not self.toggle_dynamic_style_transfer_active:
            # 动态模式结束后，恢复初始界面
            self.frame_label.config(image=self.frame_initial_image)
            self.frame_label.image = self.frame_initial_image
            self.original_content_image_label.config(text="等待导入内容图片...")
            self.original_content_image_label.config(image='')
            self.original_content_image_label.image = None
            self.style_image = None # 风格图片清空
        
    def start_video_capture(self):
        if self.video_cap is None:
            self.video_cap = cv2.VideoCapture(0)
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.__update_video__)
            self.thread.daemon = True
            self.thread.start()
    
    def __import_content_image__(self):
        """导入内容图片,并将其单独保存,用于固定风格迁移"""
        file_path = filedialog.askopenfilename(
                title="选择内容图片",
                filetypes=[("图片", "*.jpg *.png *.jpeg *.bmp *.tif *.tiff")])
        if file_path:
            self.content_image = Image.open(file_path)
            self.content_image.save(r"content_image\content_image.jpg")
            img_content = cv2.resize(cv2.cvtColor(cv2.imread(r"content_image\content_image.jpg"), 
                                            cv2.COLOR_BGR2RGB), (200, 200))  
            img_content = ImageTk.PhotoImage(Image.fromarray(img_content))
            self.original_content_image_label.config(image=img_content)
            self.original_content_image_label.image = img_content

    def __import_style_image__(self):
        """导入风格图片用于固定风格迁移"""
        file_path = filedialog.askopenfilename(
                title="选择风格图片",
                filetypes=[("图片", "*.jpg *.png *.jpeg *.bmp *.tif *.tiff")])
        if file_path:   
            self.style_image = Image.open(file_path).convert('RGB')
            self.style_image.save(r"style_image\style_image.jpg")
            if self.toggle_dynamic_style_transfer_active:
                # 如果开启动态迁移
                self.style_image = self.preprocess_image(self.style_image, width=256)
                # 计算风格图像特征
                style_features = self.vgg16(self.style_image)
                style_mean_std = self.mean_std(style_features)
                # 计算风格图像权重
                weights = self.meta_net(style_mean_std)
                self.transform_net.set_weights(weights, 0)
            img_style = cv2.resize(cv2.cvtColor(cv2.imread(r"style_image\style_image.jpg"),
                                     cv2.COLOR_BGR2RGB), (200, 200)) 
            img_style = ImageTk.PhotoImage(Image.fromarray(img_style))
            self.original_content_image_label.config(image=img_style)
            self.original_content_image_label.image = img_style
            
        
    def __save_output_image__(self):
        """保存固定风格迁移输出图片"""
        self.original_content_image_label.config(text="等待导入内容图片...")
        self.original_content_image_label.config(image='')
        self.original_content_image_label.image = None
        if self.output_image is not None:
            file_path = tk.filedialog.asksaveasfilename(
                title="保存输出图片",
                filetypes=[("图片", "*.jpg;*.png;*.jpeg;*.bmp;*.tif;*.tiff")],
                defaultextension=".jpg")
            if file_path:
                cv2.imwrite(file_path, cv2.cvtColor(self.output_image, cv2.COLOR_RGB2BGR))
                
    def __clear__showed_image__(self):
        """清除显示图片"""
        self.original_content_image_label.config(text="等待导入内容图片...")
        self.original_content_image_label.config(image='')
        self.original_content_image_label.image = None
                
    def mean_std(self, features):
        """输入 VGG16 计算的四个特征,输出每张
        特征图的均值和标准差,长度为1920"""
        mean_std_features = []
        for x in features:
            x = x.view(*x.shape[:2], -1)
            x = torch.cat([x.mean(-1), torch.sqrt(x.var(-1) +  1e-5)], dim=-1)
            n = x.shape[0]
            x2 = x.view(n, 2, -1).transpose(2, 1).contiguous().view(n, -1) # 【mean, ..., std, ...] to [mean, std, ...]
            mean_std_features.append(x2)
        mean_std_features = torch.cat(mean_std_features, dim=-1)
        return mean_std_features
    
    def preprocess_image(self,image, width=256):
        """输入 PIL.Image 风格图片对象,输出预处理后的四维tensor"""
        transform = transforms.Compose([
            transforms.Resize((width, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transform(image).unsqueeze(0).to(self.device)
        return image
    
    def postprocess_image(self,image):
        """输入 tensor 风格图片对象,输出预处理后的 PIL.Image"""
        image = image.squeeze(0).cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        return image
    
    
if __name__ == '__main__':
    # 测试代码
    root = tk.Tk()
    app = ImageStyleTransferApp(root)
    root.mainloop()