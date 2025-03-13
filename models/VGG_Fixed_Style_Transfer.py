"""
任务: 实现固定风格固定内容的普通风格迁移, 并且将代码封装为接口函数,
      输入epoches和image即可实现风格迁移.
时间: 2024/11/12-Redal
"""
import torch
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models


class ContentLoss(nn.Module):
    """定义内容损失函数"""
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    """定义风格损失函数"""
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    @staticmethod
    def gram_matrix(input):
        """计算Gram矩阵"""
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)


class Normalization(nn.Module):
    """定义归一化层"""
    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1).to(device)
        self.std = std.clone().detach().view(-1, 1, 1).to(device)
    def forward(self, img):
        return (img - self.mean) / self.std



class VGG_Fixed_Style_Transfer:
    """定义固定风格迁移模型"""
    def __init__(self, content_img, style_img, device, epoches=400):
        """初始化风格迁移相关参量"""
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),])
        self.run = None 
        self.content_img = self.transform(content_img).unsqueeze(0).to(device)
        self.style_img = self.transform(style_img).unsqueeze(0).to(device)
        self.input_img = self.content_img.clone().requires_grad_(True).to(device)

        self.device = device
        self.epoches = epoches
        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    def get_style_model_and_losses(self):
        normalization = Normalization(torch.tensor([0.485, 0.456, 0.406]), 
                    torch.tensor([0.229, 0.224, 0.225]), self.device).to(self.device)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization).to(self.device)

        i = 0
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)
            if name in self.content_layers_default:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers_default:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)].to(self.device)
        return model, style_losses, content_losses

    def run_style_transfer(self, style_weight=1000000, content_weight=1):
        print(f'====Building the style transfer model====')
        model, style_losses, content_losses = self.get_style_model_and_losses()
        model.eval().requires_grad_(False)
        optimizer = optim.LBFGS([self.input_img])

        self.run = [0]
        while self.run[0] <= self.epoches:            
            def closure():
                with torch.no_grad():
                    # 更正更新的输入图像的值
                    self.input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(self.input_img)
                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                style_score *= style_weight
                content_score *= content_weight
                loss = style_score + content_score
                loss.backward()
                self.run[0] += 1
                if self.run[0] % 50 == 0:
                    print(f"run {self.run[0]}: style loss: {style_score.item():.4f}, content loss: {content_score.item():.4f}")
                return style_score + content_score
            optimizer.step(closure)
        with torch.no_grad():
            self.input_img.clamp_(0, 1)
        return self.input_img



if __name__ == '__main__':
    # 测试函数封装代码
    style_img_path = r'style_image\style_1.png'
    content_img_path = r'content_image\content_3.jpeg'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'====Using device: {device}====')
    content_img = Image.open(content_img_path)
    style_img = Image.open(style_img_path)

    style_transfer = VGG_Fixed_Style_Transfer(content_img, style_img, device, epoches=800)
    output = style_transfer.run_style_transfer()
    img = output[0].cpu().detach().permute(1, 2, 0).numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save('output.jpg')
    plt.figure()
    plt.imshow(img)
    plt.show()