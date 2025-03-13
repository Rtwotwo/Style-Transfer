"""
任务: 实现使用TransformNet和MetaNet训练风格迁移模型训练函数
      对数据集进行训练,并且做数据的tensorboard可视化
时间: 2024/11/14-Redal
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from glob import glob
from tensorboardX import SummaryWriter
import numpy as np
import copy
from tqdm import tqdm
from collections import defaultdict
from models import *


is_hvd = False
tag = 'nohvd'
base = 32
style_weight = 50
content_weight = 1
tv_weight = 1e-6
epochs = 22
batch_size = 8
width = 256
verbose_hist_batch = 100
verbose_image_batch = 800
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = f'metanet_base{base}_style{style_weight}_tv{tv_weight}_tag{tag}'
print(f'model_name: {model_name}')


class CustomDataset(torch.utils.data.Dataset):
    """自定义数据迭代器加载数据"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error opening image at path {img_path}: {e}")
            dummy_image = Image.new('RGB', (256, 256), color='white')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image


class Smooth:
    # 对输入的数据进行滑动平均
    def __init__(self, windowsize=100):
        self.window_size = windowsize
        self.data = np.zeros((self.window_size, 1), dtype=np.float32)
        self.index = 0
    def __iadd__(self, x):
        if self.index == 0:
            self.data[:] = x
        self.data[self.index % self.window_size] = x
        self.index += 1
        return self
    def __float__(self):
        return float(self.data.mean())
    def __format__(self, f):
        return self.__float__().__format__(f)


vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()
transform_net = TransformNet(base).to(device)
transform_net.get_param_dict()
metanet = MetaNet(transform_net.get_param_dict()).to(device)


data_transform = transforms.Compose([
    transforms.RandomResizedCrop(width, scale=(256/480, 1), ratio=(1, 1)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
style_dataset = CustomDataset(r"/data2/dataset/WikiArt/Style", transform=data_transform)
content_dataset = CustomDataset(r"/data2/dataset/WikiArt/Content", transform=data_transform)
content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
style_data_loader = torch.utils.data.DataLoader(style_dataset, batch_size=1, shuffle=True, num_workers=0)
print(style_dataset)
print('-'*20)
print(content_dataset)


visualization_style_image = next(iter(style_data_loader)).to(device)
visualization_content_images = torch.stack([next(iter(content_data_loader)) for _ in range(4)]).to(device)
writer = SummaryWriter('runs/'+model_name)
del visualization_style_image, visualization_content_images


optimizer = optim.Adam(list(transform_net.parameters()) + list(metanet.parameters()), 1e-3)
n_batch = len(content_data_loader)
metanet.train()
transform_net.train()

for epoch in range(epochs):
    smoother = defaultdict(Smooth)
    with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:
        for batch, content_images in pbar:
            n_iter = epoch * n_batch + batch
            # 每 20 个 batch 随机挑选一张新的风格图像，计算其特征
            if batch % 20 == 0:
                style_image = next(iter(style_data_loader)).to(device)
                style_features = vgg16(style_image)
                style_mean_std = mean_std(style_features)
            # 检查纯色
            x = content_images.cpu().numpy()
            if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                continue
            
            optimizer.zero_grad()
            # 使用风格图像生成风格模型
            weights = metanet(mean_std(style_features))
            transform_net.set_weights(weights, 0)
            # 使用风格模型预测风格迁移图像
            content_images = content_images.to(device)   
            transformed_images = transform_net(content_images)
            # 使用 vgg16 计算特征
            content_features = vgg16(content_images)
            transformed_features = vgg16(transformed_images)
            transformed_mean_std = mean_std(transformed_features)

            content_loss = content_weight * F.mse_loss(transformed_features[2], content_features[2])
            style_loss = style_weight * F.mse_loss(transformed_mean_std, 
                                    style_mean_std.expand_as(transformed_mean_std))
            y = transformed_images
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + 
                                    torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))
            loss = content_loss + style_loss + tv_loss 
            
            loss.backward()
            optimizer.step()
            
            smoother['content_loss'] += content_loss.item()  
            smoother['style_loss'] += style_loss.item()
            smoother['tv_loss'] += tv_loss.item()
            smoother['loss'] += loss.item()

            max_value = max([x.max().item() for x in weights.values()])
            writer.add_scalar('loss/loss', loss, n_iter)
            writer.add_scalar('loss/content_loss', content_loss, n_iter)
            writer.add_scalar('loss/style_loss', style_loss, n_iter)
            writer.add_scalar('loss/total_variation', tv_loss, n_iter)
            writer.add_scalar('loss/max', max_value, n_iter)
            
            s = 'Epoch: {} '.format(epoch+1)
            s += 'Content: {:.2f} '.format(smoother['content_loss'])
            s += 'Style: {:.1f} '.format(smoother['style_loss'])
            s += 'Loss: {:.2f} '.format(smoother['loss'])
            s += 'Max: {:.2f}'.format(max_value)
            
            pbar.set_description(s)
            del transformed_images, weights
    
    if not os.path.exists('models'):      
        os.makedirs('models')   
    if not os.path.exists('checkpoints'):  
        os.makedirs('checkpoints')

    # torch.save(metanet.state_dict(), 'checkpoints/{}_{}.pth'.format(model_name, epoch+1))
    # torch.save(transform_net.state_dict(), 'checkpoints/{}_transform_net_{}.pth'.format(model_name, epoch+1))
    torch.save(metanet.state_dict(), 'state_dict/{}.pth'.format(model_name))
    torch.save(transform_net.state_dict(), 'state_dict/{}_transform_net.pth'.format(model_name))
# 关闭TensorBoard记录
writer.close()