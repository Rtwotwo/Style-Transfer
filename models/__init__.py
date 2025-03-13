import cv2
import torch
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from models.MetaNet_Random_Style_Transfer import VGG, TransformNet, MetaNet
import numpy as np

# 设备选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
def load_models(model_name, device):
    vgg16 = models.vgg16(pretrained=True)
    vgg16 = VGG(vgg16.features[:23]).to(device).eval()
    
    transform_net = TransformNet(base=32).to(device)
    metanet = MetaNet(transform_net.get_param_dict()).to(device)
    
    transform_net.load_state_dict(torch.load(f'state_dict/metanet_base32_style50_tv1e-06_tagnohvd_transform_net.pth', map_location=device, weights_only=True))
    metanet.load_state_dict(torch.load(f'state_dict/metanet_base32_style50_tv1e-06_tagnohvd.pth', map_location=device, weights_only=True))
    
    return vgg16, transform_net, metanet


def mean_std(features):
    """输入 VGG16 计算的四个特征,输出每张特征图的均值和标准差,长度为1920"""
    mean_std_features = []
    for x in features:
        x = x.view(*x.shape[:2], -1)
        x = torch.cat([x.mean(-1), torch.sqrt(x.var(-1) +  1e-5)], dim=-1)
        n = x.shape[0]
        x2 = x.view(n, 2, -1).transpose(2, 1).contiguous().view(n, -1) # 【mean, ..., std, ...] to [mean, std, ...]
        mean_std_features.append(x2)
    mean_std_features = torch.cat(mean_std_features, dim=-1)
    return mean_std_features

# 图像预处理
def preprocess_image(image, width=256):
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((width, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = transform(image).unsqueeze(0).to(device)
    return image

# 图像后处理
def postprocess_image(image):
    image = image.squeeze(0).cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return image

# 实时风格迁移
def real_time_style_transfer(vgg16, transform_net, metanet, style_image_path, device):
    # 加载风格图像
    style_image = Image.open(style_image_path).convert('RGB')
    style_image = preprocess_image(style_image)
    
    # 计算风格图像的特征
    style_features = vgg16(style_image)
    style_mean_std = mean_std(style_features)
    
    # 生成风格模型的权重
    weights = metanet(style_mean_std)
    transform_net.set_weights(weights, 0)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 将 OpenCV 图像转换为 PIL 图像
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 预处理图像
        input_image = preprocess_image(pil_image)
        
        # 进行风格迁移
        with torch.no_grad():
            transformed_image = transform_net(input_image)
        
        # 后处理图像
        transformed_image = postprocess_image(transformed_image)
        
        # 将 PIL 图像转换回 OpenCV 图像
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        
        # 显示结果
        cv2.imshow('Original', frame)
        cv2.imshow('Stylized', transformed_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_name = 'metanet_base32_style50_tv1e-06_tagnohvd'
    vgg16, transform_net, metanet = load_models(model_name, device)
    
    style_image_path = 'style_image\style_2.jpg'
    real_time_style_transfer(vgg16, transform_net, metanet, style_image_path, device)