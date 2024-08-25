# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import time
import cv2
import torch
import torch.nn as nn
import argparse
import os.path
from tools.config_dir import config_model_dir
from nets.U3D_OFFICIAL_MFF import UNet3D
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('--predict_name',default='predict_runs')
parser.add_argument('--model_path',default='checkpoint/checkpoint.pth')
parser.add_argument('--stack_basedir_path',default='data/Datasets_StackMFF/4D-Light-Field/image stack',help='image stack path')
parser.add_argument('--out_format',default='png')
args = parser.parse_args()

# 准备数据
def stack_y_channels(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
    y_channels = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = img_ycc[:, :, 0]

        #resize成16的倍数
        original_size = y.shape
        height, width = original_size

        new_height = ((height + 15) // 16) * 16
        new_width = ((width + 15) // 16) * 16
        y=cv2.resize(y,(new_width,new_height))
        y_channels.append(y)
    y_channels = np.stack(y_channels, axis=-1)
    return y_channels, image_paths,original_size

stack_dir_list=os.listdir(args.stack_basedir_path)
print(stack_dir_list)
predict_save_path=config_model_dir(subdir_name=args.predict_name)

for stack_name in tqdm(stack_dir_list):

    stack_path=os.path.join(args.stack_basedir_path,stack_name)

    img_stack_np,image_paths,ori_shape=stack_y_channels(stack_path)

    H, W, depth = img_stack_np.shape
    def get_transform():
        return transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.unsqueeze(0)),
            ])

    processed_stack = get_transform()
    img_stack = processed_stack(img_stack_np)
    img_stack=torch.unsqueeze(img_stack,0)

    #准备模型
    model=UNet3D()
    model=nn.DataParallel(model)
    if torch.cuda.device_count() > 1:
        model.cuda()
    model.load_state_dict(torch.load(args.model_path,map_location=lambda storage, loc: storage))

    #开始推理
    model.eval()
    output = model(img_stack)

    def tensor2uint(img):
        img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        return np.uint8((img * 255.0).round())
    y_output = np.squeeze(tensor2uint(output))

    #保存Y通道
    y_save=cv2.resize(y_output,(ori_shape[1],ori_shape[0]))
    cv2.imwrite(os.path.join(predict_save_path, '{}.{}'.format(stack_name,args.out_format)), y_save)

    #颜色信息通过Y通道内找最相近的值找到索引图像层,寻找颜色索引矩阵color_index
    target = np.array(y_output).astype(np.int64)
    img_stack_np=img_stack_np.astype(np.int64)

    dist_list=[]
    for depth_index in range(depth):
        dist = np.abs(img_stack_np[:,:,depth_index] - target)
        dist_list.append(dist)
    dist_list=np.stack(dist_list,-1)
    color_index=np.argmin(dist_list, axis=2)

    import matplotlib.pyplot as plt

    cb_channels=[]
    cr_channels=[]
    #每一个像素都根据索引矩阵取原始Cb和Cr
    for img_path in image_paths:
        img = Image.open(img_path).convert('YCbCr')
        y = np.array(img.split()[0])
        y = cv2.resize(y,(W,H))
        cb = np.array(img.split()[1])
        cb = cv2.resize(cb, (W, H))
        cr = np.array(img.split()[2])
        cr = cv2.resize(cr, (W, H))
        cb_channels.append(cb)
        cr_channels.append(cr)

    #每一像素逐一合成颜色
    color_img=np.zeros((H,W,3)).astype(np.int32)

    for i in range(H):
        for j in range(W):
            color_index_number=color_index[i,j]
            cb_pixel=cb_channels[color_index_number][i,j]
            cr_pixel = cr_channels[color_index_number][i, j]
            y_pixel=y_output[i,j]

            color_img[i, j, 0] = y_pixel
            color_img[i, j, 1] = cb_pixel
            color_img[i, j, 2] = cr_pixel

    color_img=color_img.astype(np.int8)
    color_img = Image.fromarray(color_img, 'YCbCr')
    rgb_img = color_img.convert('RGB')
    rgb_img_save=rgb_img.resize((ori_shape[1],ori_shape[0]))
    # 保存图片
    rgb_img_save.save(os.path.join(predict_save_path, '{}_rgb.{}'.format(stack_name,args.out_format)),quality=100)
