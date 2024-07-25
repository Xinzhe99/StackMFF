# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import glob
import os.path
from torch.nn import functional as F
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    #训练的所有文件夹
    train_stack_path = os.path.join(filepath, 'train_stack')
    subfolders = [f for f in os.listdir(train_stack_path) if os.path.isdir(os.path.join(train_stack_path, f))]
    train_stack = []
    for subfolder in subfolders:
        train_stack.append(os.path.join(train_stack_path, subfolder))

    # 验证的所有文件夹
    test_stack_path = os.path.join(filepath, 'test_stack')
    subfolders = [f for f in os.listdir(test_stack_path) if os.path.isdir(os.path.join(test_stack_path, f))]
    test_stack = []
    for subfolder in subfolders:
        test_stack.append(os.path.join(test_stack_path, subfolder))

    #训练标签
    train_label_img=glob.glob(os.path.join(filepath, 'train_lable','*.jpg'))
    #验证标签
    test_label_img=glob.glob(os.path.join(filepath, 'test_lable','*.jpg'))

    return train_stack, train_label_img,test_stack,test_label_img

def stack_y_channels(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith('.jpg') or f.endswith('.png')]

    y_channels = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('YCbCr')
        y = np.array(img.split()[0])
        y_channels.append(y)
    random.shuffle(y_channels)#手动打乱，数据增强
    y_channels = np.stack(y_channels, axis=-1)
    return y_channels

def default_loader(path,mode='stack',padding=0):
    if mode=='img':
        y_channel=np.array(Image.open(path).convert('YCbCr').split()[0])
        h=y_channel.shape[0]
        w = y_channel.shape[1]

        padding_matrix=np.zeros((h, w,padding))
        padding_matrix[:,:,0]=y_channel
        y_channel_padding=padding_matrix

        return y_channel_padding#yCbCr

    elif mode=='stack':
        return stack_y_channels(path)
    else:
        print('Please check mode!')

def get_transform():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

class myImageFloder(data.Dataset):
    def __init__(self, img_stack,label_img,loader=default_loader):

        self.img_stack = img_stack
        self.label_img = label_img
        self.loader = loader

    def __getitem__(self, index):
        img_stack = self.img_stack[index]
        label_img = self.label_img[index]


        img_stack = self.loader(img_stack,mode='stack')

        #需要padding图片到图像栈一样的维度
        padding_channel=img_stack.shape[2]

        label_img = self.loader(label_img, mode='img',padding=padding_channel)

        processed= get_transform()

        img_stack = processed(img_stack)
        label_img = processed(label_img)

        return img_stack, label_img

    def __len__(self):
        return len(self.label_img)


