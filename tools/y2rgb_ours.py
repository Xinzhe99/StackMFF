# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import os
import cv2
import numpy as np
from PIL import Image

stack_basedir_path=r'E:\ImagefusionDatasets\test\test_2'
fusion_y=r'E:\ImagefusionDatasets\test_result\test_2.jpg'
save_path=r'E:\ImagefusionDatasets\test_result\test_2_rgb.jpg'

def RGB2YCbCr(img_rgb):
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    # RGB to YCbCr
    Y = 0.257 * R + 0.564 * G + 0.098 * B + 16
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128

    return Y, Cb, Cr


def YCbCr2RGB(img_YCbCr):
    Y = img_YCbCr[:, :, 0]
    Cb = img_YCbCr[:, :, 1]
    Cr = img_YCbCr[:, :, 2]

    # YCbCr to RGB
    R = 1.164 * (Y - 16) + 1.596 * (Cr - 128)
    G = 1.164 * (Y - 16) - 0.392 * (Cb - 128) - 0.813 * (Cr - 128)
    B = 1.164 * (Y - 16) + 2.017 * (Cb - 128)

    image_RGB = np.dstack((R, G, B))
    return image_RGB

def stack_y_channels(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith('.jpg') or f.endswith('.png')]

    y_channels = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('YCbCr')
        y = np.array(img.split()[0])
        y_channels.append(y)

    y_channels = np.stack(y_channels, axis=-1)

    # if y_channels.shape[-1] < 7:
    #     pad_width = ((0, 0), (0, 0), (0, 16 - y_channels.shape[-1]))
    #     y_channels = np.pad(y_channels, pad_width, mode='constant', constant_values=0)

    return y_channels,image_paths

stack_dir_list=os.listdir(stack_basedir_path)
fusion_y=cv2.imread(fusion_y,0)
#颜色信息通过Y通道内找最相近的值找到索引图像层,寻找颜色索引矩阵color_index
target = np.array(fusion_y).astype(np.int64)

img_stack_np,image_paths=stack_y_channels(stack_basedir_path)
H, W, depth = img_stack_np.shape
img_stack_np=img_stack_np.astype(np.int64)

dist_list=[]
for depth_index in range(depth):
    dist = np.abs(img_stack_np[:,:,depth_index] - target)
    dist_list.append(dist)
dist_list=np.stack(dist_list,-1)
color_index=np.argmin(dist_list, axis=2)
color_index_smoothed= cv2.GaussianBlur(color_index.astype(np.uint8), (11, 11), 0)
import matplotlib.pyplot as plt
# plt.imsave(os.path.join(predict_save_path, 'index depth.{}'.format(args.out_format)), color_index)
# plt.imsave(os.path.join(predict_save_path, 'smooth index depth.{}'.format(args.out_format)), color_index_smoothed)
cb_channels=[]
cr_channels=[]
#每一个像素都根据索引矩阵取原始Cb和Cr
for img_path in image_paths:
    img = Image.open(img_path).convert('YCbCr')
    y = np.array(img.split()[0])
    cb = np.array(img.split()[1])
    cr = np.array(img.split()[2])
    cb_channels.append(cb)
    cr_channels.append(cr)

#每一像素逐一合成颜色
color_img=np.zeros((H,W,3)).astype(np.int32)
for i in range(H):
    for j in range(W):
        color_index_number=color_index[i,j]
        cb_pixel=cb_channels[color_index_number][i,j]
        cr_pixel = cr_channels[color_index_number][i, j]
        y_pixel=fusion_y[i,j]

        color_img[i, j, 0] = y_pixel
        color_img[i, j, 1] = cb_pixel
        color_img[i, j, 2] = cr_pixel

color_img=color_img.astype(np.int8)

color_img = Image.fromarray(color_img, 'YCbCr')
rgb_img = color_img.convert('RGB')

# 保存图片
rgb_img.save(save_path,quality=100)