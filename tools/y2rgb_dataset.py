# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import os
import cv2
import numpy as np


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

A_dir = r'E:\pycharmproject\SwinFusion-master\Dataset\valsets\Lytro\A_Y'  # 源图像A的文件夹路径
B_dir = r'E:\pycharmproject\SwinFusion-master\Dataset\valsets\Lytro\B_Y'  # 源图像B的文件夹路径
Fused_dir = r'E:\pycharmproject\SwinFusion-master\results\SwinFusion_Lytro'  # 融合图像的Y通道的文件夹路径
save_dir = r'E:\pycharmproject\SwinFusion-master\results\trans'  # 彩色融合图像的文件夹路径

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

file_names = os.listdir(A_dir)
num = len(file_names)

for i in range(num):
    name_A = os.path.join(A_dir, file_names[i])
    name_B = os.path.join(B_dir, file_names[i])
    name_fused = os.path.join(Fused_dir, file_names[i])
    save_name = os.path.join(save_dir, file_names[i])

    image_A = cv2.imread(name_A)
    image_B = cv2.imread(name_B)
    I_result = cv2.imread(name_fused,cv2.IMREAD_UNCHANGED)

    Y1, Cb1, Cr1 = RGB2YCbCr(image_A)
    Y2, Cb2, Cr2 = RGB2YCbCr(image_B)

    H, W = Cb1.shape
    Cb = np.ones((H, W))
    Cr = np.ones((H, W))

    for k in range(H):
        for n in range(W):
            if abs(Cb1[k, n] - 128) == 0 and abs(Cb2[k, n] - 128) == 0:
                Cb[k, n] = 128
            else:
                middle_1 = Cb1[k, n] * abs(Cb1[k, n] - 128) + Cb2[k, n] * abs(Cb2[k, n] - 128)
                middle_2 = abs(Cb1[k, n] - 128) + abs(Cb2[k, n] - 128)
                Cb[k, n] = middle_1 / middle_2

            if abs(Cr1[k, n] - 128) == 0 and abs(Cr2[k, n] - 128) == 0:
                Cr[k, n] = 128
            else:
                middle_3 = Cr1[k, n] * abs(Cr1[k, n] - 128) + Cr2[k, n] * abs(Cr2[k, n] - 128)
                middle_4 = abs(Cr1[k, n] - 128) + abs(Cr2[k, n] - 128)
                Cr[k, n] = middle_3 / middle_4

    I_final_YCbCr = np.dstack((I_result, Cb, Cr))
    I_final_RGB = YCbCr2RGB(I_final_YCbCr)
    cv2.imwrite(save_name, I_final_RGB)

    print(save_name)