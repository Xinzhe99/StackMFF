# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import numpy as np
import cv2
import os

def Simulate_Dof(img,img_depth,num_regions,name):
    #parameters
    # img : ndarry
    # img_depth : ndarry
    # num_regions : int

    # 把一张图片，根据模糊等级，模糊N次
    imgs_blurred_list = []
    kernel_list= [2 * i + 1 for i in range(num_regions)]

    for i in kernel_list:
        img_blured=cv2.GaussianBlur(img,(i,i),0)
        imgs_blurred_list.append(img_blured)

    #根据depth的值来制作模糊蒙版
    # 生成0-255的参考划分点
    ref_points = np.linspace(0, 255, num_regions+1)

    # 进行量化,分段，得到一个矩阵，每个值表示该像素在的划分段
    quantized = np.digitize(img_depth, ref_points) - 1

    #每个蒙版都是在该分段的为True
    masks = []
    for i in range(num_regions):
        # 生成对应值的mask
        mask = (quantized == i)#mask中
        masks.append(mask)

    #按不同的mask取出模糊图像中清晰的部分，组成为一张图像
    sys_result=np.zeros_like(img)

    for index_mask,mask in enumerate(masks):
        #合成散焦图像
        for ind,mask in enumerate(masks):
            target_index=abs(ind-index_mask)
            sys_result[mask]=imgs_blurred_list[target_index][mask]
        # 保存blured结果
        cv2.imwrite(os.path.join(save_sys_path, name  + str(index_mask) + '.jpg'), sys_result)


num_regions = 16 # 分割区域
# 加载深度图
depth_path = r'C:\Users\dell\Desktop\Working\U3D-Fusion\major_revision_self\exp\datasets\depth.npy'
img_path=r'C:\Users\dell\Desktop\Working\U3D-Fusion\major_revision_self\exp\datasets\jonathan-borba-CnthDZXCdoY-unsplash.jpg'
save_sys_path = r'C:\Users\dell\Desktop\Working\U3D-Fusion\major_revision_self\exp\datasets'
img=cv2.imread(img_path)
img=cv2.resize(img,(520, 520))
print(img.shape)
depth = np.load(depth_path)
depth =cv2.resize(depth, (520, 520))
# 假设depth是一个浮点型数组，并且我们需要找到它的最大值来归一化
depth_max = np.max(depth)
# 归一化深度图到0-255，并转换为uint8类型
depth_normalized = (depth / depth_max * 255).astype(np.uint8)
print(depth_normalized.shape)

# # 显示图像
# cv2.imshow('Depth Map', depth_resized)
# cv2.waitKey(0)  # 等待任意键按下
# cv2.destroyAllWindows()  # 关闭所有窗口

Simulate_Dof(img,depth_normalized,16,'')