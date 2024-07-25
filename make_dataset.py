# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import glob
import cv2
import numpy as np
import os
from tqdm import tqdm

#模拟景深
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

    current_sys_folder = os.path.join(save_sys_path, name)
    current_mask_folder = os.path.join(save_mask_path, name)
    os.makedirs(current_sys_folder, exist_ok=True)
    os.makedirs(current_mask_folder, exist_ok=True)

    for index_mask,mask in enumerate(masks):
        mask_result = mask.astype(np.uint8) * 255
        # 保存mask结果
        cv2.imwrite(os.path.join(current_mask_folder, f'{index_mask}.png'), mask_result)

        #合成散焦图像
        for ind,mask in enumerate(masks):
            target_index=abs(ind-index_mask)
            sys_result[mask]=imgs_blurred_list[target_index][mask]

        # 保存blured结果
        cv2.imwrite(os.path.join(current_sys_folder, f'{index_mask}.jpg'), sys_result)


num_regions = 16 # 分割区域
# 源数据集地址
# 训练的原图
ori_train_dataset_path = 'data/OpenImagesV7/train'
# 训练的深度图
depth_train_dataset_path = 'data/OpenImagesV7/train_depth'

# 测试的原图
ori_test_dataset_path = 'data/OpenImagesV7/val'
# 测试的深度图
depth_test_dataset_path = 'data/OpenImagesV7/val_depth'

# 保存地址
sys_train = '/data/Datasets_train_StackMFF/train_stack'
sys_train_mask = '/data/Datasets_train_StackMFF/train_mask'
sys_val = '/data/Datasets_train_StackMFF/test_stack'
sys_val_mask = '/data/Datasets_train_StackMFF/test_mask'

if not os.path.exists(sys_train):  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(sys_train)
if not os.path.exists(sys_val):  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(sys_val)
if not os.path.exists(sys_train_mask):  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(sys_train_mask)
if not os.path.exists(sys_val_mask):  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(sys_val_mask)

# 获取所有源图像
train_ori_list = glob.glob(os.path.join(ori_train_dataset_path, '*.jpg'))
test_ori_list = glob.glob(os.path.join(ori_test_dataset_path, '*.jpg'))


#训练集和测试集 训练为0测试为1
for dataset_class in range(2):
    if dataset_class ==0:
        print('开始制作训练集')
        dataset_list_path=train_ori_list
        dataset_depth_path=depth_train_dataset_path
        save_sys_path=sys_train
        save_mask_path = sys_train_mask
    elif dataset_class==1:
        print('开始制作验证集')
        dataset_list_path = test_ori_list
        dataset_depth_path=depth_test_dataset_path
        save_sys_path = sys_val
        save_mask_path = sys_val_mask

    #训练/验证
    for index,pic_path in tqdm(enumerate(dataset_list_path)):
        filename = os.path.basename(pic_path)
        name, ext = os.path.splitext(filename)
        img = cv2.imread(pic_path)

        img_depth=cv2.imread(os.path.join(dataset_depth_path,name+'.png'),0)
        #simulate DOF
        Simulate_Dof(img, img_depth, num_regions,name)
