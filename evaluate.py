# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import numpy as np
import os
import pandas as pd
import cv2
import glob
from natsort import natsorted
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
#Parameters to set todo
base_path='data/diff_methods'
ground_truth_path='data/Datasets_StackMFF'
method_2_compare=["CVT","DWT","DCT","DTCWT","DSIFT","NSCT","IFCNN-MAX","U2Fusion","SDNet","MFF-GAN","SwinFusion","StackMFF"]
datasets=['4D-Light-Field','FlyingThings3D','Middlebury','Mobile Depth']
metrics=['MSE','MAE','RMSE','logRMS']

def get_image_formats(folder):
    formats = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            filename, ext = os.path.splitext(file)
            ext = ext[1:].lower() # remove .
            if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
                if ext not in formats:
                    formats.append(ext)
    return formats[0]
def read_image(path):
    img = cv2.imread(path, 0)
    if img is None:
        # 如果读取失败，尝试更改扩展名
        base, ext = os.path.splitext(path)
        new_ext = '.jpg' if ext.lower() == '.png' else '.png'
        new_path = base + new_ext
        img = cv2.imread(new_path, 0)
    return img

def MSE_function(A, F):
    A = A / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE = np.sum(np.sum((F - A)**2))/(m*n)

    return MSE


def RMSE_function(A, F):
    """
    计算两幅图像之间的均方根误差（RMSE）
    :param A: 图像 A 的像素值数组
    :param F: 图像 F 的像素值数组
    :return: 两幅图像之间的 RMSE
    """
    # 将像素值归一化到 [0, 1] 区间
    A = A / 255.0
    F = F / 255.0

    # 获取图像的尺寸
    m, n = F.shape

    # 计算均方误差（MSE）
    MSE = np.sum((F - A) ** 2) / (m * n)

    # 计算均方根误差（RMSE）
    RMSE = np.sqrt(MSE)

    return RMSE


def logRMS_function(A, F):
    """
    计算两幅图像之间的对数均方根误差（logRMS）
    :param A: 图像 A 的像素值数组
    :param F: 图像 F 的像素值数组
    :return: 两幅图像之间的 logRMS
    """
    # 将像素值归一化到 [0, 1] 区间
    A = A / 255.0
    F = F / 255.0

    # 获取图像的尺寸
    m, n = F.shape

    # 计算对数均方根误差（logRMS）
    logRMS = np.sqrt(np.sum((np.log1p(F) - np.log1p(A)) ** 2) / (m * n))

    return logRMS


def MAE_function(A, F):
    # 确保输入是 numpy 数组
    A = np.array(A, dtype=np.float64)
    F = np.array(F, dtype=np.float64)

    # 将像素值归一化到 [0, 1] 范围
    A = A / 255.0
    F = F / 255.0

    # 获取图像尺寸
    m, n = F.shape

    # 计算绝对误差的和
    absolute_diff = np.abs(F - A)

    # 计算平均绝对误差
    MAE = np.sum(absolute_diff) / (m * n)

    return MAE
if __name__ == '__main__':
    #获取groundtruth的图片路径
    for dataset in datasets:
        print('testing:',dataset)
        ground_truth_img=natsorted(glob.glob(os.path.join(ground_truth_path,dataset,'AiF','*.png')))
        method_results = {}
        df = pd.DataFrame(columns=(['Method']+metrics))
        #遍历方法
        for method_index,method in enumerate(method_2_compare):
            #获取该方法结果的所有图片的路径
            ext='*.'+get_image_formats(os.path.join(base_path,method,dataset))
            img_path_list=natsorted(glob.glob(os.path.join(base_path,method,dataset,ext)))
            metric_results = {}
            #遍历里面的每张图片
            metric_dict={}
            for metric in metrics:
                metric_dict[metric] = 0  # 初始化为0或空值
            for img_truth_ind,img_truth in tqdm(enumerate(ground_truth_img)):

                img_name=os.path.basename(img_truth)
                img_truth = cv2.imread(img_truth, 0)
                result_path = os.path.join(base_path, method, dataset, img_name)
                img_result = read_image(result_path)
                #每张图片都要计算它的不同指标，并累计加上去
                #遍历所有的评测指标
                for metric_index,metric in enumerate(metrics):


                    if metric=='MSE':
                        value= MSE_function(img_result,img_truth)

                    if metric=='MAE':
                        value= MAE_function(img_result,img_truth)

                    if metric=='RMSE':
                        value= RMSE_function(img_result,img_truth)

                    if metric=='logRMS':
                        value= logRMS_function(img_result,img_truth)

                    metric_dict[metric] += value

            for metric in metric_dict:
                metric_dict[metric] /= len(img_path_list)
                metric_dict[metric]=round(metric_dict[metric],4)

            method_results[method]=metric_dict


        for key, value in method_results.items():
            print(key, ':', value)
            temp = {'Method': key}
            new_row = {**temp,**value}
            df = df.append(new_row, ignore_index=True)
            print(df)

        df.to_excel('compare result_{}.xlsx'.format(dataset))
