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
parser.add_argument('--predict_name', default='predict_runs')
parser.add_argument('--model_path',
                    default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/3dcon/checkpoint/checkpoint.pth')
parser.add_argument('--stack_basedir_path',
                    default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/test_luo',
                    help='image stack path')
parser.add_argument('--out_format', default='png')
args = parser.parse_args()


def stack_y_channels(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith(('.jpg', '.png', '.bmp'))]
    y_channels = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = img_ycc[:, :, 0]

        original_size = y.shape
        height, width = original_size

        new_height = ((height + 15) // 16) * 16
        new_width = ((width + 15) // 16) * 16
        y = cv2.resize(y, (new_width, new_height))
        y_channels.append(y)
    y_channels = np.stack(y_channels, axis=-1)
    return y_channels, image_paths, original_size


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ])


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


stack_dir_list = os.listdir(args.stack_basedir_path)
print(stack_dir_list)
predict_save_path = config_model_dir(subdir_name=args.predict_name)
t_total = 0

# 准备模型
model = UNet3D()
model = nn.DataParallel(model)
if torch.cuda.device_count() > 1:
    model.cuda()
model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))
model.eval()

for stack_name in tqdm(stack_dir_list):
    stack_path = os.path.join(args.stack_basedir_path, stack_name)

    img_stack_np, image_paths, ori_shape = stack_y_channels(stack_path)

    H, W, depth = img_stack_np.shape

    processed_stack = get_transform()
    img_stack = processed_stack(img_stack_np)
    img_stack = torch.unsqueeze(img_stack, 0)

    t_start = time.time()

    # 开始推理
    with torch.no_grad():
        output = model(img_stack)
    t_total += time.time() - t_start

    y_output = np.squeeze(tensor2uint(output))

    # 保存Y通道
    y_save = cv2.resize(y_output, (ori_shape[1], ori_shape[0]))
    cv2.imwrite(os.path.join(predict_save_path, f'{stack_name}.{args.out_format}'), y_save)

    # 颜色信息通过Y通道内找最相近的值找到索引图像层,寻找颜色索引矩阵color_index
    target = y_output.astype(np.int64)
    img_stack_np = cv2.resize(img_stack_np, (W, H)).astype(np.int64)

    diff = np.abs(img_stack_np - target[:, :, np.newaxis])
    color_index = np.argmin(diff, axis=2)

    cb_channels = []
    cr_channels = []
    # 每一个像素都根据索引矩阵取原始Cb和Cr
    for img_path in image_paths:
        img = Image.open(img_path).convert('YCbCr')
        _, cb, cr = [np.array(img.split()[i]) for i in range(3)]
        cb = cv2.resize(cb, (W, H))
        cr = cv2.resize(cr, (W, H))
        cb_channels.append(cb)
        cr_channels.append(cr)

    cb_stack = np.stack(cb_channels, axis=-1)
    cr_stack = np.stack(cr_channels, axis=-1)

    # 使用高级索引来一次性获取所有像素的cb和cr值
    indices = np.indices((H, W))
    cb_values = cb_stack[indices[0], indices[1], color_index]
    cr_values = cr_stack[indices[0], indices[1], color_index]

    # 合成颜色图像
    color_img = np.stack([y_output, cb_values, cr_values], axis=-1).astype(np.int8)
    color_img = Image.fromarray(color_img, 'YCbCr')
    rgb_img = color_img.convert('RGB')
    rgb_img_save = rgb_img.resize((ori_shape[1], ori_shape[0]))

    # 保存图片
    rgb_img_save.save(os.path.join(predict_save_path, f'{stack_name}_rgb.{args.out_format}'), quality=100)

print('mean_time_each_stack:', t_total / len(stack_dir_list))
