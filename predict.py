# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import time
import cv2
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from numba import jit, prange
from tools.config_dir import config_model_dir
from nets.U3D_OFFICIAL_MFF import UNet3D

parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('--predict_name', default='predict')
parser.add_argument('--model_path',
                    default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/3dcon/checkpoint/checkpoint.pth')
parser.add_argument('--stack_path',
                    default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/test_luo/coral_best_crop_stable_resize',
                    help='image stack path')
parser.add_argument('--out_format', default='jpg')
args = parser.parse_args()

@jit(nopython=True, parallel=True)
def compute_color_index(img_stack_np, y_output):
    H, W, D = img_stack_np.shape
    color_index = np.zeros((H, W), dtype=np.int64)
    for i in prange(H):
        for j in prange(W):
            min_diff = np.inf
            for d in range(D):
                diff = abs(int(img_stack_np[i, j, d]) - int(y_output[i, j]))
                if diff < min_diff:
                    min_diff = diff
                    color_index[i, j] = d
    return color_index

@jit(nopython=True)
def compute_color_img(y_output, cb_stack, cr_stack, color_index):
    H, W = y_output.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            color_img[i, j, 0] = y_output[i, j]
            color_img[i, j, 1] = cb_stack[i, j, color_index[i, j]]
            color_img[i, j, 2] = cr_stack[i, j, color_index[i, j]]
    return color_img

def stack_y_channels(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith(('.jpg', '.png', '.bmp'))]
    y_channels = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = img_ycc[:, :, 0]

        original_size = y.shape
        new_height = ((original_size[0] + 15) // 16) * 16
        new_width = ((original_size[1] + 15) // 16) * 16
        y = cv2.resize(y, (new_width, new_height))
        y_channels.append(y)

    return np.stack(y_channels, axis=-1), image_paths, original_size

img_stack_np, image_paths, ori_shape = stack_y_channels(args.stack_path)
H, W, depth = img_stack_np.shape

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ])

processed_stack = get_transform()
img_stack = processed_stack(img_stack_np)
img_stack = torch.unsqueeze(img_stack, 0)

predict_save_path = config_model_dir(subdir_name='predict_runs')

model = UNet3D()
model = nn.DataParallel(model)
if torch.cuda.device_count() > 1:
    model.cuda()
model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))

t_start = time.time()
model.eval()
with torch.no_grad():
    output = model(img_stack)
print(f"Inference time: {time.time() - t_start:.4f} seconds")

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())

y_output = cv2.resize(np.squeeze(tensor2uint(output)), (ori_shape[1], ori_shape[0]))

img_stack_np = cv2.resize(img_stack_np, (ori_shape[1], ori_shape[0]))

t_start = time.time()
color_index = compute_color_index(img_stack_np, y_output)
print(f"Color index computation time: {time.time() - t_start:.4f} seconds")

cb_channels = []
cr_channels = []
for img_path in image_paths:
    img = cv2.imread(img_path)
    img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cb, cr = img_ycc[:, :, 1], img_ycc[:, :, 2]
    cb_channels.append(cb)
    cr_channels.append(cr)

cb_stack = np.stack(cb_channels, axis=-1)
cr_stack = np.stack(cr_channels, axis=-1)

t_start = time.time()
color_img = compute_color_img(y_output, cb_stack, cr_stack, color_index)
print(f"Color image computation time: {time.time() - t_start:.4f} seconds")

rgb_img = cv2.cvtColor(color_img, cv2.COLOR_YCrCb2BGR)

cv2.imwrite(os.path.join(predict_save_path, f'result_color.{args.out_format}'), rgb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
print(f'Image is saved in {os.path.join(predict_save_path, f"result_color.{args.out_format}")}')
