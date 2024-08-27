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
from tools.config_dir import config_model_dir
from nets.U3D_OFFICIAL_MFF import UNet3D
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('--predict_name', default='predict')
parser.add_argument('--model_path',
                    default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/3dcon/checkpoint/checkpoint.pth')
parser.add_argument('--stack_path',
                    default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/test_luo/coral_best_crop_stable_resize',
                    help='image stack path')
parser.add_argument('--out_format', default='jpg')
args = parser.parse_args()

class ImageStackDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                            f.endswith(('.jpg', '.png', '.bmp'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = img_ycc[:, :, 0]
        return y, img_ycc[:, :, 1], img_ycc[:, :, 2]

def stack_y_channels(folder_path):
    dataset = ImageStackDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    y_channels = []
    cb_channels = []
    cr_channels = []
    for y, cb, cr in dataloader:
        y_channels.append(y[0].numpy())
        cb_channels.append(cb[0].numpy())
        cr_channels.append(cr[0].numpy())

    y_stack = np.stack(y_channels, axis=-1)
    original_size = y_stack.shape[:2]
    new_height = ((original_size[0] + 15) // 16) * 16
    new_width = ((original_size[1] + 15) // 16) * 16
    y_stack = cv2.resize(y_stack, (new_width, new_height))
    return y_stack, cb_channels, cr_channels, original_size

img_stack_np, cb_channels, cr_channels, ori_shape = stack_y_channels(args.stack_path)
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

cb_stack = np.dstack(cb_channels)
cr_stack = np.dstack(cr_channels)

def fuse_images_vectorized(y_stack, cb_stack, cr_stack, fused_y):
    diff = np.abs(y_stack - fused_y[:, :, np.newaxis])
    color_index = np.argmin(diff, axis=2)
    fused_image = np.zeros((*fused_y.shape, 3), dtype=np.uint8)
    fused_image[:, :, 0] = fused_y
    fused_image[:, :, 1] = np.take_along_axis(cb_stack, color_index[:, :, np.newaxis], axis=2).squeeze()
    fused_image[:, :, 2] = np.take_along_axis(cr_stack, color_index[:, :, np.newaxis], axis=2).squeeze()
    return cv2.cvtColor(fused_image, cv2.COLOR_YCrCb2BGR)

t_start = time.time()
fused_rgb = fuse_images_vectorized(img_stack_np, cb_stack, cr_stack, y_output)
print(f"Color image computation time: {time.time() - t_start:.4f} seconds")
cv2.imwrite(os.path.join(predict_save_path, f'result_color.{args.out_format}'), fused_rgb)
