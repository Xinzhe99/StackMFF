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
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Import your custom modules
from tools.config_dir import config_model_dir
from nets.U3D_OFFICIAL_MFF import UNet3D

# Argument Parsing
parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('--predict_name', default='predict_runs', help='Name for the prediction run directory')
parser.add_argument('--model_path',
                    default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/3dcon/checkpoint/checkpoint.pth',
                    help='Path to the trained model checkpoint')
parser.add_argument('--stack_basedir_path',
                    default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/test_luo',
                    help='Base directory containing image stacks')
parser.add_argument('--out_format', default='jpg', help='Output image format (e.g., jpg, png)')
args = parser.parse_args()

class ImageStackDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                            f.lower().endswith(('.jpg', '.png', '.bmp'))]
        if not self.image_paths:
            raise ValueError(f"No images found in folder: {folder_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Unable to read image {img_path}")
        img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = img_ycc[:, :, 0]
        cb = img_ycc[:, :, 1]
        cr = img_ycc[:, :, 2]
        return y, cb, cr

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

def fuse_images_vectorized(y_stack, cb_stack, cr_stack, fused_y):
    diff = np.abs(y_stack - fused_y[:, :, np.newaxis])
    color_index = np.argmin(diff, axis=2)
    fused_image = np.zeros((*fused_y.shape, 3), dtype=np.uint8)
    fused_image[:, :, 0] = fused_y
    fused_image[:, :, 1] = np.take_along_axis(cb_stack, color_index[:, :, np.newaxis], axis=2).squeeze()
    fused_image[:, :, 2] = np.take_along_axis(cr_stack, color_index[:, :, np.newaxis], axis=2).squeeze()
    return cv2.cvtColor(fused_image, cv2.COLOR_YCrCb2BGR)

# Retrieve List of Stack Directories
stack_dir_list = [d for d in os.listdir(args.stack_basedir_path) if
                  os.path.isdir(os.path.join(args.stack_basedir_path, d))]
print(f"Found {len(stack_dir_list)} stacks to process.")

# Prepare Output Directory
predict_save_path = config_model_dir(subdir_name=args.predict_name)
os.makedirs(predict_save_path, exist_ok=True)

# Initialize Total Inference Time
t_total = 0

# Model Preparation
model = UNet3D()
model = nn.DataParallel(model)
if torch.cuda.device_count() > 1:
    model.cuda()
model.load_state_dict(
    torch.load(args.model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.eval()

# Processing Each Stack
for stack_name in tqdm(stack_dir_list, desc='Processing Stacks'):
    stack_path = os.path.join(args.stack_basedir_path, stack_name)

    try:
        img_stack_np, cb_channels, cr_channels, ori_shape = stack_y_channels(stack_path)
    except ValueError as e:
        print(e)
        continue

    H, W, depth = img_stack_np.shape

    # Apply Transformations
    transform = get_transform()
    img_stack = transform(img_stack_np)
    img_stack = torch.unsqueeze(img_stack, 0)  # Add batch dimension
    if torch.cuda.is_available():
        img_stack = img_stack.cuda()

    # Inference
    t_start = time.time()
    with torch.no_grad():
        output = model(img_stack)
    inference_time = time.time() - t_start
    t_total += inference_time
    print(f"Inference time for stack '{stack_name}': {inference_time:.4f} seconds")

    # Convert Model Output to Unsigned Integer Image
    y_output = np.squeeze(tensor2uint(output))
    y_output_resized = cv2.resize(y_output, (ori_shape[1], ori_shape[0]))

    # Save Y Channel as Grayscale Image
    y_save_path = os.path.join(predict_save_path, f'{stack_name}_y.{args.out_format}')
    cv2.imwrite(y_save_path, y_output_resized)
    print(f"Saved Y channel image to: {y_save_path}")

    # Compute Color Index Using Numba
    img_stack_np_resized = cv2.resize(img_stack_np, (ori_shape[1], ori_shape[0]))

    cb_stack = np.dstack(cb_channels)
    cr_stack = np.dstack(cr_channels)

    # Compute Color Image Using Numba
    color_img = fuse_images_vectorized(img_stack_np_resized, cb_stack, cr_stack, y_output_resized)
    print(f"Computed color image for stack '{stack_name}'")

    # Save Color Image
    color_save_path = os.path.join(predict_save_path, f'{stack_name}.{args.out_format}')
    cv2.imwrite(color_save_path, color_img)
    print(f"Saved color image to: {color_save_path}")

# Compute and Display Mean Inference Time
if stack_dir_list:
    mean_inference_time = t_total / len(stack_dir_list)
    print(f"Mean inference time per stack: {mean_inference_time:.4f} seconds")
else:
    print("No stacks were processed.")
