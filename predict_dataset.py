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
from tqdm import tqdm

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


# Numba-Accelerated Function to Compute Color Index
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


# Numba-Accelerated Function to Compute Color Image
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


# Function to Stack Y Channels from Images in a Folder
def stack_y_channels(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.lower().endswith(('.jpg', '.png', '.bmp'))]
    if not image_paths:
        raise ValueError(f"No images found in folder: {folder_path}")

    y_channels = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping.")
            continue
        img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = img_ycc[:, :, 0]

        original_size = y.shape
        new_height = ((original_size[0] + 15) // 16) * 16
        new_width = ((original_size[1] + 15) // 16) * 16
        y_resized = cv2.resize(y, (new_width, new_height))
        y_channels.append(y_resized)

    if not y_channels:
        raise ValueError(f"No valid images processed in folder: {folder_path}")

    return np.stack(y_channels, axis=-1), image_paths, original_size


# Function to Define Image Transformations
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ])


# Function to Convert Tensor to Unsigned Integer Image
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


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
        img_stack_np, image_paths, ori_shape = stack_y_channels(stack_path)
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
    color_index = compute_color_index(img_stack_np_resized, y_output_resized)
    print(f"Computed color index for stack '{stack_name}'")

    # Extract and Resize Cb and Cr Channels from All Images
    cb_channels = []
    cr_channels = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Using default Cb and Cr values.")
            cb = np.zeros((ori_shape[0], ori_shape[1]), dtype=np.uint8)
            cr = np.zeros((ori_shape[0], ori_shape[1]), dtype=np.uint8)
        else:
            img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            cb = cv2.resize(img_ycc[:, :, 1], (ori_shape[1], ori_shape[0]))
            cr = cv2.resize(img_ycc[:, :, 2], (ori_shape[1], ori_shape[0]))
        cb_channels.append(cb)
        cr_channels.append(cr)

    cb_stack = np.stack(cb_channels, axis=-1)
    cr_stack = np.stack(cr_channels, axis=-1)

    # Compute Color Image Using Numba
    color_img = compute_color_img(y_output_resized, cb_stack, cr_stack, color_index)
    print(f"Computed color image for stack '{stack_name}'")

    # Convert YCbCr to BGR Color Space
    color_img_bgr = cv2.cvtColor(color_img, cv2.COLOR_YCrCb2BGR)

    # Save Color Image
    color_save_path = os.path.join(predict_save_path, f'{stack_name}_rgb.{args.out_format}')
    cv2.imwrite(color_save_path, color_img_bgr)
    print(f"Saved color image to: {color_save_path}")

# Compute and Display Mean Inference Time
if stack_dir_list:
    mean_inference_time = t_total / len(stack_dir_list)
    print(f"Mean inference time per stack: {mean_inference_time:.4f} seconds")
else:
    print("No stacks were processed.")
