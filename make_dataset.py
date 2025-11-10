#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Function: Generate multi-focus image stacks by simulating depth of field (DOF) effects.
This script takes original images and their corresponding depth maps to create a series of 
images with different focus regions, simulating the depth of field effect.

Input Structure:
- Original images (jpg format) in a specified directory
- Corresponding depth maps (png format) in another directory
  Depth maps should have the same name as the original images but with .png extension

Output Structure:
- Stack images (jpg format) saved in the specified output directory
- Mask images (png format) saved in the specified mask output directory
  Each stack contains N images with different focus regions
  Each mask corresponds to a focus region in the stack

Usage Example:
python make_dataset.py --ori_dataset_path path/to/original/images 
                      --depth_dataset_path path/to/depth/maps 
                      --sys_path path/to/save/stacks 
                      --sys_mask_path path/to/save/masks 
                      --num_regions 16
"""

import glob
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse


def simulate_dof(img, img_depth, num_regions, name, save_sys_path, save_mask_path):
    """Simulate depth of field effect to generate multi-focus image stacks.
    
    Args:
        img: Original image (ndarray)
        img_depth: Depth map of the image (ndarray)
        num_regions: Number of focus regions to generate
        name: Name of the image (used for saving)
        save_sys_path: Path to save the generated stack images
        save_mask_path: Path to save the mask images
    """
    # Resize images to 384x384
    img = cv2.resize(img, (384, 384))
    img_depth = cv2.resize(img_depth, (384, 384))
    
    # Blur the image with different kernel sizes
    imgs_blurred_list = []
    kernel_list = [2 * i + 1 for i in range(num_regions)]
    
    for i in kernel_list:
        img_blured = cv2.GaussianBlur(img, (i, i), 0)
        imgs_blurred_list.append(img_blured)
    
    # Create blur masks based on depth values
    # Generate reference points for quantization
    ref_points = np.linspace(0, 255, num_regions + 1)
    
    # Quantize depth map into regions
    quantized = np.digitize(img_depth, ref_points) - 1
    
    # Generate masks for each region
    masks = []
    for i in range(num_regions):
        mask = (quantized == i)
        masks.append(mask)
    
    # Synthesize defocused images
    sys_result = np.zeros_like(img)
    
    current_sys_folder = os.path.join(save_sys_path, name)
    current_mask_folder = os.path.join(save_mask_path, name)
    os.makedirs(current_sys_folder, exist_ok=True)
    os.makedirs(current_mask_folder, exist_ok=True)
    
    for index_mask, mask in enumerate(masks):
        mask_result = mask.astype(np.uint8) * 255
        # Save mask result
        cv2.imwrite(os.path.join(current_mask_folder, f'{index_mask}.png'), mask_result)
        
        # Synthesize defocused image
        for ind, mask_item in enumerate(masks):
            target_index = abs(ind - index_mask)
            sys_result[mask_item] = imgs_blurred_list[target_index][mask_item]
        
        # Save blurred result
        cv2.imwrite(os.path.join(current_sys_folder, f'{index_mask}.jpg'), sys_result)


def main():
    parser = argparse.ArgumentParser(description='Generate multi-focus image stacks by simulating depth of field effects.')
    parser.add_argument('--ori_dataset_path', type=str, required=True, 
                        help='Path to the original images directory')
    parser.add_argument('--depth_dataset_path', type=str, required=True, 
                        help='Path to the depth maps directory')
    parser.add_argument('--sys_path', type=str, required=True, 
                        help='Path to save the generated image stacks')
    parser.add_argument('--sys_mask_path', type=str, required=True, 
                        help='Path to save the mask images')
    parser.add_argument('--num_regions', type=int, default=16, 
                        help='Number of focus regions to generate (default: 16)')
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    if not os.path.exists(args.sys_path):
        os.makedirs(args.sys_path)
    if not os.path.exists(args.sys_mask_path):
        os.makedirs(args.sys_mask_path)
    
    # Get all original images
    image_list = glob.glob(os.path.join(args.ori_dataset_path, '*.jpg'))
    
    print('Start generating dataset...')
    
    # Process all images
    for index, pic_path in tqdm(enumerate(image_list)):
        filename = os.path.basename(pic_path)
        name, ext = os.path.splitext(filename)
        img = cv2.imread(pic_path)
        
        img_depth = cv2.imread(os.path.join(args.depth_dataset_path, name + '.png'), 0)
        # Simulate DOF
        simulate_dof(img, img_depth, args.num_regions, name, args.sys_path, args.sys_mask_path)


if __name__ == '__main__':
    main()
