# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
# @Description : This script is used for multi-focus image fusion prediction using the StackMFF model.
#                It takes a stack of focused images as input and produces a fused image as output.
#                The output image will be saved in the 'results' folder with the same name as the input folder.
#
# @Usage Example:
#   python predict.py --stack_path path/to/image/stack --model_path weights/checkpoint.pth --out_format png
#   python predict.py --stack_path path/to/image/stack --model_path weights/checkpoint.pth  # Auto-detect format
#   python predict.py --stack_path path/to/image/stack  # Use default model and auto-detect format

import time
import cv2
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from stackmff import StackMFF
from torch.utils.data import Dataset, DataLoader

def main(args):
    # Validate input arguments
    if not os.path.exists(args.stack_path):
        raise FileNotFoundError(f"Stack path does not exist: {args.stack_path}")
    
    if not os.path.isdir(args.stack_path):
        raise NotADirectoryError(f"Stack path is not a directory: {args.stack_path}")
    
    # Determine output format - use specified format or auto-detect from input
    if args.out_format and args.out_format.lower() != 'none':
        output_format = args.out_format.lower()
    else:
        # Auto-detect format from input images
        output_format = get_input_image_format(args.stack_path)
        if output_format is None:
            # Fallback to jpg if no valid images found
            output_format = 'jpg'
    
    # Validate output format
    valid_formats = ['jpg', 'jpeg', 'png', 'bmp']
    if output_format not in valid_formats:
        raise ValueError(f"Invalid output format: {output_format}. Valid formats are: {valid_formats}")

    class ImageStackDataset(Dataset):
        def __init__(self, folder_path):
            # Support more image formats
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                                f.lower().endswith(valid_extensions)]
            
            if len(self.image_paths) == 0:
                raise ValueError(f"No valid images found in folder: {folder_path}")

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            # Add error handling for image reading
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Could not read image: {img_path}")
                img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                y = img_ycc[:, :, 0]
                return y, img_ycc[:, :, 1], img_ycc[:, :, 2]
            except Exception as e:
                raise RuntimeError(f"Error processing image {img_path}: {str(e)}")

    def stack_y_channels(folder_path):
        try:
            dataset = ImageStackDataset(folder_path)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            y_channels = []
            cb_channels = []
            cr_channels = []
            for y, cb, cr in dataloader:
                y_channels.append(y[0].numpy())
                cb_channels.append(cb[0].numpy())
                cr_channels.append(cr[0].numpy())

            if len(y_channels) == 0:
                raise ValueError("No valid images found in the folder")
            
            y_stack = np.stack(y_channels, axis=-1)
            original_size = y_stack.shape[:2]
            
            # Validate dimensions
            if len(original_size) != 2:
                raise ValueError(f"Invalid image dimensions: {original_size}")
                
            new_height = ((original_size[0] + 15) // 16) * 16
            new_width = ((original_size[1] + 15) // 16) * 16
            
            # Add boundary checks
            if new_height <= 0 or new_width <= 0:
                raise ValueError(f"Invalid resized dimensions: height={new_height}, width={new_width}")
            
            if new_height != original_size[0] or new_width != original_size[1]:
                y_stack = cv2.resize(y_stack, (new_width, new_height))
            return y_stack, cb_channels, cr_channels, original_size
        except Exception as e:
            raise RuntimeError(f"Error in stack_y_channels: {str(e)}")

    try:
        img_stack_np, cb_channels, cr_channels, ori_shape = stack_y_channels(args.stack_path)
    except Exception as e:
        raise RuntimeError(f"Failed to process image stack: {str(e)}")
    
    H, W, depth = img_stack_np.shape

    def get_transform():
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    processed_stack = get_transform()
    img_stack_tensor = processed_stack(img_stack_np)
    # Ensure img_stack_tensor is torch.Tensor type
    if not isinstance(img_stack_tensor, torch.Tensor):
        img_stack_tensor = torch.tensor(img_stack_tensor)

    # Add batch dimension and depth dimension, StackMFF expects input shape (B, 1, D, H, W)
    img_stack_tensor = torch.unsqueeze(img_stack_tensor, 0)  # Add batch dimension
    img_stack_tensor = torch.unsqueeze(img_stack_tensor, 0)  # Add channel dimension

    # Create results save directory - directly in results folder
    results_dir = os.path.join('.', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Use the stack folder name as the output file name
    stack_folder_name = os.path.basename(os.path.normpath(args.stack_path))
    # Remove any file extension if present (shouldn't be, but just in case)
    stack_folder_name = os.path.splitext(stack_folder_name)[0]

    model = StackMFF()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Enhanced model loading with better error handling
    if args.model_path and os.path.exists(args.model_path):
        try:
            # Load model weights
            state_dict = torch.load(args.model_path, map_location=device)
            
            # Handle DataParallel module prefix
            # Check if the state_dict keys start with 'module.'
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    # Remove 'module.' prefix
                    new_state_dict[k[7:]] = v
                else:
                    # Keep the key as is
                    new_state_dict[k] = v
            
            # Try to load the state dict
            model.load_state_dict(new_state_dict)
            print(f"Successfully loaded model from {args.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {args.model_path}: {str(e)}")
    else:
        print(f"Warning: Model path {args.model_path} does not exist. Using untrained model.")

    t_start = time.time()
    model.eval()
    with torch.no_grad():
        try:
            img_stack_tensor = img_stack_tensor.to(device)
            output = model(img_stack_tensor)
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {str(e)}")
    print(f"Inference time: {time.time() - t_start:.4f} seconds")

    def tensor2uint(img):
        img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        return np.uint8((img * 255.0).round())

    # Resize output to match original size
    output_uint = tensor2uint(output)
    # Convert numpy array to OpenCV compatible format
    y_output = cv2.resize(np.array(output_uint, dtype=np.uint8), (ori_shape[1], ori_shape[0]))

    # Resize input stack to match original size
    img_stack_np = cv2.resize(np.array(img_stack_np, dtype=np.float32), (ori_shape[1], ori_shape[0]))

    cb_stack = np.dstack(cb_channels)
    cr_stack = np.dstack(cr_channels)

    def fuse_images_vectorized(y_stack, cb_stack, cr_stack, fused_y):
        try:
            diff = np.abs(y_stack - fused_y[:, :, np.newaxis])
            color_index = np.argmin(diff, axis=2)
            fused_image = np.zeros((*fused_y.shape, 3), dtype=np.uint8)
            fused_image[:, :, 0] = fused_y
            fused_image[:, :, 1] = np.take_along_axis(cb_stack, color_index[:, :, np.newaxis], axis=2).squeeze()
            fused_image[:, :, 2] = np.take_along_axis(cr_stack, color_index[:, :, np.newaxis], axis=2).squeeze()
            return cv2.cvtColor(fused_image, cv2.COLOR_YCrCb2BGR)
        except Exception as e:
            raise RuntimeError(f"Error in fuse_images_vectorized: {str(e)}")

    t_start = time.time()
    try:
        fused_rgb = fuse_images_vectorized(img_stack_np, cb_stack, cr_stack, y_output)
    except Exception as e:
        raise RuntimeError(f"Failed to fuse color images: {str(e)}")
    print(f"Color image computation time: {time.time() - t_start:.4f} seconds")
    
    # Validate output before saving
    if fused_rgb is None or fused_rgb.size == 0:
        raise RuntimeError("Generated fused image is empty")
    
    # Save directly in results folder with stack folder name and determined output format
    output_file_path = os.path.join(results_dir, f'{stack_folder_name}.{output_format}')
    success = cv2.imwrite(output_file_path, fused_rgb)
    if not success:
        raise RuntimeError(f"Failed to save result to: {output_file_path}")
        
    print(f"Result saved to: {output_file_path}")


def get_input_image_format(folder_path):
    """Get the format of images in the input folder"""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    for f in os.listdir(folder_path):
        if f.lower().endswith(valid_extensions):
            _, ext = os.path.splitext(f)
            return ext[1:]  # Remove the dot
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--predict_name', default='predict')
    parser.add_argument('--model_path',
                        default='weights/checkpoint.pth')
    parser.add_argument('--stack_path',
                        default='path/to/stack',
                        help='image stack path')
    parser.add_argument('--out_format', default='none', 
                        help='output image format (jpg, jpeg, png, bmp) or "none" for auto-detection')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse_args()
        main(args)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)