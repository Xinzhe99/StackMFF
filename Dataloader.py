# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import glob
import os.path
import collections
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.tiff', '.TIF', '.TIFF'
]

import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    # Training folders
    train_stack_path = os.path.join(filepath, 'dof_stack')
    subfolders = [f for f in os.listdir(train_stack_path) if os.path.isdir(os.path.join(train_stack_path, f))]
    
    # Training labels
    all_label_img = glob.glob(os.path.join(filepath, 'AiF','*.jpg'))
    
    # Ensure dof_stack and AiF are paired by name
    all_data = []
    
    # Get all label file names (without extension)
    label_names = [os.path.splitext(os.path.basename(label_path))[0] for label_path in all_label_img]
    
    # Iterate through dof_stack folders, only keep samples with corresponding label files
    for subfolder in subfolders:
        folder_name = subfolder
        if folder_name in label_names:
            # Find the corresponding label file path
            label_index = label_names.index(folder_name)
            all_data.append((os.path.join(train_stack_path, subfolder), all_label_img[label_index]))
    
    # Sort by folder name to ensure consistent order
    all_data.sort(key=lambda x: os.path.basename(x[0]))
    
    # Separate stack and label
    all_stack = [item[0] for item in all_data]
    all_label_img_matched = [item[1] for item in all_data]
    
    # Split into 0.8 training set and 0.2 validation set
    total_samples = len(all_stack)
    train_end = int(0.8 * total_samples)
    
    train_stack = all_stack[:train_end]
    train_label_img = all_label_img_matched[:train_end]
    
    val_stack = all_stack[train_end:]
    val_label_img = all_label_img_matched[train_end:]
    
    # Do not return test set
    return train_stack, train_label_img, val_stack, val_label_img

def get_stack_depth(folder_path):
    """Get the number of images in a stack folder"""
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    return len(image_paths)

def stack_y_channels(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith('.jpg') or f.endswith('.png')]

    y_channels = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('YCbCr')
        y = np.array(img.split()[0])
        y_channels.append(y)
    random.shuffle(y_channels)  # Manual shuffle for data augmentation
    y_channels = np.stack(y_channels, axis=-1)
    return y_channels

def default_loader(path, mode='stack', padding=0, target_size=(384, 384)):
    if mode == 'img':
        # Add resize operation
        img = Image.open(path).convert('YCbCr')
        img = img.resize(target_size, Image.Resampling.BILINEAR)  # Use bilinear interpolation for resize
        y_channel = np.array(img.split()[0])

        h = y_channel.shape[0]
        w = y_channel.shape[1]

        padding_matrix = np.zeros((h, w, padding))
        padding_matrix[:, :, 0] = y_channel
        y_channel_padding = padding_matrix

        return y_channel_padding

    elif mode == 'stack':
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if
                       f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

        y_channels = []
        for img_path in image_paths:
            # Add resize operation
            img = Image.open(img_path).convert('YCbCr')
            img = img.resize(target_size, Image.Resampling.BILINEAR)  # Use bilinear interpolation for resize
            y = np.array(img.split()[0])
            y_channels.append(y)

        random.shuffle(y_channels)  # Manual shuffle for data augmentation
        y_channels = np.stack(y_channels, axis=-1)
        return y_channels

    else:
        print('Please check mode!')

def get_transform():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

class myImageFloder(data.Dataset):
    def __init__(self, img_stack, label_img, loader=default_loader, target_size=(384, 384)):
        self.img_stack = img_stack
        self.label_img = label_img
        self.loader = loader
        self.target_size = target_size

    def __getitem__(self, index):
        img_stack = self.img_stack[index]
        label_img = self.label_img[index]

        img_stack = self.loader(img_stack, mode='stack', target_size=self.target_size)

        # Need to pad image to the same dimension as image stack
        padding_channel = img_stack.shape[2]

        label_img = self.loader(label_img, mode='img', padding=padding_channel, target_size=self.target_size)

        processed = get_transform()

        img_stack = processed(img_stack)
        label_img = processed(label_img)

        return img_stack, label_img

    def __len__(self):
        return len(self.label_img)

class VariableDepthDataset(data.Dataset):
    """Dataset that groups samples by stack depth"""
    def __init__(self, img_stack, label_img, loader=default_loader, target_size=(384, 384)):
        self.img_stack = img_stack
        self.label_img = label_img
        self.loader = loader
        self.target_size = target_size
        
        # Group samples by stack depth
        self.depth_groups = collections.defaultdict(list)
        for i, stack_path in enumerate(img_stack):
            depth = get_stack_depth(stack_path)
            self.depth_groups[depth].append((stack_path, label_img[i]))
        
        # Create a list of (depth, index_in_group) for each sample
        self.sample_indices = []
        for depth, samples in self.depth_groups.items():
            for idx in range(len(samples)):
                self.sample_indices.append((depth, idx))
    
    def __getitem__(self, index):
        depth, idx = self.sample_indices[index]
        stack_path, label_path = self.depth_groups[depth][idx]
        
        img_stack = self.loader(stack_path, mode='stack', target_size=self.target_size)
        padding_channel = img_stack.shape[2]
        label_img = self.loader(label_path, mode='img', padding=padding_channel, target_size=self.target_size)
        
        processed = get_transform()
        img_stack = processed(img_stack)
        label_img = processed(label_img)
        
        return img_stack, label_img
    
    def __len__(self):
        return len(self.sample_indices)
    
    def get_depth_groups(self):
        """Return the grouped samples by depth"""
        return self.depth_groups