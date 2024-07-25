import os
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse

def save_stack_images_from_hdf5(hdf5_path, output_dir,data_type):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        if data_type in hdf5_file.keys():
            stack_train = hdf5_file[data_type][:]
            if data_type=='stack_train':
                names = hdf5_file['name_train'][:]
            elif data_type=='stack_val':
                names = hdf5_file['name_val'][:]
            for i, name in enumerate(tqdm(names, desc='Processing stack_train images')):
                # Convert bytes to string if necessary
                if isinstance(name[0], bytes):
                    name_str = name[0].decode('utf-8')
                else:
                    name_str = name[0]

                folder_path = os.path.join(output_dir, data_type, name_str)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                stack = stack_train[i]
                for j, img in enumerate(stack):
                    img_path = os.path.join(folder_path, f'stack_{j}.png')
                    save_image(img, img_path)


def save_image(img, img_path):
    img = Image.fromarray(img.astype(np.uint8))
    img.save(img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and save images from HDF5 file.')
    parser.add_argument('hdf5_path', type=str, help='Path to the HDF5 file')
    parser.add_argument('output_dir', type=str, help='Directory to save the extracted images')
    args = parser.parse_args()
    save_stack_images_from_hdf5(args.hdf5_path, args.output_dir, 'stack_train')
    save_stack_images_from_hdf5(args.hdf5_path, args.output_dir, 'stack_val')
