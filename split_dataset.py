import os
import random
import shutil

# 定义路径
source_dir = 'data/OpenImagesV7'
train_dir = os.path.join(source_dir, 'train')
val_dir = os.path.join(source_dir, 'val')

# 创建目标文件夹（如果不存在）
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# 随机打乱图片列表
random.shuffle(image_files)

# 计算分割点
split_point = int(len(image_files) * 0.9)

# 移动文件
for i, image in enumerate(image_files):
    source_path = os.path.join(source_dir, image)
    if i < split_point:
        dest_path = os.path.join(train_dir, image)
    else:
        dest_path = os.path.join(val_dir, image)

    shutil.move(source_path, dest_path)
    print(f"Moved {image} to {'train' if i < split_point else 'val'}")

print(f"Moved {split_point} images to train and {len(image_files) - split_point} images to val")
