import os
import random
from PIL import Image

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

# 移动并调整文件大小
for i, image in enumerate(image_files):
    source_path = os.path.join(source_dir, image)
    if i < split_point:
        dest_dir = train_dir
    else:
        dest_dir = val_dir

    # 打开图片
    with Image.open(source_path) as img:
        # 调整大小
        img_resized = img.resize((384, 384), Image.LANCZOS)

        # 保存调整大小后的图片
        filename, ext = os.path.splitext(image)
        new_filename = f"{filename}_384x384{ext}"
        dest_path = os.path.join(dest_dir, new_filename)
        img_resized.save(dest_path)

    # 删除原图
    os.remove(source_path)
    print(f"Resized and moved {image} to {'train' if i < split_point else 'val'}")

print(f"Moved {split_point} images to train and {len(image_files) - split_point} images to val")
