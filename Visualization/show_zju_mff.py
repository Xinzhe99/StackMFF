import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 文件夹路径
main_folder_path_A = r'C:\Users\dell\Desktop\Working\U3D-Fusion\ZJU-MFF\PartA\image_stack'
main_folder_path_B = r'C:\Users\dell\Desktop\Working\U3D-Fusion\ZJU-MFF\PartB'


def get_random_images(main_folder, num_images=10):
    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
    selected_images = []
    for subfolder in subfolders:
        image_files = [f for f in os.listdir(subfolder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            selected_image = random.choice(image_files)
            selected_images.append(os.path.join(subfolder, selected_image))

    if len(selected_images) < num_images:
        print(f"警告：在 {main_folder} 中只找到 {len(selected_images)} 张图片，少于预期的 {num_images} 张。")
    return random.sample(selected_images, min(num_images, len(selected_images)))


# 获取两部分的图片
selected_images_A = get_random_images(main_folder_path_A)
selected_images_B = get_random_images(main_folder_path_B)

# 创建2行10列的图形
fig, axes = plt.subplots(2, 10, figsize=(20, 4))


def display_images(axes_row, image_paths):
    for ax, img_path in zip(axes_row, image_paths):
        img = Image.open(img_path)
        img_resized = img.resize((512, 512), Image.LANCZOS)
        img_array = np.array(img_resized)
        ax.imshow(img_array)
        ax.axis('off')


# 显示Part A的图片
display_images(axes[0], selected_images_A)

# 显示Part B的图片
display_images(axes[1], selected_images_B)

plt.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.96, wspace=0.005, hspace=0.04)

# plt.show()


pdf_path = r'C:\Users\dell\Desktop\Working\U3D-Fusion\major_revision_self\exp\datasets\zju_mff.pdf'
plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')