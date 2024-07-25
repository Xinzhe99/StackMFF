import os
import matplotlib.pyplot as plt
import cv2

# 读取多聚焦图像栈
image_folder = r'C:\Users\dell\Desktop\Working\U3D-Fusion\ZJU-MFF\PartB\PCB2'
image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')])

# 选择间隔的16张图像
selected_images = image_files[::len(image_files)//16][:16]

# 创建一个4x4的子图用于显示间隔的16张图像
fig, axs = plt.subplots(4, 4, figsize=(16, 16))

for i, img_path in enumerate(selected_images):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    row = i // 4
    col = i % 4
    axs[row, col].imshow(img)
    axs[row, col].axis('off')

# 调整子图的间距
plt.subplots_adjust(wspace=0.02,hspace=-0.63)
plt.show()
