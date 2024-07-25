import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 文件夹路径
folder_path = r'C:\Users\dell\Desktop\Working\U3D-Fusion\ZJU-MFF\PartA\img_clear'  # 替换为你的文件夹路径

# 获取文件夹中所有图片的文件名
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))


print(image_files)
# 确保有16张图片


# 创建2行8列的图形
fig, axes = plt.subplots(2, 10, figsize=(20, 5))

# 逐个读取并展示图片
for i, ax in enumerate(axes.flat):
    img_path = os.path.join(folder_path, image_files[i])
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis('off')  # 隐藏坐标轴

plt.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.96, wspace=0.1, hspace=-0.2)

plt.show()
# 保存图形为PDF文件
# pdf_path = r'C:\Users\dell\Desktop\Working\U3D-Fusion\major_revision_self\exp\datasets\zju_mff.pdf'  # 替换为你想要的保存路径
# plt.savefig(pdf_path, format='pdf', dpi=300)  # dpi参数控制PDF的分辨率