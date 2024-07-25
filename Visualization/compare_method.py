# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import matplotlib.pyplot as plt
import os
def get_image_formats(folder):
    formats = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            filename, ext = os.path.splitext(file)
            ext = ext[1:].lower() # remove .
            if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
                if ext not in formats:
                    formats.append(ext)
    return formats[0]

methods=["CVT","DWT","DCT","DTCWT","DSIFT","NSCT","IFCNN-MAX","U2Fusion","SDNet","MFF-GAN","SwinFusion","StackMFF"]
#第index张图片
index="table"#sideboard,
basepath=r'C:\Users\dell\Desktop\Working\U3D-Fusion\major_revision_self\exp\all_exp'
dataset='4D-Light-Field'#Mobile Depth,Middlebury,4D-Light-Field,FlyingThings3D


point_x=60#4D-Light-Field:sideboard:165;Mobile Depth:balls:360;Middlebury:Motorcycle:397
point_y=230#4D-Light-Field:sideboard:155;Mobile Depth:balls:0;Middlebury:Motorcycle:175
height =80
width =80

font = {'family': 'Arial'}
fontsize=24

# Configuration
num_cols = 6  # Desired number of columns per row
num_methods = len(methods)
num_rows = (num_methods + num_cols - 1) // num_cols  # Calculate number of rows needed

# Create subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 6))#mobile depth
# fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 6))#Middlebury:Motorcycle
axs = axs.flatten()  # Flatten the 2D array of axes to easily iterate over

flag = 0
for j in range(num_methods):
    ext=get_image_formats( os.path.join(basepath, methods[flag],dataset))
    img_name='{}.{}'.format(index,ext)
    pic_path = os.path.join(basepath, methods[flag],dataset, img_name)
    pic_path_rgb= os.path.join(basepath, methods[flag],dataset, '{}_rgb.{}'.format(index,ext))
    if os.path.exists(pic_path_rgb):
        img = plt.imread(pic_path_rgb)
        print(f"Loaded image: {pic_path_rgb}")
    else:
        img = plt.imread(pic_path)
        print(f"Loaded image: {pic_path}")

    axs[j].imshow(img)
    if methods[flag]=='StackMFF':
        axs[j].set_title('Proposed', font=font, fontsize=fontsize+8)
    else:
        axs[j].set_title(methods[flag], font=font, fontsize=fontsize + 8)
    axs[j].set_xticks([])  # Hide x-axis ticks
    axs[j].set_yticks([])  # Hide y-axis ticks
    axs[j].axis('off')  # Hide axis

    # 绘制子图框
    rect = plt.Rectangle((point_x, point_y), width, height, edgecolor='w', linestyle='--', fill=False,linewidth=2)
    axs[j].add_patch(rect)

    #放大子图框
    # 创建子图框,显示在右下角
    # #Mobile Depth:balls
    # relative_x=0.4
    # relative_y = 0
    # axins = axs[j].inset_axes([0.525,0.016, 1-relative_x, 1-relative_x])

    # # 4d
    relative_x = 0.6
    relative_y = 0
    axins = axs[j].inset_axes([0.59, 0.014, 1 - relative_x, 1 - relative_x])

    # #Middlebury
    # relative_x = 0.5
    # relative_y = 0
    # axins = axs[j].inset_axes([0.575, 0.014, 1 - relative_x, 1 - relative_x])


    #FlyingThings3D
    # relative_x = 0.5
    # relative_y = 0
    # axins = axs[j].inset_axes([0.602, 0.014, 1 - relative_x, 1 - relative_x])

    axins.spines['top'].set_color('white')
    axins.spines['right'].set_color('white')
    axins.spines['bottom'].set_color('white')
    axins.spines['left'].set_color('white')

    axins.spines['top'].set_linestyle('--')
    axins.spines['right'].set_linestyle('--')
    axins.spines['bottom'].set_linestyle('--')
    axins.spines['left'].set_linestyle('--')

    axins.spines['top'].set_linewidth(2)
    axins.spines['right'].set_linewidth(2)
    axins.spines['bottom'].set_linewidth(2)
    axins.spines['left'].set_linewidth(2)

    axins.imshow(img[point_y:point_y + width, point_x:point_x + width])
    # 隐藏子图框的刻度
    axins.set_xticks([])
    axins.set_yticks([])
    flag += 1

for k in range(num_methods, num_rows * num_cols):
    fig.delaxes(axs[k])
plt.tight_layout()
plt.subplots_adjust(wspace=-0.855, hspace=0.2)#4d:wspace=-0.855, hspace=0.2;#flying3d:wspace=-0.4, hspace=0.2;Mobile Depth:balls:-0.3,0.2
plt.savefig('4dlightfield_table_compare.pdf', format='pdf')
plt.show()