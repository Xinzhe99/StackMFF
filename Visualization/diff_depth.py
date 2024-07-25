# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import matplotlib.pyplot as plt
import os
#todo need set
methods=["3","6","9","12","15"]
#第index张图片
index=1
is_gray=False
if is_gray==True:
    basepath = r'E:\matlabproject\fusion_eva_new\Objective-evaluation-for-image-fusion-main\result_grayscale'
    # 框选的位置与长宽
    point_x = 55
    point_y = 220
    # point_x=50
    # point_y=340
    height = 80
    width = 80
else:
    basepath=r'C:\Users\dell\Desktop\Working\U3D-Fusion\diff_depth'
    # 框选的位置与长宽
    # point_x = 130
    # point_y = 100
    point_x=180
    point_y=125
    height = 80
    width = 80

#显示的数量行列
row=1
col=5

#grayscale框选的位置与长宽

fig, axs = plt.subplots(row, col,figsize=(14, 16))
flag=0
font = {'family': 'Arial'}
fontsize=18
print(len(methods))
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
if row!=1:
    for i in range(0,row):
        for j in range(0,col):
            ext=get_image_formats(os.path.join(basepath,methods[flag]))
            pic_path=os.path.join(basepath,methods[flag],'{}.{}'.format(index,ext))
            img = plt.imread(pic_path)
            if is_gray==True:
                axs[i, j].imshow(img,cmap='gray',vmin=0, vmax=255)
            else:
                axs[i, j].imshow(img)
            axs[i, j].set_title(methods[flag],font=font,fontsize=fontsize+8)
            axs[i, j].set_xticks([])  # 隐藏x刻度
            axs[i, j].set_yticks([])  # 隐藏y刻度
            axs[i, j].axis('off')  # 隐藏坐标轴
            height=80
            width=80
            rect = plt.Rectangle((point_x, point_y), width, height, edgecolor='white', linestyle='--', fill=False,linewidth=2)
            axs[i, j].add_patch(rect)

            #绘制子图框
            # 创建子图框,显示在右下角
            relative_x=0.4#todo 改这个可以修改显示的大小
            relative_y = 0
            axins = axs[i, j].inset_axes([relative_x, relative_y, 1-relative_x, 1-relative_x])
            axins.spines['top'].set_color('white')
            axins.spines['right'].set_color('white')
            axins.spines['bottom'].set_color('white')
            axins.spines['left'].set_color('white')

            axins.spines['top'].set_linestyle('--')
            axins.spines['right'].set_linestyle('--')
            axins.spines['bottom'].set_linestyle('--')
            axins.spines['left'].set_linestyle('--')
            if is_gray==True:
                axins.imshow(img[point_y:point_y+width, point_x:point_x+width],cmap='gray',vmin=0, vmax=255)
            else:
                axins.imshow(img[point_y:point_y + width, point_x:point_x + width])
            # 隐藏子图框的刻度
            axins.set_xticks([])
            axins.set_yticks([])
            flag+=1
else:
    for i in range(0, col):
        ext = get_image_formats(os.path.join(basepath, methods[flag]))
        pic_path = os.path.join(basepath, methods[flag], '{}.{}'.format(index, ext))
        img = plt.imread(pic_path)
        if is_gray == True:
            axs[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            axs[i].imshow(img)
        axs[i].set_title('Depth='+methods[flag], font=font, fontsize=fontsize + 12)
        axs[i].set_xticks([])  # 隐藏x刻度
        axs[i].set_yticks([])  # 隐藏y刻度
        axs[i].axis('off')  # 隐藏坐标轴
        height = 80
        width = 80
        rect = plt.Rectangle((point_x, point_y), width, height, edgecolor='white', linestyle='--', fill=False,linewidth=2)
        axs[i].add_patch(rect)

        # 绘制子图框
        # 创建子图框,显示在右下角
        relative_x = 0.4  # todo 改这个可以修改显示的大小
        relative_y = 0
        axins = axs[i].inset_axes([relative_x, relative_y, 1 - relative_x, 1 - relative_x])
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
        if is_gray == True:
            axins.imshow(img[point_y:point_y + width, point_x:point_x + width], cmap='gray', vmin=0, vmax=255)
        else:
            axins.imshow(img[point_y:point_y + width, point_x:point_x + width])
        # 隐藏子图框的刻度
        axins.set_xticks([])
        axins.set_yticks([])
        flag += 1

plt.tight_layout()
plt.subplots_adjust(wspace=0.03, hspace=-0.1)
# plt.show()
plt.savefig('diff_compare.pdf', format='pdf')

