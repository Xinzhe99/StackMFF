#导入 matplotlib 库
import matplotlib.pyplot as plt
import Orange
plt.style.use('ggplot')

#定义算法的平均排名
_alg_ = [8,9,2,9,7,9,9,6,4,3,5,1][::-1]# 4D-Light-Field & FlyingThings3D & Middlebury & Mobile Depth
_alg_2 =[7,7,2,7,3,7,6,12,3,11,5,1][::-1]#
_alg_3 =[6,8,2,6,2,8,8,11,2,11,5,1][::-1]#
_alg_4 =[4,6,1,4,2,6,10,11,9,12,8,2][::-1]#
#定义算法的数量
y2= [1,2,3,4,5,6,7,8,9,10,11,12]
linewidth=3
bias=0.25
#定义临界值CD
CD1 = Orange.evaluation.compute_CD(_alg_, 4, alpha='0.05', test='nemenyi')
CD2 = Orange.evaluation.compute_CD(_alg_2, 4, alpha='0.05', test='nemenyi')
CD3 = Orange.evaluation.compute_CD(_alg_3, 4, alpha='0.05', test='nemenyi')
CD4 = Orange.evaluation.compute_CD(_alg_4, 4, alpha='0.05', test='nemenyi')
print(CD1)
print(CD2)
print(CD3)
print(CD4)
#计算半径h_CD
h_CD1 = CD1 / 2
h_CD2 = CD2 / 2
h_CD3 = CD3 / 2
h_CD4 = CD4 / 2
#创建图像
plt.figure(figsize=(18, 14))

#绘制算法平均排名图
y1=[a+bias for a in y2]
y3=[a-bias for a in y2]
y4=[a-2*bias for a in y2]
plt.scatter(_alg_, y1,s=100,c='blue',alpha=0.6,label='4D-Light-Field')
plt.scatter(_alg_2, y2,s=100,c='red',alpha=0.6,label='FlyingThings3D')
plt.scatter(_alg_3, y3,s=100,c='yellow',alpha=0.6,label='Middlebury')
plt.scatter(_alg_4, y4,s=100,c='green',alpha=0.6,label='Mobile Depth')
# for 循环用于绘制每个算法的置信区间
for i in range(len(y2)):
    # 计算置信区间的上边界
    yy = [y2[i],y2[i]]
    print(yy)
    # 计算置信区间的下边界
    xx = [_alg_[i]-h_CD1,_alg_[i]+h_CD1]
    xx2 = [_alg_2[i] - h_CD2, _alg_2[i] + h_CD2]
    xx3 = [_alg_3[i] - h_CD3, _alg_3[i] + h_CD3]
    xx4 = [_alg_4[i] - h_CD4, _alg_4[i] + h_CD4]
    # 绘制置信区间
    plt.plot(xx, [a + b for a, b in zip(yy, [bias, bias])], linewidth=linewidth,c='black',linestyle='-')
    plt.plot(xx2, yy, linewidth=linewidth, c='black',linestyle='--')
    plt.plot(xx3, [a - b for a, b in zip(yy, [bias, bias])], linewidth=linewidth, c='black',linestyle=':')
    plt.plot(xx4, [a - 2*b for a, b in zip(yy, [bias, bias])], linewidth=linewidth, c='black', linestyle='-.')
    if i==0:
        plt.legend(loc='lower right', fontsize=12)
#设置y轴标签
plt.yticks(range(0,13,1), labels=['CVT','DWT','DCT','DSIFT','DTCWT','NSCT','IFCNN-MAX','U2Fusion','SDNet','MFF-GAN','SwinFusion','Proposed',''][::-1], size=28)

#设置x轴标签
# 设置x轴标签
plt.xticks(range(1,13,1), labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], size=28)
# 添加图例

# #设置x轴标题
plt.xlabel("Rank", size=20,loc='right',labelpad=-22)

# plt.xlim(0, 7)
#设置图像标题
plt.title("Nemenyi post-hoc test result", size=36)
plt.legend(loc='upper right', fontsize=22)
#保存图像
plt.tight_layout()  # 自动调整布局
plt.savefig("nemenyi.pdf", format='pdf')
# plt.show()