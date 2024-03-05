import pandas as pd
import matplotlib.pyplot as plt

def draw_a_w(dir,name):
    # 读取表格数据
    df = pd.read_csv(dir)  # 替换为你的文件路径

    # 将第一列作为横坐标
    x = df.iloc[:, 0]

    # 对于其他每一列，绘制一条曲线
    for column in df.columns[1:]:
        y = df[column]
        plt.plot(x, y, label=column)

    # 添加图例
    plt.legend()

    # 添加横向的虚线网格背景
    plt.grid(axis='y', linestyle='--')
    # 添加横纵坐标的标签
    plt.xlabel('Epoch')  # 替换为你的横坐标标签
    plt.ylabel('The value of a_w')  # 替换为你的纵坐标标签

    plt.savefig('result/picture/' + name + '.jpg')  # 替换为你的文件路径

    # 显示图形
    plt.show()

dir = 'result/cifar100_csv_a/resnet34_cifar100/res34_cifar100_4.csv'
name = 'res34_cifar100_4bit'
draw_a_w(dir,name)