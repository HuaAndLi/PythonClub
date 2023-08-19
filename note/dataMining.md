[TOC]

# 数据挖掘基础环境安装与使用

1.1 库的安装
    matplotlib==2.2.2
    numpy==1.14.2
    pandas==0.20.3
    TA-Lib==0.4.16 技术指标库 conda install -c conda-forge ta-lib
    tables==3.4.2 hdf5
    jupyter==1.0.0 数据分析与展示的平台
1.2 Jupyter Notebook使用
    1.2.1 Jupyter Notebook介绍
        1）web版的ipython
        2）名字
        ju - Julia
        py - Python
        ter - R
        Jupiter 木星 宙斯
        3）编程、写文档、记笔记、展示
        4）.ipynb
    1.2.2 为什么使用Jupyter Notebook?
        1）画图方面的优势
        2）数据展示方面的优势
    1.2.3 Jupyter Notebook的使用-helloworld
        1 界面启动、创建文件
            在终端输入jupyter notebook / ipython notebook
            快速上手的方法：
                快捷键
                    运行代码 shift + enter
        2 cell操作
            cell：一对In Out会话被视作一个代码单元，称为cell
            编辑模式：
                enter
                鼠标直接点
            命令模式：
                esc
                鼠标在本单元格之外点一下
            2）快捷键操作
                执行代码：shift + enter
                命令模式：
                A，在当前cell的上面添加cell
                B，在当前cell的下面添加cell
                双击D：删除当前cell
                编辑模式：
                多光标操作：Ctrl键点击鼠标（Mac:CMD+点击鼠标）
                回退：Ctrl+Z（Mac:CMD+Z）
                补全代码：变量、方法后跟Tab键
                为一行或多行代码添加/取消注释：Ctrl+/（Mac:CMD+/）
            3 markdown演示

# Matplotlib 画图



## 2.1 Matplotlib之HelloWorld
 **2.1.1 什么是Matplotlib - 画二维图表的python库**

- 专门用于开发2D图表（包括3D图表）
- 使用起来及其简单
- 以渐进、交互式方式实现数据可视化

​            mat - matrix 矩阵
​                二维数据 - 二维图表
​            plot - 画图
​            lib - library 库
​            matlab 矩阵实验室
​                mat - matrix
​                lab 实验室
**2.1.2 为什么要学习Matplotlib - 画图**

可视化是在整个数据挖掘的关键辅助工具，可以清晰的理解数据，从而调整我们的分析方法。

`不是给别人看的，是我们自己看了分析数据的`

- ·能将数据进行可视化，更直观的呈现
- ·使数据更加客观、更具说服力

​            数据可视化 - 帮助理解数据，方便选择更合适的分析方法
​            js库 - D3 echarts
​            奥卡姆剃刀原理 - 如无必要勿增实体
**2.1.3 实现一个简单的Matplotlib画图**

```python
import matplotlib.pyplot as plt

plt.figure()
plt.plot([1,0,9], [4,5,6])
plt.show()
```



<div><img src=".\img\data_mining\Snipaste_2023-08-17_17-03-41.png" alt="Snipaste_2023-08-17_17-03-41" style="width:50%;" /></div>



**2.1.4 认识Matplotlib图像结构**

<div><img src="img\data_mining\Snipaste_2023-08-17_17-20-07.png" alt="Snipaste_2023-08-17_17-20-07" style="width:50%;" /></div>

**2.1.5 拓展知识点：Matplotlib三层结构**

1）容器层

​		画板层Canvas - 自动创建了

​		画布层Figure

​		绘图区/坐标系 - 一个画布可以有多个绘图区

​            x、y轴张成的区域

​			2）辅助显示层 - 使图像更加详细

​			3）图像层 - 画各种图表

<div><img src="img\data_mining\Snipaste_2023-08-17_17-29-29.png" alt="Snipaste_2023-08-17_17-29-29" style="width:80%;" /></div>

·总结：
o Canvas(画板)位于最底层，用户一般接触不到
。Figure(画布)建立在Canvas.之上
。Axes(绘图区)建立在Figure之上
。坐标轴(axis)、图例(legend)等辅助显示层以及图像层都是建立在Axes之上



 ## 2.2 折线图(plot)与基础绘图功能



### 2.2.1折线图绘制与保存图片

**1 matplotlib.pyplot模块**

matplotlib.pyplot包含了一系列类似于matlab的画图函数。它的函数**作用于当前图形((figure)的当前坐标系**
**(axes)。**

```python
import matplotlib.pyplot as plt
```

**2 折线图绘制与显示**

展现上海一周的天气比如从星期一到星期日的天气温度如下

```python
#1)创建画布(容器层)
plt.figure()
#2) 绘制折线图(图像层)
plt.plot([1, 2, 3,4, 5, 6 ,7], [17,17, 18, 15, 11, 1, 13])
#3)显示图像
plt.show()
```



**3 设置画布属性与图片保存**

figsize : 画布大小
dpi : dot per inch 图像的清晰度

```python
#1) 创建画布，并设置画布属性
plt.figure(figsize=(20,8),dpi=80)
#2) 保存图片到指定路径
plt.savefig("test.png")#写在 plt.show() 前面

```

注意：plt.show()会释放figure资源，如果在显示图像之后保存图片将只能保存空图片。

### 2.2.2完善原始折线图1（辅助显示层）

**1 准备数据并画出初始折线图**

```python
# 需求：画出某城市11点到12点1小时内每分钟的温度变化折线图，温度范围在15度~18度
import matplotlib.pyplot as plt
import random

# 1、准备数据 x y
x = range(60)
y_shanghai = [random.uniform(15, 18) for i in x]

# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制图像
plt.plot(x, y_shanghai)

# 4、显示图
plt.show()
```

**2 添加自定义X,y刻度**

```python
# 修改x、y刻度
# 准备x的刻度说明
x_label = ["11点{}分".format(i) for i in x]
plt.xticks(x[::5], x_label[::5])
plt.yticks(range(0, 40, 5))
```

**3 中文显示问题解决**

 mac的一次配置，一劳永逸
ubantu每创建一次新的虚拟环境，需要重新配置
windows
      1）安装字体
             mac/wins：双击安装
             ubantu：双击安装
      2）删除matplotlib缓存文件
      3）配置文件

注意：windows可以不安装字体，

```python
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
```



**4 添加网格显示**

```python
# 添加网格显示
plt.grid(linestyle="--", alpha=0.5)
```

**5 添加描述信息**

```python
# 添加描述信息
plt.xlabel("时间变化")
plt.ylabel("温度变化")
plt.title("某城市11点到12点每分钟的温度变化状况")
```



### 2.2.3完善原始折线图2（图像层）

**需求：再添加一个城市**

```python
# 需求：再添加一个城市的温度变化
# 收集到北京当天温度变化情况，温度在1度到3度。 

# 1、准备数据 x y
x = range(60)
y_shanghai = [random.uniform(15, 18) for i in x]
y_beijing = [random.uniform(1, 3) for i in x]

# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制图像
plt.plot(x, y_shanghai, color="r", linestyle="-.", label="上海")
plt.plot(x, y_beijing, color="b", label="北京")

# 显示图例
plt.legend()

# 修改x、y刻度
# 准备x的刻度说明
x_label = ["11点{}分".format(i) for i in x]
plt.xticks(x[::5], x_label[::5])
plt.yticks(range(0, 40, 5))

# 添加网格显示
plt.grid(linestyle="--", alpha=0.5)

# 添加描述信息
plt.xlabel("时间变化")
plt.ylabel("温度变化")
plt.title("上海、北京11点到12点每分钟的温度变化状况")

# 4、显示图
plt.show()
```



<div><img src="img\data_mining\Snipaste_2023-08-17_20-22-53.png" alt="Snipaste_2023-08-17_20-22-53" style="width:80%;" /></div>

### 2.2.4多个坐标系显示-plt.subplots(面向对象的画图方法)

figure, axes = plt.subplots(nrows=1, ncols=2, **fig_kw)
            axes[0].方法名()
            axes[1]

```python
# 需求：再添加一个城市的温度变化
# 收集到北京当天温度变化情况，温度在1度到3度。 

# 1、准备数据 x y
x = range(60)
y_shanghai = [random.uniform(15, 18) for i in x]
y_beijing = [random.uniform(1, 3) for i in x]

# 2、创建画布
# plt.figure(figsize=(20, 8), dpi=80)
figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), dpi=80)

# 3、绘制图像
axes[0].plot(x, y_shanghai, color="r", linestyle="-.", label="上海")
axes[1].plot(x, y_beijing, color="b", label="北京")

# 显示图例
axes[0].legend()
axes[1].legend()

# 修改x、y刻度
# 准备x的刻度说明
x_label = ["11点{}分".format(i) for i in x]
axes[0].set_xticks(x[::5])
axes[0].set_xticklabels(x_label)
axes[0].set_yticks(range(0, 40, 5))
axes[1].set_xticks(x[::5])
axes[1].set_xticklabels(x_label)
axes[1].set_yticks(range(0, 40, 5))

# 添加网格显示
axes[0].grid(linestyle="--", alpha=0.5)
axes[1].grid(linestyle="--", alpha=0.5)

# 添加描述信息
axes[0].set_xlabel("时间变化")
axes[0].set_ylabel("温度变化")
axes[0].set_title("上海11点到12点每分钟的温度变化状况")
axes[1].set_xlabel("时间变化")
axes[1].set_ylabel("温度变化")
axes[1].set_title("北京11点到12点每分钟的温度变化状况")

# 4、显示图
plt.show()
```

<div><img src="img\data_mining\Snipaste_2023-08-17_20-21-12.png" alt="Snipaste_2023-08-17_20-21-12" style="width:80%;" /></div>



### 2.2.5折线图的应用场景

某事物、某指标随时间的变化状况
            拓展：画各种数学函数图像

```python
import numpy as np
# 1、准备x，y数据
x = np.linspace(-1, 1, 1000)
y = 2 * x * x #函数 2x^2 

# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制图像
plt.plot(x, y)

# 添加网格显示
plt.grid(linestyle="--", alpha=0.5)

# 4、显示图像
plt.show()
```

<div><img src="img\data_mining\Snipaste_2023-08-17_20-16-54.png" alt="Snipaste_2023-08-17_20-16-54" style="width:80%;" /></div>



## 2.3 散点图

### 2.3.1 常见图形种类及意义
​    折线图plot
​    散点图scatter
​        关系/规律
​    柱状图bar
​        统计/对比
​    直方图histogram
​        分布状况
​    饼图pie π
​        占比

### 2.3.2 散点图绘制

```python
# 需求：探究房屋面积和房屋价格的关系

# 1、准备数据
x = [225.98, 247.07, 253.14, 457.85, 241.58, 301.01,  20.67, 288.64,
       163.56, 120.06, 207.83, 342.75, 147.9 ,  53.06, 224.72,  29.51,
        21.61, 483.21, 245.25, 399.25, 343.35]

y = [196.63, 203.88, 210.75, 372.74, 202.41, 247.61,  24.9 , 239.34,
       140.32, 104.15, 176.84, 288.23, 128.79,  49.64, 191.74,  33.1 ,
        30.74, 400.02, 205.35, 330.64, 283.45]
# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制图像
plt.scatter(x, y)

# 4、显示图像
plt.show()
```

<div><img src="img\data_mining\Snipaste_2023-08-17_20-19-48.png" alt="Snipaste_2023-08-17_20-19-48" style="width:80%;" /></div>

### 2.3.3 散点图应用场景

探究不同变量之间的内在关系



##    2.4 柱状图(bar)

### 2.4.1 柱状图绘制

**需求1-对比每部电影的票房收入**

```python
# 1、准备数据
movie_names = ['雷神3：诸神黄昏','正义联盟','东方快车谋杀案','寻梦环游记','全球风暴', '降魔传','追捕','七十七天','密战','狂兽','其它']
tickets = [73853,57767,22354,15969,14839,8725,8716,8318,7916,6764,52222]

# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制柱状图
x_ticks = range(len(movie_names))
plt.bar(x_ticks, tickets, color=['b','r','g','y','c','m','y','k','c','g','b'])

# 修改x刻度
plt.xticks(x_ticks, movie_names)

# 添加标题
plt.title("电影票房收入对比")

# 添加网格显示
plt.grid(linestyle="--", alpha=0.5)

# 4、显示图像
plt.show()
```

<div><img src="img\data_mining\Snipaste_2023-08-17_20-09-36.png" alt="Snipaste_2023-08-17_20-09-36" style="width:80%;" /></div>

**需求2-如何对比电影票房收入才更能加有说服力？**

```python
# 1、准备数据
movie_name = ['雷神3：诸神黄昏','正义联盟','寻梦环游记']

first_day = [10587.6,10062.5,1275.7]
first_weekend=[36224.9,34479.6,11830]

# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制柱状图
plt.bar(range(3), first_day, width=0.2, label="首日票房")
plt.bar([0.2, 1.2, 2.2], first_weekend, width=0.2, label="首周票房")

# 显示图例
plt.legend()

# 修改刻度
plt.xticks([0.1, 1.1, 2.1], movie_name)

# 4、显示图像
plt.show()
```

<div><img src="img\data_mining\Snipaste_2023-08-17_20-12-55.png" alt="Snipaste_2023-08-17_20-12-55" style="width:80%;" /></div>

### 2.4.2柱状图应用场景

适合用在分类数据对比场景上
·数量统计
·用户数量对比分析



   ##  2.5 直方图(histogram)

### 2.5.1 直方图介绍

​            组数：在统计数据时，我们把数据按照不同的范围分成几个组，分成的组的个数称为组数
​            组距：每一组两个端点的差
​            已知 最高175.5 最矮150.5 组距5
​            求 组数：(175.5 - 150.5) / 5 = 5

### 2.5.2 直方图与柱状图的对比

<div><img src="img\data_mining\Snipaste_2023-08-17_21-01-16.png" alt="Snipaste_2023-08-17_21-01-16" style="width:50%;" /></div>

直方图展示数据的分布，柱状图比较数据的大小。

直方图X轴为定量数据，柱状图X轴为分类数据

直方图柱子无间隔，柱状图柱子有间隔

直方图柱子宽度可不一，柱状图柱子宽度须一致



### 2.5.3 直方图绘制

x = time
bins 组数 = (max(time) - min(time)) // 组距
3 直方图注意点

```python
# 需求：电影时长分布状况
# 1、准备数据
time = [131,  98, 125, 131, 124, 139, 131, 117, 128, 108, 135, 138, 131, 102, 107, 114, 119, 128, 121, 142, 127, 130, 124, 101, 110, 116, 117, 110, 128, 128, 115,  99, 136, 126, 134,  95, 138, 117, 111,78, 132, 124, 113, 150, 110, 117,  86,  95, 144, 105, 126, 130,126, 130, 126, 116, 123, 106, 112, 138, 123,  86, 101,  99, 136,123, 117, 119, 105, 137, 123, 128, 125, 104, 109, 134, 125, 127,105, 120, 107, 129, 116, 108, 132, 103, 136, 118, 102, 120, 114,105, 115, 132, 145, 119, 121, 112, 139, 125, 138, 109, 132, 134,156, 106, 117, 127, 144, 139, 139, 119, 140,  83, 110, 102,123,107, 143, 115, 136, 118, 139, 123, 112, 118, 125, 109, 119, 133,112, 114, 122, 109, 106, 123, 116, 131, 127, 115, 118, 112, 135,115, 146, 137, 116, 103, 144,  83, 123, 111, 110, 111, 100, 154,136, 100, 118, 119, 133, 134, 106, 129, 126, 110, 111, 109, 141,120, 117, 106, 149, 122, 122, 110, 118, 127, 121, 114, 125, 126,114, 140, 103, 130, 141, 117, 106, 114, 121, 114, 133, 137,  92,121, 112, 146,  97, 137, 105,  98, 117, 112,  81,  97, 139, 113,134, 106, 144, 110, 137, 137, 111, 104, 117, 100, 111, 101, 110,105, 129, 137, 112, 120, 113, 133, 112,  83,  94, 146, 133, 101,131, 116, 111,  84, 137, 115, 122, 106, 144, 109, 123, 116, 111,111, 133, 150]

# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制直方图
distance = 2 #组距
group_num = int((max(time) - min(time)) / distance)#组数

plt.hist(time, bins=group_num, density=True)

# 修改x轴刻度
plt.xticks(range(min(time), max(time) + 2, distance))

# 添加网格
plt.grid(linestyle="--", alpha=0.5)

# 4、显示图像
plt.show()
```

<div><img src="img\data_mining\Snipaste_2023-08-17_21-06-40.png" alt="Snipaste_2023-08-17_21-06-40" style="width:80%;" /></div>

注意点：

注意组距，可以多次改变组距调整

注意Y轴所代表的变量，Y轴上的变量可以是频次 (数据出现了多少次)、频率 (频次/总次数)、频率/组距，不同的变量会让直方图描述的数据分布意义不同。



### 2.5.3 直方图的应用场景

用于表示分布情况

通过直方图还可以观察和估计哪些数据比较集中，异常或者孤立的数据分布在何处

例如:用户年龄分布，商品价格分布



## 2.6 饼图(pie)

%1.2f%%
print("%1.2f%%") 末尾两个%是转义

```python
# 1、准备数据
movie_name = ['雷神3：诸神黄昏','正义联盟','东方快车谋杀案','寻梦环游记','全球风暴','降魔传','追捕','七十七天','密战','狂兽','其它']

place_count = [60605,54546,45819,28243,13270,9945,7679,6799,6101,4621,20105]

# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制饼图
plt.pie(place_count, labels=movie_name, colors=['b','r','g','y','c','m','y','k','c','g','y'], autopct="%1.2f%%")

# 显示图例
plt.legend()

#为了让显示的饼图保持圆形，需要添加axis保证长宽一样
plt.axis('equal')

# 4、显示图像
plt.show()
```



<div><img src="D:\PythonClub\note\img\data_mining\Snipaste_2023-08-17_21-40-19.png" alt="Snipaste_2023-08-17_21-40-19" style="width:80%;" /></div>



**饼图应用场景**

·分类的占比情况（不超过9个分类)】
例如：班级男女分布占比，公司销售额占比



# Numpy 高效的运算工具

Numpy 高效的运算工具
Numpy的优势
ndarray属性
基本操作
    ndarray.方法()
    numpy.函数名()
ndarray运算
    逻辑运算
    统计运算
    数组间运算
合并、分割、IO操作、数据处理

**重点：任何语言任何的库不可能全部学到、记住，用到去查api。**



## 3.1.Numpy的优势

**3.1.1 Numpy介绍 - 数值计算库**
        num - numerical 数值化的
        py - python
        ndarray
            n - 任意个
            d - dimension 维度
            array - 数组
**3.1.2 ndarray介绍**

NumPy提供了一个N维数组类型ndarray，它描述了相同类型的“items”的集合

```python
import numpy as np
score = np.array([[80, 89, 86, 67, 79],
[78, 97, 89, 67, 81],
[90, 94, 78, 67, 74],
[91, 91, 90, 67, 69],
[76, 87, 75, 67, 86],
[70, 79, 84, 67, 84],
[94, 92, 93, 67, 64],
[86, 85, 83, 67, 80]])
print(score)
print(type(score))
```

**3.1.3 ndarray与Python原生list运算效率对比**

自己测试一下就知道了。

```python
import numpy as np
import random
import time

# 生成一个大数组
python_list = []

for i in range(100000000):
    python_list.append(random.random())

ndarray_list = np.array(python_list)

print(len(ndarray_list))

# 原生pythonlist求和
t1 = time.time()
a = sum(python_list)
t2 = time.time()
d1 = t2 - t1

# ndarray求和
t3 = time.time()
b = np.sum(ndarray_list)
t4 = time.time()
d2 = t4 - t3

print(d1)
print(d2)


```



**3.1.4 ndarray的优势**
        1）存储风格
            ndarray - 相同类型 - 通用性不强
            list - 不同类型 - 通用性很强
        2）并行化运算
            ndarray支持向量化运算
        3）底层语言
            C语言，解除了GIL(全局解释器锁)



## 3.2认识N维数组-ndarray属性

**3.2.1 ndarray的属性**

数组属性反映了数组本身固有的信息

| 属性名字         | 属性解释                   |
| ---------------- | -------------------------- |
| ndarray.shape    | 数组维度的元组             |
| ndarray.ndim     | 数组维数                   |
| ndarray.size     | 数组中的元素数量           |
| ndarray.itemsize | 一个数组元素的长度（字节） |
| ndarray.dtype    | 数组元素的类型             |

​        

shape
      ndim
      size
dtype
      itemsize
在创建ndarray的时候，如果没有指定类型
      默认
          整数 int64
          浮点数 float64

```python
import numpy as np

score = np.array([[80, 89, 86, 67, 79],
                  [78, 97, 89, 67, 81],
                  [90, 94, 78, 67, 74],
                  [91, 91, 90, 67, 69],
                  [76, 87, 75, 67, 86],
                  [70, 79, 84, 67, 84],
                  [94, 92, 93, 67, 64],
                  [86, 85, 83, 67, 80]])

print(score.shape) #(8, 5)
print(score.ndim) #2
print(score.size) #40
print(score.dtype) #int32 因为所以数据都不需要8个字节所以 int32 就够用了。内部优化
print(score.itemsize) #4

```

**3.2.2 ndarray的形状**

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3, 4])
c = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.shape)  # (2, 3)
print(b.shape)  # (4,)
print(c.shape)  # (2, 2, 3)
```



**3.2.3 ndarray的类型**

dtype是numpy.dtype类型，先看看对于数组来说都有哪些类型

| 名称          | 描述                                              | 简写  |
| ------------- | ------------------------------------------------- | ----- |
| np.bool       | 用一个字节存储的布尔类型(True或False)             | b     |
| np.int8       | 一个字节大小，-128至127                           | i     |
| np.int16      | 整数，-32768至32767                               | 'i2'  |
| np.int32      | 整数，-2^31至2^32-1                               | 'i4   |
| np.int64      | 整数，-2^63至2^63-1                               | 'i8'  |
| np.uint8      | 无符号整数，0至255                                | 'u'   |
| np.uint16     | 无符号整数，0至65535                              | 'u2   |
| np.uint32     | 无符号整数，0至2^32-1                             | 'u4   |
| np.uint64     | 无符号整数，0至2^64-1                             | 'u8   |
| np.float16    | 半精度浮点数：16位，正负号1位，指数5位，精度10位  | 'f2'  |
| np.float32    | 单精度浮点数：32位，正负号1位，指数8位，精度23位  | 'f4'  |
| np.float64    | 双精度浮点数：64位，正负号1位，指数11位，精度52位 | 'f8   |
| np.complex64  | 复数，分别用两个32位浮点数表示实部和虚部          | 'c8   |
| np.complex128 | 复数，分别用两个64位浮点数表示实部和虚部          | 'c16' |
| np.object_    | python对象                                        | O     |
| np.string_    | 字符串                                            | S     |
| np.unicode_   | unicode类型                                       | U     |

```python
import numpy as np

data = np.array([1.1, 2.2, 3.3])
print(data.dtype)

# 创建数组的时候指定类型
data = np.array([1.1, 2.2, 3.3], dtype="float32")
print(data.dtype)

data = np.array([1.1, 2.2, 3.3], dtype=np.float32)
print(data.dtype)
```





## 3.3基本操作

adarray.方法()
np.函数名()
        np.array()

### **3.3.1 生成数组的方法**

1）生成0和1
            np.zeros(shape)
            np.ones(shape)

```python
# 1 生成0和1的数组
np.zeros(shape=(3, 4), dtype="float32")

np.ones(shape=[2, 3], dtype=np.int32)
```

2）从现有数组中生成
	np.array() 

​	np.copy() 深拷贝

​	np.asarray() 浅拷贝  修改原有的，他也被修改

```python
import numpy as np

score = np.array([[80, 89, 86, 67, 79],
                  [78, 97, 89, 67, 81],
                  [90, 94, 78, 67, 74],
                  [91, 91, 90, 67, 69],
                  [76, 87, 75, 67, 86],
                  [70, 79, 84, 67, 84],
                  [94, 92, 93, 67, 64],
                  [86, 85, 83, 67, 80]])

# np.array()
data1 = np.array(score)
print(data1)
# np.asarray()
data2 = np.asarray(score)
print(data2)
# np.copy()
data3 = np.copy(score)
print(data3)

score[1, 1] = 1000
print(data1)
print(data2)# 浅拷贝也被修改了
print(data3)

```

​	

3）生成固定范围的数组
	np.linspace(0, 10, 100)
                [0, 10] 100个 等距离

 	np.arange(a, b, c)
 	        range(a, b, c)
 	            [a, b)  c是步长

```python
print(np.linspace(0, 10, 5))#[ 0.   2.5  5.   7.5 10. ]
print(np.arange(0, 11, 5))#[ 0  5 10]
```

4）生成随机数组
        分布状况 - 直方图
        1）均匀分布
            每组的可能性相等

```python
#均匀分布
data1 = np.random.uniform(low=-1, high=1, size=1000000)
import matplotlib.pyplot as plt

# 1、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 2、绘制直方图
plt.hist(data1, 1000)

# 3、显示图像
plt.show()

```

<div>
    <img src="img\data_mining\Snipaste_2023-08-18_16-32-12.png" alt="Snipaste_2023-08-18_16-32-12" style="width:80%;" />
</div>



​        2）正态分布

<div><img src="img\data_mining\Snipaste_2023-08-18_16-37-48.png" alt="Snipaste_2023-08-18_16-37-48" style="width:80%;" /></div>

​            σ 幅度、波动程度、集中程度、稳定性、离散程度

```python
# 正态分布
data2 = np.random.normal(loc=1.75, scale=0.1, size=1000000) # loc：均值，scale：标准差，size:形状
# 1、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 2、绘制直方图
plt.hist(data2, 1000)

# 3、显示图像
plt.show()
```

<div><img src="img\data_mining\Snipaste_2023-08-18_16-34-31.png" alt="Snipaste_2023-08-18_16-34-31" style="width:80%;" /></div>



**案例：随机生成8只股票2周的交易日涨幅数据**

```python
stock_change = np.random.normal(loc=0, scale=1, size=(8, 10))
```



### **3.3.2 数组的索引、切片**

```python
# 获取第一个股票的前3个交易日的涨跌幅数据
stock_change[0, :3]
```

### **3.3.3 形状修改**

​    ndarray.reshape(shape) 返回新的ndarray，原始数据没有改变(行列不变)

```python
stock_change.reshape((10, 8))
```

​    ndarray.resize(shape) 没有返回值，对原始的ndarray进行了修改,一样行列不变

```python
stock_change.resize((10, 8))
print(stock_change.shape)
stock_change.resize((8, 10))
print(stock_change.shape)
```

​    ndarray.T 转置 行变成列，列变成行

```python
stock_change.T
```

### **3.3.4 类型修改**

​    ndarray.astype(type)

```
stock_change.astype("int32")
```

​    ndarray序列化到本地    ndarray.tostring()

```
stock_change.tostring()
```

### **3.3.5 数组的去重**

​    set()

```python
temp = np.array([[1, 2, 3, 4],[3, 4, 5, 6]])
print(np.unique(temp))# 二维直接去重，变成一维
print(set(temp.flatten()))#转一维再去重
```



## 3.4.ndarray运算

### 3.4.1 逻辑运算

布尔索引

```python
stock_change = np.random.normal(loc=0, scale=1, size=(8, 10))

# 逻辑判断, 如果涨跌幅大于0.5就标记为True 否则为False
print(stock_change > 0.5)
# 大于0.5的标记成1.1，布尔索引
stock_change[stock_change > 0.5] = 1.1

print(stock_change)

```




### 3.4.2 通用判断函数

 np.all(布尔值)

​	只要有一个False就返回False，只有全是True才返回True

np.any(布尔值)

​	只要有一个True就返回True，只有全是False才返回False

```python
# 判断stock_change[0:2, 0:5]是否全是上涨的
print(stock_change[0:2, 0:5] > 0)

print(np.all(stock_change[0:2, 0:5] > 0)) # 可能False

# 判断前5只股票这段期间是否有上涨的
print(np.any(stock_change[:5, :] > 0)) # 可能True
```



### 3.4.3 np.where (三元运算符)

​            np.where(布尔值, True的位置的值, False的位置的值)

```python
# 判断前四个股票前四天的涨跌幅 大于0的置为1，否则为0
temp = stock_change[:4, :4]

print(np.where(temp > 0, 1, 0))


# 判断前四个股票前四天的涨跌幅 大于0.5并且小于1的，换为1，否则为0
# 判断前四个股票前四天的涨跌幅 大于0.5或者小于-0.5的，换为1，否则为0
# (temp > 0.5) and (temp < 1)
np.logical_and(temp > 0.5, temp < 1)
print(np.where(np.logical_and(temp > 0.5, temp < 1), 1, 0)) #逻辑与

print(np.where(np.logical_or(temp > 0.5, temp < -0.5), 11, 3)) #逻辑或

```



### 3.4.4统计运算

统计指标函数
            min, max, mean, median, var, std
            np.函数名
            ndarray.方法名
返回最大值、最小值所在位置
            np.argmax(temp, axis=)
            np.argmin(temp, axis=)

```python
# 前四只股票前四天的最大涨幅
temp # shape: (4, 4) 0  1
temp.max(axis=0)# 第一层
np.max(temp, axis=-1) # 第二层
np.argmax(temp, axis=-1) #第二层，返回下标
```



## 3.5数组间的运算

### 3.5.1 场景

学生最终成绩 = 平时 、 期末

### 3.5.2 数组与数的运算

```python
arr = np.array([[1, 2, 3, 2, 1, 4], [5, 6, 1, 2, 3, 1]])
arr * 2
"""
[[ 2  4  6  4  2  8]
 [10 12  2  4  6  2]]
"""
```

### 3.5.3 数组与数组的运算

```python
arr1 = np.array([[1, 2, 3, 2, 1, 4], [5, 6, 1, 2, 3, 1]])
arr2 = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])
arr1 + arr2# 不可以运算，不满足广播机制
```



### 3.5.4 广播机制

执行broadcast 的前提在于两个ndarray 执行的是element-wise的运算，Broadcast机制的功能是为了方便不同形状的ndarray(numpy库的核心数据结构)进行数学运算。
当操作两个数组时，numpy会逐个比较它们的shape(构成的元组tuple)，只有在下述情况下，两个数组才能够进行数组与数组的运算。

- 维度相等
- shape(其中相对应的一个地方为1)

例如：

```
Image (3d array): 	256 x 256 x 3
Scale (1d array):             x 3
Result (3d array): 	256 X 256 X 3

A (4d array): 		9 x 1 x 7 X 1
B (3d array):   		8 x 1 X 5
Result (4d array): 	9 x 8 x 7 x 5

A (2d array): 		5 x 4
B (1d array):    		1
Result (2d array): 	5 x 4

A (3d array):	 	15 X 3 X 5
B (3d array): 		15 x 1 X 1
Result (3d array): 	15 x 3 X 5
```

```python
arr1 = np.array([[1, 2, 3, 2, 1, 4], [5, 6, 1, 2, 3, 1]]) # (2, 6)
arr2 = np.array([[1], [3]]) # (2, 1)
print(arr1 + arr2)
"""
array([[2, 3, 4, 3, 2, 5],
       [8, 9, 4, 5, 6, 4]])
"""
print(arr1 * arr2)
"""
array([[ 1,  2,  3,  2,  1,  4],
       [15, 18,  3,  6,  9,  3]])
"""
print(arr1 / arr2)
"""
array([[1.        , 2.        , 3.        , 2.        , 1.        ,
        4.        ],
       [1.66666667, 2.        , 0.33333333, 0.66666667, 1.        ,
        0.33333333]])
"""
```



### 3.5.5 矩阵运算

**1 什么是矩阵**
	矩阵matrix 二维数组
	矩阵 & 二维数组
	两种方法存储矩阵
          1）ndarray 二维数组
                        矩阵乘法：
                            np.matmul
                            np.dot
          2）matrix数据结构

​						矩阵乘法直接*

```python
# ndarray存储矩阵
data = np.array([[80, 86],
[82, 80],
[85, 78],
[90, 90],
[86, 82],
[82, 90],
[78, 80],
[92, 94]])

# matrix存储矩阵
data_mat = np.mat([[80, 86],
[82, 80],
[85, 78],
[90, 90],
[86, 82],
[82, 90],
[78, 80],
[92, 94]])

print(type(data_mat))#type(data_mat)
```

**2 矩阵乘法运算**
	形状
         (m, n) * (n, l) = (m, l)
	运算规则
          A (2, 3) B(3, 2)
          A * B = (2, 2)

```python
# 成绩 【平时，期末】
data = np.array([[80, 86],
                 [82, 80],
                 [85, 78],
                 [90, 90],
                 [86, 82],
                 [82, 90],
                 [78, 80],
                 [92, 94]])
weights = np.array([[0.3],
                    [0.7]])  # 权重

print(np.matmul(data, weights))

```



## 3.6.合并、分割

### 3.6.1合并

水平合并、垂直合并

```python
a = stock_change[:2, 0:4] #(2, 4)
b = stock_change[4:6, 0:4] #(2, 4)
#水平合并
np.hstack((a, b))
np.concatenate((a, b), axis=1)

#垂直合并
np.vstack((a, b))
np.concatenate((a, b), axis=0)

```



### 3.6.2分割

np.split()



## 3.7.IO操作与数据处理

### 3.7.1 Numpy读取

```python
data = np.genfromtxt("test.csv", delimiter=",")

""" 读取结果出问题，不用Numpy读取数据
array([[  nan,   nan,   nan,   nan],
       [  1. , 123. ,   1.4,  23. ],
       [  2. , 110. ,   nan,  18. ],
       [  3. ,   nan,   2.1,  19. ]])
"""

```

###  3.7.2 如何处理缺失值

两种思路：
	直接删除含有缺失值的样本
	替换/插补
        按列求平均，用平均值进行填补

```python
def fill_nan_by_column_mean(t):
    for i in range(t.shape[1]):
        # 计算nan的个数
        nan_num = np.count_nonzero(t[:, i][t[:, i] != t[:, i]])
        if nan_num > 0:
            now_col = t[:, i]
            # 求和
            now_col_not_nan = now_col[np.isnan(now_col) == False].sum()
            # 和/个数
            now_col_mean = now_col_not_nan / (t.shape[0] - nan_num)
            # 赋值给now_col
            now_col[np.isnan(now_col)] = now_col_mean
            # 赋值给t，即更新t的当前列
            t[:, i] = now_col
    return t

fill_nan_by_column_mean(data)
```



​        



#  Pandas 数据处理工具


​    Pandas基础处理
​        Pandas是什么？为什么用？
​        核心数据结构
​            DataFrame
​            Panel
​            Series
​        基本操作
​        运算
​        画图
​        文件的读取与存储

​	Pandas高级处理
​    	缺失值处理
   	 数据离散化
​    	合并
​    	交叉表与透视表
​    	分组与聚合
​    	综合案例

## 4.1Pandas介绍

**4.1.1 Pandas介绍 - 数据处理工具**

​        panel + data + analysis
​        panel面板数据 - 计量经济学 三维数据
**4.1.2 为什么使用Pandas**

​        便捷的数据处理能力
​        读取文件方便
​        封装了Matplotlib、Numpy的画图和计算

**4.1.3 DataFrame**

 结构：既有行索引，又有列索引的二维数组

```python
import numpy as np
import pandas as pd

# 创建一个符合正态分布的10个股票5天的涨跌幅数据
stock_change = np.random.normal(0, 1, (10, 5))

# 添加行索引
stock = ["股票{}".format(i) for i in range(10)]

# 添加列索引
date = pd.date_range(start="20180101", periods=5, freq="B")

print(pd.DataFrame(stock_change, index=stock, columns=date))
```



属性：
            shape 形状
            index 行标题
            columns 列标题
            values - ndarry 
            T 转置

```python
data.shape#(10, 5)
data.index#	Index(['股票0', '股票1', '股票2', '股票3', '股票4', '股票5', '股票6', '股票7', '股票8', '股票9'], dtype='object')
data.columns#DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
               '2018-01-05'],
              dtype='datetime64[ns]', freq='B')
data.values"""array([[-0.07726903,  0.40607587,  1.26740233,  1.48676212, -1.35987104],
       [ 0.28361364,  0.43101642, -0.77154311,  0.48286211, -0.30724683],
       [-0.98583786, -1.96339732,  0.31658224, -1.96541561, -0.39274454],
       [ 2.38020637,  1.47056011, -0.45253103, -0.77381961,  0.4822656 ],
       [ 2.05044671, -0.0743407 ,  0.10900497,  0.00982431, -0.06639766],
       [-1.62883603,  2.370443  , -0.14230101, -1.73515932,  1.6128039 ],
       [ 0.59420384,  0.09903473, -2.82975368,  0.63599429, -0.40809638],
       [ 1.27884397, -0.42832722,  1.07118356, -0.04453698, -0.19217219],
       [ 0.35350472, -0.73933626,  0.81653138, -0.40873922,  1.24391025],
       [-0.66201232, -0.53088568, -2.01276069,  0.03709581,  0.86862061]])"""
data.T """	股票0	股票1	股票2	股票3	股票4	股票5	股票6	股票7	股票8	股票9
2018-01-01	-0.077269	0.283614	-0.985838	2.380206	2.050447	-1.628836	0.594204	1.278844	0.353505	-0.662012
2018-01-02	0.406076	0.431016	-1.963397	1.470560	-0.074341	2.370443	0.099035	-0.428327	-0.739336	-0.530886
2018-01-03	1.267402	-0.771543	0.316582	-0.452531	0.109005	-0.142301	-2.829754	1.071184	0.816531	-2.012761
2018-01-04	1.486762	0.482862	-1.965416	-0.773820	0.009824	-1.735159	0.635994	-0.044537	-0.408739	0.037096
2018-01-05	-1.359871	-0.307247	-0.392745	0.482266	-0.066398	1.612804	-0.408096	-0.192172	1.243910	0.868621"""
```

 方法：
            head() 前几行
            tail() 后几行

```python
data.head(3)#前3行
data.tail(2)#后5行
```



DataFrame索引的设置
            1）修改行列索引值

```python
# data.index[2] = "股票88" 不能单独修改索引
stock_ = ["股票_{}".format(i) for i in range(10)]
data.index = stock_ #只能一起改
```



​            2）重设索引

```python
data.reset_index(drop=False) #drop=Flase 索引直接添加到头列，否则直接删掉
```



​            3）设置新索引

```python
df = pd.DataFrame({'month': [1, 4, 7, 10],
                   'year': [2012, 2014, 2013, 2014],
                   'sale': [55, 40, 84, 31]})
print(df)
# 以月份设置新的索引
print(df.set_index("month", drop=True)) # 设置新的索引后，扔掉这列

# 设置多个索引，以年和月份
new_df = df.set_index(["year", "month"])

print(new_df.index)
"""MultiIndex([(2012,  1),
           (2014,  4),
           (2013,  7),
           (2014, 10)],
          names=['year', 'month'])"""

print(new_df.index.names)  # ['year', 'month']

print(new_df.index.levels)  # [[2012, 2013, 2014], [1, 4, 7, 10]]
```

注:通过刚才的设置，这样DataFrame就变成了一个具有Multilndex的DataFrame。



​    2 Panel 被丢弃

​        DataFrame的容器

​    3 Series
​        带索引的一维数组
​        属性
​            index
​            values

​		

```python
sr = data.iloc[1, :]
sr.index
sr.values

pd.Series(np.arange(3, 9, 2), index=["a", "b", "c"]) #创建Series
pd.Series({'red':100, 'blue':200, 'green': 500, 'yellow':1000})
```



​    总结：
​        DataFrame是Series的容器
​        Panel是DataFrame的容器

## 4.2 基本数据操作

### 4.2.1 索引操作

​        1）直接索引  

​			不能直接数字索引

​            先列后行
​        2）按名字索引
​            loc
​        3）按数字索引
​            iloc
​        4）组合索引
​            数字、名字

### 4.2.2 赋值操作

对DataFrace当中的close列进行重新赋值为1

```python
#直接修改原来的值
data['close'] = 1
#或者
data.close = 1
```



### 4.2.3 排序

​        对内容排序
​            dataframe
​            series
​        对索引排序
​            dataframe
​            series

这部分记笔记太多了，去看官方文档吧！！！！



## 4.3 DataFrame运算

​    算术运算
​    逻辑运算
​        逻辑运算符
​            布尔索引
​        逻辑运算函数
​            query()
​            isin()
​    统计运算
​        min max mean median var std
​        np.argmax()
​        np.argmin()
​    自定义运算
​        apply(func, axis=0)True
​            func:自定义函数

## 4.4 Pandas画图

​    sr.plot()

## 4.5 文件读取与存储

​    4.5.1 CSV
​        pd.read_csv(path)
​            usecols=
​            names=
​        dataframe.to_csv(path)
​            columns=[]
​            index=False
​            header=False
​    4.5.2 HDF5
​        hdf5 存储 3维数据的文件
​            key1 dataframe1二维数据
​            key2 dataframe2二维数据
​        pd.read_hdf(path, key=)
​        df.to_hdf(path, key=)
​    4.5.3 JSON
​        pd.read_json(path)
​            orient="records"
​            lines=True
​        df.to_json(patn)
​            orient="records"
​            lines=True





## 4.6 高级处理-缺失值处理

​    1）如何进行缺失值处理
​        两种思路：
​            1）删除含有缺失值的样本
​            2）替换/插补
​        4.6.1 如何处理nan
​            1）判断数据中是否存在NaN
​                pd.isnull(df)
​                pd.notnull(df)
​            2）删除含有缺失值的样本
​                df.dropna(inplace=False)
​               替换/插补
​                df.fillna(value, inplace=False)
​         4.6.2 不是缺失值nan，有默认标记的
​            1）替换 ？-> np.nan
​                df.replace(to_replace="?", value=np.nan)
​            2）处理np.nan缺失值的步骤
​    2）缺失值处理实例

## 4.7 高级处理-数据离散化

​    性别 年龄
A    1   23
B    2   30
C    1   18
​    物种 毛发
A    1
B    2
C    3
​    男 女 年龄
A   1  0  23
B   0  1  30
C   1  0  18

    狗  猪  老鼠 毛发
A   1   0   0   2
B   0   1   0   1
C   0   0   1   1
one-hot编码&哑变量
4.7.1 什么是数据的离散化
    原始的身高数据：165，174，160，180，159，163，192，184
4.7.2 为什么要离散化
4.7.3 如何实现数据的离散化
    1）分组
        自动分组sr=pd.qcut(data, bins)
        自定义分组sr=pd.cut(data, [])
    2）将分组好的结果转换成one-hot编码
        pd.get_dummies(sr, prefix=)

## 4.8 高级处理-合并

​    numpy
​        np.concatnate((a, b), axis=)
​        水平拼接
​            np.hstack()
​        竖直拼接
​            np.vstack()
​    1）按方向拼接
​        pd.concat([data1, data2], axis=1)
​    2）按索引拼接
​        pd.merge实现合并
​        pd.merge(left, right, how="inner", on=[索引])

## 4.9 高级处理-交叉表与透视表

​    找到、探索两个变量之间的关系
​    4.9.1 交叉表与透视表什么作用
​    4.9.2 使用crosstab(交叉表)实现
​        pd.crosstab(value1, value2)
​    4.9.3 pivot_table

## 4.10 高级处理-分组与聚合

​    4.10.1 什么是分组与聚合
​    4.10.2 分组与聚合API
​        dataframe
​        sr





# 金融数据分析与挖掘





