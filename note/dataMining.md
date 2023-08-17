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
                #  一级标题
                - 缩进

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



3 设置画布属性与图片保存



### 2.2.2完善原始折线图1（辅助显示层）

1 准备数据并画出初始折线图
2 添加自定义X,y刻度
3 中文显示问题解决
4 添加网格显示
5 添加描述信息

### 2.2.3完善原始折线图2（图像层）

1 多次plot
2 设置图形风格

### 2.2.4多个坐标系显示-plt.subplots(面向对象的画图方法)

### 2.2.5折线图的应用场景



**2.2.1 折线图绘制与保存图片**

​            3 设置画布属性与图片保存
​                figsize : 画布大小
​                dpi : dot per inch 图像的清晰度
​            3 中文显示问题解决
​                mac的一次配置，一劳永逸
​                ubantu每创建一次新的虚拟环境，需要重新配置
​                windows
​                1）安装字体
​                    mac/wins：双击安装
​                    ubantu：双击安装
​                2）删除matplotlib缓存文件
​                3）配置文件
​        2.2.4 多个坐标系显示-plt.subplots(面向对象的画图方法)
​            figure, axes = plt.subplots(nrows=1, ncols=2, **fig_kw)
​            axes[0].方法名()
​            axes[1]
​        2.2.5 折线图的应用场景
​            某事物、某指标随时间的变化状况
​            拓展：画各种数学函数图像
2.3.1 常见图形种类及意义
​    折线图plot
​    散点图scatter
​        关系/规律
​    柱状图bar
​        统计/对比
​    直方图histogram
​        分布状况
​    饼图pie π
​        占比

## 2.3 散点图

​    

2.3.2 散点图绘制



##     2.4 柱状图(bar)

​        2.4.1 柱状图绘制
   ##  2.5 直方图(histogram)

​        2.5.1 直方图介绍
​            组数：在统计数据时，我们把数据按照不同的范围分成几个组，分成的组的个数称为组数
​            组距：每一组两个端点的差
​            已知 最高175.5 最矮150.5 组距5
​            求 组数：(175.5 - 150.5) / 5 = 5
​        2.5.2 直方图与柱状图的对比

            1. 直方图展示数据的分布，柱状图比较数据的大小。
                        2. 直方图X轴为定量数据，柱状图X轴为分类数据。
            
                        3. 直方图柱子无间隔，柱状图柱子有间隔
                                4. 直方图柱子宽度可不一，柱状图柱子宽度须一致
                            2.5.3 直方图绘制
                                    x = time
                                    bins 组数 = (max(time) - min(time)) // 组距
                            3 直方图注意点
            
                ## 2.6 饼图(pie)
                
                ​        %1.2f%%
                ​        print("%1.2f%%")

# Numpy 高效的运算工具



#  Pandas 数据处理工具



# 金融数据分析与挖掘





