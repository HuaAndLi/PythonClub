[TOC]



# 1、深度学习介绍

百度

# 2、TensorFlow框架的使用

​    1）TensorFlow的结构
​    2）TensorFlow的各个组件
​        图
​        会话
​        张量
​        变量
​    3）简单的线性回归案例 - 将TensorFlow用起来

## 1.1 深度学习与机器学习的区别

## 1.1.1 特征提取方面

## 1.1.2 数据量和计算性能要求

## 1.1.3 算法代表

## 1.2 深度学习的应用场景

## 1.3 深度学习框架介绍

### 1.3.1 常见深度学习框架对比

### 1.3.2 TensorFlow的特点

### 1.3.3 TensorFlow的安装

​        1 CPU版本
​        2 GPU版本
​            CPU:诸葛亮
​                综合能力比较强
​                核芯的数量更少
​                更适用于处理连续性（sequential）任务。
​            GPU:臭皮匠
​                专做某一个事情很好
​                核芯的数量更多
​                更适用于并行（parallel）任务

## 2.1 TF数据流图

### 2.1.1 案例：TensorFlow实现一个加法运算

​    2 TensorFlow结构分析
​        一个构建图阶段
​            流程图：定义数据（张量Tensor）和操作（节点Op）
​        一个执行图阶段
​            调用各方资源，将定义好的数据和操作运行起来

### 2.1.2 数据流图介绍

​        TensorFlow
​        Tensor - 张量 - 数据
​        Flow - 流动

## 2.2 图与TensorBoard

### 2.2.1 什么是图结构

​        图结构：
​            数据（Tensor） + 操作（Operation）

### 2.2.2 图相关操作

​        1 默认图
​            查看默认图的方法
​                1）调用方法
​                    用tf.get_default_graph()
​                2）查看属性
​                    .graph
​        2 创建图
​            new_g = tf.Graph()
​            with new_g.as_default():
​                定义数据和操作

### 2.2.3 TensorBoard:可视化学习

​        1 数据序列化-events文件
​            tf.summary.FileWriter(path, graph=sess.graph)
​        2 tensorboard

### 2.2.4 OP

​        数据：Tensor对象
​        操作：Operation对象 - Op
​        1 常见OP
​            操作函数        &                           操作对象
​            tf.constant(Tensor对象)           输入Tensor对象 -Const-输出 Tensor对象
​            tf.add(Tensor对象1, Tensor对象2)   输入Tensor对象1, Tensor对象2 - Add对象 - 输出 Tensor对象3
​        2 指令名称
​            一张图 - 一个命名空间

## 2.3 会话

### 2.3.1 会话

​        tf.Session：用于完整的程序当中
​        tf.InteractiveSession：用于交互式上下文中的TensorFlow ，例如shell
​        1）会话掌握资源，用完要回收 - 上下文管理器
​        2）初始化会话对象时的参数
​            graph=None
​            target：如果将此参数留空（默认设置），
​            会话将仅使用本地计算机中的设备。
​            可以指定 grpc:// 网址，以便指定 TensorFlow 服务器的地址，
​            这使得会话可以访问该服务器控制的计算机上的所有设备。
​            config：此参数允许您指定一个 tf.ConfigProto
​            以便控制会话的行为。例如，ConfigProto协议用于打印设备使用信息
​        3)run(fetches,feed_dict=None)
​        3 feed操作
​            a = tf.placeholder(tf.float32, shape=)
​            b = tf.placeholder(tf.float32, shape=)

## 2.4 张量Tensor

​    print()
​    ndarray

### 2.4.1 张量(Tensor)

​        张量 在计算机当中如何存储？
​        标量 一个数字                 0阶张量
​        向量 一维数组 [2, 3, 4]       1阶张量
​        矩阵 二维数组 [[2, 3, 4],     2阶张量
​                    [2, 3, 4]]
​        ……
​        张量 n维数组                  n阶张量
​        1 张量的类型
​        2 张量的阶
​        创建张量的时候，如果不指定类型
​        默认 tf.float32
​            整型 tf.int32
​            浮点型 tf.float32

### 2.4.2 创建张量的指令

### 2.4.3 张量的变换

​        ndarray属性的修改
​            类型的修改
​                1）ndarray.astype(type)
​                tf.cast(tensor, dtype)
​                    不会改变原始的tensor
​                    返回新的改变类型后的tensor
​                2）ndarray.tostring()
​            形状的修改
​                1）ndarray.reshape(shape)
​                    -1 自动计算形状
​                2）ndarray.resize(shape)
​                静态形状 - 初始创建张量时的形状
​                1）如何改变静态形状
​                    什么情况下才可以改变/更新静态形状？
​                        只有在形状没有完全固定下来的情况下
​                    tensor.set_shape(shape)
​                2）如何改变动态形状
​                    tf.reshape(tensor, shape)
​                    不会改变原始的tensor
​                    返回新的改变形状后的tensor
​                    动态创建新张量时，张量的元素个数必须匹配

### 2.4.4 张量的数学运算

## 2.5 变量OP

​    TensorFlow - 变量
​    存储模型参数

### 2.5.1 创建变量

​        变量需要显式初始化，才能运行值

### 2.5.2 使用tf.variable_scope()修改变量的命名空间

​        使得结构更加清晰

## 2.6 高级API

### 2.6.1 其它基础API

### 2.6.2 高级API

## 2.7 案例：实现线性回归

### 2.7.1 线性回归原理复习

​        1）构建模型
​            y = w1x1 + w2x2 + …… + wnxn + b
​        2）构造损失函数
​            均方误差
​        3）优化损失
​            梯度下降

### 2.7.2 案例：实现线性回归的训练

​        准备真实数据
​            100样本
​            x 特征值 形状 (100, 1)
​            y_true 目标值 (100, 1)
​            y_true = 0.8x + 0.7
​        假定x 和 y 之间的关系 满足
​            y = kx + b
​            k ≈ 0.8 b ≈ 0.7
​            流程分析：
​            (100, 1) * (1, 1) = (100, 1)
​            y_predict = x * weights(1, 1) + bias(1, 1)
​            1）构建模型
​            y_predict = tf.matmul(x, weights) + bias
​            2）构造损失函数
​            error = tf.reduce_mean(tf.square(y_predict - y_true))
​            3）优化损失
​            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
​            5 学习率的设置、步数的设置与梯度爆炸

### 2.7.3 增加其他功能

​        1 增加变量显示
​            1）创建事件文件
​            2）收集变量
​            3）合并变量
​            4）每次迭代运行一次合并变量
​            5）每次迭代将summary对象写入事件文件
​        2 增加命名空间
​        3 模型的保存与加载
​            saver = tf.train.Saver(var_list=None,max_to_keep=5)
​            1）实例化Saver
​            2）保存
​                saver.save(sess, path)
​            3）加载
​                saver.restore(sess, path)
​        4 命令行参数使用
​            1）tf.app.flags
​            tf.app.flags.DEFINE_integer("max_step", 0, "训练模型的步数")
​            tf.app.flags.DEFINE_string("model_dir", " ", "模型保存的路径+模型名字")
​            2）FLAGS = tf.app.flags.FLAGS
​            通过FLAGS.max_step调用命令行中传过来的参数
​            3、通过tf.app.run()启动main(argv)函数

# 3.数据的读取、神经网络

## 3.1 文件读取流程

​    多线程 + 队列

###     3.1.1 文件读取流程

​        1）构造文件名队列
​            file_queue = tf.train.string_input_producer(string_tensor,shuffle=True)
​        2）读取与解码
​            文本：
​                读取：tf.TextLineReader()
​                解码：tf.decode_csv()
​            图片：
​                读取：tf.WholeFileReader()
​                解码：
​                    tf.image.decode_jpeg(contents)
​                    tf.image.decode_png(contents)
​            二进制：
​                读取：tf.FixedLengthRecordReader(record_bytes)
​                解码：tf.decode_raw()
​            TFRecords
​                读取：tf.TFRecordReader()
​            key, value = 读取器.read(file_queue)
​            key：文件名
​            value：一个样本
​        3）批处理队列
​            tf.train.batch(tensors, batch_size, num_threads = 1, capacity = 32, name=None)
​        手动开启线程
​            tf.train.QueueRunner()
​            开启会话：
​                tf.train.start_queue_runners(sess=None, coord=None)

## 3.2 图片数据

###     3.2.1 图像基本知识

​        文本  特征词 -> 二维数组
​        字典  one-hot -> 二维数组
​        图片  像素值
​        1 图片三要素
​            黑白图、灰度图
​                一个通道
​                    黑[0, 255]白
​            彩色图
​                三个通道
​                    一个像素点 三个通道值构成
​                    R [0, 255]
​                    G [0, 255]
​                    B [0, 255]
​        2 TensorFlow中表示图片
​            Tensor对象
​                指令名称、形状、类型
​                shape = [height, width, channel]
​        3 图片特征值处理
​            [samples, features]
​            为什么要缩放图片到统一大小？
​            1）每一个样本特征数量要一样多
​            2）缩小图片的大小
​            tf.image.resize_images(images, size)
​        4 数据格式
​            存储：uint8
​            训练：float32

###     3.2.4 案例：狗图片读取

​        1）构造文件名队列
​            file_queue = tf.train.string_input_producer(string_tensor,shuffle=True)
​        2）读取与解码
​            读取：
​                reader = tf.WholeFileReader()
​                key, value = reader.read(file_queue)
​            解码：
​                image_decoded = tf.image.decode_jpeg(value)
​        3）批处理队列
​            image_decoded = tf.train.batch([image_decoded], 100, num_threads = 2, capacity=100)
​        手动开启线程

## 3.3 二进制数据

​    tensor对象
​        shape:[height, width, channel] -> [32, 32, 3] [0, 1, 2] -> []
​        [[32 * 32的二维数组],
​        [32 * 32的二维数组],
​        [32 * 32的二维数组]]
​            --> [3, 32, 32] [channel, height, width] 三维数组的转置 [0, 1, 2] -> [1, 2, 0]
​            [3, 2] -转置-> [2, 3]
​        1）NHWC与NCHW
​        T = transpose 转置

### 3.3.2 CIFAR10 二进制数据读取

​    流程分析：
​        1）构造文件名队列
​        2）读取与解码
​        3）批处理队列
​        开启会话
​        手动开启线程

## 3.4 TFRecords

###     3.4.1 什么是TFRecords文件

###     3.4.2 Example结构解析

​        cifar10
​            特征值 - image - 3072个字节
​            目标值 - label - 1个字节
​        example = tf.train.Example(features=tf.train.Features(feature={
​        "image":tf.train.Feature(bytes_list=tf.train. BytesList(value=[image])
​        "label":tf.train.Feature(int64_list=tf.train. Int64List(value=[label]))
​        }))
​        example.SerializeToString()

###     3.4.3 案例：CIFAR10数据存入TFRecords文件

​        流程分析

###     3.4.4 读取TFRecords文件API

​        1）构造文件名队列
​        2）读取和解码
​            读取
​            解析example
​            feature = tf.parse_single_example(value, features={
​            "image":tf.FixedLenFeature([], tf.string),
​            "label":tf.FixedLenFeature([], tf.int64)
​            })
​            image = feature["image"]
​            label = feature["label"]
​            解码
​            tf.decode_raw()
​        3）构造批处理队列

## 3.5 神经网络基础

###     3.5.1 神经网络

​        输入层
​            特征值和权重 线性加权
​            y = w1x1 + w2x2 + …… + wnxn + b
​            细胞核-激活函数
​                sigmoid
​                sign
​        隐藏层
​        输出层
​    单个神经元 - 感知机
​    感知机(PLA: Perceptron Learning Algorithm))
​        x1, x2
​        w1x1 + w2x2 + b = 常数
​        w2x2 = -w1x1 - b + 常数
​        x2 = kx1 + b
​        x2 = kx1 + b
​        x1 x2
​        与问题
​        0   0 0
​        0   1 0
​        1   0 0
​        1   1 1
​        异或问题
​        0   0 0
​        0   1 1
​        1   0 1
​        1   1 0
​        单个神经元不能解决一些复杂问题
​        1）多层神经元
​        2）增加激活函数

## 3.6 神经网络原理

​    逻辑回归
​        y = w1x1 + w2x2 + …… + wnxn + b
​        sigmoid -> [0, 1] -> 二分类问题
​        损失函数：对数似然损失
​    用神经网络进行分类
​        假设函数
​            y_predict =
​            softmax - 多分类问题
​        构造损失函数
​            loss = 交叉熵损失
​        优化损失
​            梯度下降

###         3.6.1 softmax回归 - 多分类问题

​            假设要进行三分类
​            2.3, 4.1, 5.6

###         3.6.2 交叉熵损失

## 3.7 案例：Mnist手写数字识别

###     3.7.1 数据集介绍

​        1 特征值
​            [None, 784] * W[784, 10] + Bias = [None, 10]
​            构建全连接层：
​            y_predict = tf.matmul(x, W) + Bias
​            构造损失：
​            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict,name=None)
​            如何计算准确率?
​            np.argmax(y_predict, axis=1)
​            tf.argmax(y_true, axis=1)
​                y_predict [None, 10]
​                y_true [None, 10]
​            tf.equal()
​            如何提高准确率？
​                1）增加训练次数
​                2）调节学习率
​                3）调节权重系数的初始化值
​                4）改变优化器

# 4.卷积神经网络

## 4.1 卷积神经网络简介

​    1）与传统多层神经网络对比
​        输入层
​        隐藏层
​            卷积层
​            激活层
​            池化层
​                pooling layer
​                subsample
​            全连接层
​        输出层
​    2）发展历史
​    3）卷积网络ImageNet比赛错误率

## 4.2 卷积神经网络原理

​    卷积神经网络 - 结构
​        卷积层
​            通过在原始图像上平移来提取特征
​        激活层
​            增加非线性分割能力
​        池化层
​            减少学习的参数，降低网络的复杂度（最大池化和平均池化）
​        全连接层

### 4.2.2 卷积层（Convolutional Layer）

​    卷积核 - filter - 过滤器 - 卷积单元 - 模型参数
​        个数
​        大小 1*1 3*3 5*5
​            卷积如何计算？
​            输入
​                5*5*1 filter 3*3*1 步长 1
​            输出
​                3*3*1
​        步长
​            输入
​                5*5*1 filter 3*3*1 步长2
​            输出
​                2*2*1
​        零填充的大小
​    6 总结-输出大小计算公式
​    7 多通道图片如何观察
​        输入图片
​            5*5*3 filter 3*3*3 + bias 2个filter 步长2
​            H1=5
​            D1=3
​            K=2
​            F=3
​            S=2
​            P=1
​            H2=(5-3+2)/2+1=3
​            D2=2

​    输出
​        3*3*2
卷积网络API
​    tf.nn.conv2d(input, filter, strides=, padding=)
​    input：输入图像
​        要求：形状[batch,heigth,width,channel]
​        类型为float32,64
​    filter:
​        weights
​        变量initial_value=random_normal(shape=[F, F, 3/1, K])
​    strides:
​        步长 1
​         [1, 1, 1, 1]
​    padding: “SAME”
​        “SAME”：越过边缘取样
​        “VALID”：不越过边缘取样
1）掌握filter要素的相关计算公式
2）filter大小
​    1x1，3x3，5x5
  步长 1
3）每个过滤器会带有若干权重和1个偏置
4.2.3 激活函数
​    sigmoid
​    1/(1+e^-x)
​        1）计算量相对大
​        2）梯度消失
​        3）输入的值的范围[-6, 6]
​    Relu的好处
​        1）计算速度快
​        2）解决了梯度消失
​        3）图像没有负的像素值
​    tf.nn.relu(features)

### 4.2.4 池化层(Polling)

​    利用了图像上像素点之间的联系
​    tf.nn.max_pool(value, ksize=, strides=, padding=)
​        value:
​            4-D Tensor形状[batch, height, width, channels]
​        ksize：
​           池化窗口大小，[1, 2, 2, 1]
​        strides:
​            步长大小，[1, 2, 2, 1]
​        padding：“SAME”

## 4.3 案例：CNN-Mnist手写数字识别

###     4.3.1 网络设计

​        第一个卷积大层：
​            卷积层：
​                32个filter 大小5*5 步长：1 padding="SAME"
​                 tf.nn.conv2d(input, filter, strides=, padding=)
​                 input：输入图像 [None, 28, 28, 1]
​                     要求：形状[batch,heigth,width,channel]
​                     类型为float32,64
​                 filter:
​                     weights = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 1, 32]))
​                     bias = tf.Variable(initial_value=tf.random_normal(shape=[32]))
​                     变量initial_value=random_normal(shape=[F, F, 3/1, K])
​                 strides:
​                     步长 1
​                      [1, 1, 1, 1]
​                 padding: “SAME”
​                     “SAME”：越过边缘取样
​                     “VALID”：不越过边缘取样
​                 输出形状：
​                 [None, 28, 28, 32]
​            激活：
​                Relu
​                tf.nn.relu(features)
​            池化：
​                输入形状：[None, 28, 28, 32]
​                大小2*2 步长2
​                输出形状：[None, 14, 14, 32]
​        第二个卷积大层：
​            卷积层：
​                64个filter 大小5*5 步长：1 padding="SAME"
​                输入：[None, 14, 14, 32]
​                tf.nn.conv2d(input, filter, strides=, padding=)
​                input：[None, 14, 14, 32]
​                    要求：形状[batch,heigth,width,channel]
​                    类型为float32,64
​                filter:
​                    weights = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, 64]))
​                    bias = tf.Variable(initial_value=tf.random_normal(shape=[64]))
​                    变量initial_value=random_normal(shape=[F, F, 3/1, K])
​                strides:
​                    步长 1
​                     [1, 1, 1, 1]
​                padding: “SAME”
​                    “SAME”：越过边缘取样
​                    “VALID”：不越过边缘取样
​                输出形状：
​                [None, 14, 14, 64]
​            激活：
​                Relu
​                tf.nn.relu(features)
​            池化：
​                输入形状：[None, 14, 14, 64]
​                大小2*2 步长2
​                输出形状：[None, 7, 7, 64]
​        全连接
​            tf.reshape()
​            [None, 7, 7, 64]->[None, 7*7*64]
​            [None, 7*7*64] * [7*7*64, 10] = [None, 10]
​            y_predict = tf.matmul(pool2, weithts) + bias
​        调参->提高准确率？
​        1）学习率
​        2）随机初始化的权重、偏置的值
​        3）选择好用的优化器
​        4）调整网络结构

## 4.4 网络结构与优化

###     4.4.1 网络的优化和改进

###     4.4.2 卷积神经网络的拓展了解

​        1 常见网络模型
​        2 卷积网络其它用途

## 4.5 实战：验证码图片识别

​    验证码识别实战
​        1）数据集
​            图片1 -> NZPP 一个样本对应4个目标值 -> sigmoid交叉熵
​            一张手写数字的图片 -> 0~9之间的某一个数 一个样本对应一个目标值 -> softmax交叉熵
​            切割 -> 不具备通用性
​            [0,0,1,0……]
​            NZPP -> [13, 25, 15, 15]
​            [4, 26]
​            -> [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
​                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
​                [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
​                [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]
​        2）对数据集中
​            特征值 目标值 怎么用
​        3）如何分类？
​            如何比较输出结果和真实值的正确性？
​            如何衡量损失？
​                手写数字识别案例
​                    softmax+交叉熵
​                        [4, 26] -> [4*26]
​                sigmoid交叉熵
​            准确率如何计算？
​                核心：对比真实值和预测值最大值所在位置
​                手写数字识别案例
​                y_predict[None, 10]
​                tf.argmax(y_predict, axis=1)
​                y_predict[None, 4, 26]
​                tf.argmax(y_predict, axis=2/-1)
​                [True,
​                True,
​                True,
​                False] -> tf.reduce_all() -> False
​        4）流程分析
​            1）读取图片数据
​                filename -> 标签值
​            2）解析csv文件，将标签值NZPP->[13, 25, 15, 15]
​            3）将filename和标签值联系起来
​            4）构建卷积神经网络->y_predict
​            5）构造损失函数
​            6）优化损失
​            7）计算准确率
​            8）开启会话、开启线程
​        5）代码实现
