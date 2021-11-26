# Dive into Deep Learning v2
This Markdown is noted for the course of Mu Li

## 序言

### 概述
教材URL： https://zh-v2.d2l.ai/chapter_preface/index.html
b站课程： https://space.bilibili.com/1567748478?spm_id_from=333.788.b_765f7570696e666f.2
**课程框架**
![](https://pic.imgdb.cn/item/619fb8f72ab3f51d914d429f.jpg)

* Part One，基础知识和预备知识。
  * Section1 深度学习入门课程
  * Section2 实践深度学习所需的前提条件
* Part Two，现代深度学习技术
  * Section3&4 深度学习的最基本概念和技术
  * Section5 深度学习计算的各种关键组件
  * Section6&7 卷积神经网络（CNN)
  * Section8&9 循环神经网络（RNN）
  * Section10 Attention is all your need
* Part Three，可伸缩性、效率、应用程序
  * Section11 训练重的集中常用优化算法
  * Section12 深度学习代码计算性能的几个关键因素
  * Section13 深度学习在cv中的主要应用
  * Section14&15 预测训练语言模型并将其应用到nlp任务。


### 黑箱性质


目前，某些直觉只能通过试错、小幅调整代码并观察结果来发展。理想情况下，一个优雅的数学理论可能会精确地告诉我们如何调整代码以达到期望的结果。不幸的是，这种优雅的理论目前还没有出现。尽管我们尽了最大努力，但仍然缺乏对各种技术的正式解释，这既是因为描述这些模型的数学可能非常困难，也是因为对这些主题的认真研究最近才进入高潮。

## 1.前言

深度学习推动： 计算机视觉、自然语言处理、医疗保健、基因组学等创新领域。

*学习算法*(learning algorithm)： 使用数据集来选择参数的元程序

*训练过程*: 
1. 从一个随机初始化参数的模型开始
2. 获取一些数据样本(包含Label)
3. 调整参数
4. 重复第2步和第3步，直至模型令人满意

![](https://pic.imgdb.cn/item/61a051ec2ab3f51d9178eece.jpg)

### 1.2 关键组件
1. 数据（data
2. 模型（model
3. 目标函数（objective function
4. 优化算法

#### 数据

每个数据集，由一个个样本(example)组成，大多时候，遵循**独立同分布**(independently and identically distributed,i.i.d)
*样本别名*： 数据点(data point)、数据实例(data instance)
通常 <u>每个样本</u> 由一组称为 **特征**(features) 或 **协变量**(covariates)


图像数据为例：
样本： 单张照片(200$\times$200 pixels)
特征： RGB三通道(像素值)
特征数： $200\times200\times3=120000$ 个数值组成
                        
每个样本的特征类别数量相同 $\implies$ 特征向量长度固定，该长度称为**维数**(dimensionality)

#### 模型
深度学习，关注功能强大的模型
——由神经网络错综复杂的交织在一起，包含层层数据转换

#### 目标函数
定义模型的优劣程度的度量
*损失函数*(loss function or cost function)

**常见损失函数**：
* 平方误差(squared error)
* 错误率
* 交叉熵 cross-entropy
* 。。。。。。

**训练数据集**： training dataset
**测试数据集**： test set

常见问题： **过拟合**(overfitting)

#### 优化算法
深度学习中，大多数优化算法都是基于 **梯度下降**(gradient descent

### 1.3 各种机器学习问题


#### 1.  监督学习
supervised learning

A example -- a pair of "feature-label"

* 回归问题： 涉及值的多少
* 分类问题： 涉及，什么是什么
  * 标记问题 *多标签分类*(multi-label classification)

* 搜索问题： 为集合中的每个元素分配相应的相关性分数，然后检索评级最高的元素。
* 推荐系统(recommender system)

####  序列学习

要求系统能够拥有“记忆”能力： 语音处理、视频处理、医学方面的应用（病人机体形况监视
  1. 标记与解析
  2. 自动语音识别
  3. 文本到语音
  4. 机器翻译


#### 2. 无监督学习 
unsupervised learning

* 聚类(clustering)： 没有标签的情况下对数据进行分类
* *主成分分析*(principal component analysis)： 找到少量的参数来准确地**捕捉**数据的**线性相关属性**。
* 因果关系(causality)和概率图模型(probabilistic graphical models)问题
* 生成对抗网络(generative adversarial networks): GAN

### 1.4 与环境互动

* 离线学习(offline learning)

![](https://pic.imgdb.cn/item/61a0a52e2ab3f51d919e1dde.jpg)

* 在线学习(online learning)
相对于离线学习，增加了 **智能体**agent 与 **环境** env 互动的流程

### 1.5 强化学习
reinforcement learning

Deep Q network for Atari/AlphaGo

![](https://pic.imgdb.cn/item/61a0a63c2ab3f51d919ea075.jpg)

* 马尔可夫决策过程 (markov decision process)
  * 上下文赌博机(contextual bandit problem)
  * 多臂赌博机(multi-armed bandit problem)


### 1.6 深度学习的一些trick

* dropout 减轻过拟合，往整个神经网络中注入噪声
* **注意力机制** 在不增加 可学习参数的情况下增加系统的记忆力和复杂度。 
* GAN 神经网络的想象力

### 1.7 summary
* 机器学习 研究计算机系统如何利用经验（通常为数据）来提高特定任务的性能。
* 表示学习作为机器学习的一类，其研究的重点是如何自动找到合适的数据表示方式。是通过学习多层次的转换来进行的多层次的表示学习。
* 由廉价传感器和互联网规模应用所产生的大量数据，以及（通过GPU）算力的突破来触发的。
* 有效的深度学习框架的开源使得这一点的设计和实现变得非常容易。

