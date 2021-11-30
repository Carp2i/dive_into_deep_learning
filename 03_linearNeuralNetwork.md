## 线性神经网络
Linear Nerual Network

### 线性回归 Linear Regression
在美国买房例子
* 假设1： 影响房价的关键因素是 **卧室个数**，**卫生间个数**，和**居住面积**，记为 $x_1, x_2, x_3$
* 假设2： 成交价是关键因素的加权和
$$
y = w_1x_1+w_2x_2+w_3x_3+b
$$
权重和偏差的实际值在后面决定

#### 1. 线性模型
* 给定n维输入 $\mathbf{x}=\left[x_1,x_2,\cdots,x_n\right]$
* 线性模型有一个n维权重和一个标量偏差
$$
\mathbf{w}=\left[w_1,w_2,\cdots,w_n\right]^T, b
$$
* 输出是输入的加权和
$$
y = w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$
向量版本： $y=<\mathbf{w},\mathbf{x}>+b$

#### 2. 衡量估计质量
* 比较真实值和估计值，例如房屋售价和估价
* 假设 $y$ 是真实值， $\hat{y}$ 是估计值，我们可以比较
$$
l(y,\hat{y})=\frac{1}{2}(y-\hat{y})^2
$$
这个叫做 **平方损失**

#### 3.训练数据
* 收集一些数据点来决定参数值（权重和偏差），例如过去6约卖的房子
* 这被称之为训练数据
* 通常越多越好
* 假设我们有n个样本，记
$$
\mathbf{X}=\left [x_1,x_2,\cdots,x_n\right]^T \quad \mathbf{y}=\left[y_1,y_2,\cdots,y_n\right]^T
$$

#### 4. 参数学习
* 训练损失
$$
l(\mathbf{X},\mathbf{y},\mathbf{w},b)=\frac{1}{2n}\sum\limits_{i=1}^n(y_i-<\mathbf{x}_i,\mathbf{w}>-b)^2=\frac{1}{2n}\Vert \mathbf{y}-\mathbf{Xw}-b \Vert^2
$$
* 最小化损失来学习参数
$$
\mathbf{w}^*,\mathbf{b}^* = \arg \min\limits_{\mathbf{w},b} l(\mathbf{X,y,w,}b)
$$

#### 显示解
线性回归是少有的p问题，可以通过矩阵运算来求得 **显示解**

#### 线性回归Summary

![](https://pic.imgdb.cn/item/61a327f92ab3f51d91b37583.jpg)

### 基础优化算法

#### Gradient Descent
* 挑选一个随机初始值 $\mathbf{w}_0$
* 重复迭代参数 t=1,2,3
$$
\mathbf{w}_t=\mathbf{w}_{t-1}-\eta\frac{\partial l}{\partial \mathbf{w}_{t-1}}
$$
    * 沿梯度方向 将增加损失函数值
    * **学习率**： 步长的超参数(hyperparameter)

![](https://pic.imgdb.cn/item/61a32a362ab3f51d91b46400.jpg)

过小的learning rate时间成本高，过大的learning rate难以收敛

#### 小批量batch 随机梯度下降
* 在整个训练集上算梯度太贵
  * 一个深度神经网络模型可能需要数分钟至数小时
* 可以随机采样b个样本$i_1,i_2,\cdots,i_b$来近似损失
$$
\frac{1}{b}\sum\limits_{i\in I_b}l(\mathbf{X}_i, y_i, \mathbf{w})
$$
  * b 是**批量大小**(batch_size)，另一个重要的超参数

batch_size的选择：
1. 不能太大，内存消耗增加，浪费计算，若所有样本都是相统的
2. 不能太小，每次计算太小，不适合并行来最大利用计算资源

#### Summary of Gradient Descent

![](https://pic.imgdb.cn/item/61a32bc82ab3f51d91b4f445.jpg)


### 线性回归SUMMARY
* 机器学习模型中的关键要素是 **训练数据**， **损失函数**， **优化算法**， **模型**
* 矢量化是数学表达上更简洁，同时运行的更快
* 最小化目标函数和执行**最大似然估计**等价
* 线性回归模型也是神经网络


## Softmax Regression

虽然叫做 Regression 但是解决的问题还是**分类**

* 回归估计一个连续纸
* 分类预测一个离散类别

MNIST： 手写数字识别(10类)
ImageNet: 自然物体分类(1000类)

回归
  * 单连续数值输出
  * 自然区间$\mathbb{R}$
  * 跟真实值的区别作为损失

分类
  * 通常多个输出
  * 输出i是预测为第i类的置信度

### 从回归到多分类
1. 对类别进行一位有效编码 one-hot encoding
$$
\mathbf{y} = \left[y_1,y_2,\cdots,y_n\right] \\
y_i = \left\{ 
\begin{aligned}
1& \quad , i =y \\
0& \quad , otherwise
\end{aligned}
\right.
$$

2. 使用均方损失训练ian
3. 最大值最为预测
$$
\hat{y} = \argmax\limits_i o_i
$$

#### 分类问题
1. 对样本硬性类别感兴趣，即属于哪个类别
2. 希望得到软性类别（**属于各个类别概率**）。

#### 网络结构

仿射函数（affine function）
有4个特征和3个可能的输出类别，需要12个标量来表示权重（带下标的 $w$)，3个标量来表示偏置（带下标的$b$)
下面为每个输入计算三个 *未归一化的预测*(logit): $o_1$、$o_2$ 和 $o_3$ 

$$
o_1 = x_1w_{11}+x_2w_{12}+x_3w_{13}+x_4w_{14}+b_1 \\
o_2 = x_1w_{21}+x_2w_{22}+x_3w_{23}+x_4w_{24}+b_2 \\
o_3 = x_1w_{31}+x_2w_{32}+x_3w_{33}+x_4w_{34}+b_3 \\
$$

与线性回归一样，softmax回归也是一个单层神经网络

![](https://pic.imgdb.cn/item/61a42f212ab3f51d9108d376.jpg)


### 全连接层的参数开销

对于任何具有 d 个输入和 q 个输出的全连接层，参数开销为 $O(dq)$

2021年的论文） 可以将$d \to 1$的输出成本降低到 $O(\frac{dq}{n})$


#### softmax 运算
将模型的输出视作为概率，优化参数以最大化观测数据的概率

保证在任何数据上的输出都是非负的且总和为1
每个求幂后的结果除以他们的总和为1.

在分类器输出0.5的所有样本中，希望这些样本有一般实际上属于预测类。这个属性叫做 *校准* calibration

$$
\hat{\mathbf{y}} = softmax(0) \quad \hat{y_i} = \frac{\exp( o_j)}{\sum_k \exp(o_k)}
$$
在预测过程： $\argmax\limits_j \hat{y}_i = \argmax\limits_j o_j.$

### 小批量样本矢量化
假设，一个小批量的样本$\mathbf{X}$其中特征维度（输入数量）为d, 批量大小为n。 那么小批量特征为 $\mathbf{X} \in \mathbf{R}^{n\times d}$, 权重为 $\mathbb{W} \in \mathbf{R}^{d\times q}$, 偏置为$\mathbb{b}\in \mathbf{R}^{1\times q}$

softmax回归的适量计算表达式为：
$$
\mathbf{O} = \mathbf{XW} + \mathbb{b}, \\
\hat{\mathbf{Y}} = softmax(\mathbf{O})
$$

### 损失函数
#### 对数似然
softmax函数给出一个向量 $\hat{\mathbb{y}}$，视为给定任意输入$\mathbb{x}$的每个类的估计条件概率

整个数据集$\left\{X，Y \right\}$ 具有n个样本

$$
P(Y\vert X) = \prod\limits_{i=1}^n P(y^{(i)},\hat{x}^{(i)})
$$

根据**最大似然估计**，最大化$P(Y\vert X)$，相当于最小化负对数似然：
$$
-\log P(Y\vert X) = \sum\limits_{i=1}^n -\log P(y^{(i)},\hat{x}^{(i)}) = \sum\limits^n_{i=1}l(\mathbb{y}^{(i)}, \hat{\mathbb{y}}^(i)), 
$$
其中，对于任何标签 $\mathbb{y}$ 何模型预测 $\hat{\mathbb{y}}$，损失函数为：
$$
l(\mathbb{y},\hat{\mathbb{y}}) = -\sum\limits_{j=1}^q y_i\log\hat{y}_i
$$
上式的损失函数，成为 *交叉熵损失* (cross-entropy loss)

由于 y 是一个长度为 q 的独热编码向量，所以除了一个项以外的所有项 j 都消失了。由于所有 y^j 都是预测的概率，所以它们的对数永远不会大于 0 。 因此，如果正确地预测实际标签，即，如果实际标签 P(y∣x)=1 ，则损失函数不能进一步最小化。 注意，这往往是不可能的。例如，数据集中可能存在标签噪声（某些样本可能被误标），或输入特征没有足够的信息来完美地对每一个样本分类。

### 常用损失函数
#### L2 Loss
$$
l(y, y')= \frac{1}{2}(y-y')^2
$$

#### L1 Loss
绝对值损失函数

$$
l(y, y')= \vert y-y' \vert
$$

#### Huber's Robust Loss

$$
l(y, y') =
\left\{
\begin{aligned}
\vert y-y' \vert -\frac{1}{2} \quad if \vert y-y' \vert > 1 \\
\frac{1}{2}(y-y')^2 \quad otherwise
\end{aligned}
\right.
$$


### softmax及其导数
$$
\begin{aligned}
l(\mathbb{y},\hat{\mathbb{y}}) &= -\sum\limits^q_{j=1}y_j\log\frac{\exp(o_j)}{\sum^q_{k=1}\exp({o_k})} \\
&= \sum\limits^q_{j=1}y_i\log\sum\limits^q_{k=1}\exp(o_k)-\sum\limits^q_{j=1}y_jo_j \\
&=\log\sum\limits^q_{k=1}\exp(o_k)-\sum\limits^q
_{j=1}y_jo_j
\end{aligned}
$$

为了更好地理解发生了什么，考虑相对于任何未归一化的预测 $o_j$ 的导数。
$$
\partial_{o_j}l(\mathbf{y},\hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum^q_{k=1}\exp(o_k)} - y_j = softmax(o)_j - y_j
$$

softmax的梯度计算非常容易。

#### cross-entropy loss
信息论中的概念

## 信息论基础
*信息论* 涉及编码、解码、发送以及尽可能简洁地处理信息或数据

### 熵
信息论的核心思想是 量化数据中的信息内容，在信息论中，该数据之被称为：
分布$P$的熵(entropy)。
$$
H[P] = \sum\limits_j -P(j)\log{P(j)}
$$

信息论基本定理指出：
为了从分布 $p$ 中随机抽取的数据进行比那吗，至少需要$H[P]$ nat 对其进行编码

上式的对数底为e： 一个纳特是 $\frac{1}{\log 2} \approx 1.44$位


### 惊异

cloud Shannon 用下式来量化一个人的 *惊异*(surprisal)
在观察事件j，并赋予它（主观）概率$P(j)$
$$
\log{\frac{1}{P(j)}} = -\log{P(j)}
$$

**交叉熵**是分配的概率真正匹配数据生成过程时的 *预期惊异*(expected surprisal)

![](https://pic.imgdb.cn/item/61a4bdd42ab3f51d91988f11.jpg)


### 重新审视Softmax的实现

![](https://pic.imgdb.cn/item/61a5aa382ab3f51d91b508a5.jpg)

$$
\begin{aligned}
\log (\hat{y}_j) &= \log{\left(\frac{\exp(o_j)}{\sum_k\exp(o_k)} \right)} \\
& =\log(\exp(o_j))-\log(\sum_k\exp(o_k))\\
&=o_j -\log{\left(\sum_k\exp(o_k)\right)}
\end{aligned}
$$