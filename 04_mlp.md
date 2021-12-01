# Multilayer perceptron
### noted by Car_pe

感知机： 

![](https://pic.imgdb.cn/item/61a5ed8b2ab3f51d91e95549.jpg)

## 隐藏层

在网络中加入 *一个或多个* **隐藏层** 来克服线性模型的限制，使其能处理更输出到上面的层，指导生成最后的输出。

最简单方法： 徐多全连接层堆叠在一起

![](https://pic.imgdb.cn/item/61a5d9152ab3f51d91db2d45.jpg)

## 线性到非线性

矩阵 $\mathbf{X} \in \mathbb{R}^{n\times d}$ 来表示 $n$ 个样本的小批量，其中每个样本具有 $d$个输入(features)。
对于具有 $h$ 个隐藏单元的单隐藏层多层感知机，$\mathbf{H} \in \mathbb{R}^{n\times h}$ 表示隐藏层的输出，称为 *隐藏表示*(hidden representations)  aka hidden-layer variable or hidden variable

全连接中，隐藏层与输出层**参数等价**
按如下方式计算单隐藏层多层感知机的输出 $\mathbf{O}\in \mathbf{R}^{n\times q}$:
$$
\mathbf{H} = \mathbf{XW}^{(1)} + \mathbf{b}^{(1)}, \\
\mathbf{O} = \mathbf{HW}^{(2)} + \mathbf{b}^{(2)}.
$$

1. 添加隐藏层后，模型需要跟踪和更新额外的参数
2. 只是单纯的仿射变换，没有任何额外好处

## 激活函数 activation function $\sigma$

引入激活函数，MLP将无法退化成线性模型：
$$
\mathbf{H}=\sigma(\mathbf{XW}^{(1)}+\mathbf{b}^{1}), \\
\mathbf{O}=\mathbf{HW}^{(2)}+\mathbf{b}^{(2)}.
$$

由于 $X$ 中的每一行对应于小批量中的一个样本，出于记号习惯的考量，我们定义 **非线性函数**$\sigma$ 也以按行的方式作用于其输入，即一次计算一个样本。

激活函数的输出(ex, $\sigma(\cdot)$) *激活值*(activations)

## 通用近似定理
多层感知机(MLP) 是 **通用近似器**
* MLP可以通过隐藏神经元捕捉到我们输入之间复杂的相互作用
* 即使是网络只有一个隐藏层，给定足够的神经元（可能非常多）和正确的权重，也可以对**任意函数建模**，


## 常见激活函数

* 激活函数通过计算加权和并加上偏置来确定神经元是否应该被激活。
* 大多数激活函数都是非线性的
### ReLU
最受欢迎的选择 *线性整流单元*(Rectified Linear Unit, ReLU)

ReLU求导很方便，sigmoid的运算需要用到指数运算。一次指数运算$\approx$百次乘法运算

$$
ReLU(x) = max(x, 0).
$$
当输入为负时，ReLU函数的导数为0，而当输入为正时，ReLU函数的导数为1

* 求导表现得特别好：要么让参数消失，要么让参数通过。
* ReLU减轻了困扰以往神经网络的梯度消失问题。
$$
pReLU(x) = \max(0,x)+\alpha\min(0,x).
$$

### sigmoid 函数

对于一个定义域在 $\mathbb{R}$ 中的输入， *sigmoid函数* 将输入变换为区间 (0,1) 上的输出。
因此，sigmoid总是被称为 *挤压函数*(squashing function)
将 (-inf, inf) 中的任意输入压缩到区间 (0, 1)

$$
sigmoid(x)=\frac{1}{1+\exp{(-x)}}.
$$

**sigmoid 可以被视为 softmax 的特例** 输入越接近于0，越接近线性

$$
\frac{d}{dx}sigmoid(x)=\frac{\exp{(-x)}}{(1+\exp{(-x)})^2}=sigmoid(x)(1-sigmoid(x)).
$$

### tanh 函数
与 sigmoid 类似，tanh (双曲正切)函数也能将其压缩转换到区间(-1, 1)上。tanh函数公式如下：
$$
tanh(x)=\frac{1-\exp{(-2x)}}{1+\exp{(-2x)}}.
$$
tanh函数的导数是：
$$
\frac{d}{dx}tanh(x)=1-tanh^2(x)
$$

* 多层感知机MLP在输出层儿喊输入层之间增加一个或多个全连接的隐藏层，并通过激活函数转换隐藏层的输出。
* 常用的激活函数包括 ReLU函数、Sigmoid函数和tanh函数。


### 传统训练感知机

![](https://pic.imgdb.cn/item/61a5ed8b2ab3f51d91e95549.jpg)

条件： $y_i\left[<w,x_i>+b\right] \le 0$
反应 预测与label 相悖

$$
w \leftarrow w + y_iw_i \\
b \leftarrow b + y_i
$$
令 $Z = y_i\left[<w,x_i>+b\right]  $
代入条件表达式后，$Z$ 加上一个**模值平方**，使得 $Z\to 0^-$

等价于使用 batch_size=1 Gradient Descent, Loss Function: 
$$
l(y, x, w) =\max(0, -y<w,x>)
$$

 #### 收敛定理
 * 数据在半径$r$内
 * 余量$\rho$分类两类  $y(\mathbf{x}^T\mathbf{w}+b)\ge \rho$
对于$\Vert \mathbf{w} \Vert^2+b^2\le 1$
 * 感知机保证在 $\frac{r^2+1}{\rho^2}$ 步后收敛

感知机不能拟合 XOR函数
**感知机只能产生线性分割面**（Minsky&Papert，1969）

1. 感知机是一个二分类模型，最早的AI模型之一
2. 它的求解算法等价于使用批量大小为1的梯度下降
3. 不能拟合XOR函数，导致第一次AI寒冬


### 多层感知机

#### 单隐藏层-单分类
* 输入 $\mathbf{x}\in \mathbb{R}^n$
* 隐藏层 $\mathbf{W}_1\in \mathbb{R}^{m\times n},\mathbf{b}_1\in \mathbb{R}^m$
* 输出层 $\mathbf{w}_2\in\mathbb{R}^m, b_2\in \mathbb{R}$
$$
\begin{aligned}
&\mathbf{h}=\sigma(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1) \\
&o=\mathbf{w}^T_2\mathbf{h}+b_2
\end{aligned}
$$
$\sigma$ 是按元素的激活函数

## 多类分类
与softmax回归的主要区别在于，有Hidden layer

hyperparameters:
* 隐藏层数
* 每层隐藏层的大小

$$
y_1,y_2,\cdots,y_k = softmax(o_1,o_2,\cdots,o_k)
$$

![](https://pic.imgdb.cn/item/61a613fe2ab3f51d9105c6b6.jpg)

* 输入$\mathbf{x}\in\mathbb{R}^n$
* 隐藏层 $\mathbf{W}_1\in\mathbb{R}^{m\times n},\mathbf{b}_1\in\mathbb{R}^m$
* 输出层 $\mathbf{W}_2\in \mathbb{R}^{m\times k}, \mathbf{b}_2\in\mathbb{R}^k$

$$
\begin{aligned}
& \mathbf{h}=\sigma(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1) \\
& \mathbf{o}=\mathbf{W^T_2h}+\mathbf{b}_2 \\
& \mathbf{y}=\mathbf{softmax(o)}
\end{aligned} 
$$

输出不需要激活函数，激活函数主要用于避免层数塌陷。

* MLP使用隐藏层和激活函数来得到非线性模型
* 常用激活函数 Sigmoid，Tanh，ReLU
* 使用 Softmax 来处理多类分类
* 超参数为隐藏层数，和各个隐藏层大小

### 模型选择、欠拟合和过拟合

机器学习的目的：发现 *模式*(pattern), 捕捉训练集所来自的潜在总体的规律。

#### 过拟合 overfitting
将模型在训练数据上拟合得比在潜在分布中更接近的现象。

用于对抗过拟合的技术： 正则化 regularization


### 训练误差和泛化误差

训练误差(training error): 在训练数据集上计算得到的误差。
泛化误差(generalization error): 在同样的原始样本的分布中抽取的无限多的数据样本，模型误差的期望

泛化误差的定义使其难以实现。一般将模型应用于一个独立的测试集来估计泛化误差

#### 验证数据集和测试数据集
这两种数据集很容易搞错
* 验证数据集： 一个用来评估模型好坏的数据集
  * 例如拿出 50% 的训练数据
  * 不要跟训练数据集混在一起（常犯错误
* 测试数据集： 只用一次的数据集（不能用来调整超参数）。例如
  * 未来的考试
  * 我出价的房子的实际成交价
  * 用在 Kaggle 私有排行榜中的数据集

#### K-则交叉验证
* 在没有足够多数据时使用（常态）
* 算法：
  * 将训练数据分割成K块
  * For i = 1,..., k
    * 使用第i块作为验证数据集，其余作为训练数据集
  * 报告K个验证机误差的平均
* 常用： K=5 或 10

**总结*
* 训练数据集： 训练模型参数
* 验证数据集： 选择模型超参数
* 非大数据集上通常使用 k-fold cross- validation


## 统计学习理论
泛化是机器学习的基本问题

在<u>**同名定理 (eponymous theorem**)</u>中，格里文科和坎特利推导出了训练误差收敛到泛化误差的速率。后来理论被扩展到更一般种类的函数。

假设，训练数据和测试数据都是 *相统的* 分布中 *独立* 提取的。

<u>**独立同分布假设**</u> 意味着，对数据进行采样的过程没有进行“记忆”。

训练模型是，试图找一个能够尽可能拟合训练数据的函数。
若该函数灵活到可以捕捉真是模式一样容易地捕捉到干扰的模式，就会 **过拟合**。

深度学习，有徐多启发式的技术旨在防止过拟合。

### 模型复杂性
简单模型&大量数据： 泛化与训练误差相近
复杂模型&少量样本： 训练误差下降，泛化误差增大
**复杂性评估**
* 更多参数
* 参数更大取值反围
* 需要更多迭代

表达能力有限但仍能很好地姐是数据到模型可能更有显示用途。

1. 调整参数的数量。当可调整参数的数量（*自由度*）很大，模型更容易过拟合
2. 参数采用的值。权重的取值范围大，易过拟合。
3. 训练样本的数量。

## 模型选择
在机器学习中，通常在评估几个候选模型后选择最终的模型。

### 欠拟合还是过拟合
训练误差和验证误差都很严重，但他们之间仅有一点差距。
1. 模型过于简单（即表达能力不足），无法捕获试图学习的模式。 *欠拟合*(underfitting)

2. 训练误差明星低于验证误差， *过拟合*(overfitting)

最终，我们通常更关心 **验证误差**

### 模型复杂性

单个特征 $x$ 和对应实数标签 $y$ 组成的训练数据，找到下面的$d$阶多项式来估计标签$y$

$$
\hat{y} = \sum\limits^d_{i=0}x^iw_i
$$
特征是$x$的幂给出的，模型的权重是$w_i$给出的，偏置是$w_0$给出的（因为对于所有的 $x$

![](https://pic.imgdb.cn/item/61a7088d2ab3f51d91980038.jpg)

训练数据集中样本越少，月可能（更严重地）过拟合

**模型容量**
* 拟合各种函数的能力
* 低容量的模型难以集合训练数据
* 高容量的模型可以记住所有的训练数据

**估计模型容量**
* 难以在不同的种类算法之间比较
  * 例如树模型何神经网络
* 给定一个模型种类，将有两个主要因素
  * 参数的个数
  * 参数值的选择范围


### VC 维
统计学习理论的一个核心思想
对于分类模型，VC等于一个最大的数据集的大小，无论如何给定标号，都存在模型来完美分类0 

**线性分类器的VC维**
* 2维输入的感知机，VC维=3
  * 能够分类任何三个点，但不是4个(xor)

  * 支持N维输入的感知机的 $VC$ 维是 $N+1$
  * 一些多层感知机的VC维 $O(N\log_2N)$

* 提供为什么一个模型好的理论依据
  * 它可以衡量训练误差何泛化误差之间的间隔
* 但深度学习中很少使用
  * 衡量不是很准确
  * 计算深度学习模型的VC维很困难

**数据复杂度**
* 多个重要因素
  * 样本个数
  * 每个样本的元素个数
  * 时间、空间结构
  * 多样性

**总结**
* 模拟容量需要匹配数据复杂度，否则可能导致欠拟合何过拟合
* 统计机器学习提供数学工具来衡量模型复杂度
* 实际中一般靠观察训练误差和验证误差

