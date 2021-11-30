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

