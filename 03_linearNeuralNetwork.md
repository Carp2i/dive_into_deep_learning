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

