# 预备知识
### noted by CARPE

所有机器学习方法都涉及从数据中提取信息。

#### 关于数据
**基本操作**：
1. 存储
2. 操作
3. 数据预处理

## 2.1 数据操作
1. 获取数据
2. 数据读入计算机后，进行处理。

**数据存储结构——张量**(tensor)
NumPy仅支持CPU计算； Tensor支持AutoGrad
```python
import torch
x = torch.arange(12)
x
```
arange(12)： 创建 0->11 为元素的元素集合
默认格式为： 浮点数
默认计算方式： 存储于内存 CPU计算

```python
x.shape
OUTPUT: 
torch.Size([12])
x.numel()
OUTPUT:
12
X=x.reshape(3,4)
OUTPUT:
tensor([[0,1,2,3],
        [4,5,6,7],
        [8,9,10,11]])
```
shape属性反映Tensor的形状，numel()方法反映元素个数(num of elements)，reshape()方法，改变Tensor的shape， Tensor的大小(size=numel)

reshape(h,w) // *高度或宽度，有一项即可，令一项设置为-1会被隐式计算出*

```python
torch.zeros((2,3,4))        ## 生成shape全零张量
torch.ones((2,3,4))         ## 生成shape全一张量
torch.randn(3,4)            ## 生成shape正态分布随机数张量

## 利用包含Python_list的列表来为所需Tensor中的每个元素赋予确定值
torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
```

#### 按元素操作 element-wise

$$
f: \mathbb{R} \rightarrow \mathbb{R}
$$
表示 函数从任何实数($\mathbb{R}$)映射到另一个实数。

$$
f:\mathbb{R},\mathbb{R}\rightarrow\mathbb{R}
$$
二元标量运算符，接收两个输入，并产生一个输出
$$
f: \boldsymbol{c}=F(\boldsymbol{u},\boldsymbol{v})
$$
计算方法： $c_i \leftarrow f(u_i,v_i)$
推广至**按向量元素**运算：
$$
F： \mathbb{R}^d,\mathbb{R}^d\rightarrow \mathbb{R}^d。
$$


```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```
cat()方法将(X,Y)这两个Tensor在dim上连接合并

```python
## 逻辑运算符来构成张量
X==Y
OUTPUT:
tensor([[False,  True, False,  True],
        [False, False, False, False],
        [False, False, False, False]])
```

sum()方法求和


#### 广播机制
**Broadcasting mechanism**

当两个Tensor的形状不匹配的时候，两者之间的按元素运算会在矩阵 **广播**（broadcasting）条件下运行

### 切片与索引

切片：      索引，

### 节省内存

```python
### 下面的Y是不会原地更新的，而是重新开了一片内存
before = id(Y)
Y = Y + X
id(Y) == before

Output: False;
```
因为模型训练过程中，可学习参数非常多，经常需要迭代，必须学会**原地迭代**

```python
### 执行原地操作
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

Output: 
id(Z): 140272150341696
id(Z): 140272150341696
```
*如果存在自变量迭代现象*

```python
### 内存减少方法
before = id(X)
X += Y      ### or X[:] = X + Y
id(X) == before
```

### 转换为其他 Python 对象
Tensor和NumPy之间可以很好的做转换


## 2.2 数据预处理
Pandas是主角

1. 把csv文件读入
2. 做缺失值的处理（均值&去除
3. 生成Tensor


## 2.3 线性代数

### 标量

标量变量： 小写字母表示($x、y、z$)
实数标量的空间： $\mathbb{R}$
属于符号： $\in$

### 向量
向量可以视为元素值组成的列表


## 2.4 微分 differential calculus

拟合模型的任务--两个关键问题：
1. 优化optimization：
用模型拟合观测数据的过程；
2. 泛化generalization:
数学原理和实践者的智慧，能够指导我们生成出有效性超出用于训练的数据集本身的模型。

### 导数与微分

假设有一个函数$f:\mathbb{R}^n \rightarrow \mathbb{R}$

$f$的 *导数* 被定义为
$$
f'(x)=\lim\limits_{h\to 0}\frac{f(x+h)-f(x)}{h},
$$

#### 导数的几个等价符号
$$
f'(x)=y'=\frac{dy}{dx}=\frac{df}{dx}=\frac{d}{dx}f(x)=Df(x)=D_xf(x),
$$
其中符号 $\frac{d}{dx}$ 和 $D$ 是微分运算符，表示*微分* 操作。

![](https://pic.imgdb.cn/item/61a21e382ab3f51d912af548.jpg)

导数反映，切线斜率

### 偏导数

设 $y=f(x_1,x_2,\cdots,x_n)$ 是一个具有$n$个变量的函数。$y$ 关于第 $i$ 个参数 $x_i$ 的偏导数为：

$$
\frac{\partial{y}}{\partial{x_i}} = \lim\limits_{h\to 0}\frac{f(x_1,\cdots,x_{i-1},x_i+h,x_{i+1},\cdots,x_n)-f(x_1,\cdots,x_i,\cdots,x_n)}{h}
$$
为了计算$\frac{\partial y}{\partial x_i}$

**偏导数等价表示**

$$
\frac{\partial{y}}{\partial{x_i}}=\frac{\partial{f}}{\partial{x_i}}=f_{xi}=f_i=D_i f = D_{xi}f.
$$

### 梯度 gradient

设函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的输入是一个$n$维向量 $\mathbf{x}=\left[x_1,x_2,\cdots,x_n\right]^T$, 并且输出是一个标量。
函数$f(x)$相对于$\mathbf{x}$的梯度是一个包含$n$个偏导数的向量：
$$
\nabla_{\mathbf{x}}f(x)=\left[\frac{\partial{f(x)}}{\partial{x_1}},\frac{\partial{f(x)}}{\partial{x_2}},\cdots,\frac{\partial{f(x)}}{\partial{x_n}}\right]^T,
$$
$\nabla_{\mathbf{x}}f(x)$通常在没有歧义时被$\nabla f(x)$取代

假设$\mathbf{x}$为$n$维向量，在微分多元函数时经常使用一下规则：
* 对于所有 $\mathbf{A}\in \mathbb{R}^{m\times n}$, 都有$\nabla_{mathbb{x}}\mathbf{Ax}=\mathbf{A}^T$
* 对于所有 $\mathbf{A}\in \mathbb{R}^{m\times n}$, 都有$\nabla_{\mathbf{x}}\mathbf{x}^T\mathbf{A}=\mathbf{A}$
* 对于所有 $\mathbf{A}\in \mathbb{R}^{m\times n}$, 都有$\nabla_{\mathbf{x}}\mathbf{x}^T\mathbf{Ax}=(\mathbf{A}+\mathbf{A}^T)\mathbf{x}$
* $\nabla_{mathbf{x}}\Vert\mathbf{x}\Vert^2=\nabla_{mathbf{x}}\mathbf{x}^T\mathbf{x}=2\mathbf{x}$

同样，对于任何矩阵$\mathbf{X}$，都有$\nabla_{\mathbf{X}}\Vert\mathbf{X}\Vert^2_F=2\mathbf{X}$ (这里的表达式比较重要)

### Chain Rule
深度学习中，多元函数通常是 *复合*(composite)的

假设函数 $y=f(u)$ 和 $u=g(x)$ 都是可微的，根据链式法则：
$$
\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}
$$
更一般的场合：
$$
\frac{dy}{dx_i}=\frac{dy}{du_1}+\frac{dy}{du_2}\frac{du_2}{dx_i}+\cdots+\frac{dy}{du_m}\frac{du_m}{dx_i}
$$

### AutoML

* 链式法则:
$$
\frac{\partial y}{\partial x}=\frac{\partial y}{\partial u_n}\frac{\partial u_n}{\partial_{n-1}}\cdots\frac{\partial u_2}{\partial u_1}\frac{\partial u_1}{\partial x}
$$

1. 正向累计：
$$
\frac{\partial y}{\partial x}=\frac{\partial y}{\partial u_n}(\frac{\partial u_n}{\partial u_{n-1}}(\cdots(\frac{\partial u_2}{\partial u_1}\frac{\partial u_1}{\partial x})))
$$



