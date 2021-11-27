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
