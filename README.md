# 分布式矩阵计算：稀疏与稠密矩阵乘法运算
任务：实现一个支持大规模稀疏/稠密矩阵乘法的分布式计算框架，运行于多节点 HPC 集群，任务包括：
- 支持矩阵分块、任务划分（按行或按块）
- 稀疏矩阵支持CSR/CSC格式，高效乘法
- 结果可用于后续线性代数计算（如 PageRank）

---

# 一、任务本质

要求实现：

[
C=A\times B
]

其中：

* A、B 可以是稠密矩阵(Dense Matrix)
* A、B 也可以是稀疏矩阵(Sparse Matrix)

而且矩阵规模可能非常大：

例如

[
100000\times100000
]

这种规模：

* 单机内存难以存储
* 单核计算极慢

因此必须：

1. 数据分散到多个节点
2. 每个节点负责一部分计算
3. 最后汇总结果

这就是分布式矩阵计算。

---

# 二、矩阵分块（Matrix Partitioning）

README第一条：

> 支持矩阵分块

意思是：

不要把整个矩阵放在一个进程。

例如矩阵：

[
A=
\begin{bmatrix}
1&2&3&4\
5&6&7&8\
9&10&11&12\
13&14&15&16
\end{bmatrix}
]

---

## 方法1：按行划分（Row Partition）

MPI Rank 0：

[
\begin{bmatrix}
1&2&3&4\
5&6&7&8
\end{bmatrix}
]

MPI Rank 1：

[
\begin{bmatrix}
9&10&11&12\
13&14&15&16
\end{bmatrix}
]

即：

```text
Rank0 -> 前半部分行
Rank1 -> 后半部分行
```

这是最简单的方法。

---

## 方法2：二维分块（Block Partition）

划成多个子块：

[
\begin{bmatrix}
A_{11} & A_{12}\
A_{21} & A_{22}
\end{bmatrix}
]

例如：

[
A_{11}=
\begin{bmatrix}
1&2\
5&6
\end{bmatrix}
]

[
A_{12}=
\begin{bmatrix}
3&4\
7&8
\end{bmatrix}
]

等等。

然后：

```text
Rank0 -> A11
Rank1 -> A12
Rank2 -> A21
Rank3 -> A22
```

这种方式：

* 通信更均衡
* 扩展性更好

也是很多 HPC 库的做法。

---

# 三、任务划分（Task Scheduling）

README：

> 任务划分（按行或按块）

这是在说：

谁负责算哪部分结果矩阵。

---

例如：

[
C=A\times B
]

结果矩阵：

[
C=
\begin{bmatrix}
c_{11}&c_{12}\
c_{21}&c_{22}
\end{bmatrix}
]

---

按行划分：

```text
Rank0:
计算 C 的前半部分行

Rank1:
计算 C 的后半部分行
```

例如：

```text
Rank0 -> C[0:5000,:]

Rank1 -> C[5000:10000,:]
```

---

按块划分：

```text
Rank0 -> C11
Rank1 -> C12
Rank2 -> C21
Rank3 -> C22
```

这也是经典二维并行乘法。

---

# 四、OpenMP 与 MPI 如何配合

这是项目最可能考察的重点。

---

## MPI

负责节点之间

例如：

```text
Node1
Node2
Node3
Node4
```

每个节点一个 MPI Process。

---

MPI负责：

```cpp
MPI_Send
MPI_Recv
MPI_Bcast
MPI_Scatter
MPI_Gather
```

实现：

```text
数据分发
数据交换
结果汇总
```

---

## OpenMP

负责节点内部

例如：

```text
Node1
 ├ Thread0
 ├ Thread1
 ├ Thread2
 └ Thread3
```

利用共享内存并行。

例如：

```cpp
#pragma omp parallel for
for(int i=0;i<n;i++)
{
    ...
}
```

---

典型结构：

```text
MPI
 ├ Rank0
 │   ├ OpenMP Thread0
 │   ├ OpenMP Thread1
 │   └ OpenMP Thread2
 │
 ├ Rank1
 │   ├ OpenMP Thread0
 │   ├ OpenMP Thread1
 │   └ OpenMP Thread2
```

这叫：

**Hybrid MPI + OpenMP**

也是 HPC 最常见方案。

---

# 五、什么是 CSR 格式

README：

> 稀疏矩阵支持 CSR/CSC

这是很多同学第一次接触的概念。

---

假设矩阵：

[
A=
\begin{bmatrix}
1&0&0&2\
0&0&3&0\
4&0&0&5
\end{bmatrix}
]

大多数元素是0。

如果直接存：

```text
1 0 0 2
0 0 3 0
4 0 0 5
```

浪费空间。

---

## CSR

Compressed Sparse Row

按行压缩。

---

只存非零元素：

### values

```text
[1,2,3,4,5]
```

---

### col_idx

记录列号：

```text
[0,3,2,0,3]
```

---

### row_ptr

记录每行起始位置：

```text
[0,2,3,5]
```

表示：

```text
Row0 -> values[0:2]

Row1 -> values[2:3]

Row2 -> values[3:5]
```

---

最终：

```cpp
struct CSR {
    vector<double> values;
    vector<int> col_idx;
    vector<int> row_ptr;
};
```

存储量：

[
O(nnz)
]

而不是

[
O(nm)
]

其中：

[
nnz=非零元素个数
]

---

# 六、什么是 CSC 格式

CSC：

Compressed Sparse Column

按列压缩。

---

CSR：

```text
按行访问快
```

适合：

[
y=Ax
]

---

CSC：

```text
按列访问快
```

适合：

[
A^T x
]

或者一些稀疏乘法算法。

---

项目要求同时支持：

```text
CSR
CSC
```

说明你需要设计统一接口。

例如：

```cpp
class SparseMatrix
{
public:
    virtual multiply(...) = 0;
};
```

然后：

```cpp
CSRMatrix
CSCMatrix
```

继承实现。

---

# 七、高效稀疏矩阵乘法

这是整个项目最难的部分。

---

稠密矩阵：

[
O(n^3)
]

经典三重循环：

```cpp
for(i)
 for(j)
  for(k)
```

---

稀疏矩阵：

大部分元素是0。

如果仍然三重循环：

```text
大量无意义计算
```

---

例如：

[
A=
\begin{bmatrix}
1&0&0\
0&0&2\
0&0&0
\end{bmatrix}
]

只需要处理：

```text
1
2
```

两个非零值。

---

CSR乘向量：

[
y=Ax
]

只遍历：

```cpp
for(nonzero element)
```

复杂度：

[
O(nnz)
]

而不是：

[
O(n^2)
]

---

稀疏矩阵乘法（SpGEMM）

[
C=A\times B
]

实际只处理：

```text
A非零
B非零
```

的组合。

很多实现使用：

```cpp
unordered_map
```

或者：

```cpp
hash accumulator
```

累积结果。

这是工业级稀疏矩阵库的核心思想。

---

# 八、PageRank 为什么被提到

README最后：

> 结果可用于后续线性代数计算（如 PageRank）

说明项目不仅仅要得到矩阵乘法结果。

还要支持：

[
y=Ax
]

这样的运算。

因为 PageRank 本质上是：

[
x_{k+1}=Ax_k
]

不断迭代。

---

因此你的框架最好提供：

```cpp
DenseMatrix
SparseMatrix

multiply(Matrix, Matrix)

multiply(Matrix, Vector)
```

而不仅仅是：

```cpp
matrix × matrix
```

---

# 九、如果这是课程项目，我建议的实现路线

### 第一阶段（必做）

实现：

```text
MPI 行划分
Dense × Dense
```

即：

```text
Rank0 算前几行
Rank1 算后几行
```

先跑通。

---

### 第二阶段

加入：

```text
OpenMP
```

实现：

```cpp
#pragma omp parallel for
```

加速局部计算。

---

### 第三阶段

实现：

```text
CSR
```

支持：

[
Ax
]

和简单稀疏乘法。

---

### 第四阶段

实现：

```text
Block Partition
```

二维分块分布式计算。

---

如果后续你要真正开始写这个框架，我还可以进一步帮你设计一个比较完整的架构（类设计、MPI通信流程、CSR数据结构、目录组织，以及课程项目答辩时容易拿高分的性能测试方案）。
