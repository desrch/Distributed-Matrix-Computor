# 分布式矩阵计算框架总体设计

## 1. 项目目标

实现一个基于 MPI + OpenMP 的分布式矩阵计算框架，支持：

* 稠密矩阵（Dense Matrix）
* 稀疏矩阵（CSR / CSC）
* 分布式矩阵分块
* 多节点并行计算
* OpenMP 线程级加速
* Matrix × Matrix
* Matrix × Vector
* 后续 PageRank 等迭代算法

---

# 2. 总体架构

系统划分为六层：

Application Layer
↓
Algorithm Layer
↓
Distributed Runtime Layer
↓
Matrix Layer
↓
Storage Layer
↓
Communication Layer

---

# 3. 目录结构

src/

├── matrix/
│   ├── Matrix.hpp
│   ├── DenseMatrix.hpp
│   ├── CSRMatrix.hpp
│   ├── CSCMatrix.hpp
│
├── partition/
│   ├── Partitioner.hpp
│   ├── RowPartitioner.hpp
│   ├── BlockPartitioner.hpp
│
├── runtime/
│   ├── DistributedContext.hpp
│   ├── TaskScheduler.hpp
│
├── communication/
│   ├── MPIWrapper.hpp
│
├── operators/
│   ├── DenseMultiply.hpp
│   ├── SparseMultiply.hpp
│   ├── MatrixVector.hpp
│
├── algorithms/
│   ├── PageRank.hpp
│
├── utils/
│   ├── Timer.hpp
│   ├── Logger.hpp
│
└── main.cpp

---

# 4. Matrix层设计

统一矩阵接口。

class Matrix
{
public:

```
virtual int rows() const = 0;

virtual int cols() const = 0;

virtual bool isSparse() const = 0;

virtual ~Matrix() {}
```

};

---

DenseMatrix

class DenseMatrix : public Matrix
{
private:

```
int rows_;
int cols_;

std::vector<double> data_;
```

};

采用连续内存存储：

data[i * cols + j]

---

CSRMatrix

class CSRMatrix : public Matrix
{
private:

```
int rows_;
int cols_;

std::vector<double> values;

std::vector<int> col_idx;

std::vector<int> row_ptr;
```

};

---

CSCMatrix

class CSCMatrix : public Matrix
{
private:

```
int rows_;
int cols_;

std::vector<double> values;

std::vector<int> row_idx;

std::vector<int> col_ptr;
```

};

---

# 5. 分块模块

统一接口：

class Partitioner
{
public:

```
virtual PartitionInfo partition(
    int rows,
    int cols,
    int world_size
) = 0;
```

};

---

RowPartitioner

按行切分。

Rank0:
0 ~ 2499

Rank1:
2500 ~ 4999

Rank2:
5000 ~ 7499

Rank3:
7500 ~ 9999

---

BlockPartitioner

二维切分。

A11 A12
A21 A22

适合后期扩展。

---

# 6. MPI运行时

DistributedContext

负责：

MPI_Init

MPI_Finalize

MPI_Comm_rank

MPI_Comm_size

封装：

class DistributedContext
{
public:

```
int rank();

int size();

bool isMaster();
```

};

避免项目各处直接调用 MPI API。

---

# 7. 通信层

MPIWrapper

封装：

MPI_Bcast

MPI_Scatter

MPI_Gather

MPI_Allgather

MPI_Reduce

例如：

broadcastMatrix()

scatterRows()

gatherResult()

后续替换通信策略会非常方便。

---

# 8. Task Scheduler

负责决定：

哪个进程计算哪部分。

class TaskScheduler
{
public:

```
Task assignTask();
```

};

输入：

矩阵规模

进程数

分块方式

输出：

当前Rank负责的数据范围

---

# 9. 算子层

统一矩阵运算入口。

class MatrixOperator
{
public:

```
virtual Matrix multiply(
    const Matrix& A,
    const Matrix& B
) = 0;
```

};

---

DenseMultiply

负责：

Dense × Dense

---

SparseMultiply

负责：

CSR × CSR

CSR × CSC

CSR × Dense

---

MatrixVector

负责：

A × x

PageRank后续直接复用。

---

# 10. OpenMP加速位置

Dense矩阵乘法：

#pragma omp parallel for

for(i)
{
for(j)
{
...
}
}

---

CSR SpMV：

#pragma omp parallel for

for(row)
{
...
}

并行遍历行。

---

# 11. PageRank扩展

后续实现：

class PageRank
{
public:

```
std::vector<double> run(
    CSRMatrix& graph
);
```

};

核心计算：

x(k+1)=A*x(k)

直接调用：

MatrixVector

无需重新写并行框架。

---

# 12. 开发路线

阶段1：

完成

DenseMatrix
MPI运行时
RowPartitioner
DenseMultiply

实现：

Dense × Dense

---

阶段2：

加入

OpenMP

优化Dense乘法。

---

阶段3：

实现

CSRMatrix

以及：

CSR × Vector

---

阶段4：

实现

CSR × CSR

稀疏矩阵乘法。

---

阶段5：

实现

BlockPartitioner

二维分块。

---

阶段6：

实现

PageRank

验证框架可扩展性。
