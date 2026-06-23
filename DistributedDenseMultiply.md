## DistributedDenseMultiply
按照工程收益最大化原则，下一步应该尽快实现一个 **End-to-End MVP（Minimum Viable Product）**：

```text
Dense Matrix
    ↓
Row Partition
    ↓
MPI Scatter
    ↓
Local Multiply
    ↓
MPI Gather
    ↓
Result Matrix
```

实现：
* 分布式运行
* 数据分发
* 任务划分
* 结果聚合
等功能
后面的 CSR、CSC、Block Partition 本质上都只是替换其中的某几个模块。

---

## Step 1：实现 RowPartitioner

这是最简单也是最重要的模块。

定义：

```cpp
struct PartitionInfo
{
    int start_row;
    int end_row;
};
```

接口：

```cpp
class RowPartitioner
{
public:
    static PartitionInfo partition(
        int total_rows,
        int rank,
        int world_size
    );
};
```

例如：

```text
10000 rows
4 processes
```

划分结果：

```text
Rank0 : 0~2499

Rank1 : 2500~4999

Rank2 : 5000~7499

Rank3 : 7500~9999
```

---

## Step 2：实现 MPIWrapper

不要在业务逻辑里直接写 MPI。

封装：

```cpp
class MPIWrapper
{
public:

    static void broadcast(...);

    static void scatterRows(...);

    static void gatherRows(...);
};
```

后面换通信策略时非常方便。

---

## Step 3：实现 DenseMultiply

先做最简单版本：

```cpp
DenseMatrix multiply(
    const DenseMatrix& A,
    const DenseMatrix& B
);
```

经典三重循环：

```cpp
for(i)
    for(j)
        for(k)
```

先保证正确性。

优化以后再说。

---

## Step 4：实现 DistributedDenseMultiply

这是第一个真正的分布式算法。

流程：

```text
Master

A
↓
按行切分

Rank0
Rank1
Rank2
Rank3

B
↓
Broadcast

所有节点

局部计算

Ci = Ai × B

Gather

C
```

通信过程：

```text
MPI_Scatter
MPI_Bcast
MPI_Gather
```



## 开发完成后的验证

准备：

```text
A(8×8)
B(8×8)
```

启动：

```bash
mpirun -np 4 ./matrix_app
```

验证：

```text
Distributed Result
=
Single-thread Result
```

完全一致。

