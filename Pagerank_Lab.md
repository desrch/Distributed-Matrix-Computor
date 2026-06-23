# 一、PageRank实验总体设计

建议设计三个层次的实验：

```text
Experiment 1
Correctness Verification

Experiment 2
Scalability Evaluation

Experiment 3
Parallel Efficiency Evaluation
```

---

# 二、PageRank实现参数

统一采用：

[
r_{k+1}
=======

dMr_k
+
\frac{1-d}{N}\mathbf{1}
]

其中：

* damping factor = 0.85
* max_iter = 100
* tolerance = 1e-6

推荐固定：

```cpp
damping = 0.85
max_iter = 100
tol = 1e-6
```

避免实验变量过多。

---

# 三、实验数据集

不要自己随机生成图。

老师通常更认可真实图数据。

推荐使用：

SNAP

提供的公开数据集。

---

## Dataset A

web-Stanford

```text
节点数：
281903

边数：
2312497
```

特点：

```text
规模适中
适合作为基准实验
```

---

## Dataset B

web-Google

```text
节点数：
875713

边数：
5105039
```

特点：

```text
百万级节点
稀疏矩阵规模较大
```

---

## Dataset C

soc-Epinions1

```text
节点数：
75879

边数：
508837
```

特点：

```text
规模较小
用于正确性验证
```

---

## Dataset D（可选）

com-LiveJournal

```text
节点数：
4.8M

边数：
68M
```

特点：

```text
真正的大规模图
用于展示扩展性
```

如果集群资源有限可以不做。

---

# 四、实验一：正确性验证

目标：

```text
验证PageRank实现正确
```

---

选用：

```text
soc-Epinions1
```

或者：

```text
100节点手工图
```

---

比较对象：

你的实现

VS

NetworkX PageRank

可以使用：

[NetworkX官方文档](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html?utm_source=chatgpt.com)

---

比较指标：

```text
Top-10 PageRank节点
```

以及：

```text
L1 Error
```

[
||r_{ours}-r_{ref}||_1
]

---

# 五、实验二：规模扩展实验

目标：

```text
图越大
耗时如何变化
```

---

固定：

```text
MPI = 1
OpenMP = 1
```

即单进程单线程。

---

测试：

```text
100K节点

300K节点

800K节点

1M节点
```

---

记录：

```text
构图时间

CSR构建时间

PageRank时间

总时间
```

---

绘图：

```text
X轴
Graph Size

Y轴
Execution Time
```

---

# 六、实验三：MPI扩展性

最重要的实验。

---

固定：

```text
web-Google
```

---

OpenMP固定：

```text
1 Thread
```

---

MPI进程数：

```text
1

2

4

8

16
```

---

记录：

```text
总执行时间
```

---

计算：

Speedup

[
S(p)
====

\frac{T_1}{T_p}
]

---

计算：

Parallel Efficiency

[
E(p)
====

\frac{S(p)}{p}
]

---

推荐画图：

```text
MPI Processes
vs
Speedup
```

---

# 七、实验四：OpenMP扩展性

固定：

```text
MPI = 1
```

---

线程数：

```text
1

2

4

8

16
```

---

记录：

```text
PageRank Time
```

---

计算：

[
S(t)
====

\frac{T_1}{T_t}
]

---

观察：

```text
线程扩展能力
```

---

# 八、实验五：混合并行

这是最能体现系统价值的实验。

---

例如：

总核心数16

比较：

方案A

```text
MPI=16
OMP=1
```

方案B

```text
MPI=8
OMP=2
```

方案C

```text
MPI=4
OMP=4
```

方案D

```text
MPI=2
OMP=8
```

方案E

```text
MPI=1
OMP=16
```

---

观察：

```text
哪种配置最好
```

通常：

```text
纯MPI
通信开销高

纯OMP
内存压力大

Hybrid
最佳
```

---

# 九、采集通信开销

这一项非常加分。

PageRank每轮迭代都会：

```text
SpMV

↓

AllGather
或
AllReduce

↓

同步Rank向量
```

因此记录：

```cpp
comm_time

compute_time
```

---

最终统计：

```text
Computation %

Communication %
```

例如：

```text
Compute 82%

Communication 18%
```

---

# 十、最终论文/报告建议展示的指标

至少统计：

| 指标                 | 说明        |
| ------------------ | --------- |
| Graph Nodes        | 节点数       |
| Graph Edges        | 边数        |
| nnz                | CSR非零元数   |
| Iterations         | 收敛迭代次数    |
| Total Time         | 总时间       |
| SpMV Time          | 稀疏矩阵向量乘时间 |
| Communication Time | MPI通信时间   |
| Speedup            | 加速比       |
| Efficiency         | 并行效率      |
| Memory Usage       | 内存占用      |

