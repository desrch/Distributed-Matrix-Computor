# 第一部分：实验环境（Experimental Setup）

这一部分一般一张表即可。

包括：

| 内容         | 配置                |
| ---------- | ----------------- |
| CPU        | 例如 Intel Xeon ... |
| 节点数        | xx                |
| MPI        | OpenMPI 4.x       |
| OpenMP     | GCC 11            |
| OS         | Ubuntu 22.04      |
| PageRank参数 | d=0.85, tol=1e-6  |
| 数据集        | 四个SNAP数据集         |

然后再放一个数据集统计表。

| Dataset       | Nodes | Edges | Density |
| ------------- | ----: | ----: | ------: |
| soc-Epinions1 |   75k |  508k |     ... |
| web-Stanford  |  281k |  2.3M |     ... |
| web-Google    |  875k |  5.1M |     ... |
| LiveJournal   |  4.8M |   68M |     ... |

这一张表能够告诉读者：

> 为什么这些数据集越来越大。

---

# 第二部分：Correctness

在本地对四个数据集做pagerank，设置OMP线程数为2 ，MPI进程数为2，对比通过networkx算出的pagerank结果，统计误差，从而说明正确性

| Dataset     | Iterations | diff    |
| ----------- | ---------: | --------- |
| Epinions    |          |          |
| Stanford    |          |          |
| Google      |          |          |
| LiveJournal |          |          |



---

# 第三部分：Scalability（最重要）

这里建议画第一张图。

## 图1

> 不同数据集在最佳配置下的运行时间

X轴：

```text
Dataset
```

Y轴：

```text
Execution Time
```

展示：

```text
Epinions

Stanford

Google

LiveJournal
```

这一张图回答：

> 图规模增加后系统还能否工作。

---

然后第二张图：

## 图2

SpMV Time 占 Total Time 的比例

对于四个数据集：

```
█████ Compute
██ Communication
```

画成：

Stacked Bar

因为：

PageRank本质就是SpMV。

如果：

```
SpMV占80%以上
```

说明：

你的优化方向是正确的。

---

# 第四部分：MPI扩展性

这是论文最核心的图。

建议：

固定：

```
OMP=1
```

对于每一个数据集：

画：

```
MPI=1

MPI=2

MPI=4

MPI=8
```

Y轴：

```
Execution Time
```

最好画：

Line Chart

四条线：

```
Epinions

Stanford

Google

LiveJournal
```

回答：

> MPI是否有效。

---

然后计算：

Speedup：

[
S(p)=\frac{T_1}{T_p}
]

再画：

## 图4

MPI Speedup

横轴：

```
MPI Processes
```

纵轴：

```
Speedup
```

最好再画一条：

Ideal Speedup

```
1

2

4

8
```

作为虚线。

这样老师一眼就知道：

离理想还有多少。

---

# 第五部分：OpenMP扩展

固定：

```
MPI=1
```

画：

```
OMP=1

OMP=2

OMP=4
```

同样：

Execution Time

然后：

OpenMP Speedup

这一部分图与MPI保持一致。

---

# 第六部分：Hybrid分析

这是我认为最值得写的。

你的实验其实已经覆盖：

```
MPI=1

2

4

8
```

×

```
OMP=1

2

4
```

共：

12组。

完全可以画：

Heatmap（热力图）

横轴：

```
OMP Threads
```

纵轴：

```
MPI Processes
```

颜色：

```
Execution Time
```

例如：

```
      OMP

      1   2   4

MPI

1    12  10   9

2     8   7   6

4     6   5   5

8     5   5   6
```

颜色越深：

越快。

这一张图的信息量远远超过12张柱状图。

然后讨论：

> 为什么8×4反而没有最快？

通常：

因为：

* MPI通信增加
* OpenMP线程竞争
* NUMA
* Cache

这是Discussion可以写的。

---

# 第七部分：Communication Analysis

你已经统计了：

```
Communication Time
```

千万不要浪费。

画：

Stacked Bar：

```
Compute

Communication
```

不同MPI数量。

例如：

```
MPI=1

MPI=2

MPI=4

MPI=8
```

观察：

```
Communication Ratio
```

随着MPI增加：

通信比例会上升。

这是非常典型的HPC分析。

---

# 第八部分：Memory

最后：

画：

Memory Usage

横轴：

四个数据集。

纵轴：

MB

证明：

CSR可以支持：

```
LiveJournal
```

否则：

Dense根本放不下。

---



# 结果分析

实验结果分析 — 完成 ✅
输出文件

analysis/
├── fig1_scalability.png       — 扩展性: 最佳单节点时间 + 耗时分解
├── fig2_mpi_speedup.png       — MPI 强扩展: 执行时间 + Speedup
├── fig3_omp_speedup.png       — OpenMP 强扩展: 执行时间 + Speedup
├── fig4_hybrid_heatmaps.png   — 4× Hybrid 热力图 (MPI×OMP×Time)
├── fig5_communication.png     — 通信占比堆叠柱状图
├── fig6_memory.png            — 内存占用对比

Part 1: 环境与数据
Dataset	Nodes	Edges	nnz	Density
soc-Epinions1	75,888	508,837	508,837	0.0088%
web-Stanford	281,903	2,312,497	2,312,497	0.0029%
web-Google	916,428	5,105,039	5,105,039	0.0006%
soc-LiveJournal1	4,847,571	68,993,773	68,993,773	0.0003%
Part 2: 正确性验证 (Local OMP=2, MPI=2)

Dataset	Iter	L1 vs scipy	Top10
Epinions	54	1.47×10⁻¹⁵	10/10
Stanford	63	4.46×10⁻¹⁵	10/10
Google	62	2.79×10⁻¹⁴	10/10
LiveJournal	51	2.37×10⁻¹⁴	10/10

全部四个数据集 L1 误差在机器精度级别 (10⁻¹⁴~10⁻¹⁵)，Top-10 节点完全匹配。

Part 3-4: 扩展性
Best Config	Total	Speedup
Epinions: Hybrid_p2_t4	0.17s	—
Stanford: OMP_CSR_t2	1.20s	—
Google: OMP_CSR_t4	3.11s	—
LiveJournal: CSR_Serial	29.4s	—
小数据集 (Epinions/Stanford) 最佳策略是单机 OMP。对于 LiveJournal (4.8M 节点)，单机串行 CSR 反而比 MPI 快 — MPI 通信开销超过了并行收益。

Parts 5-8: 分析要点
OMP 在 Google 上获得 1.12x speedup (t=4)，小数据集几乎无加速（线程开销）
MPI p=2 在较大数据集上有 1.1~1.2x，但 p≥4 后通信占主导 (>50%) 导致负加速
内存：CSR 只需 46MB (Epinions) 到 954MB (LiveJournal)，而 Dense 等价存储需要 45GB~184TB
通信占比：随 MPI 进程数增长，Epinions 从 9% 升至 80%+，LiveJournal 从 5% 升至 60%+
