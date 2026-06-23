# 第一层对比：朴素串行 PageRank

这是最重要的基线。

## Baseline 0

单线程 PageRank

```cpp
for(iter)
{
    for(i)
    {
        for(j)
        {
            rank_new[i] +=
                A[i][j] * rank[j];
        }
    }
}
```

特点：

```text
Dense Matrix
单线程
无OpenMP
无MPI
```

复杂度：

[
O(N^2)
]

---

## Baseline 1

CSR + 单线程

即：

```text
CSR × Vector
1 MPI
1 Thread
```

这是你当前系统关闭并行后的版本。

复杂度：

[
O(nnz)
]

---

这样就能得到：

| 实现           | 时间  |
| ------------ | --- |
| Dense Serial | xxx |
| CSR Serial   | xxx |

用于证明：

```text
CSR带来的收益
```

---

# 第二层对比：MPI带来的收益

固定：

```text
OMP = 1
```

比较：

| 实现          | MPI | OMP |
| ----------- | --- | --- |
| CSR Serial  | 1   | 1   |
| MPI Version | 2   | 1   |
| MPI Version | 4   | 1   |
| MPI Version | 8   | 1   |

得到：

[
Speedup_{MPI}
=============

\frac{T_{serial}}{T_{MPI}}
]

这是证明：

```text
MPI值不值得
```

---

# 第三层对比：OpenMP带来的收益

固定：

```text
MPI=1
```

比较：

| 实现         | MPI | OMP |
| ---------- | --- | --- |
| CSR Serial | 1   | 1   |
| OMP        | 1   | 2   |
| OMP        | 1   | 4   |
| OMP        | 1   | 8   |

得到：

[
Speedup_{OMP}
=============

\frac{T_1}{T_t}
]

证明：

```text
线程并行收益
```

---

# 第四层对比：MPI vs OpenMP

这个实验很有意思。

假设总共有：

```text
8 CPU Core
```

比较：

| 方案          | 配置  |
| ----------- | --- |
| Pure MPI    | 8×1 |
| Pure OpenMP | 1×8 |
| Hybrid      | 4×2 |
| Hybrid      | 2×4 |

很多课程项目都不会做这个。

但如果做出来会很有亮点。

---

# 第五层对比：工业级库

如果你时间充裕，我强烈推荐加这一项。



## Baseline 3

SciPy Sparse

[SciPy官网](https://scipy.org?utm_source=chatgpt.com)

实现：

```python
scipy.sparse
```

内部也是 CSR。

比较：

```text
SpMV性能
```

---

这个基线价值很高。

因为：

```text
SciPy
=
成熟工业实现
```

如果你的结果接近它：

```text
说明你的实现质量不错
```

