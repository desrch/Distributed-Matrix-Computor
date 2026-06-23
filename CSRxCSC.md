


### 第一阶段：实现CSCMatrix

先实现：

```cpp
CSCMatrix CSRMatrix::toCSC();
```

于是：

```text
A -> CSR
B -> CSC
```

之后：

```text
A 的行
B 的列
```

都能高效访问。

很多工业实现都是这么做的。

---

# 第二阶段：实现朴素 CSR × CSC

先不要考虑最优性能。

目标：

```text
正确
```

即可。

---

## 数据结构

结果矩阵：

```cpp
vector<double> values;
vector<int> col_idx;
vector<int> row_ptr;
```

计算流程：

```cpp
for each row i in A
{
    for each column j in B
    {
        double sum = dot(
            A.row(i),
            B.col(j)
        );

        if(sum != 0)
        {
            store(sum);
        }
    }
}
```

复杂度比较高：

```text
O(rows * cols)
```

但很好验证。

---

# 第三阶段：实现真正的 SpGEMM

Sparse General Matrix Multiplication。

这是项目最重要的部分。

---

思路：

对于 A 的每个非零元：

```text
A(i,k)
```

只访问：

```text
B(k,*)
```

而不是整列扫描。

---

伪代码：

```cpp
for row i in A
{
    unordered_map<int,double> acc;

    for each nonzero (i,k) in A
    {
        for each nonzero (k,j) in B
        {
            acc[j] +=
                A(i,k) * B(k,j);
        }
    }

    flush(acc);
}
```

这就是现代 SpGEMM 的核心思想。

复杂度接近：

```text
仅与非零元有关
```

而不是：

```text
n³
```

---

# 第四阶段：设计独立算子层

不要把乘法写进 CSRMatrix。

推荐：

```cpp
class SparseMultiply
{
public:

    static CSRMatrix multiply(
        const CSRMatrix& A,
        const CSRMatrix& B
    );
};
```

这样未来：

```text
CSR × CSR
CSR × Dense
CSC × CSR
```

都能统一管理。

---

# 第五阶段：实现 CSR × Dense

这一步很容易被忽略。

但你的 README 明确要求：

```text
支持稀疏/稠密矩阵乘法
```

因此需要：

```cpp
DenseMatrix multiply(
    const CSRMatrix& A,
    const DenseMatrix& B
);
```

---

这是你现阶段最值得实现的模块。

因为：

```text
PageRank
Graph Embedding
GNN
```

很多时候都是：

```text
Sparse × Dense
```

而不是：

```text
Sparse × Sparse
```

---

# 第六阶段：验证体系

不要等实现完再测试。

建议建立：

```text
tests/

test_csr_vector.cpp
test_csr_dense.cpp
test_csr_csr.cpp
```

每个测试都用：

```text
Dense版本结果
=
Sparse版本结果
```

做对拍。

例如：

```cpp
DenseMatrix dense_result =
    denseMultiply(A_dense,B_dense);

CSRMatrix sparse_result =
    sparseMultiply(A_csr,B_csr);

assert(equal(...));
```

