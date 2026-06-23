# 建议实现顺序

## Step1

CSRMatrix

```cpp
class CSRMatrix : public Matrix
{
private:

    int rows_;
    int cols_;

    std::vector<double> values;
    std::vector<int> col_idx;
    std::vector<int> row_ptr;
};
```

实现：

```cpp
get(i,j)
fromDense()
toDense()
```

---

## Step2

CSR × Vector

即：

[
y = Ax
]

这是最重要的稀疏运算。

复杂度：

```text
O(nnz)
```

而不是：

```text
O(n²)
```

---

## Step3

Distributed CSR × Vector

也就是：

```text
A
↓
按行分块

Rank0
Rank1
Rank2
Rank3

局部计算

yi = Ai * x

Gather

y
```

