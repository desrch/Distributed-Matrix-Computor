#pragma once

#include "SpMVEngine.hpp"

#include <chrono>

namespace dmc {

class DenseSerialEngine : public ISpMVEngine {
public:
    std::string name() const override { return "DenseSerial"; }

    std::vector<double> multiply(const CSRMatrix& A,
                                 const std::vector<double>& x) override
    {
        DenseMatrix Ad = A.toDense();
        int M = Ad.rows();
        int N = Ad.cols();

        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<double> y(M, 0.0);
        for (int i = 0; i < M; ++i) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j)
                sum += Ad(i, j) * x[j];
            y[i] = sum;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        timing_.comp_sec += std::chrono::duration<double>(t1 - t0).count();

        return y;
    }
};

} // namespace dmc
