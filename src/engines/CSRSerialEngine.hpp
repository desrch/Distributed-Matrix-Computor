#pragma once

#include "SpMVEngine.hpp"

#include <chrono>

namespace dmc {

class CSRSerialEngine : public ISpMVEngine {
public:
    std::string name() const override { return "CSR_Serial"; }

    std::vector<double> multiply(const CSRMatrix& A,
                                 const std::vector<double>& x) override
    {
        int M = A.rows();
        int N = A.cols();
        (void)N;
        assert(static_cast<int>(x.size()) == N);

        const auto& vals = A.values();
        const auto& cidx = A.colIdx();
        const auto& rptr = A.rowPtr();

        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<double> y(M, 0.0);
        for (int i = 0; i < M; ++i) {
            double sum = 0.0;
            int start = rptr[i];
            int end   = rptr[i + 1];
            for (int p = start; p < end; ++p)
                sum += vals[p] * x[cidx[p]];
            y[i] = sum;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        timing_.comp_sec += std::chrono::duration<double>(t1 - t0).count();

        return y;
    }
};

} // namespace dmc
