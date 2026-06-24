#pragma once

#include "SpMVEngine.hpp"

#include <cassert>
#include <chrono>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace dmc {

class OpenMPCSREngine : public ISpMVEngine {
public:
    std::string name() const override {
#if defined(_OPENMP)
        return "OMP_CSR_t" + std::to_string(omp_get_max_threads());
#else
        return "OMP_CSR_t1";
#endif
    }

    std::vector<double> multiply(const CSRMatrix& A,
                                 const std::vector<double>& x) override
    {
        int M = A.rows();
        int N = A.cols();
        assert(static_cast<int>(x.size()) == N);
        (void)N;

        const auto& vals = A.values();
        const auto& cidx = A.colIdx();
        const auto& rptr = A.rowPtr();

        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<double> y(M, 0.0);

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
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
