#include "lib.hpp"

void sum_func(double *dst, const double *src, int n) {
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

ReduceFunction sum_reduce = &sum_func;