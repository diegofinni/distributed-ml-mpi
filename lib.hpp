typedef void(*ReduceFunction)(double* dst, const double* src, int n);

extern ReduceFunction sum_reduce;