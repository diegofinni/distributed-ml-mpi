
typedef void(*ReduceFunction)(double* dst, const double* src, int n);

extern ReduceFunction sum_reduce;

void reduce(double *params, int N, ReduceFunction f);

void init_mpi_env(int rank, int num_procs, int stale_bound);