#include <vector>

using namespace std;

typedef void(*ReduceFunction)(double* dst, const double* src, int n);

extern ReduceFunction sum_reduce;

void reduce(vector<double>& params, int N, ReduceFunction f);

void init_mpi_env(int rank, int num_procs);