
typedef void(*ReduceFunction)(double* dst, const double* src, int n);

extern int rank, numProc;

extern ReduceFunction sumReduce;

void ringAllReduce(double *params, int N, ReduceFunction f);
