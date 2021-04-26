
typedef void(*ReduceFunction)(double* dst, double*, int);

extern int rank, numProc;

extern ReduceFunction sumReduce;

void ringAllReduce(double *params, int N, ReduceFunction f);
