
typedef void* (*ReduceFunction)(void* dst, void*, int);

int rank, numProc;

ReduceFunction intSum, floatSum;

void ringAllReduce(int *params, int N, ReduceFunction f);
