#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"
#include "dml.h"

#define SYSEXPECT(expr) do { if(!(expr)) { perror(__func__); exit(1); } } while(0)
#define error_exit(fmt, ...) do { fprintf(stderr, "%s error: " fmt, __func__, ##__VA_ARGS__); exit(1); } while(0);

int rank, numProc;

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int N = atoi(argv[1]);

    // Get rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);

    // If there are less params to share than there are processes we
    // shut down any extra processes and adjust numProc accordingly
    if (numProc > N) {
        numProc = N;
        if (rank >= N) {
            MPI_Finalize();
            return 0;
        }
    }

    // Placeholder until we get real data
    double *params = (double*)malloc(N * sizeof(double));
    ReduceFunction f = sumReduce;
    for (int i = 0; i < N; i++) params[i] = rank + 1;
    ringAllReduce(params, N, f);

    // Print results
    for (int i = 0; i < N; i++) {
        printf("(%d) %.3f ", rank, params[i]);
    }
    printf("\n");

    // Exit program
    MPI_Finalize();
    return 0;
}