#include <cstdio>
#include <cstdlib>

#include <mpi.h>
#include "dml.hpp"

int rank, num_procs;

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int N = atoi(argv[1]);
    int bound = atoi(argv[2]);

    // Get rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // If there are less params to share than there are processes we
    // shut down any extra processes and adjust numProc accordingly
    if (num_procs > N) {
        num_procs = N;
        if (rank >= N) {
            MPI_Finalize();
            return 0;
        }
    }

    // Placeholder until we get real data
    auto *params = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) params[i] = rank + 1;

    // Init MPI env and set reduce function
    init_mpi_env(rank, num_procs, bound);
    ReduceFunction f = sum_reduce;
    reduce(params, N, f);

    // Print results
    for (int i = 0; i < N; i++) {
        printf("(%d) %.3f ", rank, params[i]);
    }
    printf("\n");
    
    // Exit program
    MPI_Finalize();
    return 0;
}