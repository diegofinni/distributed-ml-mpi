#include <cstdio>
#include <cstdlib>

#include <mpi.h>
#include "dml.hpp"

#define error_exit(fmt, ...) do { fprintf(stderr, "%s error: " fmt, __func__, ##__VA_ARGS__); exit(1); } while(0);

int rank, num_procs;

int main(int argc, char* argv[]) {
    // Grab command line arguments
    int N = atoi(argv[1]);
    int bound = atoi(argv[2]);

    // Check if arguments are valid
    if (N < 1 || bound < 0) {
        error_exit("Correct arguments not provided (int N, int bound)\n");
    }

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // If there are less parameters than processes, we cannot proceed
    if (num_procs > N) {
        MPI_Finalize();
        error_exit("Less parameters than there are processes\n");
    }

    // Placeholder until we get real data
    auto *params = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) params[i] = rank + 1;

    // Init MPI comm data structures and set reduce function
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