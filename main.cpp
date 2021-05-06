#include <cstdio>
#include <cstdlib>
#include <vector>

#include <mpi.h>
#include "decentralized.hpp"

#define error_exit(fmt, ...) do { fprintf(stderr, "%s error: " fmt, __func__, ##__VA_ARGS__); exit(1); } while(0);

int proc_rank, num_procs;

int main(int argc, char* argv[]) {
    // Grab command line arguments
    int N = atoi(argv[1]);
    int mode = atoi(argv[2]); // 0 = decentralized, 1 = centralized

    // Check if arguments are valid
    if (N < 1 && mode == 0) {
        error_exit("Too few nodes for decentralized mode (At least 1 needed\n");
    }
    else if (N < 2 && mode == 1) {
        error_exit("Too few nodes for centralized mode (At least 2 needed\n");
    }

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // If there are less parameters than processes, we cannot proceed
    if (num_procs > N && mode == 0) {
        MPI_Finalize();
        error_exit("Less parameters than there are processes (not allowed in decentralized mode)\n");
    }

    // Placeholder until we get real data
    std::vector<double> params (N, 0);
    std::fill(params.begin(), params.end(), proc_rank + 1);

    // Init MPI comm data structures and set reduce function
    init_mpi_env(proc_rank, num_procs);
    ReduceFunction f = sum_reduce;
    reduce(params, N, f);
    
    // Print results
    for (int i = 0; i < N; i++) {
        printf("(%d) %.3f ", proc_rank, params[i]);
    }
    printf("\n");
    
    // Exit program
    MPI_Finalize();
    return 0;
}