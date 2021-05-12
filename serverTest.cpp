#include <iostream>
#include <stdlib.h>
#include <mpi.h>
#include "master_node.hpp"
#include "worker_node.hpp"


int main(int argc, char* argv[]) {

    int num_epochs = atoi(argv[1]);
    double lr = atoi(argv[2]);
    int n_bound = atoi(argv[3]);

    // Initialize MPI environment
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    vector<double> init_params(10, 0);
    if (!rank) {
        init_master_node(init_params, 
            num_procs - 1,
            num_epochs,
            n_bound,
            lr);
        manage_workers();
    }
    else {
        // Nang: Place your lr code here
        work(init_params, num_epochs, rank);
    }
    MPI_Finalize();
    return 0;
}