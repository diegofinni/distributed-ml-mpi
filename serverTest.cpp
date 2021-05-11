#include <iostream>
#include <mpi.h>
#include "master_node.hpp"
#include "worker_node.hpp"

int proc_rank, num_procs;

int main(int argc, char* argv[]) {

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    vector<double> init_params(10, 0);
    double lr = 0.01;
    int num_epochs = 2;
    int n_bound = 0;

    if (!proc_rank) {
        init_master_node(init_params, 
            num_procs - 1,
            num_epochs,
            n_bound,
            lr);
        manage_workers();
    }
    else {
        work(init_params, num_epochs, proc_rank);
    }
    MPI_Finalize();
    return 0;
}