#include "worker_node.hpp"
#include <stdlib.h>


void update_params(vector<double>& params) {
    // stupid
}

void work(vector<double>& params, int num_epoch, int rank) {
    double* recv_buf = (double*) malloc(10 * sizeof(double));
    for (int i = 0; i < num_epoch; i++) {
        // Placeholder line here
        fill(params.begin(), params.end(), (i + 1) * rank);
        MPI_Send(&params.front(), params.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        if (i < num_epoch - 1) MPI_Recv(&params.front(), params.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}