#include "worker_node.hpp"
#include <stdlib.h>

int num_epoch;

void update_params(vector<double>& params, int iter) {
    MPI_Send(&params.front(), params.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    if (iter < num_epoch - 1) MPI_Recv(&params.front(), params.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Nang, replace this code with lr stuff
// Outer for loop runs num_epoch times, call update_params after each iteration
// Rank variable is unecessary
void work(vector<double>& params, int num_epoch, int rank) {
    num_epoch = num_epoch;
    double* recv_buf = (double*) malloc(10 * sizeof(double));
    for (int i = 0; i < num_epoch; i++) {
        // Placeholder line here
        fill(params.begin(), params.end(), (i + 1) * rank);
        update_params(params, i);
    }
}