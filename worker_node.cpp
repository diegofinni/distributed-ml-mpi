#include "worker_node.hpp"
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

#include <mpi.h>
#include "decentralized.hpp"
#include "lr.hpp"
#include <string>
#include <chrono>

// Timing 
using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::duration;
using chrono::milliseconds;
int num_epoch;

void update_params(vector<double>& params, int iter) {
    MPI_Send(&params.front(), params.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    if (iter < num_epoch - 1) MPI_Recv(&params.front(), params.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Nang, replace this code with lr stuff
// Outer for loop runs num_epoch times, call update_params after each iteration
// Rank variable is unecessary
void work(vector<double>& params, vector<vector<double> > data_shard, int num_epoch, int rank, int num_workers, string infile) {
    num_epoch = num_epoch;
    
    int m = data_shard.size();

    double* recv_buf = (double*) malloc(theta.size() * sizeof(double));
    for (int i = 0; i < num_epoch; i++) {
        // Placeholder line here
        train(data_shard, 1);
        
        update_params(gradient, i); // This is a lil weird variable-name wise on the master node side but we want to send over the gradient
        reset_gradient();
    }


}